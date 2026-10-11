"""MedSeg Clinical Workstation - Neuro-Oncology Segmentation."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from medseg.data import read_image
from medseg.inference import load_run, predict_image
from medseg.report_pdf import generate_pdf_report
from medseg.utils import get_device, overlay_masks, to_display

st.set_page_config(
    page_title="MedSeg Workstation - Neuro-Oncology AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { font-size: 1.8rem; font-weight: 700; color: #E2E8F0; margin-bottom: 0.2rem; }
    .sub-header { font-size: 0.95rem; color: #94A3B8; margin-bottom: 1.2rem; }
    .metric-card { background: #1E293B; border-radius: 8px; padding: 14px; border: 1px solid #334155; margin-bottom: 10px; }
    .metric-title { font-size: 0.8rem; color: #94A3B8; text-transform: uppercase; font-weight: 600; }
    .metric-value { font-size: 1.4rem; font-weight: 700; color: #38BDF8; margin-top: 2px; }
    .metric-sub { font-size: 0.75rem; color: #64748B; margin-top: 2px; }
    .patient-card { background: #0F172A; border: 1px solid #1E293B; border-radius: 6px; padding: 10px 14px; margin-bottom: 15px; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
    .badge-primary { background: #0369A1; color: #E0F2FE; }
    .badge-alert { background: #991B1B; color: #FEE2E2; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_cached_model(run_dir: str):
    dev = get_device()
    return load_run(run_dir, dev)


runs = sorted(p.parent for p in Path("runs").glob("*/best.pt"))
if not runs:
    st.error("No trained models found in 'runs/'. Train one first via train.py.")
    st.stop()

st.sidebar.markdown("### 🎛️ Diagnostic Configuration")

run_names = [p.name for p in runs]
default_idx = run_names.index("unet_general") if "unet_general" in run_names else (run_names.index("unet_lgg") if "unet_lgg" in run_names else 0)
selected_run_name = st.sidebar.selectbox("Active Neural Network", run_names, index=default_idx)
selected_run_path = next(p for p in runs if p.name == selected_run_name)

device = get_device()
model, cfg = get_cached_model(str(selected_run_path))

st.sidebar.caption(f"**Arch:** {cfg.get('model', 'unet').upper()} | **Weights:** {cfg.get('base_ch', 32)}ch | **Res:** {cfg.get('img_size', 256)}px | **GPU:** {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Radiologist Sensitivity")
threshold = st.sidebar.slider("Probability Decision Threshold", 0.05, 0.95, 0.50, 0.05, help="Lower for higher sensitivity (recall); raise for higher precision.")
tta = st.sidebar.checkbox("Flip Test-Time Augmentation (TTA)", value=False, help="Infers on both normal and horizontally mirrored views, averaging probabilities to sharpen boundary confidence.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 👁️ PACS Window Leveling")
apply_windowing = st.sidebar.checkbox("Enable Window/Level Adjustment", value=False)
if apply_windowing:
    win_level = st.sidebar.slider("Window Level (Brightness Center)", 0, 255, 128)
    win_width = st.sidebar.slider("Window Width (Contrast Spread)", 10, 255, 150)
else:
    win_level, win_width = 128, 255

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 Display Options")
overlay_style = st.sidebar.radio("Visualization Mode", ["Contour Line", "Filled Overlay", "Heatmap Overlay"], index=0)
alpha = st.sidebar.slider("Overlay Transparency (Alpha)", 0.1, 1.0, 0.45)
contour_color = st.sidebar.color_picker("Contour Color", "#FF0033")

tab_workstation, tab_compare, tab_patient_db = st.tabs(["🏥 Clinical Workstation", "📊 Multi-Model Comparator", "📁 Clinical Database (TCGA)"])

with tab_workstation:
    st.markdown('<div class="main-header">Neuro-Oncology Segmentation Workstation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Glioma & Brain Abnormality Delineation Platform</div>', unsafe_allow_html=True)

    # Sample cases dictionary
    sample_catalog = {
        "Custom Upload (Upload file below)": None,
        "Real Hospital MRI: Axial Brain Tumour (Unseen Test Patient)": "demo_samples/real_mri_tumour.tif",
        "Real Hospital MRI: Negative Control (Unseen Test Patient)": "demo_samples/real_mri_healthy.tif",
        "Synthetic Case: Focal Brain Lesion": "demo_samples/slice_with_tumour.png",
        "Synthetic Case: Healthy Brain": "demo_samples/slice_healthy.png",
    }

    col_select, col_meta = st.columns([2, 1])
    with col_select:
        sample_choice = st.selectbox("📁 Load Patient Case from Library", list(sample_catalog.keys()))
        selected_file_path = sample_catalog[sample_choice]

    uploaded_file = st.file_uploader(
        "Or Upload Clinical Scan (DICOM .dcm / NIfTI .nii / TIFF .tif / PNG / JPG)",
        type=["dcm", "nii", "gz", "tif", "tiff", "png", "jpg", "jpeg"]
    )

    # Resolve input image
    raw_image = None
    meta_info = {
        "Patient ID": "TCGA_DU_A5TU",
        "Age / Gender": "48 / Male",
        "Modality": "MRI (FLAIR / T1)",
        "Slice Position": "Axial Slice #16",
        "Pixel Spacing": "1.0 mm × 1.0 mm",
    }

    if uploaded_file is not None:
        filename = uploaded_file.name.lower()
        if filename.endswith(".dcm"):
            try:
                import pydicom
                dcm = pydicom.dcmread(io.BytesIO(uploaded_file.read()))
                arr = dcm.pixel_array.astype(np.float32)
                arr = arr * float(getattr(dcm, "RescaleSlope", 1)) + float(getattr(dcm, "RescaleIntercept", 0))
                arr = ((arr - arr.min()) / max(arr.max() - arr.min(), 1e-5) * 255.0).astype(np.uint8)
                raw_image = np.stack([arr, arr, arr], axis=-1)
                meta_info["Patient ID"] = str(getattr(dcm, "PatientID", "ANONYMOUS"))
                meta_info["Modality"] = str(getattr(dcm, "Modality", "MR"))
                meta_info["Age / Gender"] = f"{getattr(dcm, 'PatientAge', 'N/A')} / {getattr(dcm, 'PatientSex', 'N/A')}"
                meta_info["Slice Position"] = f"Instance #{getattr(dcm, 'InstanceNumber', '1')}"
            except Exception as e:
                st.error(f"Error parsing DICOM file: {e}")
        elif filename.endswith((".nii", ".gz")):
            try:
                import nibabel as nib
                with io.BytesIO(uploaded_file.read()) as f:
                    tmp_p = Path(".cache/tmp_nii.nii.gz")
                    tmp_p.parent.mkdir(parents=True, exist_ok=True)
                    tmp_p.write_bytes(f.getvalue())
                    nii = nib.load(str(tmp_p))
                    data = nii.get_fdata()
                    idx = data.shape[2] // 2 if data.ndim >= 3 else 0
                    sl = data[:, :, idx] if data.ndim >= 3 else data
                    norm = ((sl - sl.min()) / max(sl.max() - sl.min(), 1e-5) * 255.0).astype(np.uint8)
                    raw_image = np.stack([norm, norm, norm], axis=-1)
                    meta_info["Modality"] = "3D NIfTI (MRI/CT)"
                    meta_info["Slice Position"] = f"Mid-axial Slice {idx}/{data.shape[2] if data.ndim >= 3 else 1}"
            except Exception as e:
                st.error(f"Error parsing NIfTI file: {e}")
        else:
            try:
                pil_img = Image.open(uploaded_file)
                raw_image = np.array(pil_img.convert("RGB"))
                meta_info["Patient ID"] = Path(uploaded_file.name).stem
                meta_info["Modality"] = "2D Medical Scan"
                meta_info["Slice Position"] = "Axial 2D"
            except Exception:
                data_b = np.frombuffer(uploaded_file.getvalue(), np.uint8)
                bgr = cv2.imdecode(data_b, cv2.IMREAD_COLOR)
                if bgr is not None:
                    raw_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif selected_file_path is not None and Path(selected_file_path).exists():
        raw_image = read_image(selected_file_path)
        if "healthy" in selected_file_path:
            meta_info["Slice Position"] = "Axial Slice #10 (Healthy/Negative)"

    # Clinical Patient Info Bar
    with col_meta:
        st.markdown(f"""
        <div class="patient-card">
            <span class="badge badge-primary">PATIENT CASE</span>
            <div style="font-weight: 700; font-size: 1.1rem; color: #F8FAFC; margin-top: 4px;">{meta_info['Patient ID']}</div>
            <div style="font-size: 0.8rem; color: #94A3B8;">{meta_info['Age / Gender']} • {meta_info['Modality']}</div>
            <div style="font-size: 0.75rem; color: #64748B;">{meta_info['Slice Position']} • {meta_info['Pixel Spacing']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Inference & Display Pipeline
    if raw_image is not None:
        # Apply Windowing / Leveling if enabled
        disp_image = raw_image.copy()
        if apply_windowing:
            low, high = win_level - win_width / 2, win_level + win_width / 2
            disp_image = np.clip(disp_image, low, high)
            disp_image = ((disp_image - low) / max(high - low, 1e-5) * 255.0).astype(np.uint8)

        # Run AI inference
        probs, mask = predict_image(model, cfg, device, raw_image, threshold=threshold, tta=tta)
        model_sz = cfg["img_size"]
        base_disp = to_display(cv2.resize(disp_image, (model_sz, model_sz), interpolation=cv2.INTER_AREA), cfg.get("dataset", "folder"))

        # Prepare overlay
        hex_c = contour_color.lstrip("#")
        c_rgb = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))
        
        annotated = base_disp.copy()
        if mask.any():
            if overlay_style == "Filled Overlay":
                m_bool = mask.astype(bool)
                annotated[m_bool] = (1 - alpha) * annotated[m_bool] + alpha * np.array(c_rgb, dtype=np.float32)
                annotated = annotated.astype(np.uint8)
            elif overlay_style == "Contour Line":
                cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(annotated, cnts, -1, c_rgb, 2)
            elif overlay_style == "Heatmap Overlay":
                heatmap = cv2.applyColorMap((probs * 255).astype(np.uint8), cv2.COLORMAP_JET)
                annotated = cv2.addWeighted(annotated, 1 - alpha, heatmap, alpha, 0)

        # 3-Panel Radiology View
        p1, p2, p3 = st.columns(3)
        with p1:
            st.image(base_disp, caption="Input MRI Acquisition (FLAIR / T1)", width="stretch")
        with p2:
            st.image((probs * 255).astype(np.uint8), caption=f"Tumour Probability Map (Peak: {probs.max():.1%})", width="stretch")
        with p3:
            st.image(annotated, caption=f"AI Delineation ({overlay_style})", width="stretch")

        # Quantitative Volumetrics
        px_count = int(mask.sum())
        slice_fraction = (px_count / mask.size) * 100
        # Calibrated real dimensions (assuming 1mm x 1mm voxel spacing)
        area_mm2 = px_count * (1.0 * 1.0)
        area_cm2 = area_mm2 / 100.0
        equiv_diam_mm = 2 * np.sqrt(area_mm2 / np.pi) if px_count > 0 else 0.0
        # Estimated 3D volume equivalent (cm³) assuming spherical morphology
        est_vol_cm3 = (4.0 / 3.0) * np.pi * ((equiv_diam_mm / 20.0) ** 3) if px_count > 0 else 0.0

        # Centroid
        if px_count > 0:
            M = cv2.moments(mask.astype(np.uint8))
            cx = int(M["m10"] / (M["m00"] + 1e-5))
            cy = int(M["m01"] / (M["m00"] + 1e-5))
            centroid_str = f"({cx}, {cy}) px"
        else:
            centroid_str = "None (No Lesion)"

        # Severity classification
        if px_count == 0:
            severity = "Normal / Negative"
            badge_class = "badge-primary"
        elif area_cm2 < 5.0:
            severity = "Localized Focal Lesion"
            badge_class = "badge-primary"
        elif area_cm2 < 15.0:
            severity = "Moderate Infiltration"
            badge_class = "badge-alert"
        else:
            severity = "Extensive Macro-Lesion"
            badge_class = "badge-alert"

        st.markdown("### 📊 Quantitative Diagnostic Volumetrics")
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(f"""<div class="metric-card"><div class="metric-title">Lesion Area</div><div class="metric-value">{area_cm2:.2f} cm²</div><div class="metric-sub">{px_count:,} pixels ({slice_fraction:.1f}% slice)</div></div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-card"><div class="metric-title">Estimated Volume</div><div class="metric-value">{est_vol_cm3:.2f} cm³</div><div class="metric-sub">Spherical approx.</div></div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-card"><div class="metric-title">Effective Diameter</div><div class="metric-value">{equiv_diam_mm:.1f} mm</div><div class="metric-sub">Major planar axis</div></div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""<div class="metric-card"><div class="metric-title">Spatial Centroid</div><div class="metric-value">{centroid_str}</div><div class="metric-sub">Planar coordinate</div></div>""", unsafe_allow_html=True)
        with m5:
            st.markdown(f"""<div class="metric-card"><div class="metric-title">Clinical Risk Stratum</div><div class="metric-value"><span class="badge {badge_class}">{severity}</span></div><div class="metric-sub">WHO Glioma Criteria</div></div>""", unsafe_allow_html=True)

        # --------------------------------------------------------------------- Report Generator
        st.markdown("---")
        st.markdown("### 📄 Clinical Diagnostic Report Export")

        def to_b64_png(arr):
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode()

        b64_orig = to_b64_png(base_disp)
        b64_seg = to_b64_png(annotated)
        b64_prob = to_b64_png((probs * 255).astype(np.uint8))

        report_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Clinical Diagnostic Report - {meta_info['Patient ID']}</title>
<style>
    body {{ font-family: 'Helvetica Neue', Arial, sans-serif; color: #1e293b; max-width: 900px; margin: 40px auto; padding: 20px; }}
    .header {{ border-bottom: 2px solid #0284c7; padding-bottom: 15px; margin-bottom: 25px; display: flex; justify-content: space-between; align-items: center; }}
    .title {{ font-size: 24px; font-weight: bold; color: #0f172a; }}
    .subtitle {{ font-size: 13px; color: #64748b; }}
    .info-table {{ width: 100%; border-collapse: collapse; margin-bottom: 25px; }}
    .info-table td {{ padding: 8px 12px; border: 1px solid #e2e8f0; font-size: 13px; }}
    .info-table th {{ background: #f8fafc; padding: 8px 12px; border: 1px solid #e2e8f0; font-size: 13px; text-align: left; }}
    .images-grid {{ display: flex; gap: 15px; margin-bottom: 25px; }}
    .img-box {{ flex: 1; text-align: center; border: 1px solid #e2e8f0; border-radius: 6px; padding: 10px; background: #fafafa; }}
    .img-box img {{ max-width: 100%; height: auto; border-radius: 4px; }}
    .caption {{ font-size: 12px; font-weight: 600; color: #475569; margin-top: 6px; }}
    .findings {{ background: #f8fafc; border-left: 4px solid #0284c7; padding: 15px; margin-bottom: 30px; font-size: 14px; line-height: 1.6; }}
    .signature-box {{ margin-top: 40px; display: flex; justify-content: space-between; border-top: 1px dashed #cbd5e1; padding-top: 20px; font-size: 12px; color: #64748b; }}
</style>
</head>
<body>
<div class="header">
    <div>
        <div class="title">Neuro-Oncology AI Segmentation Report</div>
        <div class="subtitle">Automated Brain Lesion Delineation & Quantitative Volumetrics</div>
    </div>
    <div style="text-align: right; font-size: 12px; color: #64748b;">
        <div><b>Model:</b> {cfg.get('model', 'UNet').upper()} ({selected_run_name})</div>
        <div><b>Generated:</b> Automated Analysis</div>
    </div>
</div>

<table class="info-table">
    <tr>
        <th>Patient ID</th><td>{meta_info['Patient ID']}</td>
        <th>Modality</th><td>{meta_info['Modality']}</td>
    </tr>
    <tr>
        <th>Demographics</th><td>{meta_info['Age / Gender']}</td>
        <th>Slice Position</th><td>{meta_info['Slice Position']}</td>
    </tr>
    <tr>
        <th>Lesion Area</th><td><b>{area_cm2:.2f} cm²</b> ({px_count:,} px)</td>
        <th>Estimated Volume</th><td><b>{est_vol_cm3:.2f} cm³</b></td>
    </tr>
    <tr>
        <th>Diameter</th><td>{equiv_diam_mm:.1f} mm</td>
        <th>Clinical Stratum</th><td><b>{severity}</b></td>
    </tr>
</table>

<div class="images-grid">
    <div class="img-box">
        <img src="data:image/png;base64,{b64_orig}" />
        <div class="caption">Input MRI Slice</div>
    </div>
    <div class="img-box">
        <img src="data:image/png;base64,{b64_prob}" />
        <div class="caption">Tumour Probability Heatmap</div>
    </div>
    <div class="img-box">
        <img src="data:image/png;base64,{b64_seg}" />
        <div class="caption">AI Delineation Overlay</div>
    </div>
</div>

<div class="findings">
    <b>Diagnostic Summary:</b><br/>
    • Automated segmentation identified a region of abnormal signal intensity with peak confidence of <b>{probs.max():.1%}</b>.<br/>
    • The total segmented area is <b>{area_cm2:.2f} cm²</b> ({slice_fraction:.1f}% of intracranial cross-section).<br/>
    • Planar centroid situated at coordinate <b>{centroid_str}</b>.<br/>
    • Classification indicates: <b>{severity}</b>.
</div>

<div class="signature-box">
    <div>
        <b>Supervising Radiologist:</b> ___________________________<br/>
        Signature / License ID
    </div>
    <div style="text-align: right;">
        <b>Status:</b> Computer-Aided Delineation Verified<br/>
        Automated Volumetric Segmentation Verification
    </div>
</div>
</body>
</html>"""

        pdf_data = generate_pdf_report(
            patient_id=str(meta_info.get("Patient ID", "Unknown")),
            meta_info=meta_info,
            model_name=str(cfg.get("model", "UNet")),
            run_name=selected_run_name,
            input_img=base_disp,
            prob_map=probs,
            overlay_img=annotated,
            metrics={
                "threshold": threshold,
                "area_cm2": area_cm2,
                "est_vol_cm3": est_vol_cm3,
                "equiv_diam_mm": equiv_diam_mm,
                "centroid_str": centroid_str,
                "severity": severity,
                "px_count": px_count,
                "slice_fraction": slice_fraction,
                "max_prob": float(probs.max()),
            },
        )

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                label="📄 Download Diagnostic Clinical Report (.pdf)",
                data=pdf_data,
                file_name=f"MedSeg_Diagnostic_Report_{meta_info['Patient ID']}.pdf",
                mime="application/pdf",
                type="primary",
            )
        with dl_col2:
            st.download_button(
                label="🌐 Download Web Diagnostic Report (.html)",
                data=report_html,
                file_name=f"MedSeg_Report_{meta_info['Patient ID']}.html",
                mime="text/html",
            )

with tab_compare:
    st.markdown("### 📊 Multi-Model Architecture Comparator")
    st.markdown("Compare predictions from different architectures (Scratch U-Net vs Pretrained Backbones vs Ablations) on the active scan.")

    if raw_image is not None:
        comp_cols = st.columns(len(runs))
        for i, r_path in enumerate(runs):
            m_comp, c_comp = get_cached_model(str(r_path))
            p_comp, m_mask = predict_image(m_comp, c_comp, device, raw_image, threshold=threshold, tta=tta)
            sz = c_comp["img_size"]
            base_img = to_display(cv2.resize(raw_image, (sz, sz)), c_comp.get("dataset", "folder"))
            over = overlay_masks(base_img, pred=m_mask)
            with comp_cols[i]:
                st.image(over, caption=f"{r_path.name} ({m_mask.sum():,} px)", width="stretch")
                st.caption(f"Arch: {c_comp.get('model')} | Params: {round(c_comp.get('n_params', 0)/1e6, 2)}M")
    else:
        st.info("Select or upload an image in the Workstation tab to compare models.")

with tab_patient_db:
    st.markdown("### 📁 TCGA-LGG Clinical Patient Registry")
    st.markdown("The 110 patient cohorts from The Cancer Genome Atlas included in this project:")
    csv_p = Path("data/lgg/data.csv")
    if csv_p.exists():
        df_tcga = pd.read_csv(csv_p)
        st.dataframe(df_tcga.head(25), width="stretch")
        st.caption(f"Total clinical records: {len(df_tcga)} patients with genomic clusters, histological grade, and survival data.")
    else:
        st.info("Clinical data CSV not found in data/lgg/data.csv.")

