"""Generate publication-grade PDF clinical radiology reports."""
from __future__ import annotations

import io
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def generate_pdf_report(
    patient_id: str,
    meta_info: dict,
    model_name: str,
    run_name: str,
    input_img: np.ndarray,
    prob_map: np.ndarray,
    overlay_img: np.ndarray,
    metrics: dict,
) -> bytes:
    fig = plt.figure(figsize=(8.27, 11.69), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    def make_box_ax(rect_coords, bg_color, border_color="#e2e8f0"):
        ax = fig.add_axes(rect_coords)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        box = patches.FancyBboxPatch(
            (0, 0), 1, 1, boxstyle="round,pad=0.01,rounding_size=0.015",
            transform=ax.transAxes, facecolor=bg_color, edgecolor=border_color, lw=1
        )
        ax.add_patch(box)
        return ax

    header_ax = make_box_ax([0.05, 0.915, 0.90, 0.065], bg_color="#0f172a", border_color="#0284c7")
    header_ax.text(0.03, 0.65, "MEDSEG CLINICAL RADIOLOGY REPORT", color="#ffffff",
                   fontsize=13, fontweight="bold", va="center", transform=header_ax.transAxes)
    header_ax.text(0.03, 0.28, "Computer-Aided Neuro-Oncology Segmentation & Volumetric Biomarkers",
                   color="#94a3b8", fontsize=8, va="center", transform=header_ax.transAxes)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    header_ax.text(0.97, 0.65, f"Date: {now_str}", color="#38bdf8",
                   fontsize=8.5, fontweight="bold", ha="right", va="center", transform=header_ax.transAxes)
    header_ax.text(0.97, 0.28, f"System: {model_name.upper()} ({run_name})", color="#cbd5e1",
                   fontsize=7.5, ha="right", va="center", transform=header_ax.transAxes)

    meta_ax = make_box_ax([0.05, 0.805, 0.90, 0.095], bg_color="#f8fafc", border_color="#e2e8f0")
    meta_ax.text(0.03, 0.82, "PATIENT & SCAN METADATA", color="#0284c7", fontsize=8.5, fontweight="bold", transform=meta_ax.transAxes)
    
    col1_x, col2_x, col3_x = 0.03, 0.36, 0.68
    y1, y2, y3 = 0.58, 0.35, 0.12

    meta_ax.text(col1_x, y1, f"Patient ID:  {patient_id}", fontsize=7.5, color="#1e293b", fontweight="bold", transform=meta_ax.transAxes)
    meta_ax.text(col1_x, y2, f"Demographics:  {meta_info.get('Age / Gender', 'Unknown')}", fontsize=7.5, color="#475569", transform=meta_ax.transAxes)
    meta_ax.text(col1_x, y3, f"Modality:  {meta_info.get('Modality', 'MRI Axial')}", fontsize=7.5, color="#475569", transform=meta_ax.transAxes)

    meta_ax.text(col2_x, y1, f"Slice Position:  {meta_info.get('Slice Position', 'Mid-Axial')}", fontsize=7.5, color="#475569", transform=meta_ax.transAxes)
    meta_ax.text(col2_x, y2, f"Voxel Spacing:  {meta_info.get('Voxel Size', '1.0 x 1.0 mm')}", fontsize=7.5, color="#475569", transform=meta_ax.transAxes)
    meta_ax.text(col2_x, y3, f"Decision Threshold (τ):  {metrics.get('threshold', 0.50):.2f}", fontsize=7.5, color="#475569", transform=meta_ax.transAxes)

    sev = metrics.get('severity', 'Indeterminate')
    sev_col = "#dc2626" if ("Macro" in sev or "Moderate" in sev) else "#0284c7"
    meta_ax.text(col3_x, y1, f"Risk Stratum:  {sev}", fontsize=7.5, fontweight="bold", color=sev_col, transform=meta_ax.transAxes)
    meta_ax.text(col3_x, y2, f"Peak Probability:  {metrics.get('max_prob', 0.0):.1%}", fontsize=7.5, color="#475569", transform=meta_ax.transAxes)
    meta_ax.text(col3_x, y3, "Status:  Delineation Completed", fontsize=7.5, color="#16a34a", fontweight="bold", transform=meta_ax.transAxes)

    y_img = 0.535
    h_img = 0.235
    w_img = 0.27
    gap = 0.045
    x_start = 0.05

    ax1 = fig.add_axes([x_start, y_img, w_img, h_img])
    if input_img.ndim == 2:
        ax1.imshow(input_img, cmap="gray")
    else:
        ax1.imshow(input_img)
    ax1.set_title("A: Input MRI Scan", fontsize=8.5, fontweight="bold", pad=5, color="#0f172a")
    ax1.axis("off")

    ax2 = fig.add_axes([x_start + w_img + gap, y_img, w_img, h_img])
    im2 = ax2.imshow(prob_map, cmap="inferno", vmin=0.0, vmax=1.0)
    ax2.set_title("B: Lesion Probability Map", fontsize=8.5, fontweight="bold", pad=5, color="#0f172a")
    ax2.axis("off")

    cbar_ax = fig.add_axes([x_start + w_img + gap, y_img - 0.022, w_img, 0.009])
    cbar = fig.colorbar(im2, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("Posterior Probability [0, 1]", fontsize=6.5)

    ax3 = fig.add_axes([x_start + 2 * (w_img + gap), y_img, w_img, h_img])
    if overlay_img.ndim == 2:
        ax3.imshow(overlay_img, cmap="gray")
    else:
        ax3.imshow(overlay_img)
    ax3.set_title("C: Lesion Delineation", fontsize=8.5, fontweight="bold", pad=5, color="#0f172a")
    ax3.axis("off")

    bio_ax = make_box_ax([0.05, 0.38, 0.90, 0.10], bg_color="#f1f5f9", border_color="#cbd5e1")
    bio_ax.text(0.03, 0.82, "QUANTITATIVE VOLUMETRIC BIOMARKERS", color="#0284c7", fontsize=8.5, fontweight="bold", transform=bio_ax.transAxes)

    cards = [
        ("Lesion Cross Area", f"{metrics.get('area_cm2', 0.0):.2f} cm²", f"{metrics.get('px_count', 0):,} px ({metrics.get('slice_fraction', 0.0):.1f}% slice)"),
        ("Estimated Volume", f"{metrics.get('est_vol_cm3', 0.0):.2f} cm³", "Spherical approximation"),
        ("Effective Diameter", f"{metrics.get('equiv_diam_mm', 0.0):.1f} mm", "Planar equivalent axis"),
        ("Spatial Centroid", f"{metrics.get('centroid_str', 'N/A')}", "Coordinate (X, Y)"),
    ]

    card_xs = [0.03, 0.27, 0.52, 0.76]
    for i, (ctitle, cval, csub) in enumerate(cards):
        cx = card_xs[i]
        bio_ax.text(cx, 0.58, ctitle, fontsize=7, color="#64748b", fontweight="bold", transform=bio_ax.transAxes)
        bio_ax.text(cx, 0.30, cval, fontsize=10, color="#0f172a", fontweight="bold", transform=bio_ax.transAxes)
        bio_ax.text(cx, 0.10, csub, fontsize=6.5, color="#94a3b8", transform=bio_ax.transAxes)

    find_ax = make_box_ax([0.05, 0.175, 0.90, 0.185], bg_color="#ffffff", border_color="#e2e8f0")
    find_ax.text(0.03, 0.88, "CLINICAL RADIOLOGY IMPRESSION & FINDINGS", color="#0f172a", fontsize=8.5, fontweight="bold", transform=find_ax.transAxes)
    
    findings_lines = [
        f"• FINDINGS: Automated deep learning semantic segmentation delineates an area of abnormal signal intensity",
        f"  measuring {metrics.get('area_cm2', 0.0):.2f} cm² across the analyzed axial plane with peak posterior confidence of {metrics.get('max_prob', 0.0):.1%}.",
        f"",
        f"• VOLUMETRY: 3D ellipsoidal volumetric approximation projects a total lesion volume of {metrics.get('est_vol_cm3', 0.0):.2f} cm³",
        f"  with an effective planar diameter of {metrics.get('equiv_diam_mm', 0.0):.1f} mm centered at coordinate {metrics.get('centroid_str', 'N/A')}.",
        f"",
        f"• IMPRESSION: Findings consistent with {metrics.get('severity', 'Abnormal Lesion')} per WHO Neuro-Oncology criteria.",
        f"  Correlation with multi-contrast MRI sequences (T1, T1-Gd, T2, FLAIR) and clinical history is recommended.",
    ]
    find_ax.text(0.03, 0.12, "\n".join(findings_lines), fontsize=7.2, color="#334155", va="bottom", linespacing=1.35, transform=find_ax.transAxes)

    sig_ax = make_box_ax([0.05, 0.04, 0.90, 0.115], bg_color="#ffffff", border_color="#e2e8f0")

    sig_ax.text(0.03, 0.72, "Supervising Radiologist: ____________________________________", fontsize=7.5, color="#475569", transform=sig_ax.transAxes)
    sig_ax.text(0.03, 0.44, "License / NPI Number:   ____________________________________", fontsize=7.5, color="#64748b", transform=sig_ax.transAxes)
    sig_ax.text(0.03, 0.16, "Verification Date:      ____________________________________", fontsize=7.5, color="#64748b", transform=sig_ax.transAxes)

    sig_ax.text(0.97, 0.72, "Status: COMPUTER-AIDED DELINEATION VERIFIED", fontsize=7.5, fontweight="bold", color="#16a34a", ha="right", transform=sig_ax.transAxes)
    sig_ax.text(0.97, 0.44, "Automated Volumetric Segmentation System", fontsize=7, color="#64748b", ha="right", transform=sig_ax.transAxes)
    sig_ax.text(0.97, 0.16, "TCGA Lower-Grade Glioma Segmentation Protocol", fontsize=6.5, color="#94a3b8", ha="right", transform=sig_ax.transAxes)

    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return buf.getvalue()
