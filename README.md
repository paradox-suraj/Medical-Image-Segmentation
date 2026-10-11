# Medical Image Segmentation with U-Net and FCN

Deep learning framework in PyTorch for brain lesion and tissue segmentation on MRI and CT scans. Implements **U-Net** and **FCN-32s / 16s / 8s** architectures with joint spatial-photometric augmentations, strict patient-level dataset partitioning, and standard overlap and boundary metrics (Dice, IoU, HD95).

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Quick verification with synthetic data:

```bash
python make_synthetic_data.py --out data/synthetic --layout folder
python train.py --dataset folder --data-root data/synthetic --model unet --base-ch 8 --img-size 64 --epochs 3 --batch-size 8 --workers 0 --out runs/smoke
python evaluate.py --run runs/smoke --workers 0
```

## 2. Dataset Preparation

**Primary: Brain MRI (FLAIR abnormality), 110 patients, 3,929 slices** - Kaggle `mateuszbuda/lgg-mri-segmentation`

```bash
python download_data.py --out data/lgg
```

Manual download: place patient directories (`TCGA_CS_4941_19960909/`, ...) inside `data/lgg/`.

**CT or 3D MRI tasks (Medical Segmentation Decathlon)** - convert 3D volumes to 2D slices:

```bash
python prepare_msd.py --images data/Task03_Liver/imagesTr --labels data/Task03_Liver/labelsTr --out data/liver_2d --modality ct --label-ids 1 2
# Training usage: --dataset folder --data-root data/liver_2d
```

## 3. Training & Experiments

```bash
# Visualise augmentations
python show_augmentations.py --dataset lgg --data-root data/lgg --out figures/augmentations.png

# Primary models
python train.py --dataset lgg --data-root data/lgg --model unet   --epochs 50 --out runs/unet
python train.py --dataset lgg --data-root data/lgg --model fcn32s --epochs 40 --lr 1e-4 --out runs/fcn32s
python train.py --dataset lgg --data-root data/lgg --model fcn8s  --epochs 40 --lr 1e-4 --out runs/fcn8s

# Ablation studies
python train.py --dataset lgg --data-root data/lgg --model unet --aug none --out runs/unet_noaug
python train.py --dataset lgg --data-root data/lgg --model unet --loss bce --out runs/unet_bce
python train.py --dataset lgg --data-root data/lgg --model unet --balance --out runs/unet_balanced

# Evaluate runs on held-out test patients and compile comparison summary
for r in unet fcn32s fcn8s unet_noaug unet_bce unet_balanced; do python evaluate.py --run runs/$r; done
python compare_runs.py runs/unet runs/fcn32s runs/fcn8s runs/unet_noaug runs/unet_bce runs/unet_balanced --out results
```

## 4. Inference & Interactive Workstation

```bash
python predict.py --run runs/unet --image data/lgg/<patient>/<slice>.tif --mask data/lgg/<patient>/<slice>_mask.tif --out outputs/case1
streamlit run app.py
```

## 5. Repository Structure

| File | Purpose |
|---|---|
| `medseg/data.py` | Data loading, patient-wise splits, Dataset and DataLoader classes |
| `medseg/augment.py` | Synchronized image and mask augmentations (torchvision v2) |
| `medseg/models.py` | U-Net from scratch, FCN-32s/16s/8s with VGG16-BN, SMP wrappers |
| `medseg/losses.py` | Soft Dice, BCE, BCE-Dice combo, Focal-Tversky |
| `medseg/metrics.py` | Dice, IoU, Precision, Recall, Specificity, 95th-percentile Hausdorff Distance |
| `medseg/inference.py` | Checkpoint loading and inference with test-time augmentation |
| `train.py` | Training loop with AMP, AdamW, ReduceLROnPlateau, and early stopping |
| `evaluate.py` | Test set evaluation and qualitative metric visualizations |
| `compare_runs.py` | Metric aggregation table and comparative plotting |
| `show_augmentations.py` | Augmentation pipeline visualizer |
| `prepare_msd.py` | 3D NIfTI volume to 2D PNG slice converter |
| `download_data.py`, `make_synthetic_data.py` | Dataset download and synthetic test generation |
| `predict.py`, `app.py` | CLI slice inference and Streamlit clinical workstation |

Each run directory contains `config.json`, `split.json`, `history.csv`, `curves.png`, `best.pt`, `last.pt`, and `eval_test/`.

## 6. Common Issues

* **GPU memory exhaustion** - Reduce `--batch-size` (e.g. 8) or `--img-size` (192) or `--base-ch` (16).
* **Low recall on lesions** - Use `--balance` oversampling and `--loss bce_dice`.
* **Inflated metrics (> 0.95 Dice)** - Ensure evaluation uses patient-level splits rather than random slice splitting to prevent patient leakage.
* **DataLoader subprocess issues on Windows** - Set `--workers 0`.

