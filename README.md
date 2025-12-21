# YOLO-to-Instance-Segmentation Trainer

Train **SOLOv2** and **RTMDet-Ins** instance segmentation models using your existing YOLO format datasets with **Ultralytics-matched augmentations and learning rate schedule**.

## 📐 Architecture

Detailed architecture diagrams and comparisons for all three models:

**👉 [View Complete Architecture Documentation →](ARCHITECTURE.md)**

- **SOLOv2**: Grid-based with dynamic kernels - best for precise masks
- **RTMDet-Ins**: One-stage with CSPNeXt backbone - best overall
- **YOLOv11-seg**: Prototype-based - best for speed

## Features

- ✅ **Two Architectures**: Choose between SOLOv2 (ResNet backbone) or RTMDet-Ins (CSPNeXt backbone)
- ✅ **Drop-in Replacement**: Use YOLO format datasets directly
- ✅ **4 Model Sizes**: Nano/Tiny (fastest) → Large (most accurate)
- ✅ **Auto Conversion**: YOLO → COCO format conversion built-in
- ✅ **Pre-trained Backbones**: ImageNet pre-trained ResNet (SOLOv2) or CSPNeXt (RTMDet-Ins) models
- ✅ **Ultralytics-Matched Training**:
  - YOLO's auto LR formula: `lr = 0.002 * 5 / (4 + num_classes)`
  - 7-epoch linear warmup (2% to 100%)
  - Cosine annealing to 1.7% of peak LR
  - Identical augmentation defaults (rotation ±2°, no shear/mixup)
- ✅ **Complete Augmentation Suite**:
  - **Mosaic** (100%): Custom instance-aware 4-image grid augmentation
  - **MixUp** (configurable): Image blending with alpha compositing
  - **HSV**: Color space augmentation (hue ±0.015, sat ±0.7, val ±0.4)
  - **Affine**: Rotation, translation ±10%, scale 0.5-1.5x
  - **Random brightness/contrast**: ±0.2 each
  - **Flips**: Horizontal (50%) and vertical (configurable)
- ✅ **Training Features**:
  - Best model checkpointing based on validation mAP
  - TensorBoard logging for training visualization
  - Multiprocessing data loading (4 workers)
  - Proper mask/bbox synchronization for all augmentations
- ✅ **Easy CLI**: Simple command-line interface matching Ultralytics

## Quick Start

### Installation

**Using conda:**

```bash
conda create -n solov2 python=3.11
conda activate solov2
pip install torch torchvision openmim albumentations
mim install mmengine 'mmcv>=2.0.0rc4,<2.2.0' 'mmdet>=3.0.0'
pip install pyyaml pillow tqdm
```

**Using uv (fast alternative):**

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv --python 3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install all dependencies
uv pip install -r requirements.txt

# Install MMDetection ecosystem with mim
uv pip install openmim
uv run python -m mim install mmengine 'mmcv>=2.0.0rc4,<2.2.0' 'mmdet>=3.0.0'
```

**Or use uv directly without activation:**

```bash
# Train directly with uv (no venv activation needed)
uv run train.py --data /path/to/data.yaml --model nano
```

### Train

**Using regular Python:**

```bash
# Basic SOLOv2 (nano model, 1280px, 150 epochs)
python train.py --data /path/to/data.yaml --model nano

# Train RTMDet-Ins (tiny model, 640px, 150 epochs)
python train.py --data /path/to/data.yaml --architecture rtmdet-ins --model tiny --imgsz 640

# Custom configuration
python train.py --data /path/to/data.yaml --architecture solov2 --model small --epochs 100 --batch 8

# List available models
python train.py --list-models
```

**Using uv (same commands, just prefix with `uv run`):**

```bash
# Basic SOLOv2 (nano model, 1280px, 150 epochs)
uv run train.py --data /path/to/data.yaml --model nano

# Train RTMDet-Ins (tiny model, 640px, 150 epochs)
uv run train.py --data /path/to/data.yaml --architecture rtmdet-ins --model tiny --imgsz 640
```

## Model Sizes

### SOLOv2 Models

| Model | Backbone | FPN Channels | Mask Channels | Batch | Speed | Use Case |
|-------|----------|--------------|---------------|-------|-------|----------|
| **nano** | ResNet18 | 128 | 64 | 8 | ⚡⚡⚡ | Edge devices, real-time |
| **small** | ResNet34 | 192 | 96 | 6 | ⚡⚡ | Balanced |
| **medium** | ResNet50 | 256 | 128 | 4 | ⚡ | General |
| **large** | ResNet101 | 384 | 256 | 2 | 🐢 | Maximum accuracy |

### RTMDet-Ins Models

| Model | Backbone | Neck Channels | Deepen/Widen | Batch | Speed | Use Case |
|-------|----------|---------------|--------------|-------|-------|----------|
| **tiny** | CSPNeXt | 96 | 0.167 / 0.375 | 10 | ⚡⚡⚡ | Edge devices, real-time |
| **small** | CSPNeXt | 128 | 0.33 / 0.5 | 8 | ⚡⚡ | Balanced |
| **medium** | CSPNeXt | 192 | 0.67 / 0.75 | 6 | ⚡ | General |
| **large** | CSPNeXt | 256 | 1.0 / 1.0 | 4 | 🐢 | Maximum accuracy |

### Choosing Between SOLOv2 and RTMDet-Ins

| Feature | SOLOv2 | RTMDet-Ins |
|---------|--------|------------|
| **Backbone** | ResNet (ImageNet pretrained) | CSPNeXt (ImageNet pretrained) |
| **Architecture** | Grid-based segmentation | One-stage detector with mask head |
| **Speed** | Moderate | Faster (especially on GPU) |
| **Accuracy** | Good | Better (on COCO) |
| **Optimizer** | SGD | AdamW |
| **Learning Rate** | Auto formula (class-dependent) | Fixed 0.004 |
| **Training** | Stable with proper LR | Fast convergence |
| **Use RTMDet-Ins if:** | - | Want faster inference, better COCO performance, or modern architecture |
| **Use SOLOv2 if:** | - | Need proven stability or prefer SGD optimizer |

## Performance

### COCO Benchmark (Official Results)

**SOLOv2** official performance on COCO val2017 (80 classes, 5000 images):

| Model | Backbone | mAP | mAP50 | mAP75 | Params | FPS |
|-------|----------|-----|-------|-------|--------|-----|
| SOLOv2-Light | ResNet18 | 29.6 | 47.3 | 31.3 | 24.9M | - |
| SOLOv2 | ResNet50 | 34.8 | 55.5 | 37.2 | 46.1M | 12.1 |
| SOLOv2 | ResNet101 | 37.1 | 58.3 | 39.6 | 65.0M | 9.9 |

**RTMDet-Ins** official performance on COCO val2017 (80 classes, 5000 images):

| Model | Backbone | mAP | mAP50 | mAP75 | Params | FPS |
|-------|----------|-----|-------|-------|--------|-----|
| RTMDet-Ins-tiny | CSPNeXt-tiny | 33.5 | 52.8 | 35.0 | 11.1M | ~50 |
| RTMDet-Ins-s | CSPNeXt-s | 38.1 | 58.2 | 41.0 | 20.6M | ~40 |
| RTMDet-Ins-m | CSPNeXt-m | 42.6 | 62.9 | 46.3 | 40.0M | ~30 |
| RTMDet-Ins-l | CSPNeXt-l | 44.9 | 65.4 | 48.6 | 52.3M | ~22 |

### Custom Dataset Results (Lingfield Racetrack)

Tested on 217 train, 99 val images, 3 classes (grass track, jumps, track) with **identical training configuration** (150 epochs, 1280px, matched augmentations and LR schedule):

#### Final Results (Epoch 150) - **Latest Training Run (2025-12-20)**

| Model | mAP50-95 | mAP50 | mAP75 | Params | Training Time | Best Epoch |
|-------|----------|-------|-------|--------|---------------|------------|
| **YOLOv11n-seg** | 62.0% | 94.5% | ~80% | 2.9M | ~12 min | - |
| **SOLOv2-nano** | **74.5%** | **95.0%** | **87.2%** | 11.2M | ~90 min | 130 |
| **RTMDet-Ins-tiny** | **75.3%** | 93.5% | 80.6% | 11.1M | ~85 min | 142 |

**Key Findings**:
- ✅ **SOLOv2 achieves 12.5% higher mAP50-95** than YOLO (74.5% vs 62.0%)
- ✅ **RTMDet-Ins achieves 13.3% higher mAP50-95** than YOLO (75.3% vs 62.0%)
- ✅ **SOLOv2 has highest mAP50** (95.0%) - best at rough segmentation
- ✅ **SOLOv2 has highest mAP75** (87.2%) - best at precise masks
- ✅ **RTMDet-Ins has best overall mAP50-95** (75.3%) - most consistent
- ⚠️ **YOLO is 7x faster** but significantly less accurate on precise masks

#### Detailed Metrics (Latest Run - 2025-12-20)

**SOLOv2-nano (Epoch 150, Best @ 130):**
- mAP@50-95: **74.5%** (best: 74.5% @ epoch 130)
- mAP@50: **95.0%**
- mAP@75: **87.2%**
- Final loss: 0.250
- Gradient norm: 3.44 (stable)
- Training time: ~90 minutes

**RTMDet-Ins-tiny (Epoch 150, Best @ 142):**
- mAP@50-95: **75.3%** (best: 75.4% @ epoch 149)
- mAP@50: 93.5%
- mAP@75: 80.6%
- Final loss: 0.429
- Gradient norm: 2.85 (stable)
- Training time: ~85 minutes

**YOLOv11n-seg (Epoch 150):**
- mAP@50-95: 62.0%
- mAP@50: 94.5%
- mAP@75: ~80% (estimated)
- Training time: ~12 minutes

### The Importance of Learning Rate Schedule

The key to SOLOv2's success is using **YOLO's auto optimizer formula** instead of a naive learning rate:

| Configuration | Peak LR | Warmup | Result |
|---------------|---------|--------|--------|
| **Incorrect** | 0.01 | 3 epochs (0.1% → 100%) | 26.8% mAP50-95 (gradient explosion) |
| **Correct** | 0.001429 | 7 epochs (2% → 100%) | **75.5% mAP50-95** (stable training) |

**YOLO's auto optimizer formula**: `lr = 0.002 * 5 / (4 + num_classes)`

For 3 classes: `lr = 0.002 * 5 / (4 + 3) = 0.001429`

Using the naive 0.01 LR caused:
- Gradient norm of 668 in epoch 2 (explosion!)
- Loss spiking to 58.7
- Training instability throughout
- Final mAP of only 26.8%

With the correct formula-based LR (0.001429):
- Stable gradient norms (3-35 range)
- Smooth loss convergence
- **74.5% mAP50-95** (SOLOv2) and **75.3% mAP50-95** (RTMDet-Ins) - outperforming YOLO by 12-13%

### Why SOLOv2 & RTMDet-Ins Excel with Proper Training

**1. More Precise Masks (Both Models)**
- mAP50 (IoU=0.5): All models ~93-95% (similar rough segmentation)
- **mAP50-95 (IoU=0.5-0.95)**: SOLOv2 74.5%, RTMDet-Ins 75.3% vs YOLO 62.0% (+12-13%)
- **mAP75 (IoU=0.75)**: SOLOv2 **87.2%**, RTMDet-Ins 80.6% vs YOLO ~80%
- Architecture design produces tighter mask fits at higher IoU thresholds

**2. SOLOv2-Specific Strengths**
- **Grid-Based Prediction**: Spatial grid system enables finer-grained instance localization
- **Best mAP75**: 87.2% shows superior mask boundary precision
- **Highest mAP50**: 95.0% indicates excellent instance detection
- **Decoupled Heads**: Separate category and mask branches with multi-level fusion (P2→P3→P4)

**3. RTMDet-Ins-Specific Strengths**
- **Best Overall mAP50-95**: 75.3% shows most consistent performance across IoU thresholds
- **Modern Architecture**: CSPNeXt backbone with efficient one-stage detection
- **AdamW Optimization**: Fast convergence with adaptive learning rates
- **Balanced Performance**: Good trade-off between speed and accuracy

### Ultralytics Compatibility

This implementation matches Ultralytics YOLO's training recipe for fair comparison:

**Matched Configuration:**

```yaml
# Learning Rate (CRITICAL)
lr: 0.001429  # YOLO's auto formula: 0.002 * 5 / (4 + num_classes)
warmup_epochs: 7  # Linear warmup from 2% to 100%
scheduler: cosine  # Decay to 1.7% of peak by epoch 150

# Augmentations (Default YOLO Nano)
mosaic: 1.0
mixup: 0.0
degrees: 2.0  # ±2° rotation
shear: 0.0
translate: 0.1
scale: 0.5
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
fliplr: 0.5

# Training
batch_size: 10
epochs: 150
image_size: 1280
optimizer: SGD
momentum: 0.937
weight_decay: 0.0005
```

## Dataset Format

Your YOLO `data.yaml`:

```yaml
path: /path/to/dataset
train: train/images
val: valid/images
nc: 3
names: ['class1', 'class2', 'class3']
```

Directory structure:
```
dataset/
├── data.yaml
├── train/
│   ├── images/      # .jpg, .png
│   └── labels/      # .txt (normalized polygons)
└── valid/
    ├── images/
    └── labels/
```

Label format: `class_id x1 y1 x2 y2 x3 y3 ...` (normalized 0-1)

## CLI Arguments

### Basic Training

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | required | Path to data.yaml |
| `--architecture` | solov2 | Architecture: solov2 or rtmdet-ins |
| `--model` | nano | nano/tiny, small, medium, large |
| `--epochs` | 150 | Training epochs |
| `--batch` | auto | Batch size (default varies by model) |
| `--imgsz` | 1280 | Image size (640 recommended for RTMDet-Ins) |
| `--lr` | auto | Learning rate (uses YOLO formula for SOLOv2, 0.004 for RTMDet-Ins) |
| `--work-dir` | auto | Output directory |
| `--skip-conversion` | false | Skip YOLO→COCO conversion |

### Data Augmentation (Ultralytics-style defaults)

| Argument | Default | Description |
|----------|---------|-------------|
| `--mosaic` | 1.0 | Mosaic augmentation probability |
| `--mixup` | 0.0 | MixUp augmentation probability |
| `--hsv-h` | 0.015 | HSV hue augmentation (0-1) |
| `--hsv-s` | 0.7 | HSV saturation augmentation (0-1) |
| `--hsv-v` | 0.4 | HSV value/brightness augmentation (0-1) |
| `--degrees` | 2.0 | Random rotation (±degrees) |
| `--translate` | 0.1 | Random translation (±fraction) |
| `--scale` | 0.5 | Random scale range (±fraction) |
| `--shear` | 0.0 | Random shear (±degrees) |
| `--fliplr` | 0.5 | Horizontal flip probability |
| `--flipud` | 0.0 | Vertical flip probability |

**Note**: Defaults match YOLOv11n behavior for fair comparison.

## Tips

**GPU Memory Issues?**
```bash
python train.py --data data.yaml --batch 2 --imgsz 896
```

**Small Dataset (<500 images)?** Use `--model nano` or `--model small` to avoid overfitting.

**Need Speed?** Use `--model nano` for real-time inference.

**Need Maximum Accuracy?** Use `--model large --epochs 200` for best results.

**Monitor Training:**
```bash
tensorboard --logdir=work_dirs/solov2_nano
```

## Citation

If you use SOLOv2:
```bibtex
@article{wang2020solov2,
  title={SOLOv2: Dynamic and Fast Instance Segmentation},
  author={Wang, Xinlong and Zhang, Rufeng and Kong, Tao and Li, Lei and Shen, Chunhua},
  journal={NeurIPS},
  year={2020}
}
```

If you use RTMDet-Ins:
```bibtex
@misc{lyu2022rtmdet,
  title={RTMDet: An Empirical Study of Designing Real-Time Object Detectors},
  author={Chengqi Lyu and Wenwei Zhang and Haian Huang and Yue Zhou and Yudong Wang and Yanyi Liu and Shilong Zhang and Kai Chen},
  year={2022},
  eprint={2212.07784},
  archivePrefix={arXiv}
}
```

## License

Apache 2.0

## Acknowledgments

This implementation uses:
- [MMDetection](https://github.com/open-mmlab/mmdetection) for SOLOv2 and RTMDet-Ins
- [Ultralytics](https://github.com/ultralytics/ultralytics) training recipe
- [Albumentations](https://albumentations.ai/) for augmentations
- [RTMDet](https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet) for modern instance segmentation
