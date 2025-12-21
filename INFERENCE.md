# Video Inference Guide

This guide shows you how to run inference on videos using the trained models.

---

## Quick Start

### Single Video Inference

```bash
# SOLOv2
python inference.py \
    --video /path/to/video.mp4 \
    --model work_dirs/solov2_nano/best_coco_segm_mAP_epoch_130.pth \
    --type solov2

# RTMDet-Ins
python inference.py \
    --video /path/to/video.mp4 \
    --model work_dirs/rtmdet_ins_tiny/best_coco_segm_mAP_epoch_142.pth \
    --type rtmdet-ins
```

### Batch Inference (Multiple Videos)

```bash
# Run test script (2 videos × 2 models = 4 outputs)
./test_inference.sh
```

---

## Output Format

Each output video contains **3 views side-by-side**:

```
┌──────────────┬──────────────┬──────────────┐
│   Original   │ Colored Mask │   Overlay    │
│     RGB      │     Only     │  RGB + Mask  │
└──────────────┴──────────────┴──────────────┘
```

- **Left**: Original RGB frame
- **Middle**: Color-coded segmentation mask
  - 🟢 Green: Grass track
  - 🔵 Blue: Jumps
  - 🔴 Red: Track
- **Right**: Original frame with mask overlay (50% transparency)

---

## inference.py - Detailed Usage

### Basic Usage

```bash
python inference.py --video VIDEO --model MODEL.pth --type {solov2|rtmdet-ins}
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--video` | ✓ | Path to input video file |
| `--model` | ✓ | Path to model checkpoint (.pth) |
| `--type` | ✓ | Model type: `solov2` or `rtmdet-ins` |
| `--output` | ✗ | Output path (default: auto-generated) |
| `--conf` | ✗ | Confidence threshold (default: 0.5) |
| `--device` | ✗ | Device to use (default: cuda:0) |

### Examples

**Basic inference:**
```bash
python inference.py \
    --video input.mp4 \
    --model work_dirs/solov2_nano/best.pth \
    --type solov2
```

**Custom output and confidence:**
```bash
python inference.py \
    --video input.mp4 \
    --model work_dirs/rtmdet_ins_tiny/best.pth \
    --type rtmdet-ins \
    --output result.mp4 \
    --conf 0.3
```

**CPU inference:**
```bash
python inference.py \
    --video input.mp4 \
    --model model.pth \
    --type solov2 \
    --device cpu
```

---

## test_inference.sh - Batch Processing

This script runs inference on 2 videos with both models (4 total outputs).

### Prerequisites

1. Update video paths in `test_inference.sh`:
```bash
VIDEO1="/path/to/video1.mp4"
VIDEO2="/path/to/video2.mp4"
```

2. Ensure model checkpoints exist:
   - `work_dirs/solov2_nano/best_coco_segm_mAP_epoch_130.pth`
   - `work_dirs/rtmdet_ins_tiny/best_coco_segm_mAP_epoch_142.pth`

### Run

```bash
./test_inference.sh
```

### Output

Creates 4 videos in `./inference_outputs/`:
```
inference_outputs/
├── racetrack_100_solov2.mp4          # Video 1 + SOLOv2
├── racetrack_100_rtmdet-ins.mp4      # Video 1 + RTMDet-Ins
├── grass_50_solov2.mp4                # Video 2 + SOLOv2
└── grass_50_rtmdet-ins.mp4            # Video 2 + RTMDet-Ins
```

---

## Available Models

### Trained Checkpoints

| Model | Checkpoint | mAP50-95 | mAP75 | Best For |
|-------|-----------|----------|-------|----------|
| **SOLOv2-nano** | `work_dirs/solov2_nano/best_coco_segm_mAP_epoch_130.pth` | 74.5% | **87.2%** | Precise masks |
| **RTMDet-Ins-tiny** | `work_dirs/rtmdet_ins_tiny/best_coco_segm_mAP_epoch_142.pth` | **75.3%** | 80.6% | Best overall |

### Model Comparison

**SOLOv2** (Recommended for precise boundaries):
- Highest mAP75: 87.2%
- Highest mAP50: 95.0%
- Best at tight mask fits
- Grid-based prediction

**RTMDet-Ins** (Recommended for overall consistency):
- Highest mAP50-95: 75.3%
- Modern CSPNeXt backbone
- Faster inference
- One-stage detection

---

## Troubleshooting

### Config file not found

**Error**: `Config not found: solov2_nano_training.py`

**Solution**: The config file is auto-generated during training. Make sure you have:
```bash
ls solov2_nano_training.py  # or rtmdet_ins_tiny_training.py
```

If missing, regenerate by running training (even for 1 epoch):
```bash
python train.py --data data.yaml --model nano --epochs 1 --skip-conversion
```

### Model checkpoint not found

**Error**: `Model checkpoint not found: work_dirs/...`

**Solution**: Run training first:
```bash
# SOLOv2
python train.py --data data.yaml --architecture solov2 --model nano

# RTMDet-Ins
python train.py --data data.yaml --architecture rtmdet-ins --model tiny
```

### CUDA out of memory

**Solution**: Lower batch processing or use CPU:
```bash
python inference.py --video input.mp4 --model model.pth --type solov2 --device cpu
```

### Low quality detections

**Solution**: Adjust confidence threshold:
```bash
# More detections (may include false positives)
python inference.py --video input.mp4 --model model.pth --type solov2 --conf 0.3

# Fewer detections (higher confidence only)
python inference.py --video input.mp4 --model model.pth --type solov2 --conf 0.7
```

---

## Performance Tips

### Speed Optimization

1. **Use RTMDet-Ins** for faster inference (~5-10% faster than SOLOv2)
2. **Lower confidence threshold** reduces post-processing time
3. **GPU inference** is 10-50x faster than CPU

### Quality Optimization

1. **Use SOLOv2** for highest precision masks (87.2% mAP75)
2. **Adjust confidence**:
   - `--conf 0.3`: More detections, may include false positives
   - `--conf 0.7`: Fewer but higher quality detections
3. **Check model performance**: Use the model with best validation results for your specific use case

---

## Example Workflow

### 1. Train Models (if not done)

```bash
# Train SOLOv2
python train.py \
    --data /path/to/data.yaml \
    --architecture solov2 \
    --model nano \
    --epochs 150

# Train RTMDet-Ins
python train.py \
    --data /path/to/data.yaml \
    --architecture rtmdet-ins \
    --model tiny \
    --epochs 150
```

### 2. Single Video Test

```bash
# Quick test on one video
python inference.py \
    --video test_video.mp4 \
    --model work_dirs/solov2_nano/best_coco_segm_mAP_epoch_130.pth \
    --type solov2 \
    --conf 0.5
```

### 3. Batch Processing

```bash
# Edit test_inference.sh with your video paths
nano test_inference.sh

# Run batch inference
./test_inference.sh
```

### 4. Compare Results

Watch the output videos side-by-side to compare:
- SOLOv2 vs RTMDet-Ins performance
- Mask quality at different confidence thresholds
- Model behavior on different scenes (racetrack vs grass)

---

## Integration with Other Tools

### Convert to Different Format

```bash
# Convert to H.264 for better compatibility
ffmpeg -i output.mp4 -c:v libx264 -preset fast -crf 23 output_h264.mp4
```

### Extract Individual Views

```bash
# Extract just the mask view (middle third)
ffmpeg -i output.mp4 -vf "crop=iw/3:ih:iw/3:0" mask_only.mp4

# Extract just the overlay view (right third)
ffmpeg -i output.mp4 -vf "crop=iw/3:ih:iw*2/3:0" overlay_only.mp4
```

### Create Comparison Video

```bash
# Stack SOLOv2 and RTMDet-Ins vertically
ffmpeg -i racetrack_solov2.mp4 -i racetrack_rtmdet.mp4 \
    -filter_complex vstack comparison.mp4
```

---

## Citation

If you use this inference pipeline in your work, please cite:

```bibtex
@misc{yolo-to-instance-seg,
  title={YOLO-to-Instance-Segmentation: Fair Comparison Framework},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/yolo-to-solov2}
}
```

---

*For training documentation, see [README.md](README.md)*
*For architecture details, see [ARCHITECTURE.md](ARCHITECTURE.md)*
