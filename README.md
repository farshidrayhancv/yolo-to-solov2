# SOLOv2 Trainer

Easy-to-use SOLOv2 instance segmentation training framework that accepts Ultralytics YOLO format datasets.

## Features

- **Simple Interface**: Use your existing YOLO format datasets
- **Multiple Model Sizes**: Choose from nano, small, medium, or large models
- **Automatic Conversion**: Automatically converts YOLO format to COCO format
- **Flexible Configuration**: Customize all training parameters
- **Pre-trained Backbones**: Uses ImageNet pre-trained ResNet backbones

## Installation

### Prerequisites

```bash
# Create conda environment (optional but recommended)
conda create -n solov2 python=3.11
conda activate solov2

# Install dependencies
pip install torch torchvision
pip install openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc4,<2.2.0'
mim install 'mmdet>=3.0.0'
pip install pyyaml pillow tqdm
```

## Model Sizes

| Model | Backbone | FPN Channels | Default Batch | Speed | Accuracy | Use Case |
|-------|----------|--------------|---------------|-------|----------|----------|
| **nano** | ResNet18 | 128 | 8 | Fastest | Lowest | Real-time, edge devices |
| **small** | ResNet34 | 192 | 6 | Fast | Good | Balanced speed/accuracy |
| **medium** | ResNet50 | 256 | 4 | Medium | Better | General purpose (default) |
| **large** | ResNet101 | 384 | 2 | Slow | Best | Maximum accuracy |

## Usage

### Basic Training

Train with default settings (medium model):

```bash
python train.py --data /path/to/your/data.yaml
```

### With Custom Model Size

```bash
# Nano model (fastest)
python train.py --data /path/to/your/data.yaml --model nano

# Small model
python train.py --data /path/to/your/data.yaml --model small

# Medium model (default)
python train.py --data /path/to/your/data.yaml --model medium

# Large model (best accuracy)
python train.py --data /path/to/your/data.yaml --model large
```

### Full Configuration

```bash
python train.py \
    --data /path/to/your/data.yaml \
    --model medium \
    --epochs 150 \
    --batch 4 \
    --imgsz 1280 \
    --lr 0.01 \
    --work-dir ./my_experiment
```

### List Available Models

```bash
python train.py --list-models
```

## Dataset Format

Your `data.yaml` should follow Ultralytics YOLO format:

```yaml
# data.yaml
path: /path/to/dataset
train: train/images
val: valid/images

nc: 3  # number of classes
names: ['class1', 'class2', 'class3']  # class names
```

Dataset structure:
```
dataset/
├── data.yaml
├── train/
│   ├── images/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── labels/
│       ├── img1.txt  # segmentation polygons
│       └── img2.txt
└── valid/
    ├── images/
    └── labels/
```

Label format (normalized coordinates):
```
class_id x1 y1 x2 y2 x3 y3 ...
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data` | str | required | Path to data.yaml file |
| `--model` | str | medium | Model size: nano, small, medium, large |
| `--epochs` | int | 150 | Number of training epochs |
| `--batch` | int | auto | Batch size (auto-selected based on model) |
| `--imgsz` | int | 1280 | Input image size |
| `--lr` | float | 0.01 | Learning rate |
| `--momentum` | float | 0.937 | SGD momentum |
| `--weight-decay` | float | 0.0005 | Weight decay |
| `--warmup-epochs` | int | 3 | Number of warmup epochs |
| `--work-dir` | str | auto | Working directory for outputs |
| `--skip-conversion` | flag | False | Skip dataset conversion |
| `--list-models` | flag | False | List available models and exit |

## Output Structure

After training, you'll find:

```
work_dirs/solov2_{model_size}/
├── epoch_10.pth
├── epoch_20.pth
├── ...
└── epoch_150.pth  # Final model

solov2_{model_size}_training.log  # Training log
solov2_{model_size}_training.py   # Generated config
```

## Examples

### Example 1: Quick Start

```bash
python train.py --data ~/datasets/my_project/data.yaml
```

### Example 2: Small Model for Speed

```bash
python train.py \
    --data ~/datasets/my_project/data.yaml \
    --model small \
    --epochs 100 \
    --batch 8
```

### Example 3: Large Model for Accuracy

```bash
python train.py \
    --data ~/datasets/my_project/data.yaml \
    --model large \
    --epochs 200 \
    --imgsz 1280 \
    --batch 2
```

### Example 4: Resume with Pre-converted Dataset

```bash
# First run converts dataset
python train.py --data ~/datasets/my_project/data.yaml

# Later runs can skip conversion
python train.py \
    --data ~/datasets/my_project/data.yaml \
    --skip-conversion
```

## Performance Comparison

Based on Lingfield Racetrack dataset (217 train, 99 val images):

| Model | mAP50 | mAP50-95 | Training Time | Inference Speed |
|-------|-------|----------|---------------|-----------------|
| SOLOv2-Medium | 89.7% | 57.2% | ~12 hours | ~0.3s/image |
| YOLOv11n-seg | 94.5% | 62.1% | ~12 hours | ~0.05s/image |
| YOLOv11m-seg | 92.6% | 55.2% | ~45 hours | ~0.1s/image |

## Tips

1. **Model Selection**:
   - Use `nano` for real-time applications or edge devices
   - Use `small` for balanced speed/accuracy
   - Use `medium` for general purpose (recommended default)
   - Use `large` when maximum accuracy is critical

2. **Batch Size**:
   - Larger batch sizes generally improve training stability
   - Adjust based on your GPU memory
   - Default auto-selected values are optimized for 16GB GPU

3. **Image Size**:
   - 1280x1280 is recommended for good accuracy
   - Use 640x640 or 896x896 for faster training
   - Larger images = better accuracy but slower training

4. **Dataset Size**:
   - For small datasets (<500 images), consider using `small` or `nano` to avoid overfitting
   - For large datasets (>5000 images), `medium` or `large` will perform better

## Troubleshooting

### Out of Memory Error

Reduce batch size or image size:
```bash
python train.py --data data.yaml --batch 2 --imgsz 896
```

### Slow Training

Use a smaller model:
```bash
python train.py --data data.yaml --model small
```

### Poor Accuracy

Try larger model or more epochs:
```bash
python train.py --data data.yaml --model large --epochs 200
```

## Citation

If you use this trainer, please cite:

```bibtex
@article{wang2020solov2,
  title={SOLOv2: Dynamic and Fast Instance Segmentation},
  author={Wang, Xinlong and Zhang, Rufeng and Kong, Tao and Li, Lei and Shen, Chunhua},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```

## License

This project is released under the Apache 2.0 license.
