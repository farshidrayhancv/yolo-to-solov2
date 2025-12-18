# YOLO-to-SOLOv2 Trainer

Train SOLOv2 instance segmentation models using your existing YOLO format datasets.

## Architecture

```mermaid
graph TB
    subgraph Input["Input: YOLO Dataset"]
        A[YOLO Format<br/>Normalized Polygons]
    end

    subgraph Conversion["Automatic Conversion"]
        B[Dataset Converter<br/>YOLO → COCO]
    end

    subgraph Model["SOLOv2 Architecture"]
        direction TB
        C[Backbone<br/>ResNet 18/34/50/101<br/>ImageNet Pretrained]
        D[Neck<br/>FPN<br/>Multi-scale Features]
        E[SOLOv2 Head<br/>Category Branch<br/>Mask Branch]

        C --> D
        D --> E
    end

    subgraph Training["Training"]
        F[Losses<br/>Focal Loss + Dice Loss]
        G[Optimizer<br/>SGD + Cosine LR]
    end

    subgraph Output["Output"]
        H[Trained Model<br/>.pth checkpoints]
        I[Metrics<br/>mAP, mAP50, mAP75]
    end

    A --> B
    B --> C
    E --> F
    F --> G
    G --> H
    G --> I

    style Input fill:#e1f5ff
    style Model fill:#fff4e1
    style Output fill:#e8f5e9
```

## Features

- ✅ **Drop-in Replacement**: Use YOLO format datasets directly
- ✅ **4 Model Sizes**: Nano (fastest) → Large (most accurate)
- ✅ **Auto Conversion**: YOLO → COCO format conversion built-in
- ✅ **Pre-trained Backbones**: ImageNet pre-trained ResNet models
- ✅ **Easy CLI**: Simple command-line interface

## Quick Start

### Installation

```bash
conda create -n solov2 python=3.11
conda activate solov2
pip install torch torchvision openmim
mim install mmengine 'mmcv>=2.0.0rc4,<2.2.0' 'mmdet>=3.0.0'
pip install pyyaml pillow tqdm
```

### Train

```bash
# Basic (medium model, 1280px, 150 epochs)
python train.py --data /path/to/data.yaml

# Custom configuration
python train.py --data /path/to/data.yaml --model small --epochs 100 --batch 8

# List available models
python train.py --list-models
```

## Model Sizes

| Model | Backbone | Channels | Batch | Speed | Use Case |
|-------|----------|----------|-------|-------|----------|
| **nano** | ResNet18 | 128 | 8 | ⚡⚡⚡ | Edge devices, real-time |
| **small** | ResNet34 | 192 | 6 | ⚡⚡ | Balanced |
| **medium** | ResNet50 | 256 | 4 | ⚡ | General (default) |
| **large** | ResNet101 | 384 | 2 | 🐢 | Maximum accuracy |

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

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | required | Path to data.yaml |
| `--model` | medium | nano, small, medium, large |
| `--epochs` | 150 | Training epochs |
| `--batch` | auto | Batch size (auto per model) |
| `--imgsz` | 1280 | Image size |
| `--lr` | 0.01 | Learning rate |
| `--work-dir` | auto | Output directory |
| `--skip-conversion` | false | Skip YOLO→COCO conversion |

## Performance

Tested on Lingfield Racetrack dataset (217 train, 99 val):

| Model | mAP50 | mAP50-95 | Speed |
|-------|-------|----------|-------|
| SOLOv2-Medium | **89.7%** | **57.2%** | 0.3s/img |
| YOLOv11n-seg | 94.5% | 62.1% | 0.05s/img |

SOLOv2 offers better architecture for custom datasets and research flexibility via MMDetection.

## Tips

**GPU Memory Issues?**
```bash
python train.py --data data.yaml --batch 2 --imgsz 896
```

**Small Dataset (<500 images)?** Use `--model nano` or `--model small` to avoid overfitting.

**Need Speed?** Use `--model nano` for real-time inference.

**Need Accuracy?** Use `--model large --epochs 200` for best results.

## Citation

```bibtex
@article{wang2020solov2,
  title={SOLOv2: Dynamic and Fast Instance Segmentation},
  author={Wang, Xinlong and Zhang, Rufeng and Kong, Tao and Li, Lei and Shen, Chunhua},
  journal={NeurIPS},
  year={2020}
}
```

## License

Apache 2.0
