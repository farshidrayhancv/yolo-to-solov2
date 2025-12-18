#!/usr/bin/env python3
"""
SOLOv2 Training Script
Accepts Ultralytics-style YAML files and trains SOLOv2 models
"""

import argparse
import os
import sys
import yaml
import subprocess
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from utils.dataset_converter import convert_yolo_to_coco
from utils.config_builder import build_mmdet_config
from configs.model_configs import list_available_models


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train SOLOv2 instance segmentation model with YOLO-format dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings (medium model)
  python train.py --data /path/to/data.yaml

  # Train with specific model size
  python train.py --data /path/to/data.yaml --model medium

  # Train with custom settings
  python train.py --data /path/to/data.yaml --model small --epochs 200 --imgsz 1280 --batch 8

  # List available model sizes
  python train.py --list-models
        """
    )

    parser.add_argument('--data', type=str, help='Path to data.yaml file (Ultralytics format)')
    parser.add_argument(
        '--model',
        type=str,
        default='medium',
        choices=['nano', 'small', 'medium', 'large'],
        help='Model size (default: medium)'
    )
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs (default: 150)')
    parser.add_argument('--batch', type=int, default=None, help='Batch size (default: auto based on model size)')
    parser.add_argument('--imgsz', type=int, default=1280, help='Image size (default: 1280)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.937, help='SGD momentum (default: 0.937)')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='Weight decay (default: 0.0005)')
    parser.add_argument('--warmup-epochs', type=int, default=3, help='Warmup epochs (default: 3)')
    parser.add_argument('--work-dir', type=str, default=None, help='Working directory for outputs')
    parser.add_argument('--skip-conversion', action='store_true', help='Skip dataset conversion (use if already converted)')
    parser.add_argument('--list-models', action='store_true', help='List available model sizes and exit')

    return parser.parse_args()


def main():
    args = parse_args()

    # List models and exit
    if args.list_models:
        list_available_models()
        return

    # Validate required arguments
    if not args.data:
        print("Error: --data argument is required")
        print("Use --help for usage information")
        sys.exit(1)

    if not os.path.exists(args.data):
        print(f"Error: Data file not found: {args.data}")
        sys.exit(1)

    print("\n" + "="*70)
    print("SOLOv2 Instance Segmentation Training")
    print("="*70)

    # Load dataset configuration
    print("\nLoading dataset configuration...")
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)

    yolo_dataset_root = Path(args.data).parent
    class_names = data_config['names']
    num_classes = data_config['nc']

    print(f"Dataset: {yolo_dataset_root}")
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Model size: {args.model.upper()}")
    print(f"Image size: {args.imgsz}x{args.imgsz}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch if args.batch else 'auto'}")

    # Convert dataset to COCO format
    coco_dataset_root = yolo_dataset_root.parent / (yolo_dataset_root.name + '_coco')

    if not args.skip_conversion:
        print("\n" + "="*70)
        print("Converting YOLO dataset to COCO format...")
        print("="*70)

        for split in ['train', 'val']:
            class_names_conv, num_classes_conv = convert_yolo_to_coco(
                str(yolo_dataset_root),
                str(coco_dataset_root),
                split
            )

        print(f"\nCOCO dataset saved to: {coco_dataset_root}")
    else:
        print(f"\nSkipping conversion, using existing dataset at: {coco_dataset_root}")

    # Build MMDetection config
    print("\n" + "="*70)
    print(f"Building SOLOv2-{args.model.upper()} configuration...")
    print("="*70)

    config_content = build_mmdet_config(
        model_size=args.model,
        data_root=str(coco_dataset_root) + '/',
        class_names=class_names,
        num_classes=num_classes,
        img_size=(args.imgsz, args.imgsz),
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        work_dir=args.work_dir
    )

    # Save config file
    config_path = Path(__file__).parent / f'solov2_{args.model}_training.py'
    with open(config_path, 'w') as f:
        f.write(config_content)

    print(f"Config saved to: {config_path}")

    # Start training
    print("\n" + "="*70)
    print("Starting Training...")
    print("="*70 + "\n")

    work_dir = args.work_dir if args.work_dir else f'./work_dirs/solov2_{args.model}'
    log_file = f'solov2_{args.model}_training.log'

    train_cmd = [
        'mim', 'train', 'mmdet',
        str(config_path),
        '--work-dir', work_dir,
        '--launcher', 'none'
    ]

    print(f"Command: {' '.join(train_cmd)}")
    print(f"Logging to: {log_file}\n")

    # Run training
    try:
        with open(log_file, 'w') as log:
            process = subprocess.Popen(
                train_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            # Stream output to both console and log file
            for line in process.stdout:
                print(line, end='')
                log.write(line)
                log.flush()

            process.wait()

            if process.returncode == 0:
                print("\n" + "="*70)
                print("Training completed successfully!")
                print("="*70)
                print(f"\nModel weights saved to: {work_dir}/")
                print(f"Training log saved to: {log_file}")
            else:
                print(f"\nTraining failed with exit code {process.returncode}")
                sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        process.terminate()
        sys.exit(1)
    except Exception as e:
        print(f"\nError during training: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
