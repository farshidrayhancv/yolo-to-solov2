#!/usr/bin/env python3
"""
Video Inference Script for Instance Segmentation Models
Runs inference on videos with SOLOv2, RTMDet-Ins, and YOLOv11n-seg
Generates output videos with 3 views: Original | Mask | Overlay
"""

import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import sys

# MMDetection imports
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import DetLocalVisualizer


class VideoInference:
    """Handles video inference for instance segmentation models"""

    def __init__(self, config_path, checkpoint_path, device='cuda:0'):
        """
        Initialize model for inference

        Args:
            config_path: Path to model config file
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
        """
        self.device = device
        self.model = init_detector(config_path, checkpoint_path, device=device)

        # Class names and colors
        self.classes = ['grass track', 'jumps', 'track']
        self.colors = [
            (0, 255, 0),    # Green for grass track
            (255, 0, 0),    # Blue for jumps
            (0, 0, 255),    # Red for track
        ]

    def process_frame(self, frame):
        """
        Run inference on a single frame

        Args:
            frame: BGR image (H, W, 3)

        Returns:
            masks: List of binary masks for each instance
            labels: List of class labels for each instance
            scores: List of confidence scores for each instance
        """
        result = inference_detector(self.model, frame)

        # Extract predictions
        pred_instances = result.pred_instances
        masks = pred_instances.masks.cpu().numpy()  # (N, H, W)
        labels = pred_instances.labels.cpu().numpy()  # (N,)
        scores = pred_instances.scores.cpu().numpy()  # (N,)

        return masks, labels, scores

    def create_colored_mask(self, frame_shape, masks, labels, scores, conf_threshold=0.5):
        """
        Create colored mask visualization

        Args:
            frame_shape: Shape of the frame (H, W, 3)
            masks: Binary masks (N, H, W)
            labels: Class labels (N,)
            scores: Confidence scores (N,)
            conf_threshold: Confidence threshold for filtering

        Returns:
            colored_mask: RGB mask image (H, W, 3)
        """
        h, w = frame_shape[:2]
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

        # Filter by confidence
        valid_idx = scores >= conf_threshold
        masks = masks[valid_idx]
        labels = labels[valid_idx]
        scores = scores[valid_idx]

        # Sort by score (draw lower confidence first)
        sort_idx = np.argsort(scores)

        for idx in sort_idx:
            mask = masks[idx]
            label = labels[idx]
            color = self.colors[label]

            # Apply mask with color
            colored_mask[mask] = color

        return colored_mask

    def create_overlay(self, frame, colored_mask, alpha=0.5):
        """
        Overlay colored mask on original frame

        Args:
            frame: Original BGR image (H, W, 3)
            colored_mask: Colored mask (H, W, 3)
            alpha: Transparency factor (0-1)

        Returns:
            overlay: Blended image (H, W, 3)
        """
        overlay = cv2.addWeighted(frame, 1-alpha, colored_mask, alpha, 0)
        return overlay

    def create_three_view(self, frame, masks, labels, scores, conf_threshold=0.5):
        """
        Create 3-view output: Original | Mask | Overlay

        Args:
            frame: Original BGR image (H, W, 3)
            masks: Binary masks (N, H, W)
            labels: Class labels (N,)
            scores: Confidence scores (N,)
            conf_threshold: Confidence threshold

        Returns:
            three_view: Horizontally concatenated image (H, W*3, 3)
        """
        colored_mask = self.create_colored_mask(frame.shape, masks, labels, scores, conf_threshold)
        overlay = self.create_overlay(frame, colored_mask)

        # Concatenate horizontally: Original | Mask | Overlay
        three_view = np.hstack([frame, colored_mask, overlay])

        return three_view

    def process_video(self, input_video, output_video, conf_threshold=0.5):
        """
        Process entire video and save output

        Args:
            input_video: Path to input video
            output_video: Path to output video
            conf_threshold: Confidence threshold for detections
        """
        # Open video
        cap = cv2.VideoCapture(str(input_video))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_video}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video properties: {width}x{height} @ {fps} FPS, {total_frames} frames")

        # Create video writer (3x width for three views)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video), fourcc, fps, (width * 3, height))

        if not out.isOpened():
            raise ValueError(f"Cannot create output video: {output_video}")

        # Process frames
        frame_count = 0
        with tqdm(total=total_frames, desc=f"Processing {Path(input_video).name}") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Run inference
                masks, labels, scores = self.process_frame(frame)

                # Create 3-view output
                three_view = self.create_three_view(frame, masks, labels, scores, conf_threshold)

                # Write frame
                out.write(three_view)

                frame_count += 1
                pbar.update(1)

        # Release resources
        cap.release()
        out.release()

        print(f"✓ Saved output to: {output_video}")
        print(f"  Processed {frame_count} frames")


def main():
    parser = argparse.ArgumentParser(
        description='Run instance segmentation inference on videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run inference with SOLOv2
  python inference_video.py --model solov2 --input video.mp4 --output output.mp4

  # Run with RTMDet-Ins
  python inference_video.py --model rtmdet-ins --input video.mp4 --output output.mp4

  # Run with custom confidence threshold
  python inference_video.py --model solov2 --input video.mp4 --conf 0.3
        """
    )

    parser.add_argument('--model', type=str, required=True,
                        choices=['solov2', 'rtmdet-ins', 'yolo'],
                        help='Model to use for inference')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input video')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output video (default: auto-generated)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to model config (default: auto-detected)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (default: auto-detected)')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold (default: 0.5)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run inference (default: cuda:0)')

    args = parser.parse_args()

    # Auto-detect config and checkpoint paths
    if args.config is None or args.checkpoint is None:
        if args.model == 'solov2':
            config = 'solov2_nano_training.py'
            checkpoint = 'work_dirs/solov2_nano/best_coco_segm_mAP_epoch_130.pth'
        elif args.model == 'rtmdet-ins':
            config = 'rtmdet_ins_tiny_training.py'
            checkpoint = 'work_dirs/rtmdet_ins_tiny/best_coco_segm_mAP_epoch_142.pth'
        elif args.model == 'yolo':
            print("ERROR: YOLOv11n-seg inference not yet implemented")
            print("Please train with Ultralytics YOLO and use their inference tools")
            sys.exit(1)

        if args.config is None:
            args.config = config
        if args.checkpoint is None:
            args.checkpoint = checkpoint

    # Check if files exist
    if not Path(args.config).exists():
        print(f"ERROR: Config file not found: {args.config}")
        print("Please run training first to generate the config file")
        sys.exit(1)

    if not Path(args.checkpoint).exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        print("Please run training first to generate the checkpoint")
        sys.exit(1)

    if not Path(args.input).exists():
        print(f"ERROR: Input video not found: {args.input}")
        sys.exit(1)

    # Auto-generate output path
    if args.output is None:
        input_path = Path(args.input)
        args.output = f"{input_path.stem}_{args.model}_inference{input_path.suffix}"

    print("\n" + "="*70)
    print("Video Instance Segmentation Inference")
    print("="*70)
    print(f"Model: {args.model.upper()}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Confidence: {args.conf}")
    print(f"Device: {args.device}")
    print("="*70 + "\n")

    # Initialize inference
    print("Loading model...")
    inferencer = VideoInference(args.config, args.checkpoint, args.device)

    # Process video
    print("Processing video...")
    inferencer.process_video(args.input, args.output, args.conf)

    print("\n" + "="*70)
    print("Inference Complete!")
    print("="*70)
    print(f"Output saved to: {args.output}")
    print("\nOutput format: Original | Colored Mask | Overlay")
    print("  - Left: Original RGB frame")
    print("  - Middle: Color-coded segmentation mask")
    print("  - Right: Original frame with mask overlay")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
