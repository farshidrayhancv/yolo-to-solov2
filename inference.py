#!/usr/bin/env python3
"""
Instance Segmentation Video Inference
Simple Ultralytics-style inference script
"""

import argparse
import sys
import cv2
import numpy as np
import torch
import subprocess
import gc
from pathlib import Path
from tqdm import tqdm

# Fix PyTorch 2.6+ weights_only issue - monkey-patch torch.load
_original_torch_load = torch.load
def _patched_torch_load(f, map_location=None, **kwargs):
    # Force weights_only=False for compatibility with mmengine checkpoints
    kwargs['weights_only'] = False
    return _original_torch_load(f, map_location=map_location, **kwargs)
torch.load = _patched_torch_load

# MMDetection imports
from mmdet.apis import init_detector, inference_detector


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run instance segmentation inference on video',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run inference with SOLOv2
  python inference.py --video video.mp4 --model work_dirs/solov2_nano/best.pth --type solov2

  # Run with RTMDet-Ins
  python inference.py --video video.mp4 --model work_dirs/rtmdet_ins_tiny/best.pth --type rtmdet-ins

  # Custom output path
  python inference.py --video video.mp4 --model model.pth --type solov2 --output result.mp4
        """
    )

    parser.add_argument('--video', type=str, required=True,
                        help='Path to input video')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--type', type=str, required=True,
                        choices=['solov2', 'rtmdet-ins'],
                        help='Model type: solov2 or rtmdet-ins')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output video (default: auto-generated)')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold (default: 0.5)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device (default: cuda:0)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum frames to process (default: all)')

    return parser.parse_args()


def get_config_path(model_type):
    """Get config path for model type"""
    if model_type == 'solov2':
        return 'solov2_nano_training.py'
    elif model_type == 'rtmdet-ins':
        return 'rtmdet_ins_tiny_training.py'
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_colored_mask(frame_shape, masks, labels, scores, conf_threshold=0.5):
    """Create colored segmentation mask"""
    h, w = frame_shape[:2]
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Class colors: green (grass track), blue (jumps), red (track)
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
    ]

    # Filter by confidence
    valid = scores >= conf_threshold
    masks = masks[valid]
    labels = labels[valid]
    scores = scores[valid]

    # Sort by score (draw lowest confidence first)
    sort_idx = np.argsort(scores)

    for idx in sort_idx:
        mask = masks[idx]
        label = labels[idx]
        color = colors[label]
        colored_mask[mask] = color

    return colored_mask


def process_video(video_path, model, output_path, conf_threshold=0.5, max_frames=None):
    """Process video with model"""

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Limit frames if requested
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
        print(f"\nVideo: {width}x{height} @ {fps} FPS ({total_frames}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames - TEST MODE)")
    else:
        print(f"\nVideo: {width}x{height} @ {fps} FPS ({total_frames} frames)")

    # Create output writer using ffmpeg pipe for better H.264 encoding
    # Scale to 640px wide per view (3 views = 1920px total width)
    view_width = 640
    view_height = int(height * view_width / width)
    output_width = view_width * 3
    output_height = view_height

    # Start ffmpeg process
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{output_width}x{output_height}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',
        '-an',
        '-vcodec', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        str(output_path)
    ]

    print(f"Using H.264 encoding via ffmpeg")
    print(f"Output resolution: {output_width}x{output_height} (3 views @ {view_width}x{view_height} each)")
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # Process frames
    print("Processing frames...")
    frame_count = 0
    with tqdm(total=total_frames) as pbar:
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Run inference with no gradient tracking (saves memory)
            with torch.no_grad():
                result = inference_detector(model, frame)
                pred = result.pred_instances

                # Move tensors to CPU and convert to numpy immediately
                masks = pred.masks.cpu().numpy()
                labels = pred.labels.cpu().numpy()
                scores = pred.scores.cpu().numpy()

            # Create colored mask
            colored_mask = create_colored_mask(
                frame.shape, masks, labels, scores, conf_threshold
            )

            # Create overlay
            overlay = cv2.addWeighted(frame, 0.5, colored_mask, 0.5, 0)

            # Resize each view to fit in 1080p output (640px wide per view)
            view_width = 640
            view_height = int(height * view_width / width)
            frame_resized = cv2.resize(frame, (view_width, view_height))
            mask_resized = cv2.resize(colored_mask, (view_width, view_height))
            overlay_resized = cv2.resize(overlay, (view_width, view_height))

            # Concatenate: Original | Mask | Overlay
            three_view = np.hstack([frame_resized, mask_resized, overlay_resized])

            # Write frame to ffmpeg stdin
            try:
                ffmpeg_proc.stdin.write(three_view.tobytes())
            except BrokenPipeError:
                print("\nERROR: ffmpeg pipe broken")
                break

            # Clear memory every 100 frames to prevent balloon
            if frame_count % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            frame_count += 1
            pbar.update(1)

    # Close ffmpeg
    cap.release()
    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()

    if ffmpeg_proc.returncode != 0:
        stderr = ffmpeg_proc.stderr.read().decode('utf-8')
        print(f"\nERROR: ffmpeg failed:\n{stderr}")
        raise ValueError(f"ffmpeg encoding failed")

    print(f"\n✓ Output saved: {output_path}")


def main():
    args = parse_args()

    # Check files exist
    if not Path(args.video).exists():
        print(f"ERROR: Video not found: {args.video}")
        sys.exit(1)

    if not Path(args.model).exists():
        print(f"ERROR: Model checkpoint not found: {args.model}")
        sys.exit(1)

    # Get config path
    config_path = get_config_path(args.type)
    if not Path(config_path).exists():
        print(f"ERROR: Config not found: {config_path}")
        print("Please run training first to generate config file")
        sys.exit(1)

    # Auto-generate output name
    if args.output is None:
        video_path = Path(args.video)
        args.output = f"{video_path.stem}_{args.type}_inference.mp4"

    print("\n" + "="*70)
    print("Instance Segmentation Video Inference")
    print("="*70)
    print(f"Video: {args.video}")
    print(f"Model: {args.model}")
    print(f"Type: {args.type.upper()}")
    print(f"Output: {args.output}")
    print(f"Confidence: {args.conf}")
    print("="*70)

    # Load model
    print("\nLoading model...")
    model = init_detector(config_path, args.model, device=args.device)

    # Process video
    process_video(args.video, model, args.output, args.conf, args.max_frames)

    print("\n" + "="*70)
    print("Inference Complete!")
    print("="*70)
    print(f"Output: {args.output}")
    print("Format: Original | Colored Mask | Overlay")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
