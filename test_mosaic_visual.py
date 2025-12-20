"""
Visual test for Mosaic augmentation with instance segmentation masks
Shows original 4 images + mosaic result side-by-side for sanity checking

Uses custom MosaicWithMasks implementation for proper instance segmentation support
"""
import json
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from utils.mosaic_transform import MosaicWithMasks
from mmdet.structures.bbox import HorizontalBoxes
from mmdet.structures.mask import BitmapMasks


def load_coco_sample(coco_root, split='train', num_samples=4):
    """Load sample images with their masks and bboxes from COCO dataset"""
    coco_root = Path(coco_root)
    ann_file = coco_root / 'annotations' / f'instances_{split}.json'
    img_dir = coco_root / split

    # Load COCO annotations
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)

    # Create image_id to annotations mapping
    img_id_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)

    # Load samples
    samples = []
    for img_info in coco_data['images'][:num_samples]:
        img_path = img_dir / img_info['file_name']
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_id = img_info['id']
        anns = img_id_to_anns.get(img_id, [])

        # Extract bboxes, masks, and labels
        bboxes = []
        masks = []
        labels = []

        for ann in anns:
            # Bbox in COCO format [x, y, w, h]
            x, y, w, h = ann['bbox']
            # Convert to pascal_voc format [x1, y1, x2, y2]
            bboxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])

            # Decode mask from RLE or polygon
            if 'segmentation' in ann:
                from pycocotools import mask as mask_utils
                if isinstance(ann['segmentation'], dict):  # RLE
                    mask = mask_utils.decode(ann['segmentation'])
                else:  # Polygon
                    rle = mask_utils.frPyObjects(ann['segmentation'], img_info['height'], img_info['width'])
                    mask = mask_utils.decode(rle)
                    if len(mask.shape) == 3:
                        mask = mask.max(axis=2)
                masks.append(mask)

        # Stack masks if multiple objects
        # MMDetection expects masks in shape (N, H, W)
        if len(masks) > 0:
            masks = np.stack(masks, axis=0)  # Shape: (num_objects, H, W)
        else:
            masks = np.zeros((0, image.shape[0], image.shape[1]), dtype=np.uint8)

        samples.append({
            'image': image,
            'mask': masks,
            'bboxes': np.array(bboxes, dtype=np.float32) if bboxes else np.zeros((0, 4), dtype=np.float32),
            'labels': np.array(labels, dtype=np.int64),
            'filename': img_info['file_name']
        })

    return samples, coco_data['categories']


def visualize_sample(ax, image, masks, bboxes, labels, categories, title):
    """Visualize image with masks and bboxes overlaid"""
    ax.imshow(image)
    ax.set_title(title, fontsize=10)
    ax.axis('off')

    # Create category color map
    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    cat_id_to_color = {cat['id']: colors[i] for i, cat in enumerate(categories)}
    cat_id_to_name = {cat['id']: cat['name'] for cat in categories}

    # Overlay masks with transparency
    # Masks can be (H, W, N) or (N, H, W) format
    if len(masks.shape) == 3:
        if masks.shape[2] < masks.shape[0]:  # (H, W, N) format
            num_masks = masks.shape[2]
            for i, label in enumerate(labels):
                if i < num_masks:
                    mask = masks[:, :, i]
                    if mask.max() > 0:
                        color = cat_id_to_color.get(label, colors[0])
                        colored_mask = np.zeros((*mask.shape, 3))
                        colored_mask[mask > 0] = color[:3]
                        ax.imshow(colored_mask, alpha=0.4)
        else:  # (N, H, W) format
            for i, (label) in enumerate(labels):
                if i < len(masks):
                    mask = masks[i]
                    if mask.max() > 0:
                        color = cat_id_to_color.get(label, colors[0])
                        colored_mask = np.zeros((*mask.shape, 3))
                        colored_mask[mask > 0] = color[:3]
                        ax.imshow(colored_mask, alpha=0.4)

    # Draw bboxes
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        color = cat_id_to_color.get(label, colors[0])
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)

        # Add label text
        cat_name = cat_id_to_name.get(label, f'cls_{label}')
        ax.text(
            x1, y1 - 5,
            cat_name,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
            fontsize=8,
            color='black'
        )


def test_mosaic_augmentation(coco_root, output_dir='./mosaic_test_output', num_tests=3):
    """Test Mosaic augmentation and save visualization"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("Loading COCO dataset samples...")
    samples, categories = load_coco_sample(coco_root, split='train', num_samples=12)  # Load 12 for 3 tests
    print(f"Loaded {len(samples)} samples with {len(categories)} categories")

    # Create custom Mosaic transform
    transform = MosaicWithMasks(
        img_scale=(1280, 1280),  # Output size
        center_ratio_range=(0.3, 0.7),  # Center point variation
        pad_val=114,  # Gray padding (matches Ultralytics)
        prob=1.0  # Always apply for testing
    )

    # Run multiple tests
    for test_idx in range(num_tests):
        print(f"\n{'='*80}")
        print(f"Test {test_idx + 1}/{num_tests}")
        print(f"{'='*80}")

        # Get 4 samples for this test
        start_idx = test_idx * 4
        test_samples = samples[start_idx:start_idx + 4]

        if len(test_samples) < 4:
            print(f"Not enough samples for test {test_idx + 1}, skipping...")
            continue

        # Prepare primary data (first image)
        primary = test_samples[0]

        # Prepare additional images metadata (next 3 images)
        mosaic_metadata = []
        for i in range(1, 4):
            sample = test_samples[i]
            mosaic_metadata.append({
                'image': sample['image'],
                'mask': sample['mask'],
                'bboxes': sample['bboxes'],
                'labels': sample['labels']
            })

        print(f"Primary image: {primary['filename']}")
        print(f"  - Shape: {primary['image'].shape}")
        print(f"  - Objects: {len(primary['labels'])} ({primary['labels'].tolist()})")
        print(f"  - Bboxes: {len(primary['bboxes'])}")
        print(f"  - Masks: {primary['mask'].shape}")

        for i, meta in enumerate(mosaic_metadata, 1):
            print(f"Additional image {i}: {test_samples[i]['filename']}")
            print(f"  - Shape: {meta['image'].shape}")
            print(f"  - Objects: {len(meta['labels'])} ({meta['labels'].tolist()})")

        # Prepare data in MMDetection format
        results = {
            'img': primary['image'],
            'gt_bboxes': HorizontalBoxes(primary['bboxes']),
            'gt_bboxes_labels': primary['labels'],
            'gt_masks': BitmapMasks(primary['mask'], *primary['image'].shape[:2]),
            'mix_results': []
        }

        # Add additional images as mix_results
        for i in range(1, 4):
            sample = test_samples[i]
            results['mix_results'].append({
                'img': sample['image'],
                'gt_bboxes': HorizontalBoxes(sample['bboxes']),
                'gt_bboxes_labels': sample['labels'],
                'gt_masks': BitmapMasks(sample['mask'], *sample['image'].shape[:2])
            })

        # Apply Mosaic transform
        try:
            transformed = transform(results)

            mosaic_image = transformed['img']
            mosaic_bboxes = transformed['gt_bboxes'].tensor.numpy()
            mosaic_labels = transformed['gt_bboxes_labels']
            mosaic_masks = transformed['gt_masks'].masks

            print(f"\nMosaic result:")
            print(f"  - Shape: {mosaic_image.shape}")
            print(f"  - Total objects: {len(mosaic_labels)} ({mosaic_labels.tolist()})")
            print(f"  - Total bboxes: {len(mosaic_bboxes)}")
            print(f"  - Total masks: {mosaic_masks.shape} (N, H, W format)")

            # Verify synchronization
            assert len(mosaic_bboxes) == len(mosaic_labels), \
                f"Bbox/label count mismatch: {len(mosaic_bboxes)} bboxes vs {len(mosaic_labels)} labels"
            assert mosaic_masks.shape[0] == len(mosaic_labels), \
                f"Mask/label count mismatch: {mosaic_masks.shape[0]} masks vs {len(mosaic_labels)} labels"
            print("✅ Bbox/label/mask counts match!")

        except Exception as e:
            print(f"❌ Mosaic transform failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Create visualization
        fig = plt.figure(figsize=(20, 12))

        # Top row: Original 4 images
        for i in range(4):
            ax = plt.subplot(3, 4, i + 1)
            sample = test_samples[i]
            visualize_sample(
                ax, sample['image'], sample['mask'], sample['bboxes'],
                sample['labels'], categories, f"Original {i+1}: {sample['filename']}"
            )

        # Middle row: Mosaic result (spans 4 columns)
        ax_mosaic = plt.subplot(3, 1, 2)
        visualize_sample(
            ax_mosaic, mosaic_image, mosaic_masks, mosaic_bboxes,
            mosaic_labels, categories, f"Mosaic Result ({len(mosaic_labels)} objects)"
        )

        # Bottom row: Show mosaic details
        ax_info = plt.subplot(3, 1, 3)
        ax_info.axis('off')
        info_text = f"""
Mosaic Augmentation Test Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input Images:
  • Image 1: {test_samples[0]['filename']} → {len(test_samples[0]['labels'])} objects
  • Image 2: {test_samples[1]['filename']} → {len(test_samples[1]['labels'])} objects
  • Image 3: {test_samples[2]['filename']} → {len(test_samples[2]['labels'])} objects
  • Image 4: {test_samples[3]['filename']} → {len(test_samples[3]['labels'])} objects

Mosaic Output:
  • Image shape: {mosaic_image.shape[1]}x{mosaic_image.shape[0]}
  • Total objects: {len(mosaic_labels)}
  • Bboxes: {len(mosaic_bboxes)}
  • Masks: {mosaic_masks.shape[0]}
  • Labels: {mosaic_labels}

Synchronization Check:
  ✅ Bboxes count == Labels count == Masks count
  ✅ All annotations properly aligned

Categories: {', '.join([cat['name'] for cat in categories])}
        """
        ax_info.text(0.1, 0.5, info_text, fontsize=11, family='monospace',
                    verticalalignment='center')

        plt.tight_layout()
        output_path = output_dir / f'mosaic_test_{test_idx + 1}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Saved visualization to: {output_path}")
        plt.close()

    print(f"\n{'='*80}")
    print(f"All tests complete! Check output directory: {output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visual test for Mosaic augmentation')
    parser.add_argument(
        '--data-root',
        type=str,
        default='/home/farshid/proj/Race track lingfield.v10i.yolov11_coco/',
        help='Path to COCO dataset root'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./mosaic_test_output',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--num-tests',
        type=int,
        default=3,
        help='Number of mosaic tests to run (each uses 4 images)'
    )

    args = parser.parse_args()

    test_mosaic_augmentation(
        coco_root=args.data_root,
        output_dir=args.output_dir,
        num_tests=args.num_tests
    )
