"""
Convert YOLO segmentation format to COCO format for MMDetection
"""
import json
import os
from pathlib import Path
from PIL import Image
import yaml
from tqdm import tqdm


def convert_yolo_to_coco(yolo_root, output_root, split='train'):
    """
    Convert YOLO segmentation dataset to COCO format

    Args:
        yolo_root: Path to YOLO dataset root
        output_root: Path to save COCO format dataset
        split: 'train', 'val', or 'test'
    """

    # Read data.yaml
    with open(os.path.join(yolo_root, 'data.yaml'), 'r') as f:
        data_config = yaml.safe_load(f)

    class_names = data_config['names']
    num_classes = data_config['nc']

    print(f"\nConverting {split} split...")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")

    # Map split name (train -> train, val -> valid for YOLO)
    yolo_split = 'valid' if split == 'val' else split

    # Paths
    images_dir = Path(yolo_root) / yolo_split / 'images'
    labels_dir = Path(yolo_root) / yolo_split / 'labels'

    output_dir = Path(output_root)
    output_images_dir = output_dir / split
    output_annotations_dir = output_dir / 'annotations'

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_annotations_dir.mkdir(parents=True, exist_ok=True)

    # Initialize COCO format structure
    coco_format = {
        'info': {
            'description': 'Converted from YOLO segmentation format',
            'version': '1.0',
            'year': 2024
        },
        'licenses': [],
        'images': [],
        'annotations': [],
        'categories': []
    }

    # Add categories
    for idx, name in enumerate(class_names):
        coco_format['categories'].append({
            'id': idx + 1,  # COCO categories start from 1
            'name': name,
            'supercategory': 'object'
        })

    # Get all image files
    image_files = sorted(images_dir.glob('*.jpg')) + sorted(images_dir.glob('*.png'))

    if len(image_files) == 0:
        print(f"WARNING: No images found in {images_dir}")
        return None, None

    print(f"Found {len(image_files)} images")

    annotation_id = 1

    for image_id, image_path in enumerate(tqdm(image_files, desc=f"Processing {split}")):
        # Read image to get dimensions
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Error reading {image_path}: {e}")
            continue

        # Add image info (just filename, data_prefix will handle the split directory)
        image_info = {
            'id': image_id + 1,
            'file_name': image_path.name,
            'width': width,
            'height': height
        }
        coco_format['images'].append(image_info)

        # Copy image to output directory
        import shutil
        shutil.copy2(image_path, output_images_dir / image_path.name)

        # Read corresponding label file
        label_path = labels_dir / (image_path.stem + '.txt')

        if not label_path.exists():
            continue

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 7:  # Need at least class_id + 3 points (6 coordinates)
                continue

            class_id = int(parts[0])

            # Convert normalized coordinates to absolute pixels
            # YOLO format: class_id x1 y1 x2 y2 x3 y3 ...
            coords = list(map(float, parts[1:]))

            # Convert to absolute coordinates and create segmentation polygon
            segmentation = []
            for i in range(0, len(coords), 2):
                x = coords[i] * width
                y = coords[i + 1] * height
                segmentation.extend([x, y])

            # Calculate bounding box from polygon
            x_coords = [segmentation[i] for i in range(0, len(segmentation), 2)]
            y_coords = [segmentation[i] for i in range(1, len(segmentation), 2)]

            x_min = min(x_coords)
            y_min = min(y_coords)
            x_max = max(x_coords)
            y_max = max(y_coords)

            bbox_width = x_max - x_min
            bbox_height = y_max - y_min

            # Calculate area (approximate using bounding box)
            area = bbox_width * bbox_height

            # Add annotation
            annotation = {
                'id': annotation_id,
                'image_id': image_id + 1,
                'category_id': class_id + 1,  # COCO categories start from 1
                'segmentation': [segmentation],  # List of polygons
                'area': float(area),
                'bbox': [float(x_min), float(y_min), float(bbox_width), float(bbox_height)],  # COCO format: [x, y, width, height]
                'iscrowd': 0
            }
            coco_format['annotations'].append(annotation)
            annotation_id += 1

    # Save COCO JSON
    output_json = output_annotations_dir / f'instances_{split}.json'
    with open(output_json, 'w') as f:
        json.dump(coco_format, f, indent=2)

    print(f"\nSaved {len(coco_format['images'])} images and {len(coco_format['annotations'])} annotations to {output_json}")
    print(f"Images copied to {output_images_dir}")

    return class_names, num_classes
