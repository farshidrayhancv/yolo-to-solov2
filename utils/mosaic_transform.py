"""
Custom Mosaic augmentation for instance segmentation (with mask support)
Based on MMDetection's Mosaic but extended to handle gt_masks
"""
import numpy as np
import cv2
from typing import List, Tuple
from mmcv.transforms import BaseTransform
from mmdet.structures.bbox import HorizontalBoxes
from mmdet.structures.mask import BitmapMasks
from mmengine.registry import TRANSFORMS


@TRANSFORMS.register_module()
class MosaicWithMasks(BaseTransform):
    """
    Mosaic augmentation with instance segmentation mask support.

    Combines 4 images into a single mosaic image, properly handling:
    - Bounding boxes (gt_bboxes)
    - Instance masks (gt_masks)
    - Class labels (gt_bboxes_labels)

    Required Keys:
    - img
    - gt_bboxes (HorizontalBoxes)
    - gt_bboxes_labels (np.ndarray)
    - gt_masks (BitmapMasks)
    - mix_results (List[dict]) - 3 additional images from dataset

    Modified Keys:
    - img
    - img_shape
    - gt_bboxes
    - gt_bboxes_labels
    - gt_masks

    Args:
        img_scale (Tuple[int, int]): Target output size (width, height)
        center_ratio_range (Tuple[float, float]): Range for mosaic center point
        pad_val (int): Padding value for image borders
        prob (float): Probability of applying mosaic (default: 1.0)
    """

    def __init__(self,
                 img_scale: Tuple[int, int] = (640, 640),
                 center_ratio_range: Tuple[float, float] = (0.5, 1.5),
                 pad_val: int = 114,
                 prob: float = 1.0):
        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.pad_val = pad_val
        self.prob = prob

    def transform(self, results: dict) -> dict:
        """Apply mosaic augmentation."""
        if np.random.rand() > self.prob:
            return results

        # Get mix_results (3 additional images from dataset)
        if 'mix_results' not in results or len(results['mix_results']) < 3:
            # Not enough images for mosaic, return original
            return results

        mix_results = results['mix_results'][:3]
        all_results = [results] + mix_results

        # Mosaic center point
        mosaic_h, mosaic_w = self.img_scale
        center_x = int(np.random.uniform(*self.center_ratio_range) * mosaic_w)
        center_y = int(np.random.uniform(*self.center_ratio_range) * mosaic_h)

        # Create mosaic image and mask canvas
        mosaic_img = np.full((mosaic_h, mosaic_w, 3), self.pad_val, dtype=np.uint8)

        # Collect all bboxes, labels, and masks
        mosaic_bboxes = []
        mosaic_labels = []
        mosaic_masks = []

        # Process each of the 4 images
        for i, result in enumerate(all_results):
            img = result['img']
            h, w = img.shape[:2]

            # Determine placement (top-left, top-right, bottom-left, bottom-right)
            if i == 0:  # Top-left
                x1a, y1a, x2a, y2a = 0, 0, center_x, center_y
                x1b, y1b, x2b, y2b = w - center_x, h - center_y, w, h
            elif i == 1:  # Top-right
                x1a, y1a, x2a, y2a = center_x, 0, mosaic_w, center_y
                x1b, y1b, x2b, y2b = 0, h - center_y, min(mosaic_w - center_x, w), h
            elif i == 2:  # Bottom-left
                x1a, y1a, x2a, y2a = 0, center_y, center_x, mosaic_h
                x1b, y1b, x2b, y2b = w - center_x, 0, w, min(mosaic_h - center_y, h)
            else:  # Bottom-right
                x1a, y1a, x2a, y2a = center_x, center_y, mosaic_w, mosaic_h
                x1b, y1b, x2b, y2b = 0, 0, min(mosaic_w - center_x, w), min(mosaic_h - center_y, h)

            # Ensure coordinates are valid
            x1a, x2a = max(0, x1a), min(mosaic_w, x2a)
            y1a, y2a = max(0, y1a), min(mosaic_h, y2a)
            x1b, x2b = max(0, x1b), min(w, x2b)
            y1b, y2b = max(0, y1b), min(h, y2b)

            # Calculate actual crop size
            crop_w = min(x2a - x1a, x2b - x1b)
            crop_h = min(y2a - y1a, y2b - y1b)

            if crop_w <= 0 or crop_h <= 0:
                continue

            # Place image
            mosaic_img[y1a:y1a+crop_h, x1a:x1a+crop_w] = img[y1b:y1b+crop_h, x1b:x1b+crop_w]

            # Transform bboxes and masks
            if 'gt_bboxes' in result and len(result['gt_bboxes']) > 0:
                bboxes = result['gt_bboxes'].tensor.numpy()  # Convert to numpy
                labels = result['gt_bboxes_labels']
                masks = result['gt_masks'].masks if 'gt_masks' in result else None

                # Offset bboxes to mosaic position
                offset_x = x1a - x1b
                offset_y = y1a - y1b

                for j, (bbox, label) in enumerate(zip(bboxes, labels)):
                    x1, y1, x2, y2 = bbox

                    # Transform bbox coordinates
                    new_x1 = x1 + offset_x
                    new_y1 = y1 + offset_y
                    new_x2 = x2 + offset_x
                    new_y2 = y2 + offset_y

                    # Clip to mosaic bounds
                    new_x1 = np.clip(new_x1, 0, mosaic_w)
                    new_y1 = np.clip(new_y1, 0, mosaic_h)
                    new_x2 = np.clip(new_x2, 0, mosaic_w)
                    new_y2 = np.clip(new_y2, 0, mosaic_h)

                    # Skip if bbox is too small after clipping
                    if new_x2 - new_x1 < 1 or new_y2 - new_y1 < 1:
                        continue

                    # Transform mask FIRST to ensure we have it
                    if masks is None or j >= len(masks):
                        # Skip this object if no mask available
                        continue

                    mask = masks[j].astype(np.uint8)
                    # Create mosaic mask canvas
                    mosaic_mask = np.zeros((mosaic_h, mosaic_w), dtype=np.uint8)
                    # Place mask at the correct position
                    mask_h, mask_w = mask.shape[:2]
                    # Crop mask to match image crop
                    mask_crop = mask[y1b:y1b+crop_h, x1b:x1b+crop_w]
                    # Place in mosaic
                    mosaic_mask[y1a:y1a+crop_h, x1a:x1a+crop_w] = mask_crop

                    # Only add bbox, label, and mask together (ensures synchronization)
                    mosaic_bboxes.append([new_x1, new_y1, new_x2, new_y2])
                    mosaic_labels.append(label)
                    mosaic_masks.append(mosaic_mask)

        # Update results
        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape[:2]

        if len(mosaic_bboxes) > 0:
            results['gt_bboxes'] = HorizontalBoxes(np.array(mosaic_bboxes, dtype=np.float32))
            results['gt_bboxes_labels'] = np.array(mosaic_labels, dtype=np.int64)

            if len(mosaic_masks) > 0:
                results['gt_masks'] = BitmapMasks(mosaic_masks, mosaic_h, mosaic_w)
        else:
            # No valid annotations after mosaic
            results['gt_bboxes'] = HorizontalBoxes(np.zeros((0, 4), dtype=np.float32))
            results['gt_bboxes_labels'] = np.array([], dtype=np.int64)
            results['gt_masks'] = BitmapMasks([], mosaic_h, mosaic_w)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'center_ratio_range={self.center_ratio_range}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'prob={self.prob})'
        return repr_str
