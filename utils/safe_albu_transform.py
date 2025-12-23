"""
Safe Albumentations Transform Wrapper
Wraps the standard Albu transform with empty mask handling to prevent crashes
"""
from typing import Dict, List, Optional, Union
import numpy as np
from mmdet.datasets.transforms import Albu
from mmengine.registry import TRANSFORMS


@TRANSFORMS.register_module()
class SafeAlbu(Albu):
    """
    Extends the standard Albu transform with empty mask handling.

    If augmentation results in empty masks/bboxes, returns None to skip the sample
    instead of crashing. This mimics Ultralytics behavior where samples that lose
    all annotations during augmentation are gracefully skipped.

    This solves the IndexError at line 1770 in transforms.py where MMDetection
    tries to access results['masks'][0] on an empty list after aggressive augmentation
    removes all objects from the image.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def transform(self, results: dict) -> Optional[dict]:
        """
        Apply albumentation transforms with empty mask protection.

        Args:
            results: Data dictionary containing images, masks, and bboxes

        Returns:
            results (dict) if successful and has valid annotations
            None if augmentation resulted in empty masks/bboxes (skip sample)
        """
        # Check if input has valid annotations BEFORE augmentation
        if 'gt_bboxes' not in results or len(results.get('gt_bboxes', [])) == 0:
            return None

        if 'gt_masks' not in results or len(results.get('gt_masks', [])) == 0:
            return None

        # Store original counts for debugging
        orig_bbox_count = len(results['gt_bboxes'])
        orig_mask_count = len(results['gt_masks']) if hasattr(results['gt_masks'], '__len__') else 0

        # Apply albumentation transforms
        # Catch any errors that occur during augmentation
        try:
            # Call parent's transform, but protect against empty results
            # We override _postprocess_results to handle empty masks gracefully
            results = super().transform(results)

            # After augmentation, verify we still have valid annotations
            if results is None:
                return None

            # Check if augmentation removed all objects
            if 'gt_bboxes' not in results or len(results.get('gt_bboxes', [])) == 0:
                return None

            if 'gt_masks' not in results:
                return None

            # Check mask validity
            masks = results.get('gt_masks')
            if masks is None:
                return None

            # Different mask formats need different checks
            if hasattr(masks, 'masks'):
                # BitmapMasks format
                if len(masks.masks) == 0:
                    return None
            elif hasattr(masks, '__len__'):
                # List format
                if len(masks) == 0:
                    return None

            return results

        except IndexError as e:
            # If we catch the IndexError from empty masks, skip this sample
            # This is the specific error that crashes at line 1770
            if "index 0 is out of bounds" in str(e):
                return None
            # Re-raise other IndexErrors
            raise
        except Exception as e:
            # Log unexpected errors but don't crash
            print(f"SafeAlbu: Unexpected error during augmentation: {e}")
            print(f"  Original: {orig_bbox_count} boxes, {orig_mask_count} masks")
            # Skip this sample
            return None

    def _postprocess_results(self, results: dict, ori_masks: Optional[List]) -> Optional[dict]:
        """
        Override parent's _postprocess_results to handle empty masks gracefully.

        This is where the crash occurs at line 1770 in the original implementation.
        """
        # Check if masks are empty BEFORE trying to access them
        if 'masks' not in results or len(results.get('masks', [])) == 0:
            # Return None to signal empty annotations
            return None

        # If we have masks, call the parent implementation
        try:
            return super()._postprocess_results(results, ori_masks)
        except IndexError as e:
            # If parent crashes due to empty masks, return None
            if "index 0 is out of bounds" in str(e):
                return None
            raise

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'bbox_params={self.bbox_params}, '
        repr_str += f'keymap={self.keymap}, '
        repr_str += f'skip_img_without_anno={self.skip_img_without_anno})'
        return repr_str
