"""
Skip Empty Annotations Transform
Mimics Ultralytics behavior: when augmentation results in empty annotations, skip the sample
"""
from mmcv.transforms import BaseTransform
from mmengine.registry import TRANSFORMS


@TRANSFORMS.register_module()
class SkipEmptyAnnotations(BaseTransform):
    """
    Validates that the sample has at least one valid annotation after augmentation.
    Returns None if empty, signaling the dataloader to skip this sample.

    This mimics Ultralytics YOLO behavior where empty augmented samples are gracefully skipped.

    Required Keys:
    - gt_bboxes
    - gt_masks

    Returns:
    - results (dict) if valid annotations exist
    - None if no valid annotations (signals skip)
    """

    def transform(self, results: dict):
        """Check if sample has valid annotations."""
        # Check if we have any bounding boxes
        if 'gt_bboxes' not in results or len(results['gt_bboxes']) == 0:
            return None  # Skip this sample

        # Check if we have any masks
        if 'gt_masks' not in results or len(results['gt_masks']) == 0:
            return None  # Skip this sample

        # Check if masks are not empty (have at least one pixel)
        if hasattr(results['gt_masks'], 'masks'):
            masks = results['gt_masks'].masks
            if len(masks) == 0:
                return None

        return results

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'
