from mmhuman3d.core.conventions.keypoints_mapping.coco import COCO_KEYPOINTS
from .mmpose_hand import MMPOSE_HAND_KEYPOINTS_LEFT, MMPOSE_HAND_KEYPOINTS_RIGHT

COCO_HAND = COCO_KEYPOINTS + MMPOSE_HAND_KEYPOINTS_LEFT + MMPOSE_HAND_KEYPOINTS_RIGHT