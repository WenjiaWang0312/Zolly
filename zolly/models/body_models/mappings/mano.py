MANO_KEYPOINTS_LEFT = [
    'left_wrist',
    'left_index_1',
    'left_index_2',
    'left_index_3',
    'left_middle_1',
    'left_middle_2',
    'left_middle_3',
    'left_pinky_1',
    'left_pinky_2',
    'left_pinky_3',
    'left_ring_1',
    'left_ring_2',
    'left_ring_3',
    'left_thumb_1',
    'left_thumb_2',
    'left_thumb_3',
    'left_thumb',
    'left_index',
    'left_middle',
    'left_ring',
    'left_pinky',
]

MANO_KEYPOINTS_RIGHT = [
    'right_wrist',
    'right_index_1',
    'right_index_2',
    'right_index_3',
    'right_middle_1',
    'right_middle_2',
    'right_middle_3',
    'right_pinky_1',
    'right_pinky_2',
    'right_pinky_3',
    'right_ring_1',
    'right_ring_2',
    'right_ring_3',
    'right_thumb_1',
    'right_thumb_2',
    'right_thumb_3',
    'right_thumb',
    'right_index',
    'right_middle',
    'right_ring',
    'right_pinky',
]

MANO_JOINTS_LEFT = MANO_KEYPOINTS_LEFT.copy()[:16]
MANO_JOINTS_RIGHT = MANO_KEYPOINTS_RIGHT.copy()[:16]

MANO_KEYPOINTS_FULL = MANO_KEYPOINTS_LEFT + MANO_KEYPOINTS_RIGHT
