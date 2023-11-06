from mmhuman3d.core.conventions.keypoints_mapping.smplx import SMPLX_KEYPOINTS

SMPLX_JOINTS = SMPLX_KEYPOINTS[:55]
SMPLX_JOINTS_BODY = SMPLX_JOINTS[:22]
SMPLX_KEYPOINTS_BODY = SMPLX_JOINTS_BODY + ['left_middle_1', 'right_middle_1', 'jaw', 'left_bigtoe', 'right_bigtoe']
SMPLX_KEYPOINTS_PARTS = dict(
    torso=(
        'pelvis',
        'left_hip',
        'right_hip',
        'spine_1',
        'spine_2',
        'left_ankle',
        'right_ankle',
        'spine_3',
        'left_collar',
        'right_collar',
        'left_shoulder',
        'right_shoulder',
    ),
    limbs=(
        'left_knee',
        'right_knee',
        'left_elbow',
        'right_elbow',
    ),
    head=(
        'head',
        'jaw',
        'left_eyeball',
        'right_eyeball',
        'nose',
        'right_eye',
        'left_eye',
        'right_ear',
        'left_ear',
    ),
    left_hand=(
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
    ),
    left_palm=('left_wrist', 'left_thumb1', 'left_index_1', 'left_middle_1',
               'left_ring_1', 'left_pinky_1'),
    right_palm=('right_wrist', 'right_thumb1', 'right_index_1',
                'right_middle_1', 'right_ring_1', 'right_pinky_1'),
    right_hand=(
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
    ),
    left_foot=(
        'left_foot',
        'left_bigtoe',
        'left_smalltoe',
        'left_heel',
    ),
    right_foot=(
        'right_foot',
        'right_bigtoe',
        'right_smalltoe',
        'right_heel',
    ),
    face=(
        'right_eyebrow_1',
        'right_eyebrow_2',
        'right_eyebrow_3',
        'right_eyebrow_4',
        'right_eyebrow_5',
        'left_eyebrow_5',
        'left_eyebrow_4',
        'left_eyebrow_3',
        'left_eyebrow_2',
        'left_eyebrow_1',
        'nosebridge_1',
        'nosebridge_2',
        'nosebridge_3',
        'nosebridge_4',
        'right_nose_2',  # original name: nose_1
        'right_nose_1',  # original name: nose_2
        'nose_middle',  # original name: nose_3
        'left_nose_1',  # original name: nose_4
        'left_nose_2',  # original name: nose_5
        'right_eye_1',
        'right_eye_2',
        'right_eye_3',
        'right_eye_4',
        'right_eye_5',
        'right_eye_6',
        'left_eye_4',
        'left_eye_3',
        'left_eye_2',
        'left_eye_1',
        'left_eye_6',
        'left_eye_5',
        'right_mouth_1',  # original name: mouth_1
        'right_mouth_2',  # original name: mouth_2
        'right_mouth_3',  # original name: mouth_3
        'mouth_top',  # original name: mouth_4
        'left_mouth_3',  # original name: mouth_5
        'left_mouth_2',  # original name: mouth_6
        'left_mouth_1',  # original name: mouth_7
        'left_mouth_5',  # original name: mouth_8
        'left_mouth_4',  # original name: mouth_9
        'mouth_bottom',  # original name: mouth_10
        'right_mouth_4',  # original name: mouth_11
        'right_mouth_5',  # original name: mouth_12
        'right_lip_1',  # original name: lip_1
        'right_lip_2',  # original name: lip_2
        'lip_top',  # original name: lip_3
        'left_lip_2',  # original name: lip_4
        'left_lip_1',  # original name: lip_5
        'left_lip_3',  # original name: lip_6
        'lip_bottom',  # original name: lip_7
        'right_lip_3',  # original name: lip_8
        'right_contour_1',  # original name: face_contour_1
        'right_contour_2',  # original name: face_contour_2
        'right_contour_3',  # original name: face_contour_3
        'right_contour_4',  # original name: face_contour_4
        'right_contour_5',  # original name: face_contour_5
        'right_contour_6',  # original name: face_contour_6
        'right_contour_7',  # original name: face_contour_7
        'right_contour_8',  # original name: face_contour_8
        'contour_middle',  # original name: face_contour_9
        'left_contour_8',  # original name: face_contour_10
        'left_contour_7',  # original name: face_contour_11
        'left_contour_6',  # original name: face_contour_12
        'left_contour_5',  # original name: face_contour_13
        'left_contour_4',  # original name: face_contour_14
        'left_contour_3',  # original name: face_contour_15
        'left_contour_2',  # original name: face_contour_16
        'left_contour_1',  # original name: face_contour_17),
    ),
)
