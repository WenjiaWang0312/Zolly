import numpy as np
import torch
from typing import Union
from .smpl import SMPL_JOINTS, SMPL_14
from .mano import (MANO_KEYPOINTS_LEFT, MANO_KEYPOINTS_RIGHT, MANO_JOINTS_LEFT,
                   MANO_JOINTS_RIGHT, MANO_KEYPOINTS_FULL)
from .smplh import SMPLH_KEYPOINTS, SMPLH_JOINTS, SMPLH_52
from .smplx import SMPLX_JOINTS, SMPLX_JOINTS_BODY, SMPLX_KEYPOINTS_BODY
from .flame import (FLAME_JOINTS, FLAME_KEYPOINTS_NO_CONTOUR, FLAME_KEYPOINTS,
                    FLAME_JAW_KEYPOINTS, FLAME_JAW_JOINTS, FLAME_KEYPOINTS_LIGHT)
from .mmpose_hand import (MMPOSE_HAND_KEYPOINTS_LEFT,
                          MMPOSE_HAND_KEYPOINTS_RIGHT)
from .people_snapshot import PEOPLE_SNAPSHOT_KEYPOINTS
from .coco_hand import COCO_HAND
from .h36m import H36M_EVAL_J17, H36M_EVAL_J14, H36M_EVAL_J14_SMPL
from mmhuman3d.core.conventions.keypoints_mapping import (
    KEYPOINTS_FACTORY as _KEYPOINTS_FACTORY, convert_kps as _convert_kps,
    get_flip_pairs as _get_flip_pairs, get_keypoint_idx as _get_keypoint_idx,
    get_keypoint_idxs_by_part as _get_keypoint_idxs_by_part, get_keypoint_num
    as _get_keypoint_num, get_mapping as _get_mapping)

KEYPOINTS_FACTORY = _KEYPOINTS_FACTORY.copy()
KEYPOINTS_FACTORY.update(
    dict(mano_left=MANO_KEYPOINTS_LEFT,
         mano=MANO_KEYPOINTS_LEFT,
         mano_right=MANO_KEYPOINTS_RIGHT,
         mano_full=MANO_KEYPOINTS_FULL,
         smplx_body=SMPLX_KEYPOINTS_BODY,
         smpl_14=SMPL_14,
         smplh=SMPLH_KEYPOINTS,
         smplh_52=SMPLH_52,
         flame=FLAME_KEYPOINTS,
         flame_light=FLAME_KEYPOINTS_LIGHT,
         flame_jaw=FLAME_JAW_KEYPOINTS,
         flame_no_contour=FLAME_KEYPOINTS_NO_CONTOUR,
         mmpose_hand=MMPOSE_HAND_KEYPOINTS_LEFT,
         mmpose_hand_left=MMPOSE_HAND_KEYPOINTS_LEFT,
         mmpose_hand_right=MMPOSE_HAND_KEYPOINTS_RIGHT,
         people_snapshot=PEOPLE_SNAPSHOT_KEYPOINTS,
         coco_hand=COCO_HAND,
         h36m_eval_j17=H36M_EVAL_J17,
         h36m_eval_j14=H36M_EVAL_J14,
         h36m_eval_j14_smpl=H36M_EVAL_J14_SMPL))


def convert_kps(
    keypoints: Union[np.ndarray, torch.Tensor],
    src: str,
    dst: str,
    approximate: bool = False,
    mask: Union[np.ndarray, torch.Tensor] = None,
    keypoints_factory: dict = KEYPOINTS_FACTORY,
    return_mask: bool = True,
):
    return _convert_kps(keypoints,
                        src=src,
                        dst=dst,
                        approximate=approximate,
                        mask=mask,
                        keypoints_factory=keypoints_factory,
                        return_mask=return_mask)


JOINTS_FACTORY = dict(
    smpl=SMPL_JOINTS,
    smplh=SMPLH_JOINTS,
    flame=FLAME_JOINTS,
    flame_jaw=FLAME_JAW_JOINTS,
    flame_light=FLAME_JAW_JOINTS,
    smplx=SMPLX_JOINTS,
    smplx_body=SMPLX_JOINTS_BODY,
    mano=MANO_JOINTS_LEFT,
    mano_left=MANO_JOINTS_LEFT,
    mano_right=MANO_JOINTS_RIGHT,
)


def flip_keypoints(keypoints_names):
    flipped_keypoints_names = []
    for name in keypoints_names:
        if 'left' in name:
            corresponding_name = name.replace('left', 'right')
        elif 'right' in name:
            corresponding_name = name.replace('right', 'left')
        else:
            corresponding_name = name
        flipped_keypoints_names.append(corresponding_name)
    return flipped_keypoints_names


def flip_keypoints_mapping(src, keypoints_factory=KEYPOINTS_FACTORY):
    src_names = keypoints_factory[src]
    mapping_indexes = []
    for name in src_names:
        if 'left' in name:
            corresponding_name = name.replace('left', 'right')
        elif 'right' in name:
            corresponding_name = name.replace('right', 'left')
        else:
            corresponding_name = name
        index = src_names.index(corresponding_name)
        mapping_indexes.append(index)
    return mapping_indexes


def map_poses(poses: Union[np.ndarray, torch.Tensor],
              src: str,
              dst: str,
              keypoints_factory: dict = JOINTS_FACTORY):
    return _convert_kps(poses,
                        src=src,
                        dst=dst,
                        approximate=False,
                        keypoints_factory=keypoints_factory,
                        return_mask=False)


def format_coco_kps(kps: Union[np.ndarray, torch.Tensor],
                    convention: str = 'coco_hand',
                    keypoints_factory: dict = KEYPOINTS_FACTORY):
    kps = kps.copy() if isinstance(kps, np.ndarray) else kps.clone()
    assert convention in ['coco_hand', 'coco_wholebody']
    assert kps.ndim == 3
    mask = kps[0, :, -1]

    lwrist_index = keypoints_factory[convention].index('left_wrist')
    lhand_root_index = keypoints_factory[convention].index('left_hand_root')
    if mask[lhand_root_index] > 0 and mask[lwrist_index] == 0:
        kps[:, lwrist_index] = kps[:, lhand_root_index]
    elif mask[lhand_root_index] == 0 and mask[lwrist_index] > 0:
        kps[:, lhand_root_index] = kps[:, lwrist_index]

    rwrist_index = keypoints_factory[convention].index('right_wrist')
    rhand_root_index = keypoints_factory[convention].index('right_hand_root')
    if mask[rhand_root_index] > 0 and mask[rwrist_index] == 0:
        kps[:, rwrist_index] = kps[:, rhand_root_index]
    elif mask[rhand_root_index] == 0 and mask[rwrist_index] > 0:
        kps[:, rhand_root_index] = kps[:, rwrist_index]

    return kps


def get_flip_pairs(convention: str = 'smplx',
                   keypoints_factory: dict = KEYPOINTS_FACTORY):
    return _get_flip_pairs(convention, keypoints_factory)


def get_keypoint_idx(name: str,
                     convention: str = 'smplx',
                     approximate: bool = False,
                     keypoints_factory: dict = KEYPOINTS_FACTORY):
    return _get_keypoint_idx(name, convention, approximate, keypoints_factory)


def get_keypoint_idxs_by_part(part: str,
                              convention: str = 'smplx',
                              keypoints_factory: dict = KEYPOINTS_FACTORY):
    return _get_keypoint_idxs_by_part(part, convention, keypoints_factory)


def get_keypoint_num(convention: str = 'smplx',
                     keypoints_factory: dict = KEYPOINTS_FACTORY):
    return _get_keypoint_num(convention, keypoints_factory)


def get_joints_num(convention: str = 'smplx',
                   joints_factory: dict = JOINTS_FACTORY):
    return _get_keypoint_num(convention, joints_factory)


def get_mapping(src: str,
                dst: str,
                approximate: bool = False,
                keypoints_factory: dict = KEYPOINTS_FACTORY):
    return _get_mapping(src, dst, approximate, keypoints_factory)


__all__ = [
    'MANO_KEYPOINTS_LEFT', 'MANO_KEYPOINTS_RIGHT', 'SMPLH_KEYPOINTS',
    'KEYPOINTS_FACTORY', 'convert_kps'
]
