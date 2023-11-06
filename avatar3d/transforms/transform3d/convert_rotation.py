import torch
import numpy as np

from typing import Union

from mmhuman3d.utils.transforms import (
    rot6d_to_aa, rot6d_to_ee, rot6d_to_quat, rot6d_to_rotmat, rotmat_to_aa,
    rotmat_to_ee, rotmat_to_quat, rotmat_to_rot6d, ee_to_aa, ee_to_quat,
    ee_to_rot6d, ee_to_rotmat, aa_to_ee, aa_to_quat, aa_to_rot6d, aa_to_rotmat,
    quat_to_aa, quat_to_ee, quat_to_rot6d, quat_to_rotmat)


def flip_rotation(pose_vector: Union[torch.Tensor, np.ndarray],
                  pose_format: str = 'aa'):
    assert pose_format in ['aa', 'rotmat', 'quat', 'rot6d']
    assert isinstance(
        pose_vector,
        (torch.Tensor,
         np.ndarray)), 'pose_vector must be either torch.Tensor or np.ndarray'
    if isinstance(pose_vector, np.ndarray):
        rotation = pose_vector.copy()
    elif isinstance(pose_vector, torch.Tensor):
        rotation = pose_vector.clone()
    ori_shape = rotation.shape
    if pose_format == 'aa':
        rotation = rotation.reshape(-1, 3)
        pose_vector = pose_vector.reshape(rotation.shape)
        rotation[:, 1:] = pose_vector[:, 1:] * -1
        return rotation.reshape(ori_shape)

    elif pose_format == 'rotmat':
        rotation = rotation.reshape(-1, 9)
        pose_vector = pose_vector.reshape(rotation.shape)
        rotation[:, [1, 2, 3, 6]] = pose_vector[:, [1, 2, 3, 6]] * -1
        return rotation.reshape(ori_shape)

    elif pose_format == 'quat':
        rotation = rotation.reshape(-1, 4)
        pose_vector = pose_vector.reshape(rotation.shape)
        rotation[:, [2, 3]] = pose_vector[:, [2, 3]] * -1
        return rotation.reshape(ori_shape)

    elif pose_format == 'rot6d':
        rotation = rotation.reshape(-1, 6)
        pose_vector = pose_vector.reshape(rotation.shape)
        rotation[:, [1, 2, 3]] = pose_vector[:, [1, 2, 3]] * -1
        return rotation.reshape(ori_shape)
    else:
        raise ValueError(f'Unknown rotation format: {type(pose_format)}')
