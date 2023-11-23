from smplx.body_models import FLAME as _FLAME

from zolly.models.body_models.base import ParametricModelBase
from .mappings.flame import FLAME_JOINTS


class FLAME(ParametricModelBase, _FLAME):
    NUM_VERTS = 5023
    NUM_FACES = 9976
    JOINT_NAMES = FLAME_JOINTS
    body_pose_dims = {
        'global_orient': 1,
        'neck_pose': 1,
        'jaw_pose': 1,
        'leye_pose': 1,
        'reye_pose': 1,
    }
    full_pose_dims = {
        'global_orient': 1,
        'neck_pose': 1,
        'jaw_pose': 1,
        'leye_pose': 1,
        'reye_pose': 1,
    }

    full_param_dims = {
        'global_orient': 1 * 3,
        'neck_pose': 1 * 3,
        'jaw_pose': 1 * 3,
        'leye_pose': 1 * 3,
        'reye_pose': 1 * 3,
        'expression': 10,
        'transl': 3,
        'betas': 10,
    }
    _parents = [-1, 0, 1, 1, 1]

    def __init__(self, **kwargs) -> None:
        output = dict(super().__init__(**kwargs))
        joints, joint_mask = super().forward_joints(output)
        output.update(joints=joints, joint_mask=joint_mask)
        return output
