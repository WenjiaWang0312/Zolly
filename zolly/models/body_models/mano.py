from smplx.body_models import MANO as _MANO
from smplx.body_models import lbs
import torch
from zolly.models.body_models.base import ParametricModelBase
from zolly.transforms.transform3d import rotmat_to_aa
from zolly.utils.torch_utils import cat_pose_list
from .mappings.mano import MANO_JOINTS_LEFT, MANO_JOINTS_RIGHT


class MANO(ParametricModelBase, _MANO):
    NUM_VERTS = 778
    NUM_FACES = 1538
    JOINT_NAMES = MANO_JOINTS_LEFT
    body_pose_dims = {
        'global_orient': 1,
        'hand_pose': 15,
    }
    full_pose_dims = {
        'global_orient': 1,
        'hand_pose': 15,
    }

    full_param_dims = {
        'global_orient': 1 * 3,
        'hand_pose': 15 * 3,
        'transl': 3,
        'betas': 10,
    }
    _parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]

    def __init__(self,
                 use_handtips=True,
                 keypoint_src: str = 'mano_left',
                 keypoint_dst: str = 'mano_left',
                 keypoint_approximate: bool = False,
                 joints_regressor: str = None,
                 extra_joints_regressor: str = None,
                 **kwargs) -> None:
        super().__init__(keypoint_src=keypoint_src,
                         keypoint_dst=keypoint_dst,
                         keypoint_approximate=keypoint_approximate,
                         joints_regressor=joints_regressor,
                         extra_joints_regressor=extra_joints_regressor,
                         **kwargs)
        self.use_handtips = use_handtips

    def forward(
            self,
            betas: torch.Tensor = None,
            global_orient: torch.Tensor = None,
            hand_pose: torch.Tensor = None,
            transl: torch.Tensor = None,
            return_verts: bool = True,
            return_full_pose: bool = False,
            pose2rot: bool = True,  #True，则输入的是aa, False，则输入的是rotmat
            **kwargs):
        ''' Forward pass for the MANO model
        '''
        betas = betas if betas is not None else self.betas
        batch_size = cat_pose_list(
            hand_pose).shape[0] if hand_pose is not None else 1

        if betas.shape[0] != batch_size:
            num_repeats = int(batch_size / betas.shape[0])
            betas = betas.repeat_interleave(num_repeats, 0)

        global_orient = self.format_pose(global_orient, 'global_orient',
                                         batch_size, pose2rot)
        hand_pose = self.format_pose(hand_pose, 'hand_pose', batch_size,
                                     pose2rot)

        apply_trans = transl is not None or hasattr(self, 'transl')
        if isinstance(transl, (list, tuple)):
            transl = cat_pose_list(transl, -1)
        elif transl is None:
            if hasattr(self, 'transl'):
                transl = self.transl

        if self.use_pca:
            hand_pose = torch.einsum('bi,ij->bj',
                                     [hand_pose, self.hand_components])

        full_pose = torch.cat([global_orient, hand_pose], dim=1)

        if not pose2rot:
            full_pose = rotmat_to_aa(full_pose.reshape(batch_size * 16, 3,
                                                       3)).reshape(
                                                           batch_size, 16 * 3)

        full_pose += self.pose_mean

        vertices, joints = lbs(
            betas,
            full_pose,
            self.v_template,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            pose2rot=True,
        )
        if self.use_handtips:
            joints = self.vertex_joint_selector(vertices, joints)

        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if apply_trans:
            joints = joints + transl.unsqueeze(dim=1)
            vertices = vertices + transl.unsqueeze(dim=1)

        output = dict(vertices=vertices if return_verts else None,
                      joints=joints if return_verts else None,
                      betas=betas,
                      global_orient=global_orient,
                      hand_pose=hand_pose,
                      transl=transl,
                      full_pose=full_pose if return_full_pose else None)
        joints, joint_mask = super().forward_joints(output)
        output.update(joints=joints, joint_mask=joint_mask)
        return output


class MANO_LEFT(MANO):

    JOINT_NAMES = MANO_JOINTS_LEFT

    def __init__(self, keypoint_dst: str = 'mano_left', **kwargs) -> None:
        super().__init__(keypoint_src='mano_left',
                         is_rhand=False,
                         keypoint_dst=keypoint_dst,
                         **kwargs)

    def forward(self, left_hand_pose: torch.Tensor = None, **kwargs):
        if left_hand_pose is not None:
            left_hand_pose = left_hand_pose
        else:
            left_hand_pose = kwargs.pop('hand_pose', None)
        return super().forward(hand_pose=left_hand_pose, **kwargs)


class MANO_RIGHT(MANO):
    JOINT_NAMES = MANO_JOINTS_RIGHT

    def __init__(self, keypoint_dst: str = 'mano_right', **kwargs) -> None:
        super().__init__(keypoint_src='mano_right',
                         is_rhand=True,
                         keypoint_dst=keypoint_dst,
                         **kwargs)

    def forward(self, right_hand_pose: torch.Tensor = None, **kwargs):
        if right_hand_pose is not None:
            right_hand_pose = right_hand_pose
        else:
            right_hand_pose = kwargs.pop('hand_pose', None)
        return super().forward(hand_pose=right_hand_pose, **kwargs)
