import torch
from smplx import SMPLH as _SMPLH
from smplx.lbs import lbs
from mmhuman3d.core.conventions.segmentation import body_segmentation
from zolly.models.body_models.base import ParametricModelBase
from zolly.transforms.transform3d.convert_rotation import rotmat_to_aa
from zolly.utils.torch_utils import cat_pose_list
from .mappings.smplh import SMPLH_JOINTS, SMPLH_KEYPOINTS


class SMPLH(ParametricModelBase, _SMPLH):

    NUM_VERTS = 6890
    NUM_FACES = 13776
    JOINT_NAMES = SMPLH_JOINTS
    KP_NAMES = SMPLH_KEYPOINTS
    body_pose_dims = {
        'global_orient': 1,
        'body_pose': 21,
    }
    full_pose_dims = {
        'global_orient': 1,
        'body_pose': 21,
        'left_hand_pose': 15,
        'right_hand_pose': 15,
    }

    full_param_dims = {
        'global_orient': 1 * 3,
        'body_pose': 21 * 3,
        'left_hand_pose': 15 * 3,
        'right_hand_pose': 15 * 3,
        'transl': 3,
        'betas': 10,
    }
    _parents = [
        -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18,
        19, 20, 22, 23, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 21, 37,
        38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50
    ]

    def __init__(self,
                 model_path='',
                 gender='neutral',
                 create_transl=False,
                 use_pca=False,
                 keypoint_src: str = 'smplh',
                 keypoint_dst: str = 'human_data',
                 keypoint_approximate: bool = False,
                 joints_regressor: str = None,
                 extra_joints_regressor: str = None,
                 *args,
                 **kwargs):
        super().__init__(model_path=model_path,
                         gender=gender,
                         create_transl=create_transl,
                         use_pca=use_pca,
                         keypoint_src=keypoint_src,
                         keypoint_dst=keypoint_dst,
                         keypoint_approximate=keypoint_approximate,
                         joints_regressor=joints_regressor,
                         extra_joints_regressor=extra_joints_regressor,
                         *args,
                         **kwargs)
        self.body_part_segmentation = body_segmentation('smpl')

    def forward(self,
                betas=None,
                global_orient=None,
                body_pose=None,
                left_hand_pose=None,
                right_hand_pose=None,
                transl=None,
                return_verts: bool = True,
                return_full_pose: bool = False,
                pose2rot: bool = True,
                **kwargs):

        betas = betas if betas is not None else self.betas
        batch_size = cat_pose_list(
            body_pose).shape[0] if body_pose is not None else 1
        if betas.shape[0] != batch_size:
            num_repeats = int(batch_size / betas.shape[0])
            betas = betas.repeat_interleave(num_repeats, 0)

        global_orient = self.format_pose(global_orient, 'global_orient',
                                         batch_size, pose2rot)
        body_pose = self.format_pose(body_pose, 'body_pose', batch_size,
                                     pose2rot)

        left_hand_pose = self.format_pose(left_hand_pose, 'left_hand_pose',
                                          batch_size, pose2rot)
        right_hand_pose = self.format_pose(right_hand_pose, 'right_hand_pose',
                                           batch_size, pose2rot)

        apply_trans = transl is not None or hasattr(self, 'transl')

        if isinstance(transl, (list, tuple)):
            transl = cat_pose_list(transl, -1)
        elif transl is None:
            if hasattr(self, 'transl'):
                transl = self.transl

        if self.use_pca:
            left_hand_pose = torch.einsum(
                'bi,ij->bj', [left_hand_pose, self.left_hand_components])
            right_hand_pose = torch.einsum(
                'bi,ij->bj', [right_hand_pose, self.right_hand_components])

        full_pose = torch.cat(
            [global_orient, body_pose, left_hand_pose, right_hand_pose], dim=1)
        if not pose2rot:
            full_pose = rotmat_to_aa(full_pose.reshape(batch_size * 52, 3,
                                                       3)).reshape(
                                                           batch_size, 52 * 3)

        full_pose += self.pose_mean

        vertices, joints = lbs(betas,
                               full_pose,
                               self.v_template,
                               self.shapedirs,
                               self.posedirs,
                               self.J_regressor,
                               self.parents,
                               self.lbs_weights,
                               pose2rot=True)

        # Add any extra joints that might be needed
        joints = self.vertex_joint_selector(vertices, joints)
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if apply_trans:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)

        output = dict(vertices=vertices if return_verts else None,
                      joints=joints,
                      betas=betas,
                      global_orient=global_orient,
                      body_pose=body_pose,
                      left_hand_pose=left_hand_pose,
                      right_hand_pose=right_hand_pose,
                      transl=transl,
                      full_pose=full_pose if return_full_pose else None)
        joints, joint_mask = super().forward_joints(output)
        output.update(joints=joints, joint_mask=joint_mask)
        return output
