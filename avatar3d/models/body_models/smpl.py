from avatar3d.utils.torch_utils import cat_pose_list
from .mappings.smpl import SMPL_JOINTS
from smplx.body_models import SMPL as _SMPL
from avatar3d.models.body_models.base import ParametricModelBase


class SMPL(ParametricModelBase, _SMPL):
    """ Extension of the official SMPL implementation to support more joints """
    NUM_VERTS = 6890
    NUM_FACES = 13776
    JOINT_NAMES = SMPL_JOINTS

    body_pose_dims = {
        'global_orient': 1,
        'body_pose': 23,
    }
    full_pose_dims = {
        'global_orient': 1,
        'body_pose': 23,
    }

    full_param_dims = {
        'global_orient': 1 * 3,
        'body_pose': 23 * 3,
        'transl': 3,
        'betas': 10,
    }

    _parents = [
        -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18,
        19, 20, 21
    ]

    @property
    def body_pose_keys(self):
        return list(self.body_pose_dims.keys())

    def __init__(self,
                 model_path='',
                 gender='neutral',
                 create_transl=False,
                 use_pca=False,
                 keypoint_src: str = 'smpl_45',
                 keypoint_dst: str = 'human_data',
                 keypoint_approximate: bool = False,
                 joints_regressor: str = None,
                 extra_joints_regressor: str = None,
                 *args,
                 **kwargs):
        super(SMPL,
              self).__init__(model_path=model_path,
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

    def forward(self,
                betas=None,
                global_orient=None,
                body_pose=None,
                pose2rot=True,
                transl=None,
                *args,
                **kwargs):
        betas = betas if betas is not None else self.betas
        batch_size = cat_pose_list(
            body_pose).shape[0] if body_pose is not None else 1

        if isinstance(transl, (list, tuple)):
            transl = cat_pose_list(transl, -1)
        if betas.shape[0] != batch_size:
            num_repeats = int(batch_size / betas.shape[0])
            betas = betas.repeat_interleave(num_repeats, 0)

        body_pose = self.format_pose(body_pose,
                                     'body_pose',
                                     batch_size=batch_size,
                                     pose2rot=pose2rot)
        global_orient = self.format_pose(global_orient,
                                         'global_orient',
                                         batch_size=batch_size,
                                         pose2rot=pose2rot)
        output = dict((super(SMPL, self).forward(betas=betas,
                                                 body_pose=body_pose,
                                                 global_orient=global_orient,
                                                 pose2rot=pose2rot,
                                                 transl=transl,
                                                 *args,
                                                 **kwargs)))

        joints, joint_mask = super().forward_joints(output)
        output.update(joints=joints, joint_mask=joint_mask)
        return output
