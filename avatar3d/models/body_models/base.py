from avatar3d.utils.torch_utils import cat_pose_list
from .mappings import convert_kps
import torch
import torch.nn as nn
import numpy as np
from smplx.lbs import batch_rodrigues, vertices2joints
from typing import Union


class ParametricModelBase(nn.Module):

    def __init__(
        self,
        keypoint_src: str = '',
        keypoint_dst: str = 'human_data',
        keypoint_approximate: bool = False,
        joints_regressor: str = None,
        extra_joints_regressor: str = None,
        device: Union[str, torch.device] = 'cpu',
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.keypoint_src = keypoint_src
        self.keypoint_dst = keypoint_dst
        self.keypoint_approximate = keypoint_approximate

        # override the default SMPL joint regressor if available
        if joints_regressor is not None:
            joints_regressor = torch.tensor(np.load(joints_regressor),
                                            dtype=torch.float)
            self.register_buffer('joints_regressor', joints_regressor)

        # allow for extra joints to be regressed if available
        if extra_joints_regressor is not None:
            joints_regressor_extra = torch.tensor(
                np.load(extra_joints_regressor), dtype=torch.float)
            self.register_buffer('joints_regressor_extra',
                                 joints_regressor_extra)
        self = self.to(device)

    def to(self, device, **args):
        super().to(device=device, **args)
        self.device = device
        return self

    def forward_joints(self, smpl_output):
        if not hasattr(self, 'joints_regressor'):
            if 'joints' not in smpl_output:
                joints = vertices2joints(self.J_regressor,
                                         smpl_output['vertices'])
                joints = self.vertex_joint_selector(smpl_output['vertices'],
                                                    joints)
            else:
                joints = smpl_output['joints']
        else:
            joints = vertices2joints(self.joints_regressor,
                                     smpl_output['vertices'])

        if hasattr(self, 'joints_regressor_extra'):
            extra_joints = vertices2joints(self.joints_regressor_extra,
                                           smpl_output['vertices'])
            joints = torch.cat([joints, extra_joints], dim=1)

        joints, joint_mask = convert_kps(joints,
                                         src=self.keypoint_src,
                                         dst=self.keypoint_dst,
                                         approximate=self.keypoint_approximate)
        if isinstance(joint_mask, np.ndarray):
            joint_mask = torch.tensor(joint_mask,
                                      dtype=torch.uint8,
                                      device=joints.device)
        return joints, joint_mask

    def inverse_lbs():
        pass

    def get_canonical_pose():
        pass

    def format_pose(self, pose, name, batch_size=1, pose2rot=True):
        if pose2rot:
            if pose is not None:
                if getattr(self, 'use_pca', False) and name in [
                        'left_hand_pose', 'right_hand_pose'
                ]:
                    dims = self.num_pca_comps
                    pose = pose[..., :self.num_pca_comps]
                else:
                    dims = self.full_pose_dims[name] * 3

                pose = cat_pose_list(pose).reshape(-1, dims)

            else:
                if hasattr(self, name):
                    pose = getattr(self, name).repeat_interleave(batch_size, 0)
                else:
                    pose = torch.zeros(batch_size, self.full_pose_dims[name] *
                                       3).to(self.device)
        else:
            if pose is not None:
                if getattr(self, 'use_pca', False) and name in [
                        'left_hand_pose', 'right_hand_pose'
                ]:
                    pose = pose[..., :self.num_pca_comps]
                    pose = cat_pose_list(pose).reshape(-1, self.num_pca_comps)
                else:
                    pose = cat_pose_list(pose).reshape(
                        -1, self.full_pose_dims[name], 3, 3)
            else:
                if hasattr(self, name):
                    pose = batch_rodrigues(getattr(self, name).reshape(
                        -1, 3)).reshape(-1, self.full_pose_dims[name], 3,
                                        3).repeat_interleave(batch_size, 0)
                else:
                    pose = torch.zeros(batch_size, self.full_pose_dims[name],
                                       3, 3).to(self.device)
        return pose

    def dict_to_fullpose(self, **kwargs):
        full_pose = []
        N = kwargs.get('global_orient').reshape(-1, 3).shape[0]
        for k in self.full_pose_dims:
            pose = kwargs.get(k, torch.zeros(N, self.full_pose_dims[k], 3))
            full_pose.append(pose)
        full_pose = torch.cat(full_pose, 1)

    def fullpose_to_dict(self, full_pose):
        pose_dict = {}
        pose_dim = 0
        for k in self.full_pose_dims:
            last_pose_dim = pose_dim
            pose_dim += self.full_pose_dims[k]
            pose_dict[k] = full_pose[:, last_pose_dim:pose_dim]
        return pose_dict
