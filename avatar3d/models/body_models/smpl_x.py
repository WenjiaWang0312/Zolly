import torch
from typing import Optional
from torch import Tensor

from smplx.lbs import (lbs, find_dynamic_lmk_idx_and_bcoords,
                       vertices2landmarks, blend_shapes)
from smplx.body_models import SMPLX as _SMPLX
from mmhuman3d.core.conventions.segmentation import body_segmentation
from avatar3d.models.body_models.base import ParametricModelBase
from avatar3d.transforms.transform3d import rotmat_to_aa
from avatar3d.utils.torch_utils import cat_pose_list
from .mappings.smplx import SMPLX_JOINTS


class SMPLX(ParametricModelBase, _SMPLX):
    NUM_VERTS = 10475
    NUM_FACES = 20908

    JOINT_NAMES = SMPLX_JOINTS
    NUM_JAW_JOINTS = 1
    NUM_EYE_JOINTS = 1

    body_pose_dims = {
        'global_orient': 1,
        'body_pose': 21,
    }
    full_pose_dims = {
        'global_orient': 1,
        'body_pose': 21,
        'left_hand_pose': 15,
        'right_hand_pose': 15,
        'jaw_pose': 1,
        'leye_pose': 1,
        'reye_pose': 1,
    }

    full_param_dims = {
        'global_orient': 1 * 3,
        'body_pose': 21 * 3,
        'left_hand_pose': 15 * 3,
        'right_hand_pose': 15 * 3,
        'jaw_pose': 1 * 3,
        'leye_pose': 1 * 3,
        'reye_pose': 1 * 3,
        'expression': 10,
        'transl': 3,
        'betas': 10,
    }

    _parents = [
        -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18,
        19, 15, 15, 15, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37,
        38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53
    ]

    def __init__(self,
                 model_path='',
                 gender='neutral',
                 create_transl: bool = False,
                 create_expression: bool = False,
                 create_jaw_pose: bool = False,
                 create_leye_pose: bool = False,
                 create_reye_pose: bool = False,
                 use_pca: bool = False,
                 keypoint_src: str = 'smplx',
                 keypoint_dst: str = 'human_data',
                 keypoint_approximate: bool = False,
                 joints_regressor: str = None,
                 extra_joints_regressor: str = None,
                 *args,
                 **kwargs):
        super().__init__(model_path=model_path,
                         gender=gender,
                         create_transl=create_transl,
                         create_expression=create_expression,
                         create_jaw_pose=create_jaw_pose,
                         create_leye_pose=create_leye_pose,
                         create_reye_pose=create_reye_pose,
                         use_pca=use_pca,
                         keypoint_src=keypoint_src,
                         keypoint_dst=keypoint_dst,
                         keypoint_approximate=keypoint_approximate,
                         joints_regressor=joints_regressor,
                         extra_joints_regressor=extra_joints_regressor,
                         *args,
                         **kwargs)
        self.body_part_segmentation = body_segmentation('smplx')

    def forward(self,
                betas: Optional[Tensor] = None,
                global_orient: Optional[Tensor] = None,
                body_pose: Optional[Tensor] = None,
                left_hand_pose: Optional[Tensor] = None,
                right_hand_pose: Optional[Tensor] = None,
                transl: Optional[Tensor] = None,
                expression: Optional[Tensor] = None,
                jaw_pose: Optional[Tensor] = None,
                leye_pose: Optional[Tensor] = None,
                reye_pose: Optional[Tensor] = None,
                return_verts: bool = True,
                return_full_pose: bool = False,
                pose2rot: bool = True,
                return_shaped: bool = True,
                **kwargs) -> dict:

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
        jaw_pose = self.format_pose(jaw_pose, 'jaw_pose', batch_size, pose2rot)
        leye_pose = self.format_pose(leye_pose, 'leye_pose', batch_size,
                                     pose2rot)
        reye_pose = self.format_pose(reye_pose, 'reye_pose', batch_size,
                                     pose2rot)
        expression = torch.zeros(batch_size,
                                 self.num_expression_coeffs).to(betas.device)

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

        full_pose = torch.cat([
            global_orient, body_pose, jaw_pose, leye_pose, reye_pose,
            left_hand_pose, right_hand_pose
        ],
                              dim=1)

        if not pose2rot:
            full_pose_dim = sum(list(self.full_pose_dims.values()))
            full_pose = rotmat_to_aa(
                full_pose.reshape(batch_size * full_pose_dim, 3,
                                  3)).reshape(batch_size, full_pose_dim * 3)

        full_pose += self.pose_mean

        shape_components = torch.cat([betas, expression], dim=-1)

        shapedirs = torch.cat([self.shapedirs, self.expr_dirs], dim=-1)

        vertices, joints = lbs(
            shape_components,
            full_pose,
            self.v_template,
            shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            pose2rot=True,
        )

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).expand(
            batch_size, -1).contiguous()
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).repeat(
            self.batch_size, 1, 1)
        if self.use_face_contour:
            lmk_idx_and_bcoords = find_dynamic_lmk_idx_and_bcoords(
                vertices,
                full_pose,
                self.dynamic_lmk_faces_idx,
                self.dynamic_lmk_bary_coords,
                self.neck_kin_chain,
                pose2rot=True,
            )
            dyn_lmk_faces_idx, dyn_lmk_bary_coords = lmk_idx_and_bcoords

            lmk_faces_idx = torch.cat([lmk_faces_idx, dyn_lmk_faces_idx], 1)
            lmk_bary_coords = torch.cat([
                lmk_bary_coords.expand(batch_size, -1, -1), dyn_lmk_bary_coords
            ], 1)

        landmarks = vertices2landmarks(vertices, self.faces_tensor,
                                       lmk_faces_idx, lmk_bary_coords)

        # Add any extra joints that might be needed
        joints = self.vertex_joint_selector(vertices, joints)
        # Add the landmarks to the joints
        joints = torch.cat([joints, landmarks], dim=1)
        # Map the joints to the current dataset

        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints=joints, vertices=vertices)

        if apply_trans:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)

        v_shaped = None
        if return_shaped:
            v_shaped = self.v_template + blend_shapes(betas, self.shapedirs)

        output = dict(vertices=vertices if return_verts else None,
                      joints=joints,
                      betas=betas,
                      expression=expression,
                      global_orient=global_orient,
                      body_pose=body_pose,
                      left_hand_pose=left_hand_pose,
                      right_hand_pose=right_hand_pose,
                      jaw_pose=jaw_pose,
                      v_shaped=v_shaped,
                      full_pose=full_pose if return_full_pose else None)
        joints, joint_mask = super().forward_joints(output)
        output.update(joints=joints, joint_mask=joint_mask)
        return output
