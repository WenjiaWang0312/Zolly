import copy
import math
import torch.nn as nn
from typing import Iterable, Optional, Union

import numpy as np
import torch.nn.functional as F
from smplx.lbs import transform_mat

import torch
from avatar3d.cameras.convert_convention import (
    convert_camera_matrix,
    convert_world_view,
)
from smplx.lbs import batch_rigid_transform, batch_rodrigues
from avatar3d.cameras.utils import combine_RT
from avatar3d.cameras.convert_projection import \
    convert_perspective_to_weakperspective  # prevent yapf isort conflict
from mmhuman3d.utils.transforms import aa_to_rotmat, ee_to_rotmat, rotmat_to_aa, rotmat_to_ee
from avatar3d.models.body_models.mappings import JOINTS_FACTORY, flip_keypoints_mapping

from avatar3d.utils.torch_utils import cat_pose_list

from avatar3d.models.body_models.mappings import (MANO_JOINTS_LEFT,
                                                  MANO_JOINTS_RIGHT,
                                                  SMPLH_JOINTS,
                                                  SMPLH_KEYPOINTS)
from avatar3d.transforms.transform3d import flip_rotation
from avatar3d.models.body_models import get_model_class
import torch
import torch.nn.functional as F
from smplx.utils import find_joint_kin_chain

from avatar3d.models.body_models.mappings import (
    get_keypoint_idx,
    get_keypoint_idxs_by_part,
)
from mmhuman3d.utils.geometry import weak_perspective_projection
from mmhuman3d.models.utils.SMPLX import (get_crop_info,
                                          find_joint_global_rotation,
                                          find_joint_kin_chain, flip_rotmat,
                                          CropSampler, concat_images, SMPLXFaceCropFunc,
                                          SMPLXHandCropFunc, SMPLXFaceMergeFunc, SMPLXHandMergeFunc)
from avatar3d.utils.keypoint_utils import get_max_preds_soft
from avatar3d.transforms.transform3d import batch_rodrigues_vectors
from avatar3d.transforms.transform3d import transform_transl


def eliminate_external_matrix(
    R: Union[np.ndarray, torch.Tensor],
    T: Union[np.ndarray, torch.Tensor],
    body_model: nn.Module,
    global_orient: Optional[Union[np.ndarray, torch.Tensor]] = None,
    body_pose: Optional[Union[np.ndarray, torch.Tensor]] = None,
    transl: Optional[Union[np.ndarray, torch.Tensor]] = None,
    betas: Optional[Union[np.ndarray, torch.Tensor]] = None,
    gender: Optional[Union[np.ndarray, torch.Tensor]] = None,
    **kwargs,
):
    device = global_orient.device

    body_model_output = body_model(global_orient=global_orient,
                                   betas=betas,
                                   body_pose=body_pose,
                                   transl=transl,
                                   gender=gender)
    joints = body_model_output['joints']
    transl_root = joints[:, 0]
    global_orient_cam = rotmat_to_aa(
        torch.bmm(R.to(device), aa_to_rotmat(global_orient)))
    RT = combine_RT(R, T).to(device)
    transl_homo = torch.cat([transl_root, torch.ones_like(transl)[:, :1]], 1)
    transl_homo = torch.bmm(RT, transl_homo[..., None])
    transl_cam = transl_homo[:, :3, 0] / transl_homo[:, 3:4, 0]

    params_cam = dict()
    params_cam['betas'] = betas
    params_cam['global_orient'] = global_orient_cam
    params_cam['body_pose'] = body_pose
    params_cam['gender'] = gender
    body_model_output = body_model(**params_cam)
    pelvis_shift = body_model_output['joints'][:, 0]
    transl_cam = transl_cam - pelvis_shift
    return global_orient_cam, transl_cam


def refine_wrist_spin(rotmat, keypoint_names=SMPLH_KEYPOINTS):
    """
        解手腕的不自然旋转，方法是，求出手腕的 XYZ 顺序的欧拉角旋转，然后将 X 方向的旋转的 1/2 分担到手肘上
        因为仅分担 X 方向的旋转，所以手腕的位置保持不变
    """
    # rotmat: B, 22, 3, 3
    batch_size = rotmat.shape[0]
    device = rotmat.device
    wrist_indexl, wrist_indexr, elbow_indexl, elbow_indexr = keypoint_names.index(
        'left_wrist'), keypoint_names.index(
            'right_wrist'), keypoint_names.index(
                'left_elbow'), keypoint_names.index('right_elbow')
    rotmat = rotmat.view(-1, 22, 3, 3)
    hand_index = [wrist_indexl, wrist_indexr]
    elbow_index = [elbow_indexl, elbow_indexr]
    pi = math.pi

    for hand_idx in range(2):
        euler_to_fix = rotmat_to_ee(rotmat[:, hand_index[hand_idx]],
                                    'XYZ')  # (B, 3)
        elbow_mat = rotmat[:, elbow_index[hand_idx]]  # (B, 3, 3)
        wrist_mat = rotmat[:, hand_index[hand_idx]]  # (B, 3, 3)

        euler_to_fix[:, 1] = 0.
        euler_to_fix[:, 2] = 0.
        euler_to_fix[:, 0] = euler_to_fix[:, 0] / 2.0  # 分担到手肘上的旋转

        trans_mat = ee_to_rotmat(euler_to_fix, 'XYZ')

        elbow_mat = torch.bmm(elbow_mat, trans_mat)  # 右乘 X 方向的旋转，保证手腕的位置不变
        wrist_mat = torch.bmm(torch.inverse(trans_mat),
                              wrist_mat)  # 左乘逆变换，使手腕旋转自然的同时，保持手掌的全局朝向不变

        rotmat[:, elbow_index[hand_idx]] = elbow_mat
        rotmat[:, hand_index[hand_idx]] = wrist_mat

        # 限制手腕的y，z方向范围
        euler_to_fix = rotmat_to_ee(rotmat[:, hand_index[hand_idx]], 'XYZ')
        euler_to_fix[:, 1] = torch.max(
            torch.min(euler_to_fix[:, 1],
                      0.2 * pi * torch.ones(batch_size).to(device)),
            -0.2 * pi * torch.ones(batch_size).to(device))
        euler_to_fix[:, 2] = torch.max(
            torch.min(euler_to_fix[:, 2],
                      0.3 * pi * torch.ones(batch_size).to(device)),
            -0.3 * pi * torch.ones(batch_size).to(device))
        rotmat[:, hand_index[hand_idx]] = ee_to_rotmat(euler_to_fix, 'XYZ')

    return rotmat


def get_wrist_local(body_pose,
                    lhand_orient,
                    rhand_orient,
                    pose_format='aa',
                    refine_wrist=False,
                    keypoint_names=SMPLH_JOINTS):
    wrist_indexl, wrist_indexr, elbow_indexl, elbow_indexr = keypoint_names.index(
        'left_wrist'), keypoint_names.index(
            'right_wrist'), keypoint_names.index(
                'left_elbow'), keypoint_names.index('right_elbow')

    body_pose = cat_pose_list(body_pose).clone()
    batch_size = body_pose.shape[0]
    if pose_format == 'aa':
        body_rotmat = aa_to_rotmat(body_pose.view(batch_size, 22, 3))
        lhand_orient = aa_to_rotmat(lhand_orient)
        rhand_orient = aa_to_rotmat(rhand_orient)
    else:
        body_rotmat = body_pose
    body_rotmat = body_rotmat.view(batch_size, 22, 3, 3)

    rotmats_global = torch.zeros((batch_size, 22, 3, 3)).to(body_rotmat.device)
    rotmats_global[:, 0] = body_rotmat[:, 0].clone()
    parents = get_model_class('smplh')._parents[:22]

    for idx, parent in enumerate(parents):
        if parent != -1:
            rotmats_global[:, idx] = torch.matmul(rotmats_global[:, parent],
                                                  body_rotmat[:, idx])

    elbow_invert_matl = rotmats_global[:, elbow_indexl]
    elbow_invert_matl = torch.permute(elbow_invert_matl, [0, 2, 1])

    wrist_mat_locall = torch.bmm(elbow_invert_matl, lhand_orient)

    elbow_invert_matr = rotmats_global[:, elbow_indexr]  # B, 3, 3
    elbow_invert_matr = torch.permute(elbow_invert_matr, [0, 2, 1])
    wrist_mat_localr = torch.bmm(elbow_invert_matr, rhand_orient)

    body_rotmat[:, wrist_indexl] = wrist_mat_locall
    body_rotmat[:, wrist_indexr] = wrist_mat_localr
    if refine_wrist:
        body_rotmat = refine_wrist_spin(body_rotmat)

    if pose_format == 'aa':
        return rotmat_to_aa(body_rotmat)
    else:
        return body_rotmat


def get_wrist_global(body_pose, pose_format='aa', keypoint_names=SMPLH_JOINTS):
    wrist_indexl, wrist_indexr = keypoint_names.index(
        'left_wrist'), keypoint_names.index('right_wrist')

    body_pose = body_pose.clone()
    batch_size = body_pose.shape[0]
    if pose_format == 'aa':
        body_rotmat = aa_to_rotmat(body_pose.view(batch_size, 22, 3))
    else:
        body_rotmat = body_pose
    body_rotmat = body_rotmat.view(batch_size, 22, 3, 3)

    rotmats_global = torch.zeros((batch_size, 22, 3, 3)).to(body_rotmat.device)
    rotmats_global[:, 0] = body_rotmat[:, 0].clone()

    parents = get_model_class('smplh')._parents[:22]
    for idx, parent in enumerate(parents):
        if parent != -1:
            rotmats_global[:, idx] = torch.matmul(rotmats_global[:, parent],
                                                  body_rotmat[:, idx])
    lhand_orient = rotmats_global[:, wrist_indexl]

    rhand_orient = rotmats_global[:, wrist_indexr]
    if pose_format == 'aa':
        lhand_orient = rotmat_to_aa(lhand_orient)
        rhand_orient = rotmat_to_aa(rhand_orient)

    return lhand_orient, rhand_orient


def flip_hand_pose():
    pass


def flip_flame_pose():
    pass


def flip_full_pose(full_pose: Union[torch.Tensor, np.ndarray],
                   pose_format: str = 'aa',
                   model_type: str = 'smpl'):
    assert model_type in ['smpl', 'smplh', 'smplx']
    NUM_JOINTS = len(JOINTS_FACTORY[model_type])

    full_pose = full_pose.reshape(-1, NUM_JOINTS, 3)
    mapping_index = flip_keypoints_mapping(model_type)
    full_pose = flip_rotation(full_pose, pose_format)
    full_pose = full_pose[:, mapping_index]
    return full_pose


def merge_smplh_pose(lhand_pose_dict, rhand_pose_dict, smplh_pose_dict):
    left_wrist = lhand_pose_dict['global_orient']
    left_hand_pose = lhand_pose_dict['hand_pose']

    right_wrist = rhand_pose_dict['global_orient']
    right_hand_pose = rhand_pose_dict['hand_pose']

    body_pose = smplh_pose_dict['body_pose']
    global_orient = smplh_pose_dict['global_orient']
    betas = smplh_pose_dict['betas']

    body_pose = get_wrist_local(body_pose,
                                left_wrist,
                                right_wrist,
                                pose_format='aa',
                                refine_wrist=False)

    smplh_pose = dict(betas=betas,
                      global_orient=global_orient,
                      body_pose=body_pose,
                      left_hand_pose=left_hand_pose,
                      right_hand_pose=right_hand_pose)
    return smplh_pose


def merge_smplx_pose(lhand_pose_dict, rhand_pose_dict, smplx_pose_dict):
    left_wrist = lhand_pose_dict['global_orient']
    left_hand_pose = lhand_pose_dict['hand_pose']

    right_wrist = rhand_pose_dict['global_orient']
    right_hand_pose = rhand_pose_dict['hand_pose']

    body_pose = smplx_pose_dict['body_pose']
    global_orient = smplx_pose_dict['global_orient']
    betas = smplx_pose_dict['betas']

    body_pose = get_wrist_local(body_pose,
                                left_wrist,
                                right_wrist,
                                pose_format='aa',
                                refine_wrist=False)
    smplx_pose_dict.update(
        dict(left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose))
    return smplx_pose_dict


def split_smplh_pose(smplh_pose_dict):
    left_hand_pose = smplh_pose_dict['left_hand_pose']
    right_hand_pose = smplh_pose_dict['right_hand_pose']

    betas = smplh_pose_dict['betas']

    left_hand_transl = smplh_pose_dict.get('left_hand_transl', None)
    right_hand_transl = smplh_pose_dict.get('right_hand_transl', None)

    if 'left_hand_orient' in smplh_pose_dict:
        left_hand_orient = smplh_pose_dict['left_hand_orient']
        right_hand_orient = smplh_pose_dict['right_hand_orient']
    else:
        body_pose = smplh_pose_dict['body_pose']
        global_orient = smplh_pose_dict['global_orient']
        left_hand_orient, right_hand_orient = get_wrist_global(
            torch.cat([global_orient, cat_pose_list(body_pose)], 1),
            pose_format='aa')

    left_hand_pose_dict = dict(betas=betas,
                               transl=left_hand_transl,
                               global_orient=left_hand_orient,
                               left_hand_pose=left_hand_pose)
    right_hand_pose_dict = dict(betas=betas,
                                transl=right_hand_transl,
                                global_orient=right_hand_orient,
                                right_hand_pose=right_hand_pose)
    return left_hand_pose_dict, right_hand_pose_dict, smplh_pose_dict


def get_hand_transl_smplh(smplh_kp3d,
                          mano_left_kp3d=None,
                          mano_right_kp3d=None):
    left_hand_transl = smplh_kp3d[:, SMPLH_KEYPOINTS.index('left_wrist')]
    right_hand_transl = smplh_kp3d[:, SMPLH_KEYPOINTS.index('left_wrist')]
    if mano_left_kp3d is not None:
        left_hand_transl -= mano_left_kp3d[:,
                                           MANO_JOINTS_LEFT.index('left_wrist'
                                                                  )]
    if mano_right_kp3d is not None:
        right_hand_transl -= mano_right_kp3d[:,
                                             MANO_JOINTS_RIGHT.
                                             index('right_wrist')]
    return left_hand_transl, right_hand_transl


def batch_rigid_transform_inverse(rot_mats: torch.Tensor,
                                  joints_posed: torch.Tensor,
                                  parents: torch.Tensor,
                                  dtype=torch.float32) -> torch.Tensor:
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """
    joints_posed = torch.unsqueeze(joints_posed, dim=-1)

    # joints_tpose_rel = [joints_posed[:, 0]]

    rotation_chain = [rot_mats[:, 0]]

    for i in range(1, len(parents)):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(rotation_chain[parents[i]], rot_mats[:, i])
        rotation_chain.append(curr_res)
    rotation_chain = torch.stack(rotation_chain, dim=1)

    joints_tpose_rel = [joints_posed[:, 0]]
    joints_tpose = [joints_posed[:, 0]]
    for i in range(1, parents.shape[0]):
        curr_rot = rotation_chain[:, parents[i]]
        joints_tpose_rel.append(
            curr_rot.permute(0, 2, 1)
            @ (joints_posed[:, i] - joints_posed[:, parents[i]]))
        joints_tpose.append(joints_tpose[parents[i]] + joints_tpose_rel[i])

    joints_tpose_rel = torch.stack(joints_tpose_rel, dim=1)
    joints_tpose = torch.stack(joints_tpose, dim=1)

    rel_joints = joints_tpose.clone()

    rel_joints[:, 1:] -= joints_tpose[:, parents[1:]]

    transforms_mat = transform_mat(rot_mats.reshape(-1, 3, 3),
                                   rel_joints.reshape(-1, 3, 1)).reshape(
                                       -1, joints_tpose.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:,
                                                                            i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)
    # transforms[:, :, :3, :3] = transforms[:, :, :3, :3].permute(0, 1, 3, 2)
    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints_tpose, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])
    rel_transforms = rel_transforms.inverse().contiguous()
    return posed_joints, rel_transforms


def inverse_lbs(
    verts: torch.Tensor,  # B, 6890, 3
    pose: torch.Tensor,  # B, 24, 3, 3 or B, 72
    J_regressor: torch.Tensor,
    parents: torch.Tensor,
    lbs_weights: torch.Tensor,
    posedirs: torch.Tensor,
    pose2rot: bool = True,
):
    batch_size = verts.shape[0]
    device, dtype = verts.device, verts.dtype

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1,
                                             3)).view([batch_size, -1, 3, 3])

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(pose_feature,
                                    posedirs).view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                    posedirs).view(batch_size, -1, 3)

    # rot_mats = rot_mats.permute(0, 1, 3, 2)
    J_posed = torch.einsum('bik,ji->bjk', [verts, J_regressor])
    J_posed_new, A = batch_rigid_transform_inverse(rot_mats,
                                                   J_posed,
                                                   parents,
                                                   dtype=dtype)

    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, verts.shape[1], 1],
                               dtype=dtype,
                               device=device)
    verts_homo = torch.cat([verts, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(verts_homo, dim=-1))

    v_posed = v_homo[:, :, :3, 0]
    v_shaped = v_posed - pose_offsets

    return v_posed, v_shaped  #, J_posed2, J_posed, J_tpose


def aa_to_absmat(global_orient, body_pose, body_model):

    gt_pose = torch.cat([global_orient[:, None], body_pose], dim=1)

    rotmat = aa_to_rotmat(gt_pose.view(-1, 3)).view(-1, 24, 3, 3)
    transform_chain = [rotmat[:, 0]]
    parents = body_model._parents

    for i in range(1, len(parents)):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]], rotmat[:, i])
        transform_chain.append(curr_res)

    gt_part_rotation = torch.stack(transform_chain, dim=1)
    num_joints = len(parents)
    gt_part_rotation = gt_part_rotation.view(-1, num_joints, 3, 3)
    return gt_part_rotation


def get_crop_info_heatmap(heatmap,
                          pred_scale,
                  img_metas,
                  scale_factor: float = 1.0,
                  crop_size: int = 256):
    """Get the transformation of points on the cropped image to the points on
    the original image."""
    center = get_max_preds_soft(heatmap)
    device = center.device # B, 1, 2
    dtype = center.dtype
    batch_size = center.shape[0]
    # Get the image to crop transformations and bounding box sizes
    crop_transforms = []
    img_bbox_sizes = []
    for img_meta in img_metas:
        crop_transforms.append(img_meta['crop_transform'])
        img_bbox_sizes.append(img_meta['scale'].max())

    img_bbox_sizes = torch.tensor(img_bbox_sizes, dtype=dtype, device=device)

    crop_transforms = torch.tensor(crop_transforms, dtype=dtype, device=device)

    crop_transforms = torch.cat([
        crop_transforms,
        torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device).expand(
            [batch_size, 1, 3])
    ],
                                dim=1)

    inv_crop_transforms = torch.inverse(crop_transforms)

    # center on the cropped body image
    center_body_crop = center
    bbox_size  = pred_scale.view(-1) * crop_size / 2 * scale_factor

    orig_bbox_size = bbox_size / crop_size * img_bbox_sizes

    # Compute the center of the crop in the original image
    center = (
        torch.einsum('bij,bj->bi',
                     [inv_crop_transforms[:, :2, :2], center_body_crop.squeeze(1)]) +
        inv_crop_transforms[:, :2, 2])

    return {
        'center': center.reshape(-1, 2),
        'orig_bbox_size': orig_bbox_size,
        # 'bbox_size': bbox_size.reshape(-1),
        'inv_crop_transforms': inv_crop_transforms,
        # 'center_body_crop': 2 * center_body_crop / (crop_size-1) - 1,
    }


class SMPLXHandTranslMergeFunc():
    """This function use predictions from hand model to update the hand params
    (right_hand_pose, left_hand_pose, wrist_pose) in predictions from body
    model."""

    def __init__(self, body_model, convention='smplx', align_mode='direction'):
        self.body_model = body_model
        self.convention = convention
        self.left_hand_idxs = get_keypoint_idxs_by_part(
            'left_hand', self.convention)
        self.left_wrist_idx = get_keypoint_idx('left_wrist', self.convention)
        self.left_hand_idxs.append(self.left_wrist_idx)
        self.left_wrist_kin_chain = find_joint_kin_chain(
            self.left_wrist_idx, self.body_model.parents)

        self.right_hand_idxs = get_keypoint_idxs_by_part(
            'right_hand', self.convention)
        self.right_wrist_idx = get_keypoint_idx('right_wrist', self.convention)
        self.right_hand_idxs.append(self.right_wrist_idx)
        self.right_wrist_kin_chain = find_joint_kin_chain(
            self.right_wrist_idx, self.body_model.parents)
        self.align_mode = align_mode

    def __call__(self, pred_z, pred_body, pred_lhand, pred_rhand, pred_pose, focal_length):
        """Function
        Args:
            body_predictions (dict): The prediction from body model.
            hand_predictions (dict): The prediction from hand model.
        Returns:
            dict: Merged prediction.
        """
        global_orient = pred_pose['global_orient']
        body_pose = pred_pose['body_pose']
        pred_z_lhand = focal_length * torch.exp(-pred_lhand['pred_log_s'][:, self.left_wrist_idx])
        pred_transl_lhand = torch.cat([pred_lhand['pred_cam'[:, 1:]], pred_z_lhand], -1)
        pred_z_rhand = focal_length * torch.exp(-pred_rhand['pred_log_s'][:, self.right_wrist_idx])
        pred_transl_rhand = torch.cat([pred_rhand['pred_cam'[:, 1:]], pred_z_rhand], -1)
        if self.align_mode == 'direction':
            left_wrist_transl = transform_transl(pred_transl_lhand)
            right_wrist_transl = transform_transl(pred_transl_rhand)
        elif self.align_mode == 'projection':
            pass
            # consider the bone length of forearm
        full_transl = torch.cat([pred_body['pred_cam'][:, 1:], pred_z], -1)
        model_output = self.body_model(global_orient=global_orient, body_pose=body_pose, transl=full_transl)
        left_elbow_transl = model_output['joints'][:, self.left_elbow_idx]
        right_elbow_transl = model_output['joints'][:, self.right_elbow_idx]

        left_elbow_abs = batch_rodrigues_vectors(left_elbow_transl, left_wrist_transl)
        right_elbow_abs = batch_rodrigues_vectors(right_elbow_transl, right_wrist_transl)

        batch_size = global_orient.shape[0]
        device = global_orient.device
        hands_from_body_idxs = torch.arange(
            0, 2 * batch_size, dtype=torch.long, device=device)
        right_hand_from_body_idxs = hands_from_body_idxs[:batch_size]
        left_hand_from_body_idxs = hands_from_body_idxs[batch_size:]

        parent_rots = []
        right_elbow_parent_rot = find_joint_global_rotation(
            self.right_elbow_kin_chain[1:], global_orient, body_pose)

        left_elbow_parent_rot = find_joint_global_rotation(
            self.left_elbow_kin_chain[1:], global_orient, body_pose)
        left_to_right_elbow_parent_rot = flip_rotmat(left_elbow_parent_rot)

        parent_rots += [right_elbow_parent_rot, left_to_right_elbow_parent_rot]
        parent_rots = torch.cat(parent_rots, dim=0)

        elbow_abs = torch.cat([right_elbow_abs, left_elbow_abs], dim=0)
        # Undo the rotation of the parent joints to make the wrist rotation
        # relative again
        elbow_relative = torch.matmul(
            parent_rots.reshape(-1, 3, 3).transpose(1, 2),
            elbow_abs.reshape(-1, 3, 3))

        right_elbow_relative = elbow_relative[right_hand_from_body_idxs]
        left_elbow_relative = flip_rotmat(
            elbow_relative[left_hand_from_body_idxs])

        pred_pose['body_pose'][:, self.right_elbow_idx -
                                                    1] = right_elbow_relative
        pred_pose['body_pose'][:, self.left_elbow_idx -
                                                    1] = left_elbow_relative

        return pred_pose


class SMPLXHandHeatmapCropFunc(SMPLXHandCropFunc):
    """This function crop hand image from the original image.

    Use the output keypoints predicted by the body model to locate the hand
    position.
    """

    def __init__(self,
                 convention_body='smplx_body',
                 lhand_center_joint='left_middle_1',
                rhand_center_joint='right_middle_1',
                 img_res=256,
                 scale_factor=2.0,
                 crop_size=224):
        self.img_res = img_res
        self.convention_body = convention_body
        self.lhand_center_idx = get_keypoint_idx(lhand_center_joint, convention_body)
        self.rhand_center_idx = get_keypoint_idx(rhand_center_joint, convention_body)
    
        self.scale_factor = scale_factor
        self.hand_cropper = CropSampler(crop_size)

    def __call__(self, body_predictions, img_metas):
        """Function
        Args:
            body_predictions (dict): The prediction from body model.
            img_metas (dict): Information of the input images.
        Returns:
            all_hand_imgs (torch.tensor): Cropped hand images.
            hand_mean (torch.tensor): Mean value of hand params.
            crop_info (dict): Hand crop transforms.
        """
        pred_heatmap = body_predictions['pred_heatmap']
        pred_scale = torch.exp(body_predictions['pred_log_s'])
        # concat ori_img
        full_images = []
        for img_meta in img_metas:
            full_images.append(img_meta['ori_img'].to(device=pred_heatmap.device))
        full_imgs = concat_images(full_images)

        left_hand_points_to_crop = get_crop_info_heatmap(pred_heatmap[:, self.lhand_center_idx].unsqueeze(1),
                                                         pred_scale[:, self.lhand_center_idx].unsqueeze(1), img_metas,
                                                 self.scale_factor,
                                                 self.img_res)
        left_hand_center = left_hand_points_to_crop['center']
        left_hand_orig_bbox_size = left_hand_points_to_crop['orig_bbox_size']
        left_hand_inv_crop_transforms = left_hand_points_to_crop[
            'inv_crop_transforms']

        left_hand_cropper_out = self.hand_cropper(full_imgs, left_hand_center,
                                                  left_hand_orig_bbox_size)
        left_hand_crops = left_hand_cropper_out['images']
        # left_hand_points = left_hand_cropper_out['sampling_grid']
        left_hand_crop_transform = left_hand_cropper_out['transform']

        # right hand
        right_hand_points_to_crop = get_crop_info_heatmap(pred_heatmap[:, self.rhand_center_idx].unsqueeze(1),
                                                          pred_scale[:, self.rhand_center_idx].unsqueeze(1), img_metas,
                                                  self.scale_factor,
                                                  self.img_res)
        right_hand_center = right_hand_points_to_crop['center']
        right_hand_orig_bbox_size = right_hand_points_to_crop['orig_bbox_size']
        # right_hand_inv_crop_transforms = right_hand_points_to_crop[
        #     'inv_crop_transforms']
        right_hand_cropper_out = self.hand_cropper(full_imgs,
                                                   right_hand_center,
                                                   right_hand_orig_bbox_size)
        right_hand_crops = right_hand_cropper_out['images']
        # right_hand_points = right_hand_cropper_out['sampling_grid']
        right_hand_crop_transform = right_hand_cropper_out['transform']

        # concat
        all_hand_imgs = []
        all_hand_imgs.append(right_hand_crops)
        all_hand_imgs.append(torch.flip(left_hand_crops, dims=(-1, )))

        # [right_hand , left hand]
        all_hand_imgs = torch.cat(all_hand_imgs, dim=0)

        crop_info = dict(
            hand_inv_crop_transforms=left_hand_inv_crop_transforms,
            left_hand_crop_transform=left_hand_crop_transform,
            right_hand_crop_transform=right_hand_crop_transform)
        return all_hand_imgs, crop_info


class SMPLXFaceHeatmapCropFunc(SMPLXFaceCropFunc):
    """This function crop face image from the original image.

    Use the output keypoints predicted by the facce model to locate the face
    position.
    """

    def __init__(self,
                 convention_body='smplx_body',
                 img_res=256,
                 scale_factor=2.0,
                 crop_size=256,
                 face_center_joint='head'):
        self.img_res = img_res
        self.scale_factor = scale_factor
        self.face_cropper = CropSampler(crop_size)
        self.convention_body = convention_body
        
        self.face_center_idx = get_keypoint_idx(face_center_joint, convention_body)

    def __call__(self, body_predictions, img_metas):
        """Function
        Args:
            body_predictions (dict): The prediction from body model.
            img_metas (dict): Information of the input images.
        Returns:
            all_face_imgs (torch.tensor): Cropped face images.
            face_mean (torch.tensor): Mean value of face params.
            crop_info (dict): Face crop transforms.
        """
        pred_heatmap = body_predictions['pred_heatmap']
        pred_scale = torch.exp(body_predictions['pred_log_s'])
        # concat ori_img
        full_images = []
        for img_meta in img_metas:
            full_images.append(img_meta['ori_img'].to(device=pred_heatmap.device))
        full_imgs = concat_images(full_images)


        face_points_to_crop = get_crop_info_heatmap(pred_heatmap[:, self.face_center_idx].unsqueeze(1),
                                                         pred_scale[:, self.face_center_idx].unsqueeze(1), img_metas,
                                                 self.scale_factor,
                                                 self.img_res)
        
        face_center = face_points_to_crop['center']
        face_orig_bbox_size = face_points_to_crop['orig_bbox_size']
        face_inv_crop_transforms = face_points_to_crop['inv_crop_transforms']

        face_cropper_out = self.face_cropper(full_imgs, face_center,
                                             face_orig_bbox_size)
        face_crops = face_cropper_out['images']
        # face_points = face_cropper_out['sampling_grid']
        face_crop_transform = face_cropper_out['transform']

        all_face_imgs = [face_crops]
        all_face_imgs = torch.cat(all_face_imgs, dim=0)
        crop_info = dict(
            face_inv_crop_transforms=face_inv_crop_transforms,
            face_crop_transform=face_crop_transform)
        return all_face_imgs, crop_info


if __name__ == '__main__':
    # test inverse_lbs
    from avatar3d.models.body_models.builder import build_body_model
    from mmhuman3d.utils.mesh_utils import save_meshes_as_objs
    from pytorch3d.structures import Meshes
    root = '/nvme/wangwenjia'
    smpl_dict = dict(type='SMPL',
                     model_path=f'{root}/body_models/smpl',
                     keypoint_dst='smpl')

    smpl = build_body_model(smpl_dict)
    lbs_weights = smpl.lbs_weights
    parents = smpl.parents
    J_regressor = smpl.J_regressor
    posedirs = smpl.posedirs

    npz = np.load(f'{root}/mmhuman_data/pdhuman_1111_train.npz',
                  allow_pickle=True)
    for index in [100]:
        body_pose = torch.Tensor(npz['body_pose'][index][None])
        # body_pose = torch.zeros(1, 69)
        # body_pose[0, 42:48] = 1
        # body_pose[0, 51:57] = 0.5
        global_orient = torch.zeros(1, 3) + 0.5
        betas = torch.zeros(1, 10) + 0.5

        smpl_out = smpl(body_pose=body_pose * 0,
                        global_orient=global_orient * 0,
                        betas=betas * 0)
        verts_t = smpl_out['vertices']
        meshes = Meshes(verts_t, smpl.faces_tensor[None])
        # save_meshes_as_objs('objs/t.obj', meshes)

        smpl_out = smpl(body_pose=body_pose * 0,
                        global_orient=global_orient * 0,
                        betas=betas)
        verts_t_betas = smpl_out['vertices']
        from avatar3d.utils.mesh_utils import get_smpl_pc_mesh, get_joints_mesh, save_meshes_as_plys, get_smpl_mesh

        joints_tpose = smpl_out['joints']
        joints_mesh = get_joints_mesh(joints_tpose, radius=0.02)
        save_meshes_as_plys('objs/joints_t.ply', joints_mesh)
        mesh_tpose = get_smpl_mesh(verts_t_betas,
                                   smpl.faces_tensor[None],
                                   palette='part')
        save_meshes_as_plys('objs/t_betas.ply', mesh_tpose)
        meshes = Meshes(verts_t_betas, smpl.faces_tensor[None])
        save_meshes_as_objs('objs/t_betas.obj', meshes)

        smpl_out = smpl(body_pose=body_pose,
                        global_orient=global_orient,
                        betas=betas)
        verts_posed = smpl_out['vertices']
        joints_posed = smpl_out['joints']
        joints_mesh = get_joints_mesh(joints_posed, radius=0.02)
        save_meshes_as_plys('objs/joints_posed.ply', joints_mesh)
        meshes = Meshes(verts_posed, smpl.faces_tensor[None])
        save_meshes_as_objs('objs/posed.obj', meshes)

        pose = torch.cat([global_orient, body_pose], -1)
        verts_warped, verts_shaped = inverse_lbs(verts_posed, pose,
                                                 J_regressor, parents,
                                                 lbs_weights, posedirs)
        # meshes1 = Meshes(verts_warped, smpl.faces_tensor[None])
        # meshes2 = Meshes(verts_shaped, smpl.faces_tensor[None])

        # save_meshes_as_objs('objs/warped.obj', meshes1)
        # save_meshes_as_objs('objs/shaped.obj', meshes2)

        # meshes_p = get_pointcloud_mesh(j_posed)
        # meshes_p2 = get_pointcloud_mesh(j_posed2)
        # meshes_j = get_pointcloud_mesh(j_tpose)
        # save_meshes_as_objs('objs/j_t.obj', meshes_j)
        # save_meshes_as_objs('objs/j_posed.obj', meshes_p)
        # save_meshes_as_objs('objs/j_posed2.obj', meshes_p2)
        offset = (verts_warped - verts_t_betas).abs().max()
        print(offset)
        if offset > 1e-1:
            print('offset too large')
