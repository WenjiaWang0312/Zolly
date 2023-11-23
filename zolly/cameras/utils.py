import math

import torch

import numpy as np
from typing import Iterable, Optional, Union, Tuple

from .convert_convention import convert_K_3x3_to_4x4, convert_camera_matrix
from mmhuman3d.utils.geometry import (perspective_projection,
                                      estimate_translation,
                                      estimate_translation_np)


def convert_perspective_to_weakperspective(
        K: Union[torch.Tensor, np.ndarray],
        zmean: Union[torch.Tensor, np.ndarray, float, int],
        resolution: Union[int, Tuple[int, int], torch.Tensor,
                          np.ndarray] = None,
        in_ndc: bool = False,
        convention: str = 'opencv') -> Union[torch.Tensor, np.ndarray]:
    """Convert perspective to weakperspective intrinsic matrix.

    Args:
        K (Union[torch.Tensor, np.ndarray]): input intrinsic matrix, shape
            should be (batch, 4, 4) or (batch, 3, 3).
        zmean (Union[torch.Tensor, np.ndarray, int, float]): zmean for object.
            shape should be (batch, ) or singleton number.
        resolution (Union[int, Tuple[int, int], torch.Tensor, np.ndarray],
            optional): (height, width) of image. Defaults to None.
        in_ndc (bool, optional): whether defined in ndc. Defaults to False.
        convention (str, optional): camera convention. Defaults to 'opencv'.

    Returns:
        Union[torch.Tensor, np.ndarray]: output weakperspective pred_cam,
            shape is (batch, 4)
    """
    assert K is not None, 'K is required.'
    K, _, _ = convert_camera_matrix(K=K,
                                    convention_src=convention,
                                    convention_dst='pytorch3d',
                                    is_perspective=True,
                                    in_ndc_src=in_ndc,
                                    in_ndc_dst=True,
                                    resolution_src=resolution)
    if isinstance(zmean, np.ndarray):
        zmean = torch.Tensor(zmean)
    elif isinstance(zmean, (float, int)):
        zmean = torch.Tensor([zmean])
    zmean = zmean.view(-1)
    num_frame = max(zmean.shape[0], K.shape[0])
    new_K = torch.eye(4, 4)[None].repeat(num_frame, 1, 1)
    fx = K[:, 0, 0]
    fy = K[:, 0, 0]
    cx = K[:, 0, 2]
    cy = K[:, 1, 2]
    new_K[:, 0, 0] = fx / zmean
    new_K[:, 1, 1] = fy / zmean
    new_K[:, 0, 3] = cx
    new_K[:, 1, 3] = cy
    return new_K


def convert_weakperspective_to_perspective(
        K: Union[torch.Tensor, np.ndarray],
        zmean: Union[torch.Tensor, np.ndarray, int, float],
        resolution: Union[int, Tuple[int, int], torch.Tensor,
                          np.ndarray] = None,
        in_ndc: bool = False,
        convention: str = 'opencv') -> Union[torch.Tensor, np.ndarray]:
    """Convert perspective to weakperspective intrinsic matrix.

    Args:
        K (Union[torch.Tensor, np.ndarray]): input intrinsic matrix, shape
            should be (batch, 4, 4) or (batch, 3, 3).
        zmean (Union[torch.Tensor, np.ndarray, int, float]): zmean for object.
            shape should be (batch, ) or singleton number.
        resolution (Union[int, Tuple[int, int], torch.Tensor, np.ndarray],
            optional): (height, width) of image. Defaults to None.
        in_ndc (bool, optional): whether defined in ndc. Defaults to False.
        convention (str, optional): camera convention. Defaults to 'opencv'.

    Returns:
        Union[torch.Tensor, np.ndarray]: output weakperspective pred_cam,
            shape is (batch, 4)
    """
    if K.ndim == 2:
        K = K[None]
    if isinstance(zmean, np.ndarray):
        zmean = torch.Tensor(zmean)
    elif isinstance(zmean, (float, int)):
        zmean = torch.Tensor([zmean])
    zmean = zmean.view(-1)
    _N = max(K.shape[0], zmean.shape[0])
    s1 = K[:, 0, 0]
    s2 = K[:, 1, 1]
    c1 = K[:, 0, 3]
    c2 = K[:, 1, 3]
    new_K = torch.zeros(_N, 4, 4)
    new_K[:, 0, 0] = zmean * s1
    new_K[:, 1, 1] = zmean * s2
    new_K[:, 0, 2] = c1
    new_K[:, 1, 2] = c2
    new_K[:, 2, 3] = 1
    new_K[:, 3, 2] = 1

    new_K, _, _ = convert_camera_matrix(K=new_K,
                                        convention_src=convention,
                                        convention_dst='pytorch3d',
                                        is_perspective=True,
                                        in_ndc_src=in_ndc,
                                        in_ndc_dst=True,
                                        resolution_src=resolution)
    return new_K


def estimate_orig_cam(
        joints3d: torch.Tensor,
        joints2d: torch.Tensor,
        joints_conf: Optional[torch.Tensor] = None,
        img_size: Union[Iterable[int], int] = 224) -> torch.Tensor:
    """_summary_

    Args:
        joints3d (torch.Tensor): (B, J, 3)
        joints2d (torch.Tensor): (B, J, 2)
        joints_conf (Optional[torch.Tensor], optional): (B, J, 1).
            Defaults to None.
        img_size (Union[Iterable[int], int], optional): int.
            Defaults to 224.

    Returns:
        torch.Tensor: _description_
    """
    h, w = img_size
    if joints_conf is not None:
        valid_ids = torch.where(joints_conf.view(-1) > 0)[0]
        joints2d = joints2d[:, valid_ids]
        joints3d = joints3d[:, valid_ids]
    kp2d_max_x = joints2d[..., 0].max()
    kp2d_min_x = joints2d[..., 0].min()
    kp2d_max_y = joints2d[..., 1].max()
    kp2d_min_y = joints2d[..., 1].min()

    kp3d_max_x = torch.max(joints3d[..., 0], 1)[0]
    kp3d_min_x = torch.min(joints3d[..., 0], 1)[0]
    kp3d_max_y = torch.max(joints3d[..., 1], 1)[0]
    kp3d_min_y = torch.min(joints3d[..., 1], 1)[0]

    sx = ((kp2d_max_x - kp2d_min_x) / w) / ((kp3d_max_x - kp3d_min_x) / 2)
    sy = ((kp2d_max_y - kp2d_min_y) / h) / ((kp3d_max_y - kp3d_min_y) / 2)

    tx = (kp2d_min_x / w - (kp3d_min_x * sx + 1) / 2) / sx * 2
    ty = (kp2d_min_y / h - (kp3d_min_y * sy + 1) / 2) / sy * 2

    return torch.cat([sx[:, None], sy[:, None], tx[:, None], ty[:, None]], 1)


def estimate_transl_from_orth(
        joints3d: torch.Tensor,
        joints2d: torch.Tensor,
        joints_conf: Optional[torch.Tensor] = None,
        focal_length: Union[int, float] = 5000,
        img_size: Union[Iterable[int], int] = 224) -> torch.Tensor:
    """_summary_

    Args:
        joints3d (torch.Tensor): _description_
        joints2d (torch.Tensor): _description_
        joints_conf (Optional[torch.Tensor], optional): _description_. Defaults to None.
        focal_length (Union[int, float], optional): _description_. Defaults to 5000.
        img_size (Union[Iterable[int], int], optional): _description_. Defaults to 224.

    Returns:
        torch.Tensor: _description_
    """
    sx, sy, tx, ty = torch.unbind(
        estimate_orig_cam(joints3d, joints2d, joints_conf, img_size))
    fx = fy = focal_length
    img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
    cx = img_size[1] / 2
    cy = img_size[0] / 2
    trans_z = torch.sqrt((fx / sx) * (fy / sy))
    trans_x = trans_z * (sx * tx - cx) / fx
    trans_y = trans_z * (sy * ty - cy) / fy
    transl = torch.cat([trans_x[:, None], trans_y[:, None], trans_z[:, None]],
                       1)
    return transl


def estimate_transl_weakperspective_batch(
        joints3d: torch.Tensor,
        joints2d: torch.Tensor,
        joints2d_conf: Optional[torch.Tensor] = None,
        joints3d_conf: Optional[torch.Tensor] = None,
        focal_length: Union[int, float] = 5000,
        img_size: Union[Iterable[int], int] = 224) -> torch.Tensor:
    """_summary_

    Args:
        joints3d (torch.Tensor): _description_
        joints2d (torch.Tensor): _description_
        joints_conf (Optional[torch.Tensor], optional): _description_. Defaults to None.
        focal_length (Union[int, float], optional): _description_. Defaults to 5000.
        img_size (Union[Iterable[int], int], optional): _description_. Defaults to 224.

    Returns:
        torch.Tensor: _description_
    """

    device = joints3d.device
    joints2d = joints2d.detach().cpu()
    joints3d = joints3d.detach().cpu()

    assert joints2d.ndim == 3
    assert joints3d.ndim == 3

    transl = torch.zeros(joints3d.shape[0], 3)
    for i in range(joints3d.shape[0]):
        joints3d_i = joints3d[i]
        joints2d_i = joints2d[i]
        if joints2d_conf is not None:
            conf2d_i = joints2d_conf[i].detach().cpu()
        else:
            conf2d_i = None

        if joints3d_conf is not None:
            conf3d_i = joints3d_conf[i].detach().cpu()
        else:
            conf3d_i = None
        focal_length_i = focal_length if isinstance(
            focal_length, (int, float)) else focal_length[i]
        transl[i] = estimate_transl_weakperspective(
            joints3d=joints3d_i,
            joints2d=joints2d_i,
            joints2d_conf=conf2d_i,
            joints3d_conf=conf3d_i,
            focal_length=focal_length_i,
            img_size=img_size)
    return transl.to(device)


def estimate_transl_weakperspective(
        joints3d: torch.Tensor,
        joints2d: torch.Tensor,
        joints2d_conf: Optional[torch.Tensor] = None,
        joints3d_conf: Optional[torch.Tensor] = None,
        focal_length: Union[int, float] = 5000,
        img_size: Union[Iterable[int], int] = 224) -> torch.Tensor:
    if joints2d_conf is not None:
        valid_ids = torch.where(joints2d_conf.view(-1) > 0)[0]
        joints2d = joints2d[valid_ids]
    if joints3d_conf is not None:
        valid_ids = torch.where(joints3d_conf.view(-1) > 0)[0]
        joints3d = joints3d[valid_ids]
    x1 = torch.min(joints3d[..., 0])
    x2 = torch.max(joints3d[..., 0])
    y1 = torch.min(joints3d[..., 1])
    y2 = torch.max(joints3d[..., 1])

    u1 = torch.min(joints2d[..., 0])
    u2 = torch.max(joints2d[..., 0])
    v1 = torch.min(joints2d[..., 1])
    v2 = torch.max(joints2d[..., 1])

    fx = fy = focal_length
    img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
    cx = img_size[1] / 2
    cy = img_size[0] / 2
    trans_zx = fx * (x1 - x2) / (u1 - u2)
    trans_zy = fy * (y1 - y2) / (v1 - v2)
    trans_z = torch.sqrt(trans_zx * trans_zy)

    tx_1 = (u1 - cx) * trans_z / fx - x1
    ty_1 = (v1 - cy) * trans_z / fy - y1

    tx_2 = (u2 - cx) * trans_z / fx - x2
    ty_2 = (v2 - cy) * trans_z / fy - y2

    tx = (tx_1 + tx_2) / 2
    ty = (ty_1 + ty_2) / 2
    transl = torch.Tensor([tx, ty, trans_z]).view(3)
    return transl


def estimate_cam_weakperspective(
        joints3d: torch.Tensor,
        joints2d: torch.Tensor,
        joints2d_conf: Optional[torch.Tensor] = None,
        joints3d_conf: Optional[torch.Tensor] = None,
        img_size: Union[Iterable[int], int] = 224) -> torch.Tensor:

    if joints2d_conf is not None:
        valid_ids = torch.where(joints2d_conf.view(-1) > 0)[0]
        joints2d = joints2d[valid_ids]
    if joints3d_conf is not None:
        valid_ids = torch.where(joints3d_conf.view(-1) > 0)[0]
        joints3d = joints3d[valid_ids]
    x1 = torch.min(joints3d[..., 0])
    x2 = torch.max(joints3d[..., 0])

    y1 = torch.min(joints3d[..., 1])
    y2 = torch.max(joints3d[..., 1])

    img_size = img_size if isinstance(img_size, int) else int(img_size[0])

    u1 = torch.min(joints2d[..., 0]) / (img_size / 2) - 1
    u2 = torch.max(joints2d[..., 0]) / (img_size / 2) - 1
    v1 = torch.min(joints2d[..., 1]) / (img_size / 2) - 1
    v2 = torch.max(joints2d[..., 1]) / (img_size / 2) - 1

    sx = (u1 - u2) / (x1 - x2)
    sy = (v1 - v2) / (y1 - y2)
    s = torch.sqrt(sx * sy)

    tx_1 = u1 / s - x1
    ty_1 = v1 / s - y1

    tx_2 = u2 / s - x2
    ty_2 = v2 / s - y2

    tx = (tx_1 + tx_2) / 2
    ty = (ty_1 + ty_2) / 2
    cam = torch.Tensor([s, tx, ty]).view(3)
    return cam


def estimate_cam_weakperspective_batch(
        joints3d: torch.Tensor,
        joints2d: torch.Tensor,
        joints2d_conf: Optional[torch.Tensor] = None,
        joints3d_conf: Optional[torch.Tensor] = None,
        img_size: Union[Iterable[int], int] = 224) -> torch.Tensor:
    device = joints3d.device
    joints2d = joints2d.detach().cpu()
    joints3d = joints3d.detach().cpu()

    assert joints2d.ndim == 3  # B, J, 2
    assert joints3d.ndim == 3  # B, J, 3

    cam = torch.zeros(joints3d.shape[0], 3)
    for i in range(joints3d.shape[0]):
        joints3d_i = joints3d[i]
        joints2d_i = joints2d[i]
        if joints2d_conf is not None:
            conf2d_i = joints2d_conf[i].detach().cpu()
        else:
            conf2d_i = None

        if joints3d_conf is not None:
            conf3d_i = joints3d_conf[i].detach().cpu()
        else:
            conf3d_i = None
        cam[i] = estimate_cam_weakperspective(joints3d=joints3d_i,
                                              joints2d=joints2d_i,
                                              joints2d_conf=conf2d_i,
                                              joints3d_conf=conf3d_i,
                                              img_size=img_size)
    return cam.to(device)


def pred_cam_to_transl(pred_camera, focal_length, img_size):
    pred_cam_t = torch.stack([
        pred_camera[:, 1], pred_camera[:, 2], 2 * focal_length /
        (img_size * pred_camera[:, 0] + 1e-9)
    ],
                             dim=-1)
    return pred_cam_t


def pred_transl_to_pred_cam(pred_transl, focal_length, img_size):
    pred_cam = torch.stack([
        2 * focal_length / (img_size * pred_transl[:, 2] + 1e-9),
        pred_transl[:, 0], pred_transl[:, 1]
    ],
                           dim=-1)
    return pred_cam


def get_K(focal_length: Union[int, torch.Tensor],
          principal_point: Union[Tuple[int], torch.Tensor] = (0, 0)):
    focal_length = torch.Tensor(focal_length)
    K = torch.eye(3, 3)
    K[0, 0] = focal_length.view(1, 1)
    K[1, 1] = focal_length.view(1, 1)
    K[0, 2] = principal_point[0]
    K[1, 2] = principal_point[1]
    return convert_K_3x3_to_4x4(K[None])


def get_K_batch(focal_length: Union[int, torch.Tensor],
                principal_point: Union[Tuple[int], torch.Tensor] = (0, 0)):
    if not isinstance(focal_length, torch.Tensor):
        focal_length = torch.Tensor(focal_length).view(-1, 1)
    batch_size = focal_length.shape[0]
    if not isinstance(principal_point, torch.Tensor):
        principal_point = torch.Tensor(principal_point).view(-1, 2)
    K = torch.eye(3, 3)[None].repeat_interleave(batch_size, 0)
    K[:, 0, 0] = focal_length[:, 0]
    K[:, 1, 1] = focal_length[:, 0]
    K[:, 0, 2] = principal_point[:, 0]
    K[:, 1, 2] = principal_point[:, 1]
    return convert_K_3x3_to_4x4(K[None])


def project_points_pred_cam(points_3d, camera, focal_length, img_res):
    """Perform orthographic projection of 3D points using the camera
    parameters, return projected 2D points in image plane.

    Notes:
        batch size: B
        point number: N
    Args:
        points_3d (Tensor([B, N, 3])): 3D points.
        camera (Tensor([B, 3])): camera parameters with the
            3 channel as (scale, translation_x, translation_y)
    Returns:
        points_2d (Tensor([B, N, 2])): projected 2D points
            in image space.
    """
    batch_size = points_3d.shape[0]
    device = points_3d.device
    cam_t = torch.stack([
        camera[:, 1], camera[:, 2], 2 * focal_length /
        (img_res * camera[:, 0] + 1e-9)
    ],
                        dim=-1)
    camera_center = torch.ones([batch_size, 2]).to(device) * (img_res / 2)
    rot_t = torch.eye(3, device=device,
                      dtype=points_3d.dtype).unsqueeze(0).expand(
                          batch_size, -1, -1)
    keypoints_2d = perspective_projection(points_3d,
                                          rotation=rot_t,
                                          translation=cam_t,
                                          focal_length=focal_length,
                                          camera_center=camera_center)
    return keypoints_2d


def project_points_pred_cam_log(points_3d, camera, focal_length, img_res):
    """Perform orthographic projection of 3D points using the camera
    parameters, return projected 2D points in image plane.

    Notes:
        batch size: B
        point number: N
    Args:
        points_3d (Tensor([B, N, 3])): 3D points.
        camera (Tensor([B, 3])): camera parameters with the
            3 channel as (scale, translation_x, translation_y)
    Returns:
        points_2d (Tensor([B, N, 2])): projected 2D points
            in image space.
    """
    batch_size = points_3d.shape[0]
    device = points_3d.device
    cam_t = torch.stack([
        camera[:, 1], camera[:, 2],
        2 * focal_length * torch.exp(-camera[:, 0]) / img_res
    ],
                        dim=-1)
    camera_center = torch.ones([batch_size, 2]).to(device) * (img_res / 2)
    rot_t = torch.eye(3, device=device,
                      dtype=points_3d.dtype).unsqueeze(0).expand(
                          batch_size, -1, -1)
    keypoints_2d = perspective_projection(points_3d,
                                          rotation=rot_t,
                                          translation=cam_t,
                                          focal_length=focal_length,
                                          camera_center=camera_center)
    return keypoints_2d


def project_points_focal_length(points_3d,
                                focal_length,
                                img_res=None,
                                in_ndc=True,
                                camera_center=None):
    """Perform orthographic projection of 3D points using the camera
    parameters, return projected 2D points in image plane.

    Notes:
        batch size: B
        point number: N
    Args:
        points_3d (Tensor([B, N, 3])): 3D points.
        camera (Tensor([B, 3])): camera parameters with the
            3 channel as (scale, translation_x, translation_y)
    Returns:
        points_2d (Tensor([B, N, 2])): projected 2D points
            in image space.
    """
    batch_size = points_3d.shape[0]
    device = points_3d.device

    if camera_center is None:
        camera_center = torch.ones([batch_size, 2]).to(device) * (img_res / 2)
    if in_ndc:
        focal_length = focal_length * img_res / 2
    else:
        focal_length = focal_length
    cam_t = torch.zeros([batch_size, 3]).to(device)
    rot_t = torch.eye(3, device=device,
                      dtype=points_3d.dtype).unsqueeze(0).expand(
                          batch_size, -1, -1)
    keypoints_2d = perspective_projection(points_3d,
                                          rotation=rot_t,
                                          translation=cam_t,
                                          focal_length=focal_length,
                                          camera_center=camera_center)
    return keypoints_2d


def project_points_focal_length_pixel(points_3d,
                                      focal_length,
                                      translation,
                                      img_res=None,
                                      camera_center=None):
    """Perform orthographic projection of 3D points using the camera
    parameters, return projected 2D points in image plane.

    Notes:
        batch size: B
        point number: N
    Args:
        points_3d (Tensor([B, N, 3])): 3D points.
        camera (Tensor([B, 3])): camera parameters with the
            3 channel as (scale, translation_x, translation_y)
    Returns:
        points_2d (Tensor([B, N, 2])): projected 2D points
            in image space.
    """
    batch_size = points_3d.shape[0]
    device = points_3d.device

    if camera_center is None:
        camera_center = torch.ones([batch_size, 2]).to(device) * (img_res / 2)
    rot_t = torch.eye(3, device=device,
                      dtype=points_3d.dtype).unsqueeze(0).expand(
                          batch_size, -1, -1)
    keypoints_2d = perspective_projection(points_3d,
                                          rotation=rot_t,
                                          translation=translation,
                                          focal_length=focal_length,
                                          camera_center=camera_center)
    return keypoints_2d


def weak_perspective_projection(points, scale, translation):
    """This function computes the weak perspective projection of a set of
    points.

    Input:
        points (bs, N, 3): 3D points
        scale (bs,1): scalar
        translation (bs, 2): point 2D translation
    """
    projected_points = scale.view(
        -1, 1, 1) * (points[:, :, :2] + translation.view(-1, 1, 2))

    return projected_points


def combine_RT(R, T):
    if R.ndim == 2:
        R = R[None]

    batch_size = R.shape[0]
    T = T.view(batch_size, 3, 1)
    RT = torch.zeros(batch_size, 4, 4).to(R.device)
    RT[:, 3, 3] = 1
    RT[:, :3, :3] = R
    RT[:, :3, 3:] = T
    return RT


def pred_cam_to_full_transl(pred_cam,
                            center,
                            scale,
                            full_img_shape,
                            focal_length,
                            px=None,
                            py=None):
    """convert the camera parameters from the crop camera to the full camera.
    :param crop_cam: shape=(N, 3) weak perspective camera in cropped
       img coordinates (s, tx, ty)
    :param center: shape=(N, 2) bbox coordinates (c_x, c_y)
    :param scale: shape=(N, 1) square bbox resolution  (b / 200)
    :param full_img_shape: shape=(N, 2) original image height and width
    :param focal_length: shape=(N,)
    :return:
    """
    img_h, img_w = full_img_shape[:, 0], full_img_shape[:, 1]
    cx, cy, bbox_w = center[:, 0], center[:, 1], scale

    if px is None:
        px = img_w / 2.
    if py is None:
        py = img_h / 2.
    s = pred_cam[:, 0] + 1e-9
    tx = pred_cam[:, 1]
    ty = pred_cam[:, 2]
    trans_z = 2 * focal_length / (bbox_w * s)
    trans_x = (2 * (cx - px) / (bbox_w * s)) + tx
    trans_y = (2 * (cy - py) / (bbox_w * s)) + ty
    full_transl = torch.stack([trans_x, trans_y, trans_z], dim=-1)
    return full_transl


def pred_cam_to_full_cam(pred_cam,
                         center,
                         scale,
                         full_img_shape,
                         px=None,
                         py=None):
    """convert the camera parameters from the crop camera to the full camera.
    :param crop_cam: shape=(N, 3) weak perspective camera in cropped
       img coordinates (s, tx, ty)
    :param center: shape=(N, 2) bbox coordinates (c_x, c_y)
    :param scale: shape=(N, 1) square bbox resolution  (b / 200)
    :param full_img_shape: shape=(N, 2) original image height and width
    :param focal_length: shape=(N,)
    :return:
    """
    img_h, img_w = full_img_shape[:, 0], full_img_shape[:, 1]
    cx, cy, bbox_w = center[:, 0], center[:, 1], scale

    if px is None:
        px = img_w / 2.
    if py is None:
        py = img_h / 2.
    s = pred_cam[:, 0] + 1e-9
    tx = pred_cam[:, 1]
    ty = pred_cam[:, 2]
    trans_x = (2 * (cx - px) / (bbox_w * s)) + tx
    trans_y = (2 * (cy - py) / (bbox_w * s)) + ty
    s_full = s * bbox_w / full_img_shape.max(-1)[0]
    full_cam = torch.stack([s_full, trans_x, trans_y], dim=-1)
    return full_cam


def merge_cam_to_full_transl(pred_cam,
                             center,
                             scale,
                             full_img_shape,
                             pred_transl,
                             px=None,
                             py=None):
    """convert the camera parameters from the crop camera to the full camera.
    :param crop_cam: shape=(N, 3) weak perspective camera in cropped
       img coordinates (s, tx, ty)
    :param center: shape=(N, 2) bbox coordinates (c_x, c_y)
    :param scale: shape=(N, 1) square bbox resolution  (b / 200)
    :param full_img_shape: shape=(N, 2) original image height and width
    :param focal_length: shape=(N,)
    :return:
    """
    img_h, img_w = full_img_shape[:, 0], full_img_shape[:, 1]
    cx, cy, bbox_w = center[:, 0], center[:, 1], scale
    if px is None:
        px = img_w / 2.
    if py is None:
        py = img_h / 2.
    s = pred_cam[:, 0] + 1e-9
    tx = pred_cam[:, 1]
    ty = pred_cam[:, 2]
    trans_z = pred_transl[:, 2]
    trans_x = (2 * (cx - px) / (bbox_w * s)) + tx
    trans_y = (2 * (cy - py) / (bbox_w * s)) + ty
    full_transl = torch.stack([trans_x, trans_y, trans_z], dim=-1)
    return full_transl


def full_transl_to_pred_cam(full_transl,
                            center,
                            scale,
                            full_img_shape,
                            focal_length,
                            px=None,
                            py=None):
    """convert full transl to the camera parameters
    :param full_transl: shape=(N, 3) full transl (x, y, z)
    :param center: shape=(N, 2) bbox coordinates (c_x, c_y)
    :param scale: shape=(N, 1) square bbox resolution  (b / 200)
    :param full_img_shape: shape=(N, 2) original image height and width
    :param focal_length: shape=(N,)
    :return:
    """
    img_h, img_w = full_img_shape[:, 0], full_img_shape[:, 1]
    if px is None:
        px = img_w / 2.
    if py is None:
        py = img_h / 2.
    cx, cy, bbox_w = center[:, 0], center[:, 1], scale
    trans_x, trans_y, trans_z = torch.unbind(full_transl, -1)
    s = 2 * focal_length / (bbox_w * trans_z)
    tx = trans_x - (2 * (cx - px) / (bbox_w * s))
    ty = trans_y - (2 * (cy - py) / (bbox_w * s))
    pred_cam = torch.stack([s, tx, ty], dim=-1)
    return pred_cam


def rotate_smpl_cam(pred_cam, angle, gt_model_joints):
    batch_size = angle.shape[0]
    device = pred_cam.device
    rot_mat = torch.eye(2, 2)[None].repeat_interleave(batch_size, 0)
    rot_rad = torch.deg2rad(angle)
    sn, cs = torch.sin(rot_rad), torch.cos(rot_rad)
    rot_mat[:, 0, 0] = cs
    rot_mat[:, 0, 1] = -sn
    rot_mat[:, 1, 0] = sn
    rot_mat[:, 1, 1] = cs
    s, tx, ty = pred_cam.unbind(-1)
    tx_, ty_ = torch.bmm(
        rot_mat.to(device),
        torch.cat([
            tx.view(-1, 1) + gt_model_joints[:, 0:1, 0],
            ty.view(-1, 1) + gt_model_joints[:, 0:1, 1]
        ], 1).view(-1, 2, 1)).unbind(1)

    tx_ = tx_ - gt_model_joints[:, 0:1, 0]
    ty_ = ty_ - gt_model_joints[:, 0:1, 1]
    cam_new = torch.cat([s.unsqueeze(1), tx_, ty_], -1)
    return cam_new


def rotate_transl(transl, angle, pelvis):
    batch_size = angle.shape[0]
    device = transl.device
    rot_mat = torch.eye(2, 2)[None].repeat_interleave(batch_size, 0)
    rot_rad = torch.deg2rad(angle)
    sn, cs = torch.sin(rot_rad), torch.cos(rot_rad)
    rot_mat[:, 0, 0] = cs
    rot_mat[:, 0, 1] = -sn
    rot_mat[:, 1, 0] = sn
    rot_mat[:, 1, 1] = cs
    tx, ty, tz = transl.unbind(-1)
    tx_, ty_ = torch.bmm(
        rot_mat.to(device),
        torch.cat(
            [tx.view(-1, 1) + pelvis[:, 0],
             ty.view(-1, 1) + pelvis[:, 1]], 1).view(-1, 2, 1)).unbind(1)

    tx_ = tx_ - pelvis[:, 0]
    ty_ = ty_ - pelvis[:, 1]
    transl_new = torch.cat([tx_, ty_, tz.unsqueeze(-1)], -1)
    return transl_new


def rotate_transl_numpy(transl, angle, pelvis):
    batch_size = angle.shape[0]
    rot_mat = np.eye(2, 2)[None].repeat(batch_size, 0)
    rot_rad = np.deg2rad(angle)
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    rot_mat[:, 0, 0] = cs
    rot_mat[:, 0, 1] = -sn
    rot_mat[:, 1, 0] = sn
    rot_mat[:, 1, 1] = cs
    tx, ty, tz = np.split(transl, -1)
    tx_, ty_ = np.matmul(
        rot_mat,
        np.concatenate([
            tx.view(-1, 1) + pelvis[:, 0],
            ty.view(-1, 1) + pelvis[:, 1],
        ], 1).view(-1, 2, 1)).unbind(1)

    tx_ = tx_ - pelvis[:, 0]
    ty_ = ty_ - pelvis[:, 1]
    transl_new = np.concatenate([tx_, ty_, tz.unsqueeze(-1)], -1)
    return transl_new
