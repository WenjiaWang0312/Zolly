from typing import Optional, Tuple, Union

import numpy as np
import torch
from avatar3d.models.body_models.mappings import KEYPOINTS_FACTORY
from mmhuman3d.core.conventions.keypoints_mapping.human_data import (
    HUMAN_DATA_LIMBS_INDEX,
    HUMAN_DATA_PALETTE,
)


def search_limbs(
        data_source: str,
        mask: Optional[Union[np.ndarray, tuple, list]] = None,
        keypoints_factory: dict = KEYPOINTS_FACTORY) -> Tuple[dict, dict]:
    """Search the corresponding limbs following the basis human_data limbs. The
    mask could mask out the incorrect keypoints.

    Args:
        data_source (str): data source type.
        mask (Optional[Union[np.ndarray, tuple, list]], optional):
            refer to keypoints_mapping. Defaults to None.
        keypoints_factory (dict, optional): Dict of all the conventions.
            Defaults to KEYPOINTS_FACTORY.
    Returns:
        Tuple[dict, dict]: (limbs_target, limbs_palette).
    """
    limbs_source = HUMAN_DATA_LIMBS_INDEX
    limbs_palette = HUMAN_DATA_PALETTE
    keypoints_source = keypoints_factory['human_data']
    keypoints_target = keypoints_factory[data_source]
    limbs_target = {}
    for k, part_limbs in limbs_source.items():
        limbs_target[k] = []
        for limb in part_limbs:
            flag = False
            if (keypoints_source[limb[0]]
                    in keypoints_target) and (keypoints_source[limb[1]]
                                              in keypoints_target):
                if mask is not None:
                    if mask[keypoints_target.index(keypoints_source[
                            limb[0]])] != 0 and mask[keypoints_target.index(
                                keypoints_source[limb[1]])] != 0:
                        flag = True
                else:
                    flag = True
                if flag:
                    limbs_target.setdefault(k, []).append([
                        keypoints_target.index(keypoints_source[limb[0]]),
                        keypoints_target.index(keypoints_source[limb[1]])
                    ])
        if k in limbs_target:
            if k == 'body':
                np.random.seed(0)
                limbs_palette[k] = np.random.randint(0,
                                                     high=255,
                                                     size=(len(
                                                         limbs_target[k]), 3))
            else:
                limbs_palette[k] = np.array(limbs_palette[k])
    return limbs_target, limbs_palette


def get_max_preds_numpy(heatmaps):
    """Get keypoint predictions from score maps.
    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W
    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
    Returns:
        tuple: A tuple containing aggregated results.
        - preds (np.ndarray[N, K, 2]): Predicted keypoint location.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.detach().cpu().numpy()
    assert isinstance(heatmaps,
                      np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds


def get_max_preds_weighted(heatmaps):
    bs, num_j, h, w = heatmaps.shape
    Y, X = torch.meshgrid(torch.arange(w) / w, torch.arange(h) / h)
    X = X.to(heatmaps.device)[None, None].repeat(bs, num_j, 1, 1)
    Y = Y.to(heatmaps.device)[None, None].repeat(bs, num_j, 1, 1)
    kpx = (heatmaps * X).view(bs, num_j, -1).sum(dim=2) /\
        (1e-6 + heatmaps.view(bs, num_j, -1).sum(dim=2))
    kpy = (heatmaps * Y).view(bs, num_j, -1).sum(dim=2) /\
        (1e-6 + heatmaps.view(bs, num_j, -1).sum(dim=2))
    preds = torch.stack([kpx, kpy], dim=2)
    return preds


def get_max_preds(heatmaps):
    bs, num_j = heatmaps.shape[:2]
    h, w = heatmaps.shape[2:]
    indices = torch.argmax(heatmaps.view(bs, num_j, -1), dim=2)
    preds = torch.zeros((bs, num_j, 2),
                        dtype=torch.int64,
                        device=heatmaps.device)
    preds[..., 0] = indices % w
    preds[..., 1] = indices // w
    return preds


def get_max_preds_soft(heatmaps, beta=1e3):
    bs, num_j, h, w = heatmaps.shape
    # heatmaps: B x J x H x W
    # apply softmax function along H and W axis
    heatmaps = heatmaps.view(bs, num_j, -1)  # B x J x (H*W)
    heatmaps = torch.softmax(beta * heatmaps,
                             dim=2)  # apply softmax along H*W axis
    heatmaps = heatmaps.view(bs, num_j, h, w)  # B x J x H x W

    # create coordinate tensors
    coords_h = torch.linspace(-1,
                              1,
                              h,
                              device=heatmaps.device,
                              dtype=heatmaps.dtype)
    coords_w = torch.linspace(-1,
                              1,
                              w,
                              device=heatmaps.device,
                              dtype=heatmaps.dtype)
    coords_h = coords_h.view(1, 1, -1, 1).expand(bs, num_j, -1,
                                                 w)  # B x J x H x W
    coords_w = coords_w.view(1, 1, 1, -1).expand(bs, num_j, h,
                                                 -1)  # B x J x H x W

    # compute softargmax along H and W axis
    coord_h = torch.sum(coords_h * heatmaps, dim=(2, 3))
    coord_w = torch.sum(coords_w * heatmaps, dim=(2, 3))

    # concatenate coordinates
    coords = torch.stack([coord_w, coord_h], dim=2)  # B x J x 2

    return coords

def get_max_preds_soft_3d(heatmaps, beta=1e3):
    bs, num_j, d, h, w = heatmaps.shape
    # heatmaps: B x J x D x H x W
    # apply softmax function along D, H and W axis
    heatmaps = heatmaps.view(bs, num_j, -1)  # B x J x (D*H*W)
    heatmaps = torch.softmax(beta * heatmaps, dim=2)  # apply softmax along D*H*W axis
    heatmaps = heatmaps.view(bs, num_j, d, h, w)  # B x J x D x H x W

    # create coordinate tensors
    coords_d = torch.linspace(-1, 1, d, device=heatmaps.device, dtype=heatmaps.dtype)
    coords_h = torch.linspace(-1, 1, h, device=heatmaps.device, dtype=heatmaps.dtype)
    coords_w = torch.linspace(-1, 1, w, device=heatmaps.device, dtype=heatmaps.dtype)
    coords_d = coords_d.view(1, 1, -1, 1, 1).expand(bs, num_j, -1, h, w)  # B x J x D x H x W
    coords_h = coords_h.view(1, 1, 1, -1, 1).expand(bs, num_j, d, -1, w)  # B x J x D x H x W
    coords_w = coords_w.view(1, 1, 1, 1, -1).expand(bs, num_j, d, h, -1)  # B x J x D x H x W

    # compute softargmax along D, H and W axis
    coord_d = torch.sum(coords_d * heatmaps, dim=(2, 3, 4))
    coord_h = torch.sum(coords_h * heatmaps, dim=(2, 3, 4))
    coord_w = torch.sum(coords_w * heatmaps, dim=(2, 3, 4))

    # concatenate coordinates
    coords = torch.stack([coord_w, coord_h, coord_d], dim=2)  # B x J x 3

    return coords

def get_max_preds_soft_1d(heatmaps, beta=1e3):
    bs, num_j, w = heatmaps.shape
    # heatmaps: B x J x W
    # apply softmax function along W axis
    heatmaps = heatmaps.view(bs, num_j, -1)  # B x J x (W)
    heatmaps = torch.softmax(beta * heatmaps, dim=2)  # apply softmax along W axis
    heatmaps = heatmaps.view(bs, num_j, w)  # B x J x W

    # create coordinate tensors
    coords_w = torch.linspace(-1, 1, w, device=heatmaps.device, dtype=heatmaps.dtype)
    coords_w = coords_w.view(1, 1, -1).expand(bs, num_j, -1)  # B x J x W

    # compute softargmax along W axis
    coord_w = torch.sum(coords_w * heatmaps, dim=(2))

    # concatenate coordinates
    coords = torch.stack([coord_w], dim=2)  # B x J x 1

    return coords

def inverse_projection(
    focal_length,
    pred_log_s,
    pred_joints_uv,
    scale_full=None,
    scale_part=None,
):
    """Inverse projection to get 3D joints.
    """
    if scale_full is not None and scale_part is not None:
        scale = (scale_full.view(-1, 1) / scale_part.view(-1, 1))
    else:
        scale = 1.
    joints_z = focal_length.view(-1, 1) * torch.exp(-pred_log_s) * scale
    joints_z = joints_z.unsqueeze(-1)
    joints_xy = pred_joints_uv * torch.exp(-1 * pred_log_s).unsqueeze(-1)
    joints_xyz = torch.cat([joints_xy, joints_z], dim=2)
    return joints_xyz
