import torch
import torch.nn.functional as F
import cv2
import numpy as np

from typing import Union
from mmhuman3d.utils.demo_utils import convert_crop_cam_to_orig_img


def crop_image_grid_sample(image: torch.Tensor,
                           bbox: Union[None, torch.Tensor] = None,
                           img_size: Union[None, tuple, list, int] = None,
                           in_ndc: bool = True) -> torch.Tensor:
    # image (B, C, H, W), bbox (B, 4)
    N, _, orig_h, orig_w = image.shape

    if isinstance(img_size, int):
        H, W = img_size, img_size
    else:
        H, W = img_size

    bbox = format_bbox(bbox)
    if not in_ndc:
        bbox = bbox_screen2ndc(bbox, resolution=(orig_h, orig_w))

    x1_norm, y1_norm, x2_norm, y2_norm = torch.unbind(bbox, -1)

    h_grid = torch.linspace(0, 1, H).view(-1, 1).repeat(1, W)
    v_grid = torch.linspace(0, 1, W).repeat(H, 1)

    mesh_grid = torch.cat((v_grid.unsqueeze(2), h_grid.unsqueeze(2)),
                          dim=2)[None].repeat_interleave(N, 0).to(image.device)
    mesh_grid[..., 0] = mesh_grid[..., 0] * (x2_norm - x1_norm).view(
        N, 1, 1) + x1_norm.view(N, 1, 1)
    mesh_grid[..., 1] = mesh_grid[..., 1] * (y2_norm - y1_norm).view(
        N, 1, 1) + y1_norm.view(N, 1, 1)
    output_image = F.grid_sample(image, mesh_grid, align_corners=True)

    return output_image


def crop_image(
    image: Union[np.ndarray, torch.Tensor],
    bbox: Union[None, np.ndarray, torch.Tensor] = None,
    img_size: Union[None, tuple, list, int] = None,
    in_ndc: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    if bbox is not None:
        if in_ndc:
            bbox = bbox_ndc2screen(bbox, img_size=img_size)
        bbox = format_bbox(bbox)
        x1, y1, x2, y2 = torch.unbind(bbox.unsqueeze(-2), -1) if isinstance(
            bbox, torch.Tensor) else np.split(bbox, 4, -1)
        cropped = image[int(y1):int(y2 + 1), int(x1):int(x2 + 1)]
    else:
        cropped = image
    height, width = cropped.shape[:2]

    if height > width:
        im_bg = np.zeros((height, height, 3)) if isinstance(
            image, np.np.ndarray) else torch.zeros(height, height, 3)
        im_bg[:,
              int(height / 2 - width / 2):int(height / 2 - width / 2) +
              width] = cropped
    else:
        im_bg = np.zeros((width, width, 3)) if isinstance(
            image, np.np.ndarray) else torch.zeros(width, width, 3)
        im_bg[int(width / 2 - height / 2):int(width / 2 - height / 2) +
              height, :] = cropped
    if img_size is not None:
        if isinstance(img_size, int):
            height, width = img_size, img_size
        else:
            height, width = img_size
        if isinstance(im_bg, np.ndarray):
            im_bg = cv2.resize(im_bg, (width, height))
    return im_bg.astype(np.uint8)


def kp2d_to_bbox(
    kp2d: Union[torch.Tensor, np.ndarray],
    scale_factor: float = 1.0,
    xywh: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """Convert kp2d to bbox.

    Args:
        kp2d (np.ndarray):  shape should be (num_frame, num_points, 2/3).
        bbox_format (Literal['xyxy', 'xywh'], optional): Defaults to 'xyxy'.

    Returns:
        np.ndarray: shape will be (num_frame, num_person, 4)
    """
    if len(kp2d.shape) == 2:
        kp2d = kp2d[None]
    assert len(kp2d.shape) == 3
    num_frame, _, _ = kp2d.shape
    if isinstance(kp2d, np.ndarray):
        x1 = np.min(kp2d[..., 0:1], axis=-2)
        y1 = np.min(kp2d[..., 1:2], axis=-2)
        x2 = np.max(kp2d[..., 0:1], axis=-2)
        y2 = np.max(kp2d[..., 1:2], axis=-2)
    elif isinstance(kp2d, torch.Tensor):
        x1 = torch.min(kp2d[..., 0:1], axis=-2)[0]
        y1 = torch.min(kp2d[..., 1:2], axis=-2)[0]
        x2 = torch.max(kp2d[..., 0:1], axis=-2)[0]
        y2 = torch.max(kp2d[..., 1:2], axis=-2)[0]

    bbox = torch.cat([x1, y1, x2, y2], -1) if isinstance(
        kp2d, torch.Tensor) else np.concatenate([x1, y1, x2, y2], -1)
    assert bbox.shape == (num_frame, 4)
    if scale_factor != 1.0:
        bbox = scale_bbox(bbox, scale_factor)
    if xywh:
        bbox = bbox_xyxy2xywh(bbox)
    return bbox


def format_bbox(
        bbox: Union[np.ndarray,
                    torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    bbox = bbox.squeeze()
    if len(bbox.shape) == 1:
        bbox = bbox[None]
    return bbox[:, :4]


def expand_bbox_to_square(bbox):
    bbox = format_bbox(bbox)
    x1, y1, x2, y2 = torch.unbind(bbox.unsqueeze(-2), -1) if isinstance(
        bbox, torch.Tensor) else np.split(bbox, 4, -1)
    c_x = x1 / 2 + x2 / 2
    c_y = y1 / 2 + y2 / 2
    w_x = x2 - x1
    w_y = y2 - y1
    w = torch.max(
        torch.cat([w_x.unsqueeze(1), w_y.unsqueeze(1)], 1),
        1)[0] if isinstance(bbox, torch.Tensor) else np.max(
            np.concatenate([np.expand_dims(w_x, 1),
                            np.expand_dims(w_y, 1)], 1), 1)[0]
    x1 = c_x - w / 2
    x2 = c_x + w / 2
    y1 = c_y - w / 2
    y2 = c_y + w / 2
    bbox_square = torch.cat([x1, y1, x2, y2], -1) if isinstance(
        bbox, torch.Tensor) else np.concatenate([x1, y1, x2, y2], -1)
    return bbox_square


def bbox_screen2ndc(bbox, resolution):
    bbox = format_bbox(bbox)
    if isinstance(bbox, torch.Tensor):
        x1, y1, x2, y2 = torch.unbind(bbox.unsqueeze(-2), -1) if isinstance(
            bbox, torch.Tensor) else np.split(bbox, 4, -1)

    if isinstance(resolution, int):
        height, width = resolution, resolution
    else:
        height, width = resolution

    x1 = x1 / width * 2 - 1
    x2 = x2 / width * 2 - 1
    y1 = y1 / height * 2 - 1
    y2 = y2 / height * 2 - 1
    bbox_normed = torch.cat([x1, y1, x2, y2], -1) if isinstance(
        bbox, torch.Tensor) else np.concatenate([x1, y1, x2, y2], -1)
    return bbox_normed


def bbox_ndc2screen(bbox, resolution):
    bbox = format_bbox(bbox)
    x1, y1, x2, y2 = torch.unbind(bbox.unsqueeze(-2), -1) if isinstance(
        bbox, torch.Tensor) else np.split(bbox, 4, -1)

    if isinstance(resolution, int):
        height, width = resolution, resolution
    else:
        height, width = resolution

    x1 = (x1 + 1) / 2 * width
    x2 = (x2 + 1) / 2 * width
    y1 = (y1 + 1) / 2 * height
    y2 = (y2 + 1) / 2 * height
    bbox_unormed = torch.cat([x1, y1, x2, y2], -1) if isinstance(
        bbox, torch.Tensor) else np.concatenate([x1, y1, x2, y2], -1)
    return bbox_unormed


def get_square_bbox(resolution):
    if isinstance(resolution, int):
        height, width = resolution, resolution
    else:
        height, width = resolution

    if height > width:
        bbox = [(width - height) // 2, 0, width + (height - width) // 2,
                height]
    else:
        bbox = [
            0, (height - width) // 2, width, height + (width - height) // 2
        ]
    return np.array(bbox)


def caculate_relative_bbox(bbox, parent_box):
    bbox = format_bbox(bbox)
    parent_box = format_bbox(parent_box)
    assert bbox.shape[1] == 4
    bbox[:, 0] += parent_box[:, 0]
    bbox[:, 2] += parent_box[:, 0]

    bbox[:, 1] += parent_box[:, 1]
    bbox[:, 3] += parent_box[:, 1]
    return bbox


def bbox_xyxy2xywh(bbox):
    bbox = format_bbox(bbox)
    x1, y1, x2, y2 = torch.unbind(bbox.unsqueeze(-2), -1) if isinstance(
        bbox, torch.Tensor) else np.split(bbox, 4, -1)
    cx = x1 / 2 + x2 / 2
    cy = y1 / 2 + y2 / 2
    w = x2 - x1
    h = y2 - y1
    bbox_xywh = torch.cat([cx, cy, w, h], -1) if isinstance(
        bbox, torch.Tensor) else np.concatenate([cx, cy, w, h], -1)
    return bbox_xywh


def bbox_xyxy2ltwh(bbox):
    bbox = format_bbox(bbox)
    x1, y1, x2, y2 = torch.unbind(bbox.unsqueeze(-2), -1) if isinstance(
        bbox, torch.Tensor) else np.split(bbox, 4, -1)

    w = x2 - x1
    h = y2 - y1
    bbox_xywh = torch.cat([x1, y1, w, h], -1) if isinstance(
        bbox, torch.Tensor) else np.concatenate([x1, y1, w, h], -1)
    return bbox_xywh


def bbox_xywh2ltwh(bbox):
    bbox = format_bbox(bbox)
    cx, cy, w, h = torch.unbind(bbox.unsqueeze(-2), -1) if isinstance(
        bbox, torch.Tensor) else np.split(bbox, 4, -1)
    left = cx - w / 2
    top = cy - h / 2
    bbox_ltwh = torch.cat([left, top, w, h], -1) if isinstance(
        bbox, torch.Tensor) else np.concatenate([left, top, w, h], -1)
    return bbox_ltwh


def bbox_ltwh2xywh(bbox):
    bbox = format_bbox(bbox)
    left, top, w, h = torch.unbind(bbox.unsqueeze(-2), -1) if isinstance(
        bbox, torch.Tensor) else np.split(bbox, 4, -1)
    cx = left + w / 2
    cy = top + h / 2
    bbox_xywh = torch.cat([cx, cy, w, h], -1) if isinstance(
        bbox, torch.Tensor) else np.concatenate([cx, cy, w, h], -1)
    return bbox_xywh


def bbox_ltwh2xyxy(bbox):
    bbox = format_bbox(bbox)
    left, top, w, h = torch.unbind(bbox.unsqueeze(-2), -1) if isinstance(
        bbox, torch.Tensor) else np.split(bbox, 4, -1)
    x1 = left
    y1 = top
    x2 = left + w
    y2 = top + h
    bbox_xywh = torch.cat([x1, y1, x2, y2], -1) if isinstance(
        bbox, torch.Tensor) else np.concatenate([x1, y1, x2, y2], -1)
    return bbox_xywh


def bbox_xywh2xyxy(bbox):
    bbox = format_bbox(bbox)
    cx, cy, w, h = torch.unbind(bbox.unsqueeze(-2), -1) if isinstance(
        bbox, torch.Tensor) else np.split(bbox, 4, -1)
    x1 = cx - w / 2
    x2 = cx + w / 2
    y1 = cy - h / 2
    y2 = cy + h / 2
    bbox_xyxy = torch.cat([x1, y1, x2, y2], -1) if isinstance(
        bbox, torch.Tensor) else np.concatenate([x1, y1, x2, y2], -1)
    return bbox_xyxy


def scale_bbox(bbox, scale_factor=1.0, bbox_format='xyxy'):
    bbox = format_bbox(bbox)
    if bbox_format == 'xyxy':
        bbox = bbox_xyxy2xywh(bbox)
    cx, cy, w, h = torch.unbind(bbox.unsqueeze(-2), -1) if isinstance(
        bbox, torch.Tensor) else np.split(bbox, 4, -1)
    w = scale_factor * w
    h = scale_factor * h
    bbox = torch.cat([cx, cy, w, h], -1) if isinstance(
        bbox, torch.Tensor) else np.concatenate([cx, cy, w, h], -1)
    if bbox_format == 'xyxy':
        bbox = bbox_xywh2xyxy(bbox)
    return bbox


def clip_bbox(bbox, resolution, bbox_format='xyxy'):
    bbox = format_bbox(bbox)
    if bbox_format == 'xywh':
        bbox = bbox_xywh2xyxy(bbox)
    x1, y1, x2, y2 = torch.unbind(bbox.unsqueeze(-2), -1) if isinstance(
        bbox, torch.Tensor) else np.split(bbox, 4, -1)
    clip = torch.clip if isinstance(bbox, torch.Tensor) else np.clip

    if isinstance(resolution, int):
        height, width = resolution, resolution
    else:
        height, width = resolution

    x1 = clip(x1, 0, width)
    x2 = clip(x2, 0, width)
    y1 = clip(y1, 0, height)
    y2 = clip(y2, 0, height)
    bbox_xyxy = torch.cat([x1, y1, x2, y2], -1) if isinstance(
        bbox, torch.Tensor) else np.concatenate([x1, y1, x2, y2], -1)
    return bbox_xyxy


def bbox_iou(bbox1, bbox2, xywh=False):
    bbox1 = format_bbox(bbox1)
    bbox2 = format_bbox(bbox2)
    if xywh:
        bbox1 = bbox_xyxy2xywh(bbox1)
        bbox2 = bbox_xyxy2xywh(bbox2)
    x1, y1, x2, y2 = torch.unbind(bbox1.unsqueeze(-2), -1) if isinstance(
        bbox1, torch.Tensor) else np.split(bbox1, 4, -1)

    a1, b1, a2, b2 = torch.unbind(bbox2.unsqueeze(-2), -1) if isinstance(
        bbox2, torch.Tensor) else np.split(bbox2, 4, -1)

    ax = np.max(np.concatenate([x1, a1], -1), -1)
    ay = np.max(np.concatenate([y1, b1], -1), -1)
    bx = np.min(np.concatenate([x2, a2], -1), -1)
    by = np.min(np.concatenate([y2, b2], -1), -1)

    area_N = (x2 - x1) * (y2 - y1)
    area_M = (a2 - a1) * (b2 - b1)

    w = np.clip(bx - ax, 0, 1e6)
    h = np.clip(by - ay, 0, 1e6)
    area_X = w * h
    area_X = area_X.reshape(-1, 1)

    return area_X / (area_N + area_M - area_X)


def bbox_ioa(bbox1, bbox2, xywh=False):
    bbox1 = format_bbox(bbox1)
    bbox2 = format_bbox(bbox2)
    if xywh:
        bbox1 = bbox_xyxy2xywh(bbox1)
        bbox2 = bbox_xyxy2xywh(bbox2)
    x1, y1, x2, y2 = torch.unbind(bbox1.unsqueeze(-2), -1) if isinstance(
        bbox1, torch.Tensor) else np.split(bbox1, 4, -1)

    a1, b1, a2, b2 = torch.unbind(bbox2.unsqueeze(-2), -1) if isinstance(
        bbox2, torch.Tensor) else np.split(bbox2, 4, -1)

    ax = np.max(np.concatenate([x1, a1], -1), -1)
    ay = np.max(np.concatenate([y1, b1], -1), -1)
    bx = np.min(np.concatenate([x2, a2], -1), -1)
    by = np.min(np.concatenate([y2, b2], -1), -1)

    area_N = (x2 - x1) * (y2 - y1)

    w = np.clip(bx - ax, 0, 1e6)
    h = np.clip(by - ay, 0, 1e6)
    area_X = w * h
    area_X = area_X.reshape(-1, 1)

    return area_X / area_N


__all__ = [
    'clip_bbox', 'scale_bbox', 'bbox_xywh2xyxy', 'bbox_xyxy2xywh',
    'caculate_relative_bbox', 'get_square_bbox',
    'convert_crop_cam_to_orig_img', 'get_square_bbox', 'kp2d_to_bbox',
    'expand_bbox_to_square', 'bbox_ndc2screen', 'bbox_screen2ndc',
    'bbox_ltwh2xywh', 'bbox_ltwh2xyxy', 'bbox_iou'
]
