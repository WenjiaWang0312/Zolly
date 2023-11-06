import torch
import cv2
import numpy as np
from typing import Union

from avatar3d.utils.frame_utils import array_to_video_cv2
from .bbox_utils import format_bbox
from typing import Optional, Iterable, Tuple
from mmhuman3d.utils.demo_utils import get_different_colors
from .visualize_keypoints2d import visualize_kp2d
from avatar3d.models.body_models.mappings import KEYPOINTS_FACTORY


def draw_bbox(image: np.ndarray,
              bbox: np.ndarray,
              in_ndc: bool = False,
              color: Union[tuple, list, np.ndarray] = [255, 255, 255]):
    image = image.copy()
    orig_h, orig_w = image.shape[:2]
    bbox = format_bbox(bbox)
    x1, y1, x2, y2 = torch.unbind(bbox.unsqueeze(-2), -1) if isinstance(
        bbox, torch.Tensor) else np.split(bbox, 4, -1)
    if in_ndc:
        x1 = int((x1 + 1) / 2 * orig_w)
        x2 = int((x2 + 1) / 2 * orig_w)
        y1 = int((y1 + 1) / 2 * orig_h)
        y2 = int((y2 + 1) / 2 * orig_h)
    pt1 = [int(x1), int(y1)]
    pt2 = [int(x2), int(y1)]
    pt3 = [int(x2), int(y2)]
    pt4 = [int(x1), int(y2)]
    lines = [[pt1, pt2], [pt2, pt3], [pt3, pt4], [pt4, pt1]]
    for line in lines:
        p1, p2 = line
        cv2.line(image, p1, p2, color, thickness=1)
    return image


def draw_bbox_video(images: np.ndarray,
                    bboxes: np.ndarray,
                    in_ndc: bool = False,
                    color: Union[tuple, list, np.ndarray] = [255, 255, 255],
                    output_path: str = None):
    video = []
    for image, bbox in zip(images, bboxes):
        video.append(draw_bbox(image, bbox, in_ndc, color)[None])
    video = np.concatenate(video)
    array_to_video_cv2(video, output_path)
    if output_path is None:
        return video


def draw_circles(
    kp2d: np.ndarray,
    image_array: Optional[np.ndarray],
    radius: int = 2,
    color: Optional[Iterable[int]] = None,
    with_number: bool = False,
    font_size: int = 1,
):
    for index in range(kp2d.shape[0]):
        x, y = kp2d[index, :2]
        cv2.circle(image_array, (int(x), int(y)), radius, color, thickness=-1)

        if with_number:
            cv2.putText(image_array, str(index), (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size,
                        np.array([255, 255, 255]).astype(np.int32).tolist(), 2)
    return image_array


def draw_skeletons_image(
    kp2d: np.ndarray,
    image_array: Optional[np.ndarray],
    palette: Optional[Iterable[int]] = None,
    convention: str = 'coco',
    resolution: Optional[Union[Tuple[int, int], list]] = None,
    draw_bbox: bool = False,
    with_number: bool = False,
    mask=None,
):
    if isinstance(kp2d, torch.Tensor):
        kp2d = kp2d.detach().cpu().numpy()
    if kp2d.ndim == 2:
        kp2d = kp2d[None]
    if image_array.ndim == 3:
        image_array = image_array[None]
    return visualize_kp2d(
        kp2d=kp2d,
        image_array=image_array,
        palette=palette,
        data_source=convention,
        resolution=resolution,
        draw_bbox=draw_bbox,
        mask=mask,
        with_number=with_number,
        disable_limbs=False,
        return_array=True,
        keypoints_factory=KEYPOINTS_FACTORY,
    )[0]


def draw_skeletons_video(kp2d: np.ndarray,
                         image_array: Optional[np.ndarray],
                         video_path: str,
                         mask=None,
                         palette: Optional[Iterable[int]] = None,
                         convention: str = 'coco',
                         resolution: Optional[Union[Tuple[int, int],
                                                    list]] = None,
                         draw_bbox: bool = False,
                         with_number: bool = False,
                         overwrite: bool = True):
    visualize_kp2d(
        kp2d=kp2d,
        output_path=video_path,
        image_array=image_array,
        palette=palette,
        mask=mask,
        data_source=convention,
        resolution=resolution,
        draw_bbox=draw_bbox,
        with_number=with_number,
        disable_limbs=False,
        return_array=False,
        overwrite=overwrite,
        keypoints_factory=KEYPOINTS_FACTORY,
    )


__all__ = ['get_different_colors']
