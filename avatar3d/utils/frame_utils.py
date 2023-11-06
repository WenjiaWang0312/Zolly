import enum
from typing import Optional, Tuple, Union
import cv2
import numpy as np
import torch
import os
import torch.nn.functional as F

from mmhuman3d.utils.ffmpeg_utils import (
    video_writer, array_to_video, array_to_images, video_to_array,
    images_to_sorted_images, vid_info_reader, images_to_array, video_to_gif,
    video_to_images, images_to_video, images_to_gif, gif_to_video,
    gif_to_images, crop_video, slice_video, spatial_concat_video,
    temporal_concat_video, compress_video, pad_for_libx264)


def array_to_images_cv2(image_array,
                        folder,
                        img_format,
                        resolution: Optional[Union[Tuple[int, int],
                                                   Tuple[float,
                                                         float]]] = None):
    flags = []
    image_list = [
        os.path.join(folder, img_format % i) for i in range(len(image_array))
    ]
    for index, image_path in enumerate(image_list):
        im = image_array[index]
        if resolution is not None:
            im = cv2.resize(im, resolution)
        flags.append(cv2.imwrite(image_path, im))
    return flags


def images_to_array_cv2(image_list: list,
                        resolution: Optional[Union[Tuple[int, int],
                                                   Tuple[float,
                                                         float]]] = None):
    image_array = []
    resolution = (int(resolution[1]), int(resolution[0]))
    for image_path in image_list:
        im = cv2.imread(image_path)
        if resolution is not None:
            im = cv2.resize(im, resolution)
        image_array.append(im[None])
    image_array = np.concatenate(image_array, 0)
    return image_array


def video_to_array_cv2(video_path: list,
                       resolution: Optional[Union[Tuple[int, int],
                                                  Tuple[float,
                                                        float]]] = None):
    pass


def resize_array(image_array: Union[np.ndarray, torch.Tensor],
                 resolution,
                 interpolation=None):
    resolution = (resolution,
                  resolution) if isinstance(resolution, int) else resolution
    if image_array.ndim == 3:
        image_array = image_array[None]
    orig_resolution = image_array.shape[1:3]
    if orig_resolution != resolution:
        if isinstance(image_array, np.ndarray):
            return_array = []

            if interpolation is None or interpolation == 'bicubic':
                interpolation = cv2.INTER_CUBIC
            elif interpolation == 'bilinear':
                interpolation = cv2.INTER_LINEAR
                cv2.INTER
            elif interpolation == 'nearest':
                interpolation = cv2.INTER_NEAREST
            for im in image_array:
                im = cv2.resize(im, (resolution[1], resolution[0]),
                                interpolation)
                return_array.append(im[None])
            return np.concatenate(return_array)
        elif isinstance(image_array, torch.Tensor):
            image_array = image_array.permute(0, 3, 1, 2)
            image_array = F.interpolate(image_array,
                                        resolution,
                                        mode='bicubic' if interpolation is None
                                        else interpolation).permute(
                                            0, 2, 3, 1)
            return image_array
    else:
        return image_array


def array_to_video_cv2(
    image_array: np.ndarray,
    output_path: str,
    fps: Union[int, float] = 30,
    resolution: Optional[Union[Tuple[int, int], Tuple[float, float]]] = None
) -> None:

    height, width = resolution if resolution is not None else image_array.shape[
        1:3]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in image_array:
        if resolution is not None:
            frame = cv2.resize(frame, (width, height))
        out.write(frame)
    out.release()


def image_tensor2array(image, color='rgb'):
    image = (image - image.min()) / (image.max() - image.min())
    image = image.detach().cpu().numpy() * 255
    image = image.astype(np.uint8)
    if color == 'rgb':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


__all__ = [
    'video_writer', 'array_to_video', 'array_to_images', 'video_to_array',
    'images_to_sorted_images', 'vid_info_reader', 'images_to_array',
    'video_to_gif', 'video_to_images', 'images_to_video', 'images_to_gif',
    'gif_to_video', 'gif_to_images', 'crop_video', 'slice_video',
    'spatial_concat_video', 'temporal_concat_video', 'compress_video',
    'pad_for_libx264'
]
