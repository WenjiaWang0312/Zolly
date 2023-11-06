import mmpose
import mmcv
import os
import warnings
from tqdm import trange
import numpy as np
from mmhuman3d.core.post_processing.builder import build_post_processing
from avatar3d.utils.bbox_utils import kp2d_to_bbox, clip_bbox
from .mmdet_utils import inference_bbox as inference_bbox_mmdet
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         inference_bottom_up_pose_model)
from mmpose.datasets import DatasetInfo
from pathlib import Path

root = str(Path(mmpose.__path__[0]).parent.absolute())


def get_config_image(object_type, use_topdown=True):
    assert object_type in ['hand', 'body', 'whole_body', 'animal', 'face']
    if use_topdown:
        if object_type == 'whole_body':
            pose_config = "configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py"

            pose_checkpoint = "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth"

        elif object_type == 'body':
            pose_config = "configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py"
            pose_checkpoint = "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth"
        elif object_type == 'hand':
            pose_config = "configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/onehand10k/res50_onehand10k_256x256.py"

            pose_checkpoint = "https://download.openmmlab.com/mmpose/top_down/resnet/res50_onehand10k_256x256-e67998f6_20200813.pth"
        elif object_type == 'face':
            pose_config = "configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/aflw/hrnetv2_w18_aflw_256x256.py"
            pose_checkpoint = "https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth"

    else:
        if object_type == 'whole_body':
            pass
        elif object_type == 'body':
            pose_config = "configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hrnet_w32_coco_512x512.py"
            pose_checkpoint = "https://download.openmmlab.com/mmpose/bottom_up/hrnet_w32_coco_512x512-bcb8c247_20200816.pth"

    return pose_config, pose_checkpoint


def get_config_video(object_type, use_topdown=True):
    if use_topdown:
        if object_type == 'body':
            pose_config = "configs/body/2d_kpt_sview_rgb_vid/posewarper/posetrack18/hrnet_w48_posetrack18_384x288_posewarper_stage2.py"
            pose_checkpoint = "https://download.openmmlab.com/mmpose/top_down/posewarper/hrnet_w48_posetrack18_384x288_posewarper_stage2-4abf88db_20211130.pth"
    return pose_config, pose_checkpoint


def get_config_track():
    pass


def inference_kp2d(imgs_or_paths,
                   use_mmdet: bool = False,
                   object_type: str = 'whole_body',
                   bbox_thr=None,
                   bboxes: np.ndarray = None,
                   device: str = 'cpu',
                   use_topdown: bool = True,
                   use_smooth: bool = False):
    pose_config, pose_checkpoint = get_config_image(object_type, use_topdown)
    pose_config = os.path.join(root, pose_config)

    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(pose_config, pose_checkpoint, device=device)

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    # test a single image, the resulting box is (x1, y1, x2, y2)

    # keep the person class bounding boxes.
    if use_mmdet:
        bboxes = inference_bbox_mmdet(imgs_or_paths,
                                      device=device,
                                      batch_size=1,
                                      object_type=object_type)
        person_results = list(dict(bbox=bbox) for bbox in bboxes)
    else:
        person_results = None
        if isinstance(bboxes, np.ndarray):
            person_results = list(dict(bbox=bbox) for bbox in bboxes)
    # test a single image, with a list of bboxes.

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    kp2d = []

    for index in trange(len(imgs_or_paths)):
        images = imgs_or_paths[index]
        person_result = [person_results[index]
                         ] if person_results is not None else None

        if use_topdown:
            pose_results, _ = inference_top_down_pose_model(
                pose_model,
                imgs_or_paths=images,
                person_results=person_result,
                bbox_thr=bbox_thr,
                format='xyxy',
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=return_heatmap,
                outputs=output_layer_names)
        else:
            pose_results, _ = inference_bottom_up_pose_model(
                pose_model,
                images,
                dataset=dataset,
                dataset_info=dataset_info,
                pose_nms_thr=0.9,
                return_heatmap=return_heatmap,
                outputs=output_layer_names)
        kp2d.append(pose_results[0]['keypoints'][None])
    kp2d = np.concatenate(kp2d, 0)

    if use_smooth:
        smoother = build_post_processing(
            dict(type='savgol', window_size=11, polyorder=2))
        kp2d[..., :2] = smoother(kp2d[..., :2])
    return kp2d


def inference_kp2d_video(video,
                         use_mmdet: bool = False,
                         object_type: str = 'whole_body',
                         bbox_thr=None,
                         bboxes: np.ndarray = None,
                         device: str = 'cpu',
                         use_smooth: bool = False,
                         use_topdown=True):

    pose_config, pose_checkpoint = get_config_video(object_type)
    pose_config = os.path.join(root, pose_config)

    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(pose_config, pose_checkpoint, device=device)

    from mmpose.apis.inference import collect_multi_frames
    indices = pose_model.cfg.data.test.data_cfg['frame_indices_test']

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    # test a single image, the resulting box is (x1, y1, x2, y2)

    # keep the person class bounding boxes.
    if use_mmdet:
        bboxes = inference_bbox_mmdet(video,
                                      device=device,
                                      batch_size=1,
                                      object_type=object_type)
        person_results = list(dict(bbox=bbox) for bbox in bboxes)
    else:
        person_results = None
        if isinstance(bboxes, np.ndarray):
            person_results = list(dict(bbox=bbox) for bbox in bboxes)
    # test a single image, with a list of bboxes.

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    kp2d = []

    for frame_id in trange(len(video)):
        frames = collect_multi_frames(video, frame_id, indices)
        person_result = [person_results[frame_id]
                         ] if person_results is not None else None

        pose_results, _ = inference_top_down_pose_model(
            pose_model,
            imgs_or_paths=frames,
            person_results=person_result,
            bbox_thr=bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)
        kp2d.append(pose_results[0]['keypoints'][None])
    kp2d = np.concatenate(kp2d, 0)

    if use_smooth:
        smoother = build_post_processing(
            dict(type='savgol', window_size=11, polyorder=2))
        kp2d[..., :2] = smoother(kp2d[..., :2])
    return kp2d


def inference_bbox_mmpose(imgs_or_paths,
                          bbox_thr=None,
                          object_type='whole_body',
                          scale_factor=1.2,
                          device='cpu'):
    kp2d = inference_kp2d(imgs_or_paths=imgs_or_paths,
                          use_mmdet=False,
                          object_type=object_type,
                          bbox_thr=bbox_thr,
                          device=device)
    bboxes = kp2d_to_bbox(kp2d, scale_factor=scale_factor, bbox_format='xyxy')

    resolution = imgs_or_paths[0].shape[:2]
    bboxes = clip_bbox(bboxes, resolution)
    return bboxes


def inference_kp2d_tracking():
    pass
