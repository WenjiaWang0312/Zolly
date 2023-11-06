import os
import torch

import numpy as np
from typing import Optional, Union
from mmhuman3d.core.conventions.keypoints_mapping import get_keypoint_num

from avatar3d.datasets.human_image_dataset import HumanImageDataset
from avatar3d.utils.bbox_utils import (bbox_xywh2xyxy, bbox_xyxy2xywh,
                                       scale_bbox, kp2d_to_bbox)
from avatar3d.utils.torch_utils import concat_dict_list
from avatar3d.cameras.builder import build_cameras
from avatar3d.models.body_models.builder import build_body_model
from avatar3d.datasets.pipelines.compose import Compose
from avatar3d.models.body_models.mappings import convert_kps
from avatar3d.utils.torch_utils import slice_dict, dict2tensor


class UVDDataset(HumanImageDataset):

    def __init__(
        self,
        data_prefix: str,
        pipeline: list,
        dataset_name: str,
        use_kp3d: bool = False,
        body_model: Optional[Union[dict, None]] = None,
        ann_file: Optional[Union[str, None]] = None,
        convention: Optional[str] = 'human_data',
        src_convention_kp2d: Optional[str] = 'smplx',
        src_convention_kp3d: Optional[str] = 'smplh',
        protocol: str = 'p1',
        inter_convention: Optional[str] = None,
        test_mode: Optional[bool] = False,
        use_joint_valid: Optional[bool] = False,
        filt_kp2d: Optional[bool] = True,
        approximate: Optional[bool] = True,
    ):
        if dataset_name is not None:
            self.dataset_name = dataset_name
        self.convention = convention
        self.use_kp3d = use_kp3d
        self.src_convention_kp2d = src_convention_kp2d
        self.src_convention_kp3d = src_convention_kp3d
        self.inter_convention = inter_convention

        self.num_keypoints = get_keypoint_num(convention)

        if body_model is not None:
            self.body_model = build_body_model(body_model)
        else:
            self.body_model = None
        self.body_model_gt = self.body_model
        self.ann_file = ann_file
        self.data_prefix = data_prefix
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        self.use_joint_valid = use_joint_valid
        self.approximate = approximate
        self.load_annotations()

        valid = self.human_data[
            'valid'] if 'valid' in self.human_data else np.ones(
                (self.human_data['keypoints2d'].shape[0]))
        kp2d_valid = self.human_data['keypoints2d'][..., 2] > 0

        if self.inter_convention is not None:
            mask = convert_kps(np.ones(
                (1, get_keypoint_num(self.src_convention_kp2d), 1)),
                               self.src_convention_kp2d,
                               self.inter_convention,
                               approximate=approximate)[0]
            valid_num = convert_kps(mask,
                                    self.inter_convention,
                                    self.convention,
                                    approximate=False)[0].sum()
        else:
            valid_num = convert_kps(
                np.ones((1, get_keypoint_num(self.src_convention_kp2d), 1)),
                self.src_convention_kp2d, self.convention)[1].sum()

        if filt_kp2d:
            kp2d_valid = np.sum(kp2d_valid, 1) >= (valid_num - 5)
        else:
            kp2d_valid = np.sum(kp2d_valid, 1) > 0

        if test_mode:
            self.valid_index = np.where(
                kp2d_valid * valid * (self.human_data['transl'][:, 2] > 0.7) *
                (self.human_data['transl'][:, 2] < 10) *
                (self.human_data['distortion_max'] > 1.40) *
                (self.human_data['distortion_max'] < 5))[0]
        else:
            self.valid_index = np.where(
                kp2d_valid * valid * (self.human_data['transl'][:, 2] > 0.7) *
                (self.human_data['transl'][:, 2] < 10) *
                (self.human_data['distortion_max'] < 5))[0]

        self.num_data = len(self.valid_index)
        self.human_data = slice_dict(self.human_data,
                                     self.valid_index.tolist(),
                                     reserved_keys=[
                                         'keypoints2d_mask',
                                         'keypoints3d_mask',
                                         'keypoints2d_convention',
                                         'keypoints3d_convention'
                                     ])

        smpl_dict = dict(body_pose=self.human_data['body_pose'],
                         betas=self.human_data['betas'],
                         global_orient=self.human_data['global_orient'],
                         transl=self.human_data['transl'])
        self.human_data['smpl'] = smpl_dict

    def __len__(self, ):
        return len(self.valid_index)

    def load_annotations(self):
        if isinstance(self.ann_file, list):
            human_data = []
            for ann_file in self.ann_file:
                npz_path = os.path.join(self.data_prefix, ann_file)
                with np.load(npz_path, allow_pickle=True) as npz_file:
                    human_data.append(dict(npz_file))
            self.human_data = concat_dict_list(human_data)
        else:
            npz_path = os.path.join(self.data_prefix, self.ann_file)
            with np.load(npz_path, allow_pickle=True) as npz_file:
                self.human_data = dict(npz_file)

        self.num_data = self.human_data['body_pose'].shape[0]
        approximate = self.approximate
        keypoints3d = self.human_data['keypoints3d']

        if self.inter_convention:
            keypoints3d, _ = convert_kps(keypoints3d,
                                         src=self.src_convention_kp3d,
                                         dst=self.inter_convention,
                                         approximate=approximate)

            keypoints3d, keypoints3d_mask = convert_kps(
                keypoints3d,
                src=self.inter_convention,
                dst=self.convention,
                approximate=False)
        else:
            keypoints3d, keypoints3d_mask = convert_kps(
                keypoints3d,
                src=self.src_convention_kp3d,
                dst=self.convention,
                approximate=approximate)

        keypoints2d = self.human_data['keypoints2d']
        if self.inter_convention:
            keypoints2d, _ = convert_kps(keypoints2d,
                                         src=self.src_convention_kp2d,
                                         dst=self.inter_convention,
                                         approximate=approximate)
            keypoints2d, keypoints2d_mask = convert_kps(
                keypoints2d,
                src=self.inter_convention,
                dst=self.convention,
                approximate=False)
        else:
            keypoints2d, keypoints2d_mask = convert_kps(
                keypoints2d,
                src=self.src_convention_kp2d,
                dst=self.convention,
                approximate=approximate)

        if not self.use_kp3d:
            keypoints3d = keypoints3d * 0
            keypoints3d_mask = keypoints3d_mask * 0
        self.human_data.__setitem__('keypoints3d', keypoints3d)
        self.human_data.__setitem__('keypoints3d_convention', self.convention)
        self.human_data.__setitem__('keypoints3d_mask', keypoints3d_mask)

        self.human_data.__setitem__('keypoints2d', keypoints2d)
        self.human_data.__setitem__('keypoints2d_convention', self.convention)
        self.human_data.__setitem__('keypoints2d_mask', keypoints2d_mask)

    @staticmethod
    def kp2d_in_bbox(kp2d, bbox_xywh):
        bbox_xyxy = bbox_xywh2xyxy(bbox_xywh)
        x = kp2d[..., 0:1]
        y = kp2d[..., 1:2]
        x1, y1, x2, y2 = np.split(bbox_xyxy, 4, -1)
        return (x1 < x) * (x < x2) * (y1 < y) * (y < y2)

    def prepare_raw_data(self, idx: int):
        """Get item from self.human_data."""
        sample_idx = idx
        info = {}
        info['img_prefix'] = None
        im_path = str(self.human_data['img_name'][idx])

        keypoints3d_mask = np.expand_dims(self.human_data['keypoints3d_mask'],
                                          -1)

        if im_path.startswith('/'):
            im_path = im_path[1:]
        info['image_path'] = os.path.join(self.data_prefix, im_path)

        uvd_path = str(self.human_data['uvd_name'][idx])
        if uvd_path.startswith('/'):
            uvd_path = uvd_path[1:]
        info['uvd_path'] = os.path.join(self.data_prefix, uvd_path)

        info['dataset_name'] = self.dataset_name
        info['sample_idx'] = sample_idx

        # if 'bbox_xyxy' in self.human_data:
        #     bbox_xyxy = self.human_data['bbox_xyxy'][idx]
        #     bbox_xyxy = scale_bbox(bbox_xyxy, scale_factor=1.2)
        #     bbox_xywh = bbox_xyxy2xywh(bbox_xyxy).reshape(-1)
        #     x, y, w, h = bbox_xywh
        #     cx = x
        #     cy = y
        #     w = h = max(w, h)
        #     info['center'] = np.array([cx, cy])
        #     info['scale'] = np.array([w, h])
        if 'keypoints2d' in self.human_data:
            kp2d = self.human_data['keypoints2d'][idx].copy()
            kp2d = kp2d[(kp2d[:, 2] > 0)]
            bbox_xyxy = kp2d_to_bbox(kp2d[None], 1.25)
            bbox_xywh = bbox_xyxy2xywh(bbox_xyxy)
            x, y, w, h = bbox_xywh.reshape(4)
            cx = x
            cy = y
            w = h = max(w, h)
            info['center'] = np.array([cx, cy])
            info['scale'] = np.array([w, h])
        else:
            info['bbox_xywh'] = np.zeros((5))
            info['center'] = np.zeros((2))
            info['scale'] = np.zeros((2))

        info['keypoints2d'] = self.human_data['keypoints2d'][idx]
        info['has_keypoints2d'] = 1

        if self.use_kp3d:
            info['has_keypoints3d'] = 1
        else:
            info['has_keypoints3d'] = 0

        info['has_smpl'] = 1
        info['smpl_body_pose'] = self.human_data['body_pose'][idx]

        info['smpl_global_orient'] = self.human_data['global_orient'][idx]

        info['smpl_origin_orient'] = info['smpl_global_orient'].copy()
        info['smpl_betas'] = self.human_data['betas'][idx]
        info['smpl_transl'] = self.human_data['transl'][idx]

        keypoints3d = self.human_data['keypoints3d'][idx]
        keypoints3d = np.concatenate([keypoints3d, keypoints3d_mask], -1)
        info['keypoints3d'] = keypoints3d

        focal_length = self.human_data['focal_length'][idx]
        focal_length = focal_length.reshape(-1)

        info['orig_focal_length'] = float(focal_length[0])
        info['distortion_max'] = self.human_data['distortion_max'][idx]

        if 'K' in self.human_data:
            info['K'] = self.human_data['K'][idx].reshape(3,
                                                          3).astype(np.float32)
            info['has_K'] = 1
        else:
            H, W = self.human_data['ori_shape'][idx]
            K = np.eye(3, 3).astype(np.float32)
            K[0, 0] = info['orig_focal_length']
            K[1, 1] = info['orig_focal_length']
            K[0, 2] = W / 2.
            K[1, 2] = H / 2.
            info['K'] = K
            info['has_K'] = 1

        info['has_uvd'] = 1
        info['has_focal_length'] = 1
        info['has_transl'] = 1

        return info

    def prepare_data(self, idx: int):
        """Generate and transform data."""
        info = self.prepare_raw_data(idx)
        return self.pipeline(info)
