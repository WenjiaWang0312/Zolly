import os
import pickle
from typing import Optional, Union
import numpy as np
import torch.distributed as dist
from mmcv.runner import get_dist_info
from avatar3d.datasets.human_image_dataset import HumanImageDataset
from mmhuman3d.data.data_structures.human_data import HumanData
from avatar3d.models.body_models.mappings import convert_kps
from mmhuman3d.data.data_structures.human_data_cache import (
    HumanDataCacheReader,
    HumanDataCacheWriter,
)


class HumanImageSMPLXDataset(HumanImageDataset):

    # metric
    ALLOWED_METRICS = {
        'mpjpe', 'pa-mpjpe', 'pve', '3dpck', 'pa-3dpck', '3dauc', 'pa-3dauc',
        '3DRMSE', 'pa-pve'
    }

    def __init__(
            self,
            data_prefix: str,
            pipeline: list,
            dataset_name: str,
            body_model: Optional[Union[dict, None]] = None,
            ann_file: Optional[Union[str, None]] = None,
            convention: Optional[str] = 'humman_data',
            cache_data_path: Optional[Union[str, None]] = None,
            test_mode: Optional[bool] = False,
            num_betas: Optional[int] = 10,
            num_expression: Optional[int] = 10,
            face_vertex_ids_path: Optional[str] = None,
            hand_vertex_ids_path: Optional[str] = None,
            is_distorted: Optional[bool] = False,  # new feature
            down_scale: int = 1,  # new feature
            num_data: Optional[int] = None,  # new feature
    ):
        super().__init__(data_prefix, pipeline, dataset_name, body_model,
                         ann_file, convention, cache_data_path, test_mode)
        self.num_betas = num_betas
        self.num_expression = num_expression
        if face_vertex_ids_path is not None:
            if os.path.exists(face_vertex_ids_path):
                self.face_vertex_ids = np.load(face_vertex_ids_path).astype(
                    np.int32)
        if hand_vertex_ids_path is not None:
            if os.path.exists(hand_vertex_ids_path):
                with open(hand_vertex_ids_path, 'rb') as f:
                    vertex_idxs_data = pickle.load(f, encoding='latin1')
                self.left_hand_vertex_ids = vertex_idxs_data['left_hand']
                self.right_hand_vertex_ids = vertex_idxs_data['right_hand']
        if num_data is not None:
            self.num_data = num_data
        self.is_distorted = is_distorted
        self.down_scale = down_scale

    def load_annotations(self):
        """Load annotation from the annotation file.

        Here we simply use :obj:`HumanData` to parse the annotation.
        """
        rank, world_size = get_dist_info()
        self.get_annotation_file()
        if self.cache_data_path is None:
            use_human_data = True
        elif rank == 0 and not os.path.exists(self.cache_data_path):
            use_human_data = True
        else:
            use_human_data = False
        if use_human_data:
            self.human_data = HumanData.fromfile(self.ann_file)

            if self.human_data.check_keypoints_compressed():
                self.human_data.decompress_keypoints()
            # change keypoint from 'human_data' to the given convention
            if 'keypoints3d_smplx' in self.human_data:
                keypoints3d = self.human_data['keypoints3d_smplx']
                assert 'keypoints3d_smplx_mask' in self.human_data
                keypoints3d_mask = self.human_data['keypoints3d_smplx_mask']
                keypoints3d, keypoints3d_mask = \
                    convert_kps(
                        keypoints3d,
                        src='human_data',
                        dst=self.convention,
                        mask=keypoints3d_mask)
                self.human_data.__setitem__('keypoints3d_smplx', keypoints3d)
                self.human_data.__setitem__('keypoints3d_convention',
                                            self.convention)
                self.human_data.__setitem__('keypoints3d_smplx_mask',
                                            keypoints3d_mask)
            if 'keypoints2d_smplx' in self.human_data:
                keypoints2d = self.human_data['keypoints2d_smplx']
                assert 'keypoints2d_smplx_mask' in self.human_data
                keypoints2d_mask = self.human_data['keypoints2d_smplx_mask']
                keypoints2d, keypoints2d_mask = \
                    convert_kps(
                        keypoints2d,
                        src='human_data',
                        dst=self.convention,
                        mask=keypoints2d_mask)
                self.human_data.__setitem__('keypoints2d_smplx', keypoints2d)
                self.human_data.__setitem__('keypoints2d_convention',
                                            self.convention)
                self.human_data.__setitem__('keypoints2d_smplx_mask',
                                            keypoints2d_mask)
            self.human_data.compress_keypoints_by_mask()

        if self.cache_data_path is not None:
            if rank == 0 and not os.path.exists(self.cache_data_path):
                writer_kwargs, sliced_data = self.human_data.get_sliced_cache()
                writer = HumanDataCacheWriter(**writer_kwargs)
                writer.update_sliced_dict(sliced_data)
                writer.dump(self.cache_data_path)
            if world_size > 1:
                dist.barrier()
            self.cache_reader = HumanDataCacheReader(
                npz_path=self.cache_data_path)
            self.num_data = self.cache_reader.data_len
            self.human_data = None
        else:
            self.cache_reader = None
            self.num_data = self.human_data.data_len

    def prepare_raw_data(self, idx: int):
        """Get item from self.human_data."""
        idx = idx * self.down_scale
        sample_idx = idx
        if self.cache_reader is not None:
            self.human_data = self.cache_reader.get_item(idx)
            idx = idx % self.cache_reader.slice_size
        info = {}
        info['img_prefix'] = None
        image_path = self.human_data['image_path'][idx]

        if self.dataset_name == 'h36m':
            image_path = image_path.replace('/images/', '/')
        elif self.dataset_name == 'bedlam':
            image_path = image_path.replace('png', 'jpg')
            if '_rot_' in image_path:
                image_path = image_path.replace('_rot_', '_')

        info['image_path'] = os.path.join(self.data_prefix, 'datasets',
                                          self.dataset_name, image_path)

        if image_path.endswith('smc'):
            device, device_id, frame_id = self.human_data['image_id'][idx]
            info['image_id'] = (device, int(device_id), int(frame_id))

        info['dataset_name'] = self.dataset_name
        info['sample_idx'] = sample_idx

        if 'bbox_xywh' in self.human_data:
            info['bbox_xywh'] = self.human_data['bbox_xywh'][idx]
            x, y, w, h, s = info['bbox_xywh']
            cx = x + w / 2
            cy = y + h / 2
            w = h = max(w, h)
            info['center'] = np.array([cx, cy])
            info['scale'] = np.array([w, h])
        else:
            info['bbox_xywh'] = np.zeros((5))
            info['center'] = np.zeros((2))
            info['scale'] = np.zeros((2))

        for part_name in ['face', 'lhand', 'rhand']:
            if f'{part_name}_bbox_xywh' in self.human_data:
                info[f'{part_name}_bbox_xywh'] = self.human_data[f'{part_name}_bbox_xywh'][idx]
                x, y, w, h, s = info[f'{part_name}_bbox_xywh']
                cx = x + w / 2
                cy = y + h / 2
                w = h = max(w, h)
                info[f'{part_name}_center'] = np.array([cx, cy])
                info[f'{part_name}_scale'] = np.array([w, h]) * 1.3
            else:
                info[f'{part_name}_bbox_xywh'] = np.zeros((5))
                info[f'{part_name}_center'] = np.zeros((2))
                info[f'{part_name}_scale'] = np.zeros((2))


        # in later modules, we will check validity of each keypoint by
        # its confidence. Therefore, we do not need the mask of keypoints.

        if 'keypoints2d_smplx' in self.human_data:
            info['keypoints2d'] = self.human_data['keypoints2d_smplx'][idx]
            info['has_keypoints2d'] = 1
        else:
            info['keypoints2d'] = np.zeros((self.num_keypoints, 3))
            info['has_keypoints2d'] = 0
        if 'keypoints3d_smplx' in self.human_data:
            info['keypoints3d'] = self.human_data['keypoints3d_smplx'][idx]
            info['has_keypoints3d'] = 1
        else:
            info['keypoints3d'] = np.zeros((self.num_keypoints, 4))
            info['has_keypoints3d'] = 0

        if 'smplx' in self.human_data:
            smplx_dict = self.human_data['smplx']
            info['has_smplx'] = 1
        else:
            smplx_dict = {}
            info['has_smplx'] = 0
        if 'global_orient' in smplx_dict:
            info['smplx_global_orient'] = smplx_dict['global_orient'][idx].reshape(3)
            info['has_smplx_global_orient'] = 1
        else:
            info['smplx_global_orient'] = np.zeros((3), dtype=np.float32)
            info['has_smplx_global_orient'] = 0

        if 'body_pose' in smplx_dict:
            info['smplx_body_pose'] = smplx_dict['body_pose'][idx].reshape(21, 3)
            info['has_smplx_body_pose'] = 1
        else:
            info['smplx_body_pose'] = np.zeros((21, 3), dtype=np.float32)
            info['has_smplx_body_pose'] = 0

        if 'right_hand_pose' in smplx_dict:
            info['smplx_right_hand_pose'] = smplx_dict['right_hand_pose'][idx].reshape(15, 3)
            info['has_smplx_right_hand_pose'] = 1
        else:
            info['smplx_right_hand_pose'] = np.zeros((15, 3), dtype=np.float32)
            info['has_smplx_right_hand_pose'] = 0

        if 'left_hand_pose' in smplx_dict:
            info['smplx_left_hand_pose'] = smplx_dict['left_hand_pose'][idx].reshape(15, 3)
            info['has_smplx_left_hand_pose'] = 1
        else:
            info['smplx_left_hand_pose'] = np.zeros((15, 3), dtype=np.float32)
            info['has_smplx_left_hand_pose'] = 0

        if 'jaw_pose' in smplx_dict:
            info['smplx_jaw_pose'] = smplx_dict['jaw_pose'][idx].reshape(3)
            info['has_smplx_jaw_pose'] = 1
        else:
            info['smplx_jaw_pose'] = np.zeros((3), dtype=np.float32)
            info['has_smplx_jaw_pose'] = 0

        if 'betas' in smplx_dict:
            info['smplx_betas'] = smplx_dict['betas'][idx][:self.num_betas]
            info['has_smplx_betas'] = 1
        else:
            info['smplx_betas'] = np.zeros((self.num_betas), dtype=np.float32)
            info['has_smplx_betas'] = 0

        if 'expression' in smplx_dict:
            info['smplx_expression'] = smplx_dict['expression'][idx]
            info['has_smplx_expression'] = 1
        else:
            info['smplx_expression'] = np.zeros((self.num_expression),
                                                dtype=np.float32)
            info['has_smplx_expression'] = 0

        if 'transl' in smplx_dict:
            info['smplx_transl'] = smplx_dict['transl'][idx].astype(np.float32).reshape(3)
            info['has_transl'] = 1
        else:
            info['smplx_transl'] = np.zeros((3)).astype(np.float32)
            info['has_transl'] = 0

        if 'K' in self.human_data:
            info['K'] = self.human_data['K'][idx].reshape(3,
                                                          3).astype(np.float32)
            info['has_K'] = 1
        else:
            info['K'] = np.eye(3, 3).astype(np.float32)
            info['has_K'] = 0

        if 'focal_length' in self.human_data['meta']:
            info['ori_focal_length'] = float(
                self.human_data['meta']['focal_length'][idx].reshape(-1)[0])
            info['has_focal_length'] = 1
        else:
            info['ori_focal_length'] = 5000.
            info['has_focal_length'] = 0

        if self.is_distorted:
            info['distortion_max'] = float(
                self.human_data['distortion_max'][idx])
            info['is_distorted'] = 1
        else:
            info['distortion_max'] = 1.0
            info['is_distorted'] = 0

        return info


# if __name__ == '__main__':
#     import numpy as np
#     import cv2
#     from tqdm import tqdm
#     count = 0
#     rotated_images = []
#     shapes = []
#     d = np.load('/data1/wenjiawang/datasets/mmhuman_data/preprocessed_datasets/bedlam_train_smplx.npz')
#     for im_path in tqdm(d['image_path'][::100]):
#         im_path = im_path.replace('png', 'jpg')
#         im_path = f'/data1/wenjiawang/datasets/mmhuman_data/datasets/bedlam/{im_path}'
#         if  '_rot_' in im_path:
#             im_path = im_path.replace('_rot_', '_')
#             im = cv2.imread(im_path)
#             shapes.append(im.shape[:2])
#             # im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
#             # if not im.shape[:2] != (1280, 720):
#             #     print(im_path)
#             # cv2.imwrite(im_path, im)
#             # count += 1
#             rotated_images.append(im_path)
