import shutil
import torch
import numpy as np
import h5py
import cv2
import os
import pickle
import glob
from typing import Tuple
from torch.utils.data import Dataset
from mmhuman3d.utils.ffmpeg_utils import vid_info_reader, video_to_images
from avatar3d.cameras import build_cameras
from avatar3d.models.body_models import get_model_class
from avatar3d.models.body_models.mappings import convert_kps, format_coco_kps
from avatar3d.utils.torch_utils import build_parameters, dict2tensor
from avatar3d.transforms.transform3d import ee_to_rotmat
from avatar3d.utils.frame_utils import images_to_array_cv2, video_to_array

osj = os.path.join


class ImageDataset(Dataset):

    def __init__(
        self,
        dataset_name: str = 'people_snapshot',
        convention: str = 'coco',
        model_type: str = 'smpl',
        root: str = '',
        npz_file: str = '',
        param_path: str = None,
        mask_path: str = None,
        read_cache: bool = True,
        resolution: Tuple[int] = None,
        parameter_config=dict()) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.convention = convention
        self.root = root
        self.parameter_config = parameter_config
        self.parameters = dict()
        self.model_type = model_type
        self.body_model_cls = get_model_class(model_type)
        self._cache_folder = None
        self.read_cache = read_cache

        self.calibrated = False
        if dataset_name == 'people_snapshot':
            self.load_people_snapshot(root, read_cache, resolution)
        elif dataset_name == 'hf_avatar':
            self.load_hf_avatar()
        elif dataset_name == 'self_recon':
            self.load_selfrecon()
        elif dataset_name == 'zju_mocap':
            self.load_zju_mocap()
        elif dataset_name == 'single':
            self.load_single_image(img_path=root,
                                   npz_file=npz_file,
                                   param_path=param_path,
                                   mask_path=mask_path)
        self.parameters.update(build_parameters(**parameter_config))

        self.valid_keys = None

    def clone(self):
        argcount = self.__init__.__code__.co_argcount
        varnames = self.__init__.__code__.co_varnames[1:argcount]
        kwargs = dict()
        for name in varnames:
            kwargs[name] = getattr(self, name)
        return self.__class__.__init__(**kwargs)

    def set_valid_keys(self, keys):
        self.valid_keys = keys

    def _load(self, image_list, mask_list, seg_mask_list, parameter_config,
              npz_file):
        self.image_list = image_list
        self.mask_list = mask_list
        self.seg_mask_list = seg_mask_list
        if self.read_cache:
            if image_list:
                self.images = torch.from_numpy(
                    images_to_array_cv2(self.image_list)).float()
            else:
                self.images = None
            if mask_list:
                self.masks = torch.from_numpy(
                    images_to_array_cv2(self.mask_list)).float() / 255.
            else:
                self.masks = None
            if seg_mask_list:
                self.seg_masks = torch.from_numpy(
                    images_to_array_cv2(self.seg_mask_list)).int()
            else:
                self.seg_masks = None

        self.kp2d = None
        self.kp3d = None
        if npz_file:
            parameters = dict(np.load(npz_file))
            for k in parameters:
                if k in self.body_model_cls.full_pose_keys.union(
                    {'transl', 'cam', 'K', 'R', 'T'}):
                    self.parameters[k] = torch.from_numpy(parameters[k])
                elif k == 'keypoints2d':
                    self.kp2d = parameters['keypoints2d']
                elif k == 'keypoints3d':
                    self.kp3d = parameters['keypoints3d']
        else:
            for k in parameter_config:
                self.parameters[k] = torch.zeros(parameter_config[k])

    def load_hf_avatar():
        pass

    def load_self_recon():
        pass

    def load_zju_mocap():
        pass

    def load_single_image(self,
                          img_path,
                          npz_file,
                          param_path=None,
                          mask_path=None):
        if img_path.endswith('mp4'):
            self.images = video_to_array(img_path)
            self.len = len(self.images)

        else:
            self.len = 1
            self.images = cv2.imread(img_path)[None]

        self.kp2d = dict(np.load(npz_file, allow_pickle=True))['kp2d'].reshape(
            -1, 133, 3)
        self.kp2d = format_coco_kps(self.kp2d, convention='coco_wholebody')
        if param_path is not None:
            if param_path.endswith('pkl'):
                with open(param_path, 'rb') as f:
                    params = pickle.load(f)
            elif param_path.endswith('npz'):
                params = dict(np.load(param_path, allow_pickle=True))['kp2d']

            self.parameters = dict2tensor(params)
            for k, v in self.parameters.items():
                self.parameters[k] = v.float()
        else:
            self.parameters = dict()
        if mask_path is not None:
            mask = cv2.imread(mask_path)[..., 0]
            mask = np.where(mask > 200, 1, 0)
            self.masks = torch.Tensor(mask[None])
        else:
            self.masks = None
        self.kp3d = None

        self.calibrated = False
        self.convention = 'coco_wholebody'
        self.cameras = None
        self.resolution = self.images.shape[1:3]
        self.images = torch.Tensor(self.images)

    def convert_convention(self, dst):

        if self.kp2d is not None:
            self.kp2d = convert_kps(self.kp2d,
                                    src=self.convention,
                                    dst=dst,
                                    approximate=True)[0]
        if self.kp3d is not None:
            self.kp3d = convert_kps(self.kp3d,
                                    src=self.convention,
                                    dst=dst,
                                    approximate=True)[0]
        self.convention = dst

    def load_people_snapshot(self, root, read_cache, resolution):
        if resolution is not None:
            if isinstance(resolution, int):
                resolution = (resolution, resolution)
        self.calibrated = True
        self.convention = 'people_snapshot'
        self.masks = h5py.File(f'{root}/masks.hdf5')['masks']
        if read_cache:
            self.images = video_to_array(glob.glob(f'{root}/*mp4')[0],
                                         resolution=resolution)
            self.resolution = self.images.shape[1:3]
        else:
            video_path = glob.glob(f'{root}/*mp4')[0]
            self._cache_folder = f'{video_path}_temp'
            video_to_images(video_path,
                            self._cache_folder,
                            resolution=resolution)
            self.image_list = [
                osj(self._cache_folder, name)
                for name in os.listdir(self._cache_folder)
            ]
            self.resolution = (vid_info_reader(self.image_list[0])['height'],
                               vid_info_reader(self.image_list[0])['width'])
        self.kp2d = h5py.File(f'{root}/keypoints.hdf5')['keypoints']
        self.kp2d = np.array(self.kp2d).reshape(-1, 18, 3)
        # if os.path.exists(f'{root}/kp2d.npz'):
        #     kp2d_wholebody = dict(np.load(f'{root}/kp2d.npz'))['kp2d']
        #     kp2d, mask = convert_kps(self.kp2d,
        #                              src='people_snapshot',
        #                              dst='coco_wholebody')
        #     self.kp2d = kp2d_wholebody * (1 - mask).reshape(
        #         1, 133, 1) + kp2d * mask.reshape(1, 133, 1)
        #     self.convention = 'coco_wholebody'
        self.kp3d = None
        poses = h5py.File(f'{root}/reconstructed_poses.hdf5')
        with open(f'{root}/consensus.pkl', 'rb') as f:
            consensus = pickle.load(f, encoding='latin1')
        pose = np.array(poses['pose'])
        transl = np.array(poses['trans'])

        betas = np.array(poses['betas'])[None]
        N = min(len(self.masks), len(self.images), len(self.kp2d), len(pose))
        self.len = N
        v_personal = np.array(consensus['v_personal'])
        body_pose_dim = self.body_model_cls.full_param_dims.get(
            'body_pose', 72)

        parameters = dict(betas=betas,
                          global_orient=pose[:N, :3],
                          body_pose=pose[:N, 3:3 + body_pose_dim],
                          transl=transl[:N],
                          verts_canonical=v_personal)
        parameters = dict2tensor(parameters)

        self.parameters = parameters

        with open(f'{root}/camera.pkl', 'rb') as f:
            cam_data = pickle.load(f, encoding='latin1')
        principal_point = cam_data['camera_c'][None]
        focal_length = cam_data['camera_f'][None]
        T = cam_data['camera_t'][None]
        R = ee_to_rotmat(cam_data['camera_rt'])[None]
        H = cam_data['height']
        W = cam_data['width']
        self.T = T
        self.R = R

        cameras = dict(type='perspective',
                       resolution=(H, W),
                       R=R,
                       T=T,
                       convention='opencv',
                       in_ndc=False,
                       focal_length=focal_length,
                       principal_point=principal_point)
        self.cameras = build_cameras(cameras)
        self.cameras.update_resolution_(self.resolution)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        data = dict()

        if self.read_cache:
            if (self.valid_keys is not None
                    and 'image' in self.valid_keys) or self.valid_keys is None:
                data['image'] = self.images[index]
            if (self.valid_keys is not None
                    and 'mask' in self.valid_keys) or self.valid_keys is None:
                data['mask'] = self.masks[index]

        else:
            if self.image_list:
                if (self.valid_keys is not None and 'image'
                        in self.valid_keys) or self.valid_keys is None:
                    data['image'] = torch.from_numpy(
                        images_to_array_cv2(self.image_list[index])).float()
            if self.mask_list:
                if (self.valid_keys is not None and 'mask'
                        in self.valid_keys) or self.valid_keys is None:
                    data['mask'] = torch.from_numpy(
                        images_to_array_cv2(
                            self.mask_list[index])).float() / 1.0

        if self.kp2d is not None:
            if (self.valid_keys is not None
                    and 'kp2d' in self.valid_keys) or self.valid_keys is None:
                data['kp2d'] = torch.from_numpy(self.kp2d[index])

        if self.kp3d is not None:
            if (self.valid_keys is not None
                    and 'kp3d' in self.valid_keys) or self.valid_keys is None:
                data['kp3d'] = torch.from_numpy(self.kp3d[index])

        data['frame_id'] = index

        return data

    def __del__(self):
        if self._cache_folder:
            shutil.rmtree(self._cache_folder)
