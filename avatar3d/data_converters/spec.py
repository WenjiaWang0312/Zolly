import os
from typing import List, Tuple
import torch
import numpy as np
from tqdm import tqdm
import math
from tqdm import trange
import cv2

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.data.data_converters.base_converter import BaseModeConverter
from mmhuman3d.data.data_converters.builder import DATA_CONVERTERS

from avatar3d.models.body_models.builder import build_body_model
from avatar3d.cameras.utils import estimate_cam_weakperspective_batch, pred_cam_to_full_transl
from avatar3d.utils.torch_utils import dict2tensor, move_dict_to_device, dict2numpy
from avatar3d.cameras.builder import build_cameras
from avatar3d.transforms.transform3d import ee_to_rotmat, aa_to_rotmat, rotmat_to_aa
from avatar3d.models.body_models.mappings import convert_kps


@DATA_CONVERTERS.register_module()
class SpecConverter(BaseModeConverter):
    """AGORA dataset
    `AGORA: Avatars in Geography Optimized for Regression Analysis' CVPR`2021
    More details can be found in the `paper
    <https://arxiv.org/pdf/2104.14643.pdf>`__.

    Args:
        modes (list): 'validation' or 'train' for accepted modes
        fit (str): 'smpl' or 'smplx for available body model fits
        res (tuple): (1280, 720) or (3840, 2160) for available image resolution
    """
    ACCEPTED_MODES = ['test', 'train']
    def __init__(self, modes: List = [], fit: str = 'smpl', body_model:dict=None,
                 res: Tuple[int, int] = (1080, 1920)) -> None:  # yapf: disable
        super(SpecConverter, self).__init__(modes)
        accepted_fits = ['smpl']

        if fit not in accepted_fits:
            raise ValueError('Input fit not in accepted fits. \
                Use either smpl or smplx')
        self.fit = fit

        accepted_res = [(1080, 1920)]
        if res not in accepted_res:
            raise ValueError('Input resolution not in accepted resolution. \
                Use either (1080, 1920)')
        self.res = res
        self.body_model = body_model

    def convert_by_mode(self, dataset_path: str, out_path: str,
                        mode: str) -> dict:
        """
        Args:
            dataset_path (str): Path to directory where raw images and
            annotations are stored.
            out_path (str): Path to directory to save preprocessed npz file
            mode (str): Mode in accepted modes

        Returns:
            dict:
                A dict containing keys image_path, bbox_xywh, keypoints2d,
                keypoints3d, keypoints2d_mask, keypoints3d_mask, meta
                stored in HumanData() format
        """
        if mode == 'train':
            npz_path = os.path.join(dataset_path, 'annotations',
                                    'train_raw.npz')
            self.data = np.load(npz_path, allow_pickle=True)
        elif mode == 'test':
            npz_path = os.path.join(dataset_path, 'annotations',
                                    'test_raw.npz')
            self.data = np.load(npz_path, allow_pickle=True)
        else:
            raise NotImplementedError

        device = torch.device('cuda')

        smpl = build_body_model(self.body_model).to(device)

        targets = dict2tensor(dict(self.data))

        move_dict_to_device(targets, device)

        maxmum = targets['imgname'].shape[0]
        batch_size = 2000

        error_all = []
        transl_all = []
        pose_all = []
        kp3d_all = []
        for i in trange(math.ceil(maxmum / batch_size)):
            start = batch_size * i
            index = list(range(start, min(maxmum, start + batch_size)))

            kp2d_smpl_24 = targets['part'][index]

            R1 = targets['cam_rotmat'][index]
            f1 = targets['focal_length'][index, 0]

            H, W = 1080, 1920

            gt_pose = targets['pose'][index]
            gt_betas = targets['shape'][index]
            gt_global_orient = gt_pose[:, :3]
            gt_body_pose = gt_pose[:, 3:]

            gt_global_orient1 = rotmat_to_aa(
                torch.bmm(R1, aa_to_rotmat(gt_global_orient)))

            gt_output = smpl(
                betas=gt_betas,
                body_pose=gt_body_pose.float(),
                global_orient=gt_global_orient1,
            )
            gt_model_joints = gt_output['joints']

            gt_cam = estimate_cam_weakperspective_batch(
                gt_model_joints, kp2d_smpl_24, None, None, W)

            transl = pred_cam_to_full_transl(
                gt_cam,
                torch.Tensor([[W / 2, W / 2]]).to(device), W,
                torch.Tensor([[H, W]]).to(device), f1).float()

            device = torch.device('cuda')
            curr_batch_size = len(index)
            K = torch.eye(3, 3)[None].to(device).repeat_interleave(
                curr_batch_size, 0)
            K[:, 0, 0] = 2 * f1 / H
            K[:, 1, 1] = 2 * f1 / H

            cameras = build_cameras(
                dict(type='perspective',
                     in_ndc=True,
                     K=K,
                     resolution=(H, W),
                     convention='opencv')).to(device)

            def distance2d(kp2d1, kp2d2, h=1080, w=1920):
                x1 = kp2d1[..., 0] / w
                y1 = kp2d1[..., 1] / h
                x2 = kp2d2[..., 0] / w
                y2 = kp2d2[..., 1] / h
                return ((x1 - x2)**2 + (y1 - y2)**2)

            transl = transl.view(curr_batch_size, 1, 3)
            joints3d = gt_model_joints + transl
            projected_j3d = cameras.transform_points_screen(joints3d)

            optimizer = torch.optim.Adam([transl, gt_global_orient1],
                                         lr=0.5,
                                         betas=(0.9, 0.999))

            kp2d_, conf2 = convert_kps(kp2d_smpl_24,
                                       'smpl_24',
                                       'smpl_24',
                                       approximate=False)

            x = kp2d_smpl_24[..., 0]
            y = kp2d_smpl_24[..., 1]
            conf_kp2d = (W > x) * (x > 0) * (H > y) * (y > 0)
            conf_kp2d = conf_kp2d * 1.

            projected_j3d_, conf1 = convert_kps(projected_j3d, 'smpl_54',
                                                'smpl_24')
            conf = conf1 * conf2

            kp3d, kp3d_conf = convert_kps(gt_model_joints, 'smpl_54',
                                          'smpl_24')
            kp3d = torch.cat(
                [kp3d,
                 kp3d_conf.view(1, -1, 1).repeat(curr_batch_size, 1, 1)], -1)
            stage_dict = [
                dict(transl=True, global_orient=True, iter=300),
            ]

            for stage in stage_dict:
                iter_num = stage.get('iter', 200)
                for _ in range(iter_num):
                    if stage.get('transl', False):
                        transl.requires_grad = True
                    else:
                        transl.requires_grad = False
                    if stage.get('global_orient', False):
                        gt_global_orient1.requires_grad = True
                    else:
                        gt_global_orient1.requires_grad = False

                    gt_output = smpl(
                        betas=gt_betas,
                        body_pose=gt_body_pose.float(),
                        global_orient=gt_global_orient1,
                    )
                    gt_model_joints = gt_output['joints']

                    joints3d = gt_model_joints + transl
                    projected_j3d = cameras.transform_points_screen(joints3d)
                    projected_j3d_, _ = convert_kps(projected_j3d, 'smpl_54',
                                                    'smpl_24')

                    optimizer.zero_grad()

                    error = distance2d(projected_j3d_[:, conf.bool()],
                                       kp2d_[:, conf.bool()])
                    error = (error * conf_kp2d).mean(1)
                    loss = error.mean()

                    loss.backward()

                    optimizer.step()

            pose = torch.cat([gt_global_orient1, gt_body_pose], -1)
            pose_all.append(pose)
            transl_all.append(transl)
            error_all.append(error)
            kp3d_all.append(kp3d)
            print(loss)
        pose_all = torch.cat(pose_all, 0)
        transl_all = torch.cat(transl_all, 0)
        error_all = torch.cat(error_all, 0)
        kp3d_all = torch.cat(kp3d_all, 0)

        targets['transl'] = transl_all.view(-1, 3)
        targets['error'] = error_all
        targets['pose'] = pose_all
        targets['keypoints3d'] = kp3d_all
        targets['keypoints2d'] = targets['part']

        move_dict_to_device(targets, torch.device('cpu'))
        self.data = dict2numpy(targets)
        # use HumanData to store all data
        human_data = HumanData()

        # structs we use
        image_path_, bbox_xywh_, keypoints2d_, keypoints3d_ = [], [], [], []
        focal_length = []

        # get a list of .pkl files in the directory
        img_path = os.path.join(dataset_path, 'images', mode)

        body_model = {}
        body_model['body_pose'] = []
        body_model['global_orient'] = []
        body_model['betas'] = []
        body_model['transl'] = []

        self.imgname = self.data['imgname']
        # Bounding boxes are assumed to be in the center and scale format
        self.scale = self.data['scale'] * 200  # box height
        self.center = self.data['center']
        self.transl = self.data['transl']

        # Get gt SMPL parameters, if available
        try:
            if 'pose_0yaw_inverseyz' in self.data:
                self.pose = self.data['pose_0yaw_inverseyz'].astype(np.float64)
            else:
                self.pose = self.data['pose'].astype(np.float64)

            self.betas = self.data['shape'].astype(np.float64)
            if 'has_smpl' in self.data:
                self.has_smpl = self.data['has_smpl']
            else:
                self.has_smpl = np.ones(len(self.imgname))
        except KeyError:
            self.has_smpl = np.zeros(len(self.imgname))

        # Get gt 3D pose, if available
        try:
            self.pose_3d = self.data['keypoints3d']  # (71982, 24, 4)
            self.has_pose_3d = 1
        except KeyError:
            self.has_pose_3d = 0

        if 'focal_length' in self.data:
            self.focal_length = self.data['focal_length']  # (71982, 2)

        # Get 2D keypoints
        try:
            keypoints_gt = self.data['keypoints2d']  # (71982, 24, 3)

        except KeyError:
            keypoints_gt = np.zeros((len(self.imgname), 24, 3))
        # try:
        #     keypoints_openpose = self.data['openpose'] # (71982, 25, 3)
        # except KeyError:
        #     keypoints_openpose = np.zeros((len(self.imgname), 25, 3))
        # self.keypoints = np.concatenate([keypoints_openpose, keypoints_gt], axis=1)
        self.keypoints = keypoints_gt

        # Get gender data, if available
        # try:
        #     gender = self.data['gender']
        #     self.gender = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)
        # except KeyError:
        self.gender = -1 * np.ones(len(self.imgname)).astype(np.int32)

        # evaluation variables
        self.length = self.scale.shape[0]

        meta = {}
        # meta['gender'] = []
        # meta['age'] = []
        # meta['ethnicity'] = []
        # meta['kid'] = []
        # meta['occlusion'] = []

        # import pdb
        # pdb.set_trace()
        keypoints3d_convention = 'smpl_24'
        keypoints2d_convention = 'smpl_24'
        # keypoints3d_convention = 'SMPL_49_KEYPOINTS' # # 25 OpenPose + 24 SMPL, used by SPIN, EFT and so on
        # keypoints2d_convention = 'SMPL_KEYPOINTS' # 24
        num_body_pose = 23
        num_keypoints2d = 24
        num_keypoints3d = 24

        for i in tqdm(range(self.length)):
            # obtain keypoints
            keypoints2d = self.keypoints[i]

            # if self.res == (1280, 720):
            #     keypoints2d *= (720 / 2160)
            # import pdb
            # pdb.set_trace()
            keypoints3d = self.pose_3d[i]
            keypoints3d -= keypoints3d[0]  # root-centered

            # obtain smpl data
            pose = self.pose[i]
            body_model['body_pose'].append(pose[3:])
            global_orient = pose[:3]
            body_model['betas'].append(self.betas[i])

            body_model['transl'].append(self.transl[i])

            # global_orient = self.get_global_orient(
            #     img_path, df, idx, pidx, global_orient.reshape(-1))

            # add confidence column
            # keypoints2d = np.hstack(
            #     [keypoints2d, np.ones((num_keypoints, 1))])
            # keypoints3d = np.hstack(
            #     [keypoints3d, np.ones((num_keypoints, 1))])

            # bbox_xyxy = [
            #     min(keypoints2d[:, 0]),
            #     min(keypoints2d[:, 1]),
            #     max(keypoints2d[:, 0]),
            #     max(keypoints2d[:, 1])
            # ]
            bbox_xyxy = [
                self.center[i][0] - self.scale[i] / 2,
                self.center[i][1] - self.scale[i] / 2,
                self.center[i][0] + self.scale[i] / 2,
                self.center[i][1] + self.scale[i] / 2
            ]
            bbox_xyxy = self._bbox_expand(bbox_xyxy, scale_factor=1.1)
            bbox_xywh = self._xyxy2xywh(bbox_xyxy)

            keypoints2d_.append(keypoints2d)
            keypoints3d_.append(keypoints3d)
            bbox_xywh_.append(bbox_xywh)

            img_path = self.imgname[i]
            img_path = img_path.replace('.png', '.jpg')
            # import pdb
            # pdb.set_trace()
            image_path_.append(img_path)

            focal_length.append(self.focal_length[i])

            body_model['global_orient'].append(global_orient)

            # camera = CameraParameter(H=h, W=w)
            # camera.set_KRT(K, R, T)
            # parameter_dict = camera.to_dict()
            # cam_param_.append(parameter_dict)

        body_model['body_pose'] = np.array(body_model['body_pose']).reshape(
            (-1, num_body_pose, 3))
        body_model['global_orient'] = np.array(
            body_model['global_orient']).reshape((-1, 3))
        body_model['betas'] = np.array(body_model['betas']).reshape((-1, 10))
        body_model['transl'] = np.array(body_model['transl']).reshape((-1, 3))

        bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])

        # change list to np array
        keypoints2d_ = np.array(keypoints2d_).reshape((-1, num_keypoints2d, 3))
        keypoints2d_, mask = convert_kps(keypoints2d_, keypoints2d_convention,
                                         'human_data')
        keypoints3d_ = np.array(keypoints3d_).reshape((-1, num_keypoints3d, 4))
        keypoints3d_, _ = convert_kps(keypoints3d_, keypoints3d_convention,
                                      'human_data')

        # import pdb
        # pdb.set_trace()
        focal_length = np.array(focal_length)
        # R = np.array(R)
        # T = np.array(T).squeeze(1)

        human_data['focal_length'] = focal_length
        # human_data['R'] = R
        # human_data['T'] = T

        human_data['image_path'] = image_path_
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['keypoints2d_mask'] = mask
        human_data['keypoints3d_mask'] = mask
        human_data['keypoints2d'] = keypoints2d_
        human_data['keypoints3d'] = keypoints3d_
        human_data['meta'] = meta
        human_data['error'] = error_all.detach().cpu().numpy()
        human_data['config'] = 'spec'
        human_data['smpl'] = body_model
        # human_data['cam_param'] = cam_param_
        human_data.compress_keypoints_by_mask()
        # import pdb
        # pdb.set_trace()
        # store data
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        file_name = f'spec_{mode}_{self.fit}.npz'
        out_file = os.path.join(out_path, file_name)
        human_data.dump(out_file)


root = '/mnt/lustre/wangwenjia/datasets'

body_model_train = dict(
    type='SMPL',
    keypoint_src='smpl_54',
    keypoint_dst='smpl_54',
    model_path=f'{root}/body_models/smpl',
    keypoint_approximate=True,
    extra_joints_regressor=f'{root}/body_models/J_regressor_extra.npy')

converter = SpecConverter(modes=['train', 'test'], body_model=body_model_train)
converter.convert_by_mode(dataset_path=f'{root}/mmhuman_data/datasets/spec',
                          out_path='.',
                          mode='train')
converter.convert_by_mode(dataset_path=f'{root}/mmhuman_data/datasets/spec',
                          out_path='.',
                          mode='test')
