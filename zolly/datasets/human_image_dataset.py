import mmcv
import os
import torch
import numpy as np
from typing import Optional, Union, List
from collections import OrderedDict
from mmhuman3d.data.datasets.human_image_dataset import HumanImageDataset as _HumanImageDataset
from zolly.models.body_models.mappings import get_keypoint_idx
from zolly.utils.bbox_utils import kp2d_to_bbox


class HumanImageDataset(_HumanImageDataset):
    ALLOWED_METRICS = {
        'mpjpe', 'pa-mpjpe', 'pve', '3dpck', 'pa-3dpck', '3dauc', 'pa-3dauc',
        'ihmr', 'pa-pve', 'miou', 'pmiou'
    }

    def __init__(
        self,
        data_prefix: str,
        pipeline: list,
        dataset_name: str,
        body_model: Optional[Union[dict, None]] = None,
        ann_file: Optional[Union[str, None]] = None,
        convention: Optional[str] = 'human_data',
        cache_data_path: Optional[Union[str, None]] = None,
        test_mode: Optional[bool] = False,
        is_distorted: Optional[bool] = False,  # new feature
        num_data: Optional[int] = None,  # new feature
        start_index: int = None,
    ):
        super().__init__(data_prefix, pipeline, dataset_name, body_model,
                         ann_file, convention, cache_data_path, test_mode)
        self.start_index = start_index
        if start_index is not None:
            self.num_data = self.num_data - start_index
        if num_data is not None:
            self.num_data = num_data
        self.is_distorted = is_distorted

    def prepare_raw_data(self, idx: int):
        """Get item from self.human_data."""
        if self.start_index is not None:
            idx = idx + self.start_index
        sample_idx = idx

        if self.cache_reader is not None:
            self.human_data = self.cache_reader.get_item(idx)
            idx = idx % self.cache_reader.slice_size
        info = {}
        info['img_prefix'] = None
        image_path = self.human_data['image_path'][idx]
        if self.dataset_name == 'h36m':
            image_path = image_path.replace('/images/', '/')

        info['image_path'] = os.path.join(self.data_prefix, 'datasets',
                                          self.dataset_name, image_path)

        if image_path.endswith('smc'):
            device, device_id, frame_id = self.human_data['image_id'][idx]
            info['image_id'] = (device, int(device_id), int(frame_id))

        info['dataset_name'] = self.dataset_name
        info['sample_idx'] = sample_idx

        # in later modules, we will check validity of each keypoint by
        # its confidence. Therefore, we do not need the mask of keypoints.

        if 'keypoints2d' in self.human_data:
            info['keypoints2d'] = self.human_data['keypoints2d'][idx]
            info['has_keypoints2d'] = 1
        else:
            info['keypoints2d'] = np.zeros((self.num_keypoints, 3))
            info['has_keypoints2d'] = 0
        if 'keypoints3d' in self.human_data:
            info['keypoints3d'] = self.human_data['keypoints3d'][idx]
            info['has_keypoints3d'] = 1
        else:
            info['keypoints3d'] = np.zeros((self.num_keypoints, 4))
            info['has_keypoints3d'] = 0

        if 'bbox_xywh' in self.human_data:
            info['bbox_xywh'] = self.human_data['bbox_xywh'][idx]
            x, y, w, h, _ = info['bbox_xywh']
            cx = x + w / 2
            cy = y + h / 2
            w = h = max(w, h)
            info['center'] = np.array([cx, cy])
            info['scale'] = np.array([w, h])
        elif 'keypoints2d' in self.human_data:
            bbox_xywh = kp2d_to_bbox(info['keypoints2d'],
                                     scale_factor=1.25,
                                     xywh=True)
            cx, cy, w, h = bbox_xywh[0]
            w = h = max(w, h)
            info['center'] = np.array([cx, cy])
            info['scale'] = np.array([w, h])
        else:
            info['bbox_xywh'] = np.zeros((5))
            info['center'] = np.zeros((2))
            info['scale'] = np.zeros((2))

        if 'smpl' in self.human_data:
            smpl_dict = self.human_data['smpl']
        else:
            smpl_dict = {}

        if 'smpl' in self.human_data:
            if 'has_smpl' in self.human_data:
                info['has_smpl'] = int(self.human_data['has_smpl'][idx])
            else:
                info['has_smpl'] = 1
        else:
            info['has_smpl'] = 0
        if 'body_pose' in smpl_dict:
            info['smpl_body_pose'] = smpl_dict['body_pose'][idx]
        else:
            info['smpl_body_pose'] = np.zeros((23, 3))

        if 'global_orient' in smpl_dict:
            info['smpl_global_orient'] = smpl_dict['global_orient'][idx]
        else:
            info['smpl_global_orient'] = np.zeros((3))

        info['smpl_origin_orient'] = info['smpl_global_orient'].copy()

        if 'betas' in smpl_dict:
            info['smpl_betas'] = smpl_dict['betas'][idx]
        else:
            info['smpl_betas'] = np.zeros((10))

        if 'betas_neutral' in smpl_dict:
            if not self.test_mode:  #for pw3d training
                info['smpl_betas'] = smpl_dict['betas_neutral'][idx]

        if 'transl' in smpl_dict:
            info['smpl_transl'] = smpl_dict['transl'][idx].astype(np.float32)
            info['has_transl'] = 1
        else:
            info['smpl_transl'] = np.zeros((3)).astype(np.float32)
            info['has_transl'] = 0

        if 'focal_length' in self.human_data:
            info['ori_focal_length'] = float(
                self.human_data['focal_length'][idx].reshape(-1)[0])
            info['has_focal_length'] = 1
        else:
            info['ori_focal_length'] = 5000.
            info['has_focal_length'] = 0

        if 'K' in self.human_data:
            info['K'] = self.human_data['K'][idx].reshape(3,
                                                          3).astype(np.float32)
            info['has_K'] = 1
        else:
            info['K'] = np.eye(3, 3).astype(np.float32)
            info['has_K'] = 0

        if self.is_distorted:
            info['distortion_max'] = float(
                self.human_data['distortion_max'][idx])
            info['is_distorted'] = 1
        else:
            info['distortion_max'] = 1.0
            info['is_distorted'] = 0

        return info

    def evaluate(self,
                 outputs: list,
                 res_folder: str,
                 metric: Optional[Union[str, List[str]]] = 'pa-mpjpe',
                 **kwargs: dict):
        """Evaluate 3D keypoint results.

        Args:
            outputs (list): results from model inference.
            res_folder (str): path to store results.
            metric (Optional[Union[str, List(str)]]):
                the type of metric. Default: 'pa-mpjpe'
            kwargs (dict): other arguments.
        Returns:
            dict:
                A dict of all evaluation results.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        for metric in metrics:
            if metric not in self.ALLOWED_METRICS:
                raise KeyError(f'metric {metric} is not supported')
        if res_folder:
            res_file = os.path.join(res_folder, 'result_keypoints.json')
        # for keeping correctness during multi-gpu test, we sort all results

        res_dict = {}

        for out in outputs:
            target_id = out['image_idx']
            batch_size = len(out['keypoints_3d'])
            for i in range(batch_size):
                res_dict[int(target_id[i])] = dict(
                    keypoints=out['keypoints_3d'][i],
                    poses=out['smpl_pose'][i],
                    betas=out['smpl_beta'][i],
                    batch_miou=out['batch_miou'][i],
                    batch_pmiou=out['batch_pmiou'][i],
                    # pred_keypoints2d=out['pred_keypoints2d'][i],
                    # gt_keypoints2d=out['origin_keypoints2d'][i]
                )
                if out.get('estimate_verts', False):
                    res_dict[int(target_id[i])].update(
                        dict(
                            vertices=out['vertices'][i],
                            pred_keypoints3d=out['pred_keypoints3d'][i],
                        ))

        keypoints, poses, betas = [], [], []
        vertices = []
        pred_kp3d = []
        miou = []
        pmiou = []
        # pred_kp2d = []
        # gt_kp2d = []
        for i in range(self.num_data):
            keypoints.append(res_dict[i]['keypoints'])
            poses.append(res_dict[i]['poses'])
            betas.append(res_dict[i]['betas'])
            miou.append(res_dict[i]['batch_miou'])
            pmiou.append(res_dict[i]['batch_pmiou'])
            # pred_kp2d.append(res_dict[i]['pred_keypoints2d'])
            # gt_kp2d.append(res_dict[i]['gt_keypoints2d'])
            if 'vertices' in res_dict[i]:
                vertices.append(res_dict[i]['vertices'])
            if 'pred_keypoints3d' in res_dict[i]:
                pred_kp3d.append(res_dict[i]['pred_keypoints3d'])

        res = dict(
            keypoints=keypoints[:self.num_data],
            poses=poses[:self.num_data],
            betas=betas[:self.num_data],
            miou=miou[:self.num_data],
            pmiou=pmiou[:self.num_data],

            #    pred_keypoints2d=pred_kp2d[:self.num_data],
            #    gt_keypoints2d=gt_kp2d[:self.num_data]
        )
        if len(vertices):
            res['esitmated_vertices'] = vertices[:self.num_data]
        if len(pred_kp3d):
            res['estimated_keypoints3d'] = pred_kp3d[:self.num_data]
        if res_folder:
            mmcv.dump(res, res_file)

        name_value_tuples = []
        for _metric in metrics:
            if _metric == 'mpjpe':
                _nv_tuples = self._report_mpjpe(res)
            elif _metric == 'pa-mpjpe':
                _nv_tuples = self._report_mpjpe(res, metric='pa-mpjpe')
            elif _metric == '3dpck':
                _nv_tuples = self._report_3d_pck(res)
            elif _metric == 'pa-3dpck':
                _nv_tuples = self._report_3d_pck(res, metric='pa-3dpck')
            elif _metric == '3dauc':
                _nv_tuples = self._report_3d_auc(res)
            elif _metric == 'pa-3dauc':
                _nv_tuples = self._report_3d_auc(res, metric='pa-3dauc')
            elif _metric == 'pve':
                _nv_tuples = self._report_pve(res, metric='pa-pve')
            elif _metric == 'pa-pve':
                _nv_tuples = self._report_pve(res)
            elif _metric == 'ihmr':
                _nv_tuples = self._report_ihmr(res)
            elif _metric == 'miou':
                _nv_tuples = self._report_miou(res)
            elif _metric == 'pmiou':
                _nv_tuples = self._report_miou(res, metric='pmiou')
            else:
                raise NotImplementedError
            name_value_tuples.extend(_nv_tuples)

        name_value = OrderedDict(name_value_tuples)
        return name_value

    def _report_miou(self, res_file, metric='miou', body_part=''):
        """Cauculate mean per joint position error (MPJPE) or its variants PA-
        MPJPE.

        Report mean per joint position error (MPJPE) and mean per joint
        position error after rigid alignment (PA-MPJPE)
        """
        miou = res_file[metric]
        error = np.array(miou).mean()
        err_name = metric.upper()
        if body_part != '':
            err_name = body_part.upper() + ' ' + err_name

        info_str = [(err_name, error)]

        return info_str

    def _parse_result(self, res, mode='keypoint', body_part=None):
        """Parse results."""

        if mode == 'vertice':
            # gt
            gt_beta, gt_pose, gt_global_orient, gender = [], [], [], []
            gt_smpl_dict = self.human_data['smpl']
            for idx in range(self.num_data):
                gt_beta.append(gt_smpl_dict['betas'][idx])
                gt_pose.append(gt_smpl_dict['body_pose'][idx])
                gt_global_orient.append(gt_smpl_dict['global_orient'][idx])
                if 'meta' in self.human_data:
                    if self.human_data['meta']['gender'][idx] == 'm':
                        gender.append(0)
                    else:
                        gender.append(1)
                else:
                    gender.append(-1)
            gt_beta = torch.FloatTensor(gt_beta)
            gt_pose = torch.FloatTensor(gt_pose).view(-1, 69)
            gt_global_orient = torch.FloatTensor(gt_global_orient)
            gender = torch.Tensor(gender)
            gt_output = self.body_model(betas=gt_beta,
                                        body_pose=gt_pose,
                                        global_orient=gt_global_orient,
                                        gender=gender)
            gt_vertices = gt_output['vertices'].detach().cpu().numpy() * 1000.
            gt_mask = np.ones(gt_vertices.shape[:-1])
            # pred
            pred_pose = torch.FloatTensor(res['poses'])
            pred_beta = torch.FloatTensor(res['betas'])

            if 'esitmated_vertices' in res:
                pred_vertices = np.stack(res['esitmated_vertices'], 0) * 1000.

                gt_keypoints3d = gt_output['joints']

                right_hip_idx = get_keypoint_idx('right_hip_extra', 'h36m')
                left_hip_idx = get_keypoint_idx('left_hip_extra', 'h36m')

                gt_pelvis = (gt_keypoints3d[:, left_hip_idx] +
                             gt_keypoints3d[:, right_hip_idx]) / 2
                gt_vertices = gt_vertices - gt_pelvis.view(
                    -1, 1, 3).detach().cpu().numpy() * 1000.

            else:
                pred_output = self.body_model(
                    betas=pred_beta,
                    body_pose=pred_pose[:, 1:],
                    global_orient=pred_pose[:, 0].unsqueeze(1),
                    pose2rot=False,
                    gender=None)
                pred_vertices = pred_output['vertices'].detach().cpu().numpy(
                ) * 1000.

            assert len(pred_vertices) == self.num_data

            return pred_vertices, gt_vertices, gt_mask

        elif mode == 'keypoint':
            if 'estimated_keypoints3d' in res:
                pred_keypoints3d = res['estimated_keypoints3d']
            else:
                pred_keypoints3d = res['keypoints']

            assert len(pred_keypoints3d) == self.num_data

            # (B, 17, 3)
            pred_keypoints3d = np.array(pred_keypoints3d)

            if self.dataset_name in [
                    'pdhuman', 'pw3d', 'lspet', 'humman', 'spec_mtp', 'h36m'
            ]:
                # print('testing h36m')
                betas = []
                body_pose = []
                global_orient = []
                gender = []
                smpl_dict = self.human_data['smpl']
                for idx in range(self.num_data):
                    betas.append(smpl_dict['betas'][idx])
                    body_pose.append(smpl_dict['body_pose'][idx])
                    global_orient.append(smpl_dict['global_orient'][idx])
                    if 'meta' in self.human_data:
                        if self.human_data['meta']['gender'][idx] == 'm':
                            gender.append(0)
                        else:
                            gender.append(1)
                    else:
                        gender.append(-1)
                betas = torch.FloatTensor(betas)
                body_pose = torch.FloatTensor(body_pose).view(-1, 69)
                global_orient = torch.FloatTensor(global_orient)
                gender = torch.Tensor(gender)
                gt_output = self.body_model(betas=betas,
                                            body_pose=body_pose,
                                            global_orient=global_orient,
                                            gender=gender)
                gt_keypoints3d = gt_output['joints'].detach().cpu().numpy()
                gt_keypoints3d_mask = np.ones((len(pred_keypoints3d), 24))
            elif self.dataset_name in ['h36m']:
                gt_keypoints3d = self.human_data[
                    'keypoints3d'][:self.num_data, :, :3]
                gt_keypoints3d_mask = np.ones(
                    (len(pred_keypoints3d), pred_keypoints3d.shape[-2]))

            else:
                raise NotImplementedError()

            # SMPL_49 only!
            if gt_keypoints3d.shape[1] == 49:
                assert pred_keypoints3d.shape[1] == 49

                gt_keypoints3d = gt_keypoints3d[:, 25:, :]
                pred_keypoints3d = pred_keypoints3d[:, 25:, :]

                joint_mapper = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]
                gt_keypoints3d = gt_keypoints3d[:, joint_mapper, :]
                pred_keypoints3d = pred_keypoints3d[:, joint_mapper, :]

                # we only evaluate on 14 lsp joints
                pred_pelvis = (pred_keypoints3d[:, 2] +
                               pred_keypoints3d[:, 3]) / 2
                gt_pelvis = (gt_keypoints3d[:, 2] + gt_keypoints3d[:, 3]) / 2

            # H36M for testing!
            elif gt_keypoints3d.shape[1] == 17:
                assert pred_keypoints3d.shape[1] == 17

                H36M_TO_J17 = [
                    6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9
                ]
                H36M_TO_J14 = H36M_TO_J17[:14]
                joint_mapper = H36M_TO_J14

                pred_pelvis = pred_keypoints3d[:, 0]
                gt_pelvis = gt_keypoints3d[:, 0]

                gt_keypoints3d = gt_keypoints3d[:, joint_mapper, :]
                pred_keypoints3d = pred_keypoints3d[:, joint_mapper, :]

            # keypoint 24
            elif gt_keypoints3d.shape[1] == 24:
                assert pred_keypoints3d.shape[1] == 24

                joint_mapper = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]
                gt_keypoints3d = gt_keypoints3d[:, joint_mapper, :]
                pred_keypoints3d = pred_keypoints3d[:, joint_mapper, :]

                # we only evaluate on 14 lsp joints
                pred_pelvis = (pred_keypoints3d[:, 2] +
                               pred_keypoints3d[:, 3]) / 2
                gt_pelvis = (gt_keypoints3d[:, 2] + gt_keypoints3d[:, 3]) / 2

            else:
                pass
            if not 'estimated_keypoints3d' in res:
                pred_keypoints3d = (pred_keypoints3d -
                                    pred_pelvis[:, None, :]) * 1000
            else:
                pred_keypoints3d = pred_keypoints3d * 1000
            gt_keypoints3d = (gt_keypoints3d - gt_pelvis[:, None, :]) * 1000

            gt_keypoints3d_mask = gt_keypoints3d_mask[:, joint_mapper] > 0

            return pred_keypoints3d, gt_keypoints3d, gt_keypoints3d_mask

    def _parse_result2d(self, res):
        """Parse results."""
        pred_keypoints2d = np.array(res['pred_keypoints2d'])
        gt_keypoints2d = np.array(res['gt_keypoints2d'])
        gt_keypoints2d_mask = gt_keypoints2d[..., 2:3]
        return pred_keypoints2d, gt_keypoints2d, gt_keypoints2d_mask
