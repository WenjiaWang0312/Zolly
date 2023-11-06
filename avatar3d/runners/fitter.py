import enum
import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Iterable, Union, Tuple, List
from mmcv.runner import build_optimizer
from pytorch3d.structures import Meshes
from avatar3d.cameras.builder import build_cameras
from avatar3d.models.body_models import get_model_class
from mmhuman3d.data.datasets import RepeatDataset
from avatar3d.models.body_models.mano import MANO_LEFT
from avatar3d.models.body_models.mappings import JOINTS_FACTORY, KEYPOINTS_FACTORY, convert_kps
from avatar3d.models.body_models.mappings.mano import MANO_KEYPOINTS_LEFT, MANO_KEYPOINTS_RIGHT
from avatar3d.models.body_models.mappings.smplh import SMPLH_KEYPOINTS
from avatar3d.models.body_models.utils import get_wrist_global, get_wrist_local, merge_smplh_pose, split_smplh_pose
from avatar3d.models.body_models.builder import build_body_model
from avatar3d.models.losses.builder import build_loss
from avatar3d.datasets.builder import build_dataloader, build_dataset
from avatar3d.utils.torch_utils import (build_parameters, cat, cat_pose_list,
                                        dict2numpy, merge_loss_dict,
                                        move_dict_to_device, slice_dict,
                                        slice_pose_dict, image_tensor2numpy,
                                        unbind_pose)
from avatar3d.utils.visualize_smpl import vis_smpl


class OptimizableParameters():
    """Collects parameters for optimization."""

    def __init__(self):
        self.opt_params = {}

    def set_param(self, fit_param: bool, key: str,
                  param: Union[List[torch.Tensor], torch.Tensor]) -> None:
        """Set requires_grad and collect parameters for optimization.
        Args:
            fit_param: whether to optimize this body model parameter
            param: body model parameter
        Returns:
            None
        """
        if fit_param:
            param.requires_grad = True
            self.opt_params[key] = param
        else:
            param.requires_grad = False

    def parameters(self) -> dict:
        """Returns parameters. Compatible with mmcv's build_parameters()
        Returns:
            opt_params: a dict of body model parameters for optimization
        """
        return list(self.opt_params.values())

    def __getitem__(self, indexes) -> dict:
        if isinstance(indexes, (int)):
            indexes = [indexes]
        res_dict = {}
        for k, v in self.opt_params.items():
            if v.shape[0] == 1:
                res_dict[k] = v
            else:
                res_dict[k] = v[indexes]
        return res_dict


class FitRunnerBase(object):

    def __init__(self,
                 device: Union[str, torch.device] = 'cpu',
                 num_epoch: int = 1,
                 num_workers: int = 1,
                 stages: dict = None,
                 loss_configs: dict = None,
                 dataset_config: dict = None,
                 parameters: dict = None,
                 verbose: bool = True,
                 model_type: str = '',
                 **kwargs) -> None:
        self.device = device
        self.num_epoch = num_epoch
        self.num_workers = num_workers
        self.model_type = model_type
        self.loss_dict = {}
        self.dataset = build_dataset(dataset_config)

        if hasattr(self.dataset, 'parameters'):
            default_parameters = self.dataset.parameters
        else:
            default_parameters = dict()

        self.resolution = self.dataset.resolution

        self.init_params(parameters, default_parameters)

        self.stage_configs = stages
        for k in loss_configs:
            self.loss_dict[k] = build_loss(loss_configs[k]).to(self.device)

        self.verbose = verbose
        self.cameras = self.build_camera()

    def init_params(self, parameters, default_parameters=dict()):
        self.parameters = build_parameters(**parameters)
        self.parameters.update(default_parameters)

        self.init_parameters = dict()
        for k, v in self.parameters.items():
            self.init_parameters[k] = cat_pose_list(v).data

        for k in [
                'body_pose', 'hand_pose', 'left_hand_pose', 'right_hand_pose',
                'transl'
        ]:
            if k in self.parameters:
                if k == 'transl':
                    transl = self.parameters[k]
                    self.parameters[k] = [
                        transl[:, 0:1], transl[:, 1:2], transl[:, 2:3]
                    ]
                else:
                    self.parameters[k] = unbind_pose(self.parameters[k])

    def forward(self):
        raise NotImplementedError()

    def build_camera(self):
        if self.dataset.calibrated:
            cameras = self.dataset.cameras.to(self.device)
        else:
            resolution = self.resolution
            cameras = build_cameras(
                dict(type='perspective',
                     in_ndc=False,
                     device=self.device,
                     convention='opencv',
                     resolution=resolution,
                     R=None,
                     T=None,
                     focal_length=self.parameters.get('focal_length', 5000),
                     principal_point=[resolution[1] / 2, resolution[0] / 2]))
            cameras.K = None
        return cameras

    def _optimize_stage(self,
                        num_iter: int,
                        optimizer: dict,
                        fit_config: dict,
                        loss_weight_config: dict,
                        data_valid_keys: Tuple[str] = None,
                        batch_size: int = -1,
                        ftol: float = 1e-4,
                        convention: str = None,
                        model_type=None,
                        **kwargs):
        parameters = OptimizableParameters()

        model_type = self.model_type if model_type is None else model_type
        for k, v in self.parameters.items():

            fit_flag = fit_config.get(k, False)

            if isinstance(v, (tuple, list)):
                if k == 'transl':
                    for index, v_ in enumerate(v):
                        axis_name = ('x', 'y', 'z')[index]
                        bool_flag = fit_flag if isinstance(
                            fit_flag, bool) else axis_name in fit_flag
                        parameters.set_param(bool_flag, axis_name, v_)
                else:
                    full_pose_dims = get_model_class(model_type).full_pose_dims
                    part_names = list(full_pose_dims.keys())
                    if k in part_names:
                        pose_dim_offset = 0
                        for i in range(part_names.index(k)):
                            pose_dim_offset += full_pose_dims[part_names[i]]
                        joint_names = JOINTS_FACTORY[model_type.split('_')[0]]
                        for index, v_ in enumerate(v):
                            joint_name = joint_names[index + pose_dim_offset]
                            bool_flag = fit_flag if isinstance(
                                fit_flag, bool) else joint_name in fit_flag
                            parameters.set_param(
                                bool_flag,
                                joint_name,
                                v_,
                            )

            else:
                assert isinstance(fit_flag, bool)
                parameters.set_param(
                    fit_flag,
                    k,
                    v,
                )

        optimizer = build_optimizer(parameters, optimizer)

        pred_loss = None

        if batch_size < 0:
            batch_size = len(self.dataset)

        self.dataset.set_valid_keys(data_valid_keys)

        if num_iter > 0:
            times = batch_size * num_iter / len(self.dataset)
        else:
            times = 1

        if times >= 1:
            dataset = RepeatDataset(self.dataset, int(times))

        data_loader = build_dataloader(workers_per_gpu=self.num_workers,
                                       shuffle=False,
                                       samples_per_gpu=batch_size,
                                       dataset=dataset)
        for iter_idx, input_data in enumerate(data_loader):

            def closure():
                move_dict_to_device(input_data, self.device)
                preds = self.forward(input_data)

                optimizer.zero_grad()

                loss_dict = self._compute_loss(
                    loss_weight_config=loss_weight_config,
                    pred=preds,
                    target=input_data,
                    convention=convention)

                if self.verbose:
                    msg = ''

                    for loss_name, loss in loss_dict.items():
                        msg += f'{loss_name}={loss.mean().item():.6f}, '

                    print(msg.strip(', '))
                    # if fit_config.get('focal_length', False):
                    #     print(self.parameters['focal_length'])

                loss = loss_dict['total_loss']
                loss.backward()

                return loss

            loss = optimizer.step(closure)
            if iter_idx > 0 and pred_loss is not None and ftol > 0:
                loss_rel_change = self._compute_relative_change(
                    pred_loss, loss.item())
                if loss_rel_change < ftol:
                    print(f'[ftol={ftol}] Early stop at {iter_idx} iter!')
                    break
            pred_loss = loss.item()

    def __call__(self, **kwargs):
        for i in range(self.num_epoch):
            for stage_idx, stage_config in enumerate(self.stage_configs):
                if self.verbose:
                    print(f'epoch {i}, stage {stage_idx+1}')
                self._optimize_stage(**stage_config, )

        body_model_output = self.body_model(**self.parameters)
        ret = body_model_output
        ret.update(self.parameters)

        ret.update(kp2d=self.cameras.transform_points_screen(
            body_model_output['joints'],
            focal_length=self.parameters.get('focal_length', 5000)))
        if hasattr(self.cameras, 'focal_length'):
            self.cameras.update_focal_length(
                focal_length=self.parameters.get('focal_length', 5000))

        # for k, v in ret.items():
        #     if isinstance(v, torch.Tensor):
        #         ret[k] = v.detach().clone()
        ret = dict2numpy(ret)
        return ret

    def _compute_loss():
        raise NotImplementedError()

    @staticmethod
    def _compute_relative_change(pre_v, cur_v):
        """Compute relative loss change. If relative change is small enough, we
        can apply early stop to accelerate the optimization. (1) When one of
        the value is larger than 1, we calculate the relative change by diving
        their max value. (2) When both values are smaller than 1, it degrades
        to absolute change. Intuitively, if two values are small and close,
        dividing the difference by the max value may yield a large value.

        Args:
            pre_v: previous value
            cur_v: current value

        Returns:
            float: relative change
        """
        return np.abs(pre_v - cur_v) / max([np.abs(pre_v), np.abs(cur_v), 1])


class SMPLify(FitRunnerBase):
    """This is suitbale for all single models: mano, flame, smpl, smplh, smplx
        If you treat smplh and smplx as a whole, use this.
        If you treat smplh as smpl + mano, use SMPLifyH.
        If you treat smplx as smpl + mano + flame, use SMPLifyX.
    """

    def forward(self, input_data=None):
        if input_data is not None:
            frame_indexes = input_data['frame_id'].long()
        else:
            frame_indexes = None
        sliced_parameters = slice_pose_dict(self.parameters, frame_indexes)
        preds = self.body_model(return_full_pose=True, **sliced_parameters)
        preds.update(sliced_parameters)
        return preds

    def __init__(self,
                 device: Union[str, torch.device] = 'cpu',
                 num_epoch: int = 1,
                 num_workers: int = 1,
                 convention: str = 'smpl_45',
                 stages: dict = None,
                 body_model: Union[dict, nn.Module] = None,
                 loss_configs: dict = None,
                 dataset_config: dict = None,
                 parameters: dict = None,
                 verbose: bool = True,
                 disable_keypoints: tuple = (),
                 **kwargs) -> None:
        super().__init__(device, num_epoch, num_workers, convention, stages,
                         loss_configs, dataset_config, parameters, verbose,
                         **kwargs)
        self.disable_keypoints = disable_keypoints
        if isinstance(body_model, dict):
            self.body_model = build_body_model(body_model).to(self.device)
        elif isinstance(body_model, torch.nn.Module):
            self.body_model = body_model.to(self.device)
        else:
            raise TypeError(f'body_model should be either dict or '
                            f'torch.nn.Module, but got {type(body_model)}')

        move_dict_to_device(self.parameters, self.device)

    def visualize(self, preds, output_path, resolution=None):
        resolution = self.dataset.resolution if resolution is None else resolution
        rendered_map = vis_smpl(device=self.device,
                                verts=preds['vertices'],
                                resolution=resolution,
                                body_model=self.body_model,
                                batch_size=20,
                                end=40,
                                image_array=self.dataset.images,
                                output_path=output_path,
                                cameras=self.cameras,
                                overwrite=True,
                                verbose=True,
                                return_tensor=True)
        rendered_map = image_tensor2numpy(rendered_map)
        return rendered_map

    def _compute_loss(self,
                      loss_weight_config,
                      pred,
                      target,
                      body_model=None,
                      convention=None):
        """All the losses:
            Single Frame:
                kp3d_mse: TranslationMSELoss
                kp2d_mse: TranslationMSELoss
                silhouette: SilhouetteMSELoss
                pose_reg: JointRegularizationLoss
                pose_prior: JointPriorLoss
                shape_prior: ShapePriorLoss
                shape_threshold: ShapeThresholdPriorLoss
                limb_length: LimbLengthLoss

            Temporal:
                kp3d_smooth: SmoothTranslationLoss
                kp2d_smooth: SmoothTranslationLoss
                pose_smooth: SmoothJointLoss
                limb_smooth: SmoothTranslationLoss
                pelvis_smooth: SmoothPelvisLoss
                transl_smooth: SmoothTranslationLoss

            Cycle: (see in CycleSMPLify)
                flow_warp: FlowWarpingLoss
        """
        cameras = self.cameras
        kp3d_gt = target.get('kp3d', None)

        body_model = self.body_model if body_model is None else body_model
        if convention is not None:
            if kp3d_gt is not None:
                kp3d_gt = convert_kps(
                    kp3d_gt,
                    src=body_model.keypoint_dst,
                    dst=convention,
                )[0]
        if kp3d_gt is not None and kp3d_gt.shape[-1] == 4:
            kp3d_gt_conf = kp3d_gt[..., 3:]
        else:
            kp3d_gt_conf = None

        kp3d_pred = pred.get('joints', None)
        N = kp3d_pred.shape[0]
        kp3d_pred_conf = pred['joint_mask']
        if convention is not None:
            if kp3d_pred is not None:

                kp3d_pred, kp3d_pred_conf = convert_kps(
                    kp3d_pred,
                    src=body_model.keypoint_dst,
                    dst=convention,
                    mask=kp3d_pred_conf)
        kp3d_pred_conf = kp3d_pred_conf.unsqueeze(0).repeat_interleave(N, 0)

        # cameras.focal_length = self.parameters['focal_length']
        kp2d_pred = cameras.transform_points_screen(
            kp3d_pred + 1e-12,
            focal_length=self.parameters.get('focal_length', 5000))

        kp2d_pred = 2 * kp2d_pred[..., :2] / (
            cameras.resolution.view(-1, 1, 2) - 1) - 1
        kp2d_pred = kp2d_pred[..., :2]
        kp2d_pred_conf = kp3d_pred_conf

        for k in self.disable_keypoints:
            index = KEYPOINTS_FACTORY[body_model.keypoint_dst].index(k)
            kp2d_pred_conf[:, index] = 0

        kp2d_gt = target.get('kp2d', None)

        if convention is not None:
            if kp2d_gt is not None:
                kp2d_gt = convert_kps(kp2d_gt,
                                      src=body_model.keypoint_dst,
                                      dst=convention)[0]

        if kp2d_gt is not None and kp2d_gt.shape[-1] == 3:
            kp2d_gt_conf = kp2d_gt[..., 2:]
        else:
            kp2d_gt_conf = None

        kp2d_gt = kp2d_gt[..., :2]

        kp2d_gt = 2 * kp2d_gt / (cameras.resolution.view(-1, 1, 2) - 1) - 1
        mask_gt = target.get('mask', None)

        body_pose = pred.get('body_pose', None)
        hand_pose = pred.get('hand_pose', None)

        left_hand_pose = pred.get('left_hand_pose', None)
        right_hand_pose = pred.get('right_hand_pose', None)

        global_orient = pred['global_orient']

        betas = pred['betas']

        transl = pred.get('transl', None)

        N = kp3d_pred.shape[0]
        mesh_pred = Meshes(
            verts=pred['vertices'],
            faces=body_model.faces_tensor[None].repeat_interleave(N, 0))

        losses = {}

        for k in self.loss_dict:
            weight = loss_weight_config.get(k, 0.0)
            if weight > 0:
                if k == 'silhouette' and mask_gt is not None:

                    losses[k] = self.loss_dict[k](mesh_pred=mesh_pred,
                                                  camera_pred=cameras,
                                                  mask_target=mask_gt) * weight
                elif k == 'silhouette_reg' and mask_gt is not None:
                    losses[k] = self.loss_dict[k](mesh_pred=mesh_pred,
                                                  camera_pred=cameras,
                                                  mask_target=mask_gt) * weight
                elif k == 'laplacian':
                    losses[k] = self.loss_dict[k](mesh=mesh_pred) * weight
                elif k == 'normal_consistency':
                    losses[k] = self.loss_dict[k](mesh=mesh_pred) * weight
                elif k == 'mesh_edge':
                    losses[k] = self.loss_dict[k](mesh=mesh_pred) * weight

                elif k == 'kp3d_mse' and kp3d_gt is not None:
                    losses[k] = self.loss_dict[k](
                        pred=kp3d_pred,
                        target=kp3d_gt,
                        pred_conf=kp3d_pred_conf,
                        target_conf=kp3d_gt_conf) * weight
                elif k == 'kp2d_mse' and kp2d_gt is not None:
                    losses[k] = self.loss_dict[k](
                        pred=kp2d_pred,
                        target=kp2d_gt,
                        pred_conf=kp2d_pred_conf,
                        target_conf=kp2d_gt_conf) * weight
                elif k == 'body_pose_reg' and body_pose is not None:
                    losses[k] = self.loss_dict[k](body_pose=body_pose) * weight

                elif k == 'hand_pose_reg':
                    loss = 0
                    for pose in [left_hand_pose, right_hand_pose, hand_pose]:
                        if pose is not None:
                            loss += self.loss_dict[k](pose) * weight
                    losses[k] = loss

                elif k == 'body_pose_prior' and body_pose is not None:
                    losses[k] = self.loss_dict[k](body_pose=body_pose) * weight
                elif k == 'hand_pose_prior':
                    loss = 0
                    for pose in [left_hand_pose, right_hand_pose, hand_pose]:
                        if pose is not None:
                            loss += self.loss_dict[k](pose) * weight
                    losses[k] = loss
                elif k == 'shape_prior':
                    losses[k] = self.loss_dict[k](betas=betas) * weight
                elif k == 'shape_threshold':
                    losses[k] = self.loss_dict[k](betas=betas) * weight
                elif k == 'limb_length3d':
                    losses[k] = self.loss_dict[k](
                        pred=kp3d_pred,
                        pred_conf=None,
                        target=kp3d_gt,
                        target_conf=kp3d_gt_conf,
                        convention=convention) * weight
                elif k == 'limb_length2d':
                    losses[k] = self.loss_dict[k](
                        pred=kp2d_pred,
                        pred_conf=None,
                        target=kp2d_gt,
                        target_conf=kp2d_gt_conf,
                        convention=convention) * weight
                elif k == 'limb_direction2d':
                    losses[k] = self.loss_dict[k](
                        pred=kp2d_pred,
                        pred_conf=None,
                        target=kp2d_gt,
                        target_conf=kp2d_gt_conf,
                        convention=convention) * weight
                elif k == 'limb_direction3d':
                    losses[k] = self.loss_dict[k](
                        pred=kp3d_pred,
                        pred_conf=None,
                        target=kp3d_gt,
                        target_conf=kp3d_gt_conf,
                        convention=convention) * weight
                elif k == 'kp3d_smooth':

                    losses[k] = self.loss_dict[k](kp3d_pred) * weight
                elif k == 'kp2d_smooth':
                    losses[k] = self.loss_dict[k](kp2d_pred) * weight
                elif k == 'body_pose_smooth':
                    losses[k] = self.loss_dict[k](body_pose) * weight
                elif k == 'hand_pose_smooth':
                    loss = 0
                    for pose in [left_hand_pose, right_hand_pose, hand_pose]:
                        if pose is not None:
                            loss += self.loss_dict[k](pose) * weight
                    losses[k] = loss
                elif k == 'limb_smooth2d':
                    losses[k] = self.loss_dict[k](
                        pred=kp2d_pred, convention=convention) * weight
                elif k == 'limb_smooth3d':
                    losses[k] = self.loss_dict[k](
                        pred=kp3d_pred, convention=convention) * weight
                elif k == 'pelvis_smooth':
                    losses[k] = self.loss_dict[k](
                        global_orient=global_orient) * weight
                elif k == 'transl_smooth':
                    losses[k] = self.loss_dict[k](translation=transl) * weight
                elif k == 'transl_hinge':
                    losses[k] = self.loss_dict[k](transl[:, 2]) * weight
                elif k == 'body_pose_norm':
                    losses[k] = self.loss_dict[k](
                        body_pose, self.init_parameters['body_pose']) * weight
                elif k == 'hand_pose_norm':
                    loss = 0
                    for name, pose in zip(
                        ['left_hand_pose', 'right_hand_pose', 'hand_pose'],
                        [left_hand_pose, right_hand_pose, hand_pose]):
                        if pose is not None:
                            loss += self.loss_dict[k](
                                pose, self.init_parameters[name]) * weight
                    losses[k] = loss
                elif k == 'left_hand_pose_norm':
                    losses[k] = self.loss_dict[k](
                        body_pose,
                        self.init_parameters['left_hand_pose']) * weight
                elif k == 'right_hand_pose_norm':
                    losses[k] = self.loss_dict[k](
                        body_pose,
                        self.init_parameters['right_hand_pose']) * weight
        losses = merge_loss_dict(losses)
        return losses


class SMPLifyH_focal(SMPLify):

    def __init__(self,
                 device: Union[str, torch.device] = 'cpu',
                 num_epoch: int = 1,
                 num_workers: int = 1,
                 convention: str = 'smpl_45',
                 stages: dict = None,
                 mano_left: Union[dict, nn.Module] = None,
                 mano_right: Union[dict, nn.Module] = None,
                 body_model: Union[dict, nn.Module] = None,
                 loss_configs: dict = None,
                 dataset_config: dict = None,
                 parameters: dict = None,
                 verbose: bool = True,
                 disable_keypoints: tuple = (),
                 f_delta_threshold=10,
                 **kwargs) -> None:
        super().__init__(device, num_epoch, num_workers, convention, stages,
                         body_model, loss_configs, dataset_config, parameters,
                         verbose, disable_keypoints, **kwargs)
        self.smplh = self.body_model
        self.f_delta_threshold = f_delta_threshold

        if isinstance(mano_left, dict):
            self.mano_left = build_body_model(mano_left).to(self.device)
        elif isinstance(mano_left, torch.nn.Module):
            self.mano_left = mano_left.to(self.device)
        else:
            raise TypeError(f'body_model should be either dict or '
                            f'torch.nn.Module, but got {type(mano_left)}')

        if isinstance(mano_right, dict):
            self.mano_right = build_body_model(mano_right).to(self.device)
        elif isinstance(mano_right, torch.nn.Module):
            self.mano_right = mano_right.to(self.device)
        else:
            raise TypeError(f'body_model should be either dict or '
                            f'torch.nn.Module, but got {type(mano_right)}')
        body_pose = cat_pose_list(self.parameters['body_pose'])
        global_orient = self.parameters['global_orient']
        left_hand_orient, right_hand_orient = get_wrist_global(
            torch.cat([global_orient, body_pose], 1))
        self.parameters.update(left_hand_orient=left_hand_orient,
                               right_hand_orient=right_hand_orient)

    def forward(self, input_data=None):
        if input_data is not None:
            frame_indexes = input_data['frame_id']
        else:
            frame_indexes = None
        sliced_parameters = slice_pose_dict(self.parameters, frame_indexes)
        mano_left_params, mano_right_params, smplh_params = split_smplh_pose(
            sliced_parameters)

        mano_left_output = self.mano_left(**mano_left_params)
        mano_right_output = self.mano_right(**mano_right_params)
        smplh_output = self.smplh(**smplh_params)

        mano_left_output.update(mano_left_params)
        mano_right_output.update(mano_right_params)
        smplh_output.update(smplh_params)

        preds = dict(left_hand=mano_left_output,
                     right_hand=mano_right_output,
                     body=smplh_output)
        preds.update(sliced_parameters)
        return preds

    def visualize(self, preds, output_path, resolution=None):
        verts_body = preds['body']['vertices']
        verts_lhand = preds['left_hand']['vertices']
        verts_rhand = preds['right_hand']['vertices']
        resolution = self.dataset.resolution if resolution is None else resolution
        rendered_map = []

        verts_all = [verts_body, verts_lhand, verts_rhand]
        model_all = [self.smplh, self.mano_left, self.mano_right]
        for verts, body_model in zip(verts_all, model_all):
            rendered_map.append(
                vis_smpl(device=self.device,
                         verts=verts,
                         resolution=resolution,
                         body_model=body_model,
                         batch_size=20,
                         image_array=self.dataset.images,
                         cameras=self.cameras,
                         overwrite=True,
                         verbose=True,
                         return_tensor=True))
        if output_path.endswith('.png'):
            image = cat(rendered_map, 2)[0]
            image = image_tensor2numpy(image)
            cv2.imwrite(output_path, image)
        else:
            video = cat(rendered_map, 2)
            video = image_tensor2numpy(video)
            from avatar3d.utils.frame_utils import array_to_video_cv2
            array_to_video_cv2(video, output_path)

    def __call__(self, **kwargs):

        for i in range(self.num_epoch):
            self.f_last = float(self.parameters.get('focal_length'))
            for stage_idx, stage_config in enumerate(self.stage_configs):
                if self.verbose:
                    print(f'epoch {i}, stage {stage_idx+1}')
                self._optimize_stage(**stage_config, )

                if self.parameters['focal_length'].requires_grad:

                    focal_length = float(self.parameters.get('focal_length'))
                    print(focal_length)
                    delta = focal_length - self.f_last

                    if abs(delta) < self.f_delta_threshold:
                        break

        # body_model_output = self.body_model(**self.parameters)
        body_pose = torch.cat([
            self.parameters['global_orient'],
            cat_pose_list(self.parameters['body_pose'])
        ], 1)
        self.parameters['body_pose'] = get_wrist_local(
            body_pose,
            self.parameters['left_hand_orient'],
            self.parameters['right_hand_orient'],
            refine_wrist=False).view(body_pose.shape[0], -1)[:, 3:]
        ret = self.forward()
        ret.update(self.parameters)

        for dic in (ret['body'], ret['left_hand'], ret['right_hand']):
            dic.update(kp2d=self.cameras.transform_points_screen(
                dic['joints'],
                focal_length=self.parameters.get('focal_length', 5000)))

        if hasattr(self.cameras, 'focal_length'):
            self.cameras.update_focal_length(
                focal_length=self.parameters.get('focal_length', 5000))

        # for k, v in ret.items():
        #     if isinstance(v, torch.Tensor):
        #         ret[k] = v.detach().clone()

        ret = dict2numpy(ret)
        return ret

    def _compute_loss(self, loss_weight_config, pred, target, convention=None):

        target['left_hand'] = dict()
        target['right_hand'] = dict()
        kp2d = target.get('kp2d', None)
        kp3d = target.get('kp3d', None)
        if kp2d is not None:
            target['left_hand']['kp2d'] = convert_kps(
                kp2d,
                src=self.dataset.convention,
                dst=self.mano_left.keypoint_dst)[0]
            target['right_hand']['kp2d'] = convert_kps(
                kp2d,
                src=self.dataset.convention,
                dst=self.mano_right.keypoint_dst)[0]
        if kp3d is not None:
            target['left_hand']['kp3d'] = convert_kps(
                kp3d,
                src=self.dataset.convention,
                dst=self.mano_left.keypoint_dst)[0]
            target['right_hand']['kp3d'] = convert_kps(
                kp3d,
                src=self.dataset.convention,
                dst=self.mano_right.keypoint_dst)[0]
        losses_left_hand = super()._compute_loss(
            loss_weight_config['left_hand'],
            pred['left_hand'],
            target['left_hand'],
            self.mano_left,
            convention='mano_left')
        losses_right_hand = super()._compute_loss(
            loss_weight_config['right_hand'],
            pred['right_hand'],
            target['right_hand'],
            self.mano_right,
            convention='mano_right')
        losses_body = super()._compute_loss(loss_weight_config['body'],
                                            pred['body'],
                                            target,
                                            self.smplh,
                                            convention=convention)
        losses_left_hand.pop('total_loss')
        losses_right_hand.pop('total_loss')
        losses_body.pop('total_loss')

        losses_left_hand_ = dict()
        for k, v in losses_left_hand.items():
            losses_left_hand_['left_hand_' + k] = v

        losses_right_hand_ = dict()
        for k, v in losses_right_hand.items():
            losses_right_hand_['right_hand_' + k] = v

        focal_length = pred['focal_length']

        left_hand_joints = pred['left_hand']['joints']
        right_hand_joints = pred['right_hand']['joints']

        left_hand_transl = pred['left_hand']['transl']
        right_hand_transl = pred['right_hand']['transl']

        body_joints = pred['body']['joints']

        body_transl = pred['body']['transl']

        body_transl_f = torch.cat([
            body_transl[:, :2],
            body_transl[:, 2:3] / self.f_last * focal_length
        ], 1)

        body_joints_f = body_joints + (body_transl_f -
                                       body_transl).unsqueeze(1)

        left_hand_joints_body = convert_kps(body_joints_f,
                                            src=self.smplh.keypoint_dst,
                                            dst='mano_left')[0]
        right_hand_joints_body = convert_kps(body_joints_f,
                                             src=self.smplh.keypoint_dst,
                                             dst='mano_right')[0]

        left_hand_transl_f = torch.cat([
            left_hand_transl[:, :2],
            left_hand_transl[:, 2:3] / self.f_last * focal_length
        ], 1)

        left_hand_joints_f = left_hand_joints + (left_hand_transl_f -
                                                 left_hand_transl).unsqueeze(1)

        right_hand_transl_f = torch.cat([
            right_hand_transl[:, :2],
            right_hand_transl[:, 2:3] / self.f_last * focal_length
        ], 1)

        right_hand_joints_f = right_hand_joints + (
            right_hand_transl_f - right_hand_transl).unsqueeze(1)

        # if self.parameters['focal_length'].requires_grad is True:
        #     self.parameters['transl'] = body_transl_f.clone().detach()
        #     self.parameters['left_hand_transl'] = left_hand_transl_f.clone().detach()
        #     self.parameters['right_hand_transl'] = right_hand_transl_f.clone().detach()

        for k in self.loss_dict:
            weight = loss_weight_config['body'].get(k, 0.0)
            if weight > 0:
                if k == 'left_hand_align':
                    losses_body[k] = self.loss_dict[k](left_hand_joints_body,
                                                       left_hand_joints_f)
                if k == 'right_hand_align':
                    losses_body[k] = self.loss_dict[k](right_hand_joints_body,
                                                       right_hand_joints_f)
                if k == 'focal_hinge':
                    losses_body[k] = self.loss_dict[k](focal_length)
        losses = merge_loss_dict(losses_left_hand_, losses_right_hand_,
                                 losses_body)
        return losses


class MultiViewSMPLify(FitRunnerBase):
    pass


class CycleSMPLify(SMPLify):

    def forward(self, input_data=None):
        if input_data is not None:
            src_frame_indexes = input_data['src']['frame_index']
            src_parameters = slice_dict(self.parameters, src_frame_indexes)
            src_body_model_output = self.body_model(return_full_pose=True,
                                                    **src_parameters)
            src_body_model_output.update(src_parameters)

            dst_frame_indexes = input_data['dst']['frame_index']
            dst_parameters = slice_dict(self.parameters, dst_frame_indexes)
            dst_body_model_output = self.body_model(return_full_pose=True,
                                                    **dst_parameters)
            dst_body_model_output.update(dst_parameters)
            return dict(src=src_body_model_output, dst=dst_body_model_output)
        else:
            self.body_model(return_full_pose=True, **self.parameters)

    def _compute_loss(self, loss_weight_config, pred, target):
        cameras = self.cameras
        losses_src = super()._compute_loss(loss_weight_config, pred['src'],
                                           target['src'])
        losses_dst = super()._compute_loss(loss_weight_config, pred['dst'],
                                           target['dst'])
        losses_src.pop('total_loss')
        losses_dst.pop('total_loss')
        image_src = target['src']['image']
        image_dst = target['dst']['image']
        verts_src = pred['src']['vertices']
        verts_dst = pred['dst']['vertices']
        N = verts_src.shape[0]
        faces = self.body_model.faces_tensor
        mesh_src = Meshes(verts=verts_src, faces=faces.repeat_interleave(N, 0))
        mesh_dst = Meshes(verts=verts_dst, faces=faces.repeat_interleave(N, 0))
        camera_src = cameras
        camera_dst = cameras

        losses_cycle = dict()
        for k in self.loss_dict:
            weight = loss_weight_config.get(k, 0.0)
            if weight > 0:
                if k == 'flow_warp':
                    losses_cycle[k] = self.loss_dict[k](
                        mesh_src,
                        mesh_dst,
                        camera_src,
                        camera_dst,
                        image_src,  # N, H, W, C
                        image_dst,
                    ) * weight

        losses = merge_loss_dict(losses_dst, losses_src, losses_cycle)
        return losses
