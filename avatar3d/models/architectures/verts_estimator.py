import torch
import math
from typing import Optional, Tuple, Union
import numpy as np
from pytorch3d.structures import Meshes

from mmhuman3d.models.necks.builder import build_neck
from mmhuman3d.models.architectures.base_architecture import BaseArchitecture
from mmhuman3d.utils.geometry import batch_rodrigues

from avatar3d.utils.torch_utils import dict2numpy
from avatar3d.structures.meshes.utils import MeshSampler
from avatar3d.cameras.builder import build_cameras
from avatar3d.models.extractors.builder import build_extractor
from avatar3d.models.visualizers.builder import build_visualizer
from avatar3d.models.body_models.mappings import get_keypoint_idx, convert_kps
from avatar3d.models.body_models.builder import build_body_model
from avatar3d.models.losses.builder import build_loss
from avatar3d.models.heads.builder import build_head
from avatar3d.render.builder import build_renderer
from avatar3d.models.backbones.builder import build_backbone
from .perspectivte_mesh_estimator import PersepectiveMeshEstimator
from avatar3d.cameras.utils import (
    pred_cam_to_transl, merge_cam_to_full_transl,
    estimate_cam_weakperspective_batch, pred_cam_to_full_transl,
    pred_transl_to_pred_cam, project_points_pred_cam, full_transl_to_pred_cam,
    estimate_transl_weakperspective_batch, project_points_focal_length_pixel)


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a singleq
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class VertsEstimator(PersepectiveMeshEstimator):
    """BodyModelEstimator Architecture.

    Args:
        backbone (dict | None, optional): Backbone config dict. Default: None.
        neck (dict | None, optional): Neck config dict. Default: None
        head (dict | None, optional): Regressor config dict. Default: None.
        disc (dict | None, optional): Discriminator config dict.
            Default: None.
        registration (dict | None, optional): Registration config dict.
            Default: None.
        body_model_train (dict | None, optional): SMPL config dict during
            training. Default: None.
        body_model_test (dict | None, optional): SMPL config dict during
            test. Default: None.
        convention (str, optional): Keypoints convention. Default: "human_data"
        loss_transl_z
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(
        self,
        extractor: Optional[Union[dict, None]] = None,
        backbone: Optional[Union[dict, None]] = None,
        neck: Optional[Union[dict, None]] = None,
        verts_head: Optional[Union[dict, None]] = None,
        uv_renderer: Optional[Union[dict, None]] = None,
        depth_renderer: Optional[Union[dict, None]] = None,
        head_keys: Union[list, tuple] = (),
        pred_kp3d: bool = True,
        test_joints3d: bool = True,
        extractor_key: str = 'img',
        f_ablation: bool = False,
        resolution: Union[int, Tuple[int, int]] = (224, 224),
        body_model_train: Optional[Union[dict, None]] = None,
        body_model_test: Optional[Union[dict, None]] = None,
        mesh_sampler: Optional[Union[dict, None]] = None,
        convention: Optional[str] = 'human_data',
        convention_pred: Optional[str] = 'h36m_eval_j14_smpl',
        freeze_modules: Tuple[str] = (),
        loss_keypoints2d: Optional[Union[dict, None]] = None,
        loss_joints2d: Optional[Union[dict, None]] = None,
        loss_keypoints3d: Optional[Union[dict, None]] = None,
        loss_joints3d: Optional[Union[dict, None]] = None,
        loss_vertex: Optional[Union[dict, None]] = None,
        loss_vertex_sub1: Optional[Union[dict, None]] = None,
        loss_vertex_sub2: Optional[Union[dict, None]] = None,
        loss_camera: Optional[Union[dict, None]] = None,
        ##
        loss_transl_z: Optional[Union[dict, None]] = None,
        loss_iuv: Optional[Union[dict, None]] = None,
        loss_distortion_img: Optional[Union[dict, None]] = None,
        loss_image_grad_u: Optional[Union[dict, None]] = None,
        loss_image_grad_v: Optional[Union[dict, None]] = None,
        loss_wrapped_distortion: Optional[Union[dict, None]] = None,
        ###
        loss_smpl_pose: Optional[Union[dict, None]] = None,
        loss_smpl_betas: Optional[Union[dict, None]] = None,
        loss_vertex_smpl: Optional[Union[dict, None]] = None,
        loss_keypoints3d_smpl: Optional[Union[dict, None]] = None,
        init_cfg: Optional[Union[list, dict, None]] = None,
        visualizer: Optional[Union[int, None]] = None,
    ):
        super(VertsEstimator, self).__init__(init_cfg)

        self.extractor = build_extractor(extractor)
        self.extractor_key = extractor_key
        self.freeze_modules = freeze_modules

        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.verts_head = build_head(verts_head)
        self.head_keys = head_keys
        self.depth_renderer = build_renderer(depth_renderer)
        self.uv_renderer = build_renderer(uv_renderer)

        self.body_model_train = build_body_model(body_model_train)
        self.body_model_test = build_body_model(body_model_test)
        self.mesh_sampler = MeshSampler(**mesh_sampler)
        self.resolution = resolution
        self.f_ablation = f_ablation

        self.pred_kp3d = pred_kp3d
        self.test_joints3d = test_joints3d
        self.convention = convention
        self.convention_pred = convention_pred

        self.loss_keypoints2d = build_loss(loss_keypoints2d)
        self.loss_joints2d = build_loss(loss_joints2d)
        self.loss_keypoints3d = build_loss(loss_keypoints3d)
        self.loss_joints3d = build_loss(loss_joints3d)

        self.loss_vertex = build_loss(loss_vertex)
        self.loss_vertex_sub1 = build_loss(loss_vertex_sub1)
        self.loss_vertex_sub2 = build_loss(loss_vertex_sub2)

        self.loss_transl_z = build_loss(loss_transl_z)
        self.loss_iuv = build_loss(loss_iuv)
        self.loss_distortion_img = build_loss(loss_distortion_img)
        self.loss_image_grad_v = build_loss(loss_image_grad_v)
        self.loss_image_grad_u = build_loss(loss_image_grad_u)
        self.loss_wrapped_distortion = build_loss(loss_wrapped_distortion)
        ######
        self.loss_camera = build_loss(loss_camera)
        self.loss_smpl_pose = build_loss(loss_smpl_pose)
        self.loss_smpl_betas = build_loss(loss_smpl_betas)
        self.loss_keypoints3d_smpl = build_loss(loss_keypoints3d_smpl)
        self.loss_vertex_smpl = build_loss(loss_vertex_smpl)
        ###
        self.visualizer = build_visualizer(visualizer)

        set_requires_grad(self.body_model_train, False)
        set_requires_grad(self.body_model_test, False)

    def train_step(self, data_batch, optimizer, **kwargs):
        """Train step function.

        In this function, the detector will finish the train step following
        the pipeline:
        1. get fake and real SMPL parameters
        2. optimize discriminator (if have)
        3. optimize generator
        If `self.train_cfg.disc_step > 1`, the train step will contain multiple
        iterations for optimizing discriminator with different input data and
        only one iteration for optimizing generator after `disc_step`
        iterations for discriminator.
        Args:
            data_batch (torch.Tensor): Batch of data as input.
            optimizer (dict[torch.optim.Optimizer]): Dict with optimizers for
                generator and discriminator (if have).
        Returns:
            outputs (dict): Dict with loss, information for logger,
            the number of samples.
        """

        predictions = dict()
        for name in self.freeze_modules:
            for parameter in getattr(self, name).parameters():
                parameter.requires_grad = False

        if self.extractor:
            self.extractor.eval()
            for parameter in self.extractor.parameters():
                parameter.requires_grad = False
            extracted_data = self.extractor(data_batch[self.extractor_key])
        else:
            extracted_data = dict()

        data_batch.update(extracted_data)

        features = self.backbone(data_batch['img'])

        if self.neck is not None:
            features = self.neck(features)

        if self.verts_head is not None:
            predictions_verts = self.verts_head(features)
        else:
            predictions_verts = dict()
        predictions.update(predictions_verts)

        targets = self.prepare_targets(data_batch)

        losses = self.compute_losses(predictions, targets)
        for k, v in losses.items():
            losses[k] = v.float()

        loss, log_vars = self._parse_losses(losses)

        if self.backbone is not None:
            optimizer['backbone'].zero_grad()
        if self.neck is not None:
            optimizer['neck'].zero_grad()
        if self.verts_head is not None:
            optimizer['verts_head'].zero_grad()

        loss.backward()

        if self.backbone is not None:
            optimizer['backbone'].step()
        if self.neck is not None:
            optimizer['neck'].step()
        if self.verts_head is not None:
            optimizer['verts_head'].step()

        outputs = dict(loss=loss,
                       log_vars=log_vars,
                       num_samples=len(next(iter(data_batch.values()))))
        return outputs

    def compute_losses(self, predictions: dict, targets: dict):
        """Compute losses."""
        gt_keypoints2d = targets['keypoints2d']
        gt_keypoints3d = targets['keypoints3d']
        batch_size = gt_keypoints2d.shape[0]

        if 'has_keypoints3d' in targets:
            has_keypoints3d = targets['has_keypoints3d'].squeeze(-1)
        else:
            has_keypoints3d = None

        if self.verts_head is not None:
            pred_vertices_sub1 = predictions['pred_vertices_sub1']
            pred_vertices_sub2 = predictions.get('pred_vertices_sub2', None)

            if 'pred_vertices' in predictions:
                pred_vertices = predictions['pred_vertices']
            else:
                pred_vertices = self.mesh_sampler.upsample(pred_vertices_sub1,
                                                           n1=1,
                                                           n2=0)

            # pred_pose N, 24, 3, 3
            if self.body_model_train is not None:
                if self.pred_kp3d:
                    pred_keypoints3d = predictions['pred_keypoints3d']
                    pred_keypoints3d, conf = convert_kps(
                        pred_keypoints3d, self.convention_pred,
                        self.convention)
                    pred_keypoints3d = torch.cat([
                        pred_keypoints3d,
                        conf.view(1, -1, 1).repeat(batch_size, 1, 1)
                    ], -1)
                else:
                    pred_keypoints3d, conf = self.body_model_train.forward_joints(
                        dict(vertices=pred_vertices))
                    pred_keypoints3d = torch.cat([
                        pred_keypoints3d,
                        conf.view(1, -1, 1).repeat(batch_size, 1, 1)
                    ], -1)
                    has_keypoints3d = None

        # # TODO: temp. Should we multiply confs here?
        # pred_keypoints3d_mask = pred_output['joint_mask']
        # keypoints3d_mask = keypoints3d_mask * pred_keypoints3d_mask
        """Compute losses."""

        has_smpl = targets['has_smpl'].view(batch_size, -1)

        pred_model_joints = self.body_model_train.forward_joints(
            dict(vertices=pred_vertices))[0]

        # gt_pose N, 72
        if self.body_model_train is not None:
            gt_body_pose = targets['smpl_body_pose'].float()
            gt_betas = targets['smpl_betas'].float()
            gt_global_orient = targets['smpl_global_orient'].float()
            gt_pose = torch.cat(
                [targets['smpl_global_orient'][:, None], gt_body_pose], 1)
            gt_output = self.body_model_train(
                betas=gt_betas,
                body_pose=gt_body_pose.float(),
                global_orient=gt_global_orient,
                num_joints=gt_keypoints2d.shape[1])
            gt_vertices = gt_output['vertices']
            gt_model_joints = gt_output['joints']
            # gt_model_joint_mask = gt_output['joint_mask']

        if 'has_keypoints2d' in targets:
            has_keypoints2d = targets['has_keypoints2d'].squeeze(-1)
        else:
            has_keypoints2d = None

        ################################################
        ################################################

        losses = {}
        if self.loss_keypoints2d is not None:
            gt_keypoints2d = targets['keypoints2d']

            pred_cam = predictions['pred_cam']
            gt_focal_length = 5000

            if self.f_ablation:
                gt_focal_length = torch.ones_like(
                    targets['orig_focal_length'].float()).view(-1) * 5000
                scale = targets['scale'][:, 0].float()
                has_f_ids = torch.where(targets['has_focal_length'] > 0)[0]
                has_K_ids = torch.where(targets['has_K'] > 0)[0]
                gt_focal_length[has_f_ids] = targets['orig_focal_length'].view(
                    -1)[has_f_ids] * 224 / scale[has_f_ids]
                gt_focal_length[has_K_ids] = targets['K'][
                    has_K_ids, 0, 0].float() * 224 / scale[has_f_ids]

            losses['keypoints2d_loss'] = self.compute_keypoints2d_loss(
                pred_keypoints3d,
                pred_cam,
                gt_keypoints2d,
                has_keypoints2d=has_keypoints2d,
                focal_length=gt_focal_length,
            )

        if self.loss_joints2d is not None:
            gt_keypoints2d = targets['keypoints2d']

            pred_cam = predictions['pred_cam']

            gt_focal_length = 5000
            conf = torch.ones_like(pred_model_joints)[..., :1]
            losses['joints2d_loss'] = self.compute_joints2d_loss(
                torch.cat([pred_model_joints, conf], -1),
                pred_cam,
                gt_keypoints2d,
                has_keypoints2d=has_keypoints2d,
                focal_length=gt_focal_length,
            )

        if self.loss_keypoints3d is not None:

            # gt_keypoints3d = gt_model_joints
            # gt_keypoints3d = torch.cat([
            #     gt_keypoints3d,
            #     gt_model_joint_mask.view(1, -1, 1).repeat_interleave(
            #         batch_size, 0).float()
            # ], -1)

            batch_size = gt_keypoints3d.shape[0]
            losses['keypoints3d_loss'] = self.compute_keypoints3d_loss(
                pred_keypoints3d,
                gt_keypoints3d,
                has_keypoints3d=has_keypoints3d)

        if self.loss_joints3d is not None:

            # gt_keypoints3d = gt_model_joints
            # gt_keypoints3d = torch.cat([
            #     gt_keypoints3d,
            #     gt_model_joint_mask.view(1, -1, 1).repeat_interleave(
            #         batch_size, 0).float()
            # ], -1)

            batch_size = gt_keypoints3d.shape[0]
            conf = torch.ones_like(pred_model_joints)[..., :1]
            losses['joints3d_loss'] = self.compute_joints3d_loss(
                torch.cat([pred_model_joints, conf], -1),
                gt_keypoints3d,
                has_keypoints3d=has_keypoints3d)

        if self.loss_vertex is not None:
            losses['vertex_loss'] = self.compute_vertex_loss(
                pred_vertices,
                gt_vertices,
                gt_model_joints,
                has_smpl,
                down_scale=1,
                loss_func=self.loss_vertex)

        if self.loss_vertex_sub1 is not None:
            losses['vertex_loss_sub1'] = self.compute_vertex_loss(
                pred_vertices_sub1,
                gt_vertices,
                gt_model_joints,
                has_smpl,
                down_scale=4,
                loss_func=self.loss_vertex_sub1)

        if self.loss_vertex_sub2 is not None:
            losses['vertex_loss_sub2'] = self.compute_vertex_loss(
                pred_vertices_sub2,
                gt_vertices,
                gt_model_joints,
                has_smpl,
                down_scale=16,
                loss_func=self.loss_vertex_sub2)

        if self.loss_camera is not None:
            pred_cam = predictions['pred_cam']
            losses['camera_loss'] = self.compute_camera_loss(pred_cam)

        if self.loss_smpl_betas is not None:
            pred_betas = predictions['pred_shape']
            losses['smpl_betas_loss'] = self.compute_smpl_betas_loss(
                pred_betas, gt_betas, has_smpl)

        if self.loss_smpl_pose is not None:
            pred_betas = predictions['pred_shape'].view(-1, 10)
            pred_pose = predictions['pred_pose']
            pred_body_pose = pred_pose.view(-1, 24, 3, 3)[:, 1:]
            pred_global_orient = pred_pose.view(-1, 24, 3, 3)[:, :1]
            if self.body_model_train is not None:
                pred_output = self.body_model_train(
                    betas=pred_betas,
                    body_pose=pred_body_pose,
                    global_orient=pred_global_orient,
                    pose2rot=False,
                    num_joints=gt_keypoints2d.shape[1])
                pred_keypoints3d_smpl = pred_output['joints']
                pred_vertices_smpl = pred_output['vertices']

            losses['smpl_pose_loss'] = self.compute_smpl_pose_loss(
                pred_pose, gt_pose.float(), has_smpl)

        if self.loss_vertex_smpl is not None:
            losses['vertex_smpl_loss'] = self.compute_smpl_vertex_loss(
                pred_vertices_smpl, gt_vertices, has_smpl)

        if self.loss_keypoints3d_smpl is not None:
            losses[
                'keypoints3d_smpl_loss'] = self.compute_smpl_keypoints3d_loss(
                    pred_keypoints3d_smpl, gt_keypoints3d, has_keypoints3d)
        return losses

    def forward_test(self, **data_batch):
        """Defines the computation performed at every call when testing."""

        for name in self.freeze_modules:
            for parameter in getattr(self, name).parameters():
                parameter.requires_grad = False
        predictions = dict()
        if self.extractor:
            self.extractor.eval()
            for parameter in self.extractor.parameters():
                parameter.requires_grad = False
            extracted_data = self.extractor(data_batch['img'])
        else:
            extracted_data = dict()

        data_batch.update(extracted_data)
        batch_size = data_batch['img'].shape[0]

        features = self.backbone(data_batch['img'])

        if self.neck is not None:
            features = self.neck(features)

        if self.verts_head is not None:
            head_data = []
            for key in self.head_keys:
                head_data.append(data_batch[key])
            predictions_verts = self.verts_head(features, *head_data)
        else:
            predictions_verts = dict()

        predictions.update(predictions_verts)
        predictions.update(extracted_data)

        pred_vertices_sub1 = predictions['pred_vertices_sub1']
        if 'pred_vertices' in predictions:
            pred_vertices = predictions['pred_vertices']
        else:
            pred_vertices = self.mesh_sampler.upsample(pred_vertices_sub1,
                                                       n1=1,
                                                       n2=0)

        pred_joints3d = self.body_model_test.forward_joints(
            dict(vertices=pred_vertices))[0]
        right_hip_idx = get_keypoint_idx('right_hip_extra',
                                         self.body_model_test.keypoint_dst)
        left_hip_idx = get_keypoint_idx('left_hip_extra',
                                        self.body_model_test.keypoint_dst)

        pred_pelvis = (pred_joints3d[:, right_hip_idx, :] +
                       pred_joints3d[:, left_hip_idx, :]) / 2

        if self.loss_keypoints2d is not None:
            if self.pred_kp3d:
                pred_pelvis_offset = (
                    predictions['pred_keypoints3d'][:, right_hip_idx, :] +
                    predictions['pred_keypoints3d'][:, left_hip_idx, :]) / 2
            else:
                pred_pelvis_offset = pred_pelvis
        elif self.loss_joints2d is not None:
            pred_pelvis_offset = pred_pelvis

        if not self.test_joints3d:
            pred_keypoints3d = predictions['pred_keypoints3d']

            pred_keypoints3d, _ = convert_kps(
                pred_keypoints3d,
                self.convention_pred,
                self.body_model_test.keypoint_dst,
                approximate=False)
        else:
            pred_keypoints3d = pred_joints3d - pred_pelvis[:, None, :]

        pred_cam = predictions['pred_cam']

        center, scale = data_batch['center'], data_batch['scale'][:, 0]
        pred_transl = pred_cam_to_transl(pred_cam, 5000, 224)
        pred_transl = pred_pelvis_offset + pred_transl
        # pred_transl = pred_pelvis + pred_transl
        pred_cam = pred_transl_to_pred_cam(pred_transl, 5000, 224)

        pred_focal_length = 5000 / 112.

        predictions.update(pred_transl=pred_cam_to_transl(pred_cam, 5000, 224))
        predictions.update(full_transl=pred_cam_to_full_transl(
            pred_cam, center, scale, data_batch['ori_shape'], 5000 / 224. *
            scale).float())
        predictions.update(pred_focal_length=pred_focal_length)
        predictions.update(orig_focal_length=pred_focal_length * scale / 2)

        orig_wh = torch.cat(
            [data_batch['ori_shape'][:, 1], data_batch['ori_shape'][:, 0]], -1)
        origin_focal_length = predictions['orig_focal_length']
        if isinstance(origin_focal_length, torch.Tensor):
            origin_focal_length = origin_focal_length.view(-1).float()
        pred_keypoints2d = project_points_focal_length_pixel(
            pred_keypoints3d,
            translation=predictions['full_transl'].float(),
            focal_length=origin_focal_length,
            camera_center=data_batch['ori_shape'].float() / 2)
        pred_keypoints2d = pred_keypoints2d / orig_wh.view(batch_size, 1, 2)
        if 'keypoints2d' in data_batch:
            gt_keypoints2d = data_batch['keypoints2d']
            origin_keypoints2d = gt_keypoints2d.clone()[..., :2]

            origin_keypoints2d = (origin_keypoints2d - 112
                                  ) / 224 * scale[:, None, None] + center.view(
                                      -1, 1, 2)
            origin_keypoints2d = torch.cat([
                origin_keypoints2d / orig_wh.view(batch_size, 1, 2),
                gt_keypoints2d[..., 2:3]
            ], -1)

            predictions.update(origin_keypoints2d=origin_keypoints2d)
        if self.loss_keypoints2d is not None:
            if self.pred_kp3d:
                pred_vertices_vis = pred_vertices - pred_pelvis[:,
                                                                None] - pred_pelvis_offset[:,
                                                                                           None]
            else:
                pred_vertices_vis = pred_vertices - pred_pelvis[:, None]
        elif self.loss_joints2d is not None:
            pred_vertices_vis = pred_vertices - pred_pelvis[:, None]
        predictions.update(img=data_batch['img'],
                           sample_idx=data_batch['sample_idx'],
                           ori_shape=data_batch['ori_shape'],
                           pred_keypoints2d=pred_keypoints2d,
                           pred_vertices=pred_vertices_vis,
                           img_metas=data_batch['img_metas'])

        if self.visualizer is not None and self.vis:
            self.visualizer(predictions)

        if self.miou or self.pmiou:
            test_res = 224

            K = data_batch['K']
            gt_transl = data_batch['smpl_transl']
            ori_shape = data_batch['ori_shape']
            orig_focal_length = data_batch['orig_focal_length'].float().view(
                -1)
            px = ori_shape[:, 1].float() / 2
            py = ori_shape[:, 0].float() / 2
            has_K_ids = torch.where(data_batch['has_K'] == 1)[0]
            px[has_K_ids] = K[has_K_ids, 0, 2].float()
            py[has_K_ids] = K[has_K_ids, 1, 2].float()

            gt_body_pose = data_batch['smpl_body_pose']
            gt_global_orient = data_batch['smpl_global_orient']
            gt_betas = data_batch['smpl_betas'].float()
            gt_output = self.body_model_test(
                betas=gt_betas,
                body_pose=gt_body_pose.float(),
                global_orient=gt_global_orient.float())
            gt_vertices = gt_output['vertices']

        if self.miou or self.pmiou:
            test_res = 224
            pred_focal_length = predictions['pred_focal_length']
            K = data_batch['K']
            gt_transl = data_batch['smpl_transl']
            ori_shape = data_batch['ori_shape']
            orig_focal_length = data_batch['orig_focal_length'].float().view(
                -1)
            px = ori_shape[:, 1].float() / 2
            py = ori_shape[:, 0].float() / 2
            has_K_ids = torch.where(data_batch['has_K'] == 1)[0]
            px[has_K_ids] = K[has_K_ids, 0, 2].float()
            py[has_K_ids] = K[has_K_ids, 1, 2].float()

            gt_body_pose = data_batch['smpl_body_pose']
            gt_global_orient = data_batch['smpl_global_orient']
            gt_betas = data_batch['smpl_betas'].float()
            gt_output = self.body_model_test(
                betas=gt_betas,
                body_pose=gt_body_pose.float(),
                global_orient=gt_global_orient.float())
            gt_vertices = gt_output['vertices']

            if self.pmiou:

                gt_mask = self.render_segmask(
                    vertices=gt_vertices,
                    transl=gt_transl,
                    center=center,
                    scale=scale,
                    focal_length_ndc=orig_focal_length.float() /
                    scale.float() * 2,
                    px=px,
                    py=py,
                    img_res=test_res)

                pred_mask = self.render_segmask(
                    vertices=pred_vertices_vis,
                    transl=predictions['full_transl'].float(),
                    center=center,
                    scale=scale,
                    focal_length_ndc=pred_focal_length,
                    px=px,
                    py=py,
                    img_res=test_res)

                # import cv2

                # from avatar3d.utils.torch_utils import image_tensor2numpy

                # im = image_tensor2numpy(data_batch['img'][index].permute(
                #     1, 2, 0))
                # cv2.imwrite(f'{index}_rgb.png', im)

                # im = image_tensor2numpy(pred_mask[index, 0])
                # cv2.imwrite(f'{index}_pred.png', im)
                # im = image_tensor2numpy(gt_mask[index, 0])
                # cv2.imwrite(f'{index}_gt.png', im)

                gt_mask_ = torch.zeros(batch_size, 24, gt_mask.shape[-1],
                                       gt_mask.shape[-1]).to(gt_mask.device)
                for i in range(24):
                    gt_mask_[:, i:i + 1] = (gt_mask == i + 1)

                pred_mask_ = torch.zeros(batch_size, 24, pred_mask.shape[-1],
                                         pred_mask.shape[-1]).to(
                                             pred_mask.device)
                for i in range(24):
                    pred_mask_[:, i:i + 1] = (pred_mask == i + 1)

                gt_mask_ = gt_mask_.view(batch_size, -1)
                pred_mask_ = pred_mask_.view(batch_size, -1)
                batch_u = (((gt_mask_ + pred_mask_) > 0) * 1.0).sum(1)
                batch_i = (gt_mask_ * pred_mask_).sum(1)
                batch_piou = batch_i / batch_u

            if self.miou:

                gt_mask = self.render_segmask(
                    vertices=gt_vertices,
                    transl=gt_transl,
                    center=center,
                    scale=scale,
                    focal_length_ndc=orig_focal_length.float() /
                    scale.float() * 2,
                    px=px,
                    py=py,
                    img_res=test_res)

                pred_mask = self.render_segmask(
                    vertices=pred_vertices_vis,
                    transl=predictions['full_transl'].float(),
                    center=center,
                    scale=scale,
                    focal_length_ndc=pred_focal_length,
                    px=px,
                    py=py,
                    img_res=test_res)
                gt_mask = gt_mask.view(batch_size, -1)
                pred_mask = pred_mask.view(batch_size, -1)
                batch_u = (((gt_mask + pred_mask) > 0) * 1.0).sum(1)
                batch_i = ((gt_mask * pred_mask) > 0).sum(1)
                batch_iou = batch_i / batch_u
        else:
            batch_iou = torch.zeros(batch_size, 1)
            batch_piou = torch.zeros(batch_size, 1)

        all_preds = {}
        all_preds['batch_miou'] = batch_iou * 100
        all_preds['batch_pmiou'] = batch_piou * 100
        all_preds['estimate_verts'] = True
        all_preds['keypoints_3d'] = pred_joints3d
        all_preds['smpl_pose'] = torch.zeros(batch_size, 72)
        all_preds['smpl_beta'] = torch.zeros(batch_size, 10)
        all_preds['camera'] = pred_cam
        all_preds['vertices'] = pred_vertices - pred_pelvis[:, None]
        image_path = []
        for img_meta in data_batch['img_metas']:
            image_path.append(img_meta['image_path'])
        all_preds['image_path'] = image_path
        all_preds['image_idx'] = data_batch['sample_idx']

        all_preds['pred_keypoints2d'] = predictions['pred_keypoints2d']

        all_preds['pred_keypoints3d'] = pred_keypoints3d

        all_preds['origin_keypoints2d'] = predictions['origin_keypoints2d']
        pred_output = dict2numpy(all_preds)

        return pred_output

    def forward_train(self, **kwargs):
        """Forward function for general training.

        For mesh estimation, we do not use this interface.
        """
        raise NotImplementedError('This interface should not be used in '
                                  'current training schedule. Please use '
                                  '`train_step` for training.')

    def compute_smpl_vertex_loss(self, pred_vertices: torch.Tensor,
                                 gt_vertices: torch.Tensor,
                                 has_smpl: torch.Tensor):
        return super().compute_vertex_loss(pred_vertices, gt_vertices,
                                           has_smpl)

    def compute_smpl_keypoints3d_loss(
            self,
            pred_keypoints3d: torch.Tensor,
            gt_keypoints3d: torch.Tensor,
            has_keypoints3d: Optional[torch.Tensor] = None):
        return super().compute_keypoints3d_loss(pred_keypoints3d,
                                                gt_keypoints3d,
                                                has_keypoints3d)

    def compute_focal_length_loss(self, pred_f, gt_f):
        return self.loss_focal_length(pred_f, gt_f)

    def compute_keypoints3d_loss(
            self,
            pred_keypoints3d: torch.Tensor,
            gt_keypoints3d: torch.Tensor,
            has_keypoints3d: Optional[torch.Tensor] = None):
        """Compute loss for 3d keypoints."""
        assert pred_keypoints3d.shape[-1] == 4
        keypoints3d_conf = gt_keypoints3d[:, :, 3].float().unsqueeze(
            -1) * pred_keypoints3d[:, :, 3].float().unsqueeze(-1)
        keypoints3d_conf = keypoints3d_conf.repeat(1, 1, 3)
        pred_keypoints3d = pred_keypoints3d[:, :, :3].float()
        gt_keypoints3d = gt_keypoints3d[:, :, :3].float()

        # currently, only mpi_inf_3dhp and h36m have 3d keypoints
        # both datasets have right_hip_extra and left_hip_extra
        right_hip_idx = get_keypoint_idx('right_hip_extra', self.convention)
        left_hip_idx = get_keypoint_idx('left_hip_extra', self.convention)
        # right_hip_idx = get_keypoint_idx('right_hip', self.convention)
        # left_hip_idx = get_keypoint_idx('left_hip', self.convention)
        gt_pelvis = (gt_keypoints3d[:, right_hip_idx, :] +
                     gt_keypoints3d[:, left_hip_idx, :]) / 2

        # pred_pelvis = (pred_keypoints3d[:, right_hip_idx, :] +
        #                pred_keypoints3d[:, left_hip_idx, :]) / 2

        gt_keypoints3d = gt_keypoints3d - gt_pelvis[:, None, :]
        # pred_keypoints3d = pred_keypoints3d - pred_pelvis[:, None, :]
        loss = self.loss_keypoints3d(pred_keypoints3d,
                                     gt_keypoints3d,
                                     reduction_override='none')

        # If has_keypoints3d is not None, then computes the losses on the
        # instances that have ground-truth keypoints3d.
        # But the zero confidence keypoints will be included in mean.
        # Otherwise, only compute the keypoints3d
        # which have positive confidence.

        # has_keypoints3d is None when the key has_keypoints3d
        # is not in the datasets
        if has_keypoints3d is None:

            valid_pos = keypoints3d_conf > 0
            if keypoints3d_conf[valid_pos].numel() == 0:
                return torch.Tensor([0]).type_as(gt_keypoints3d)
            loss = torch.sum(loss * keypoints3d_conf)
            loss /= keypoints3d_conf[valid_pos].numel()
        else:

            keypoints3d_conf = keypoints3d_conf[has_keypoints3d == 1]
            if keypoints3d_conf.shape[0] == 0:
                return torch.Tensor([0]).type_as(gt_keypoints3d)
            loss = loss[has_keypoints3d == 1]
            loss = (loss * keypoints3d_conf).mean()
        return loss

    def compute_joints3d_loss(self,
                              pred_keypoints3d: torch.Tensor,
                              gt_keypoints3d: torch.Tensor,
                              has_keypoints3d: Optional[torch.Tensor] = None):
        """Compute loss for 3d keypoints."""
        assert pred_keypoints3d.shape[-1] == 4
        keypoints3d_conf = gt_keypoints3d[:, :, 3].float().unsqueeze(
            -1) * pred_keypoints3d[:, :, 3].float().unsqueeze(-1)
        keypoints3d_conf = keypoints3d_conf.repeat(1, 1, 3)
        pred_keypoints3d = pred_keypoints3d[:, :, :3].float()
        gt_keypoints3d = gt_keypoints3d[:, :, :3].float()

        # currently, only mpi_inf_3dhp and h36m have 3d keypoints
        # both datasets have right_hip_extra and left_hip_extra
        right_hip_idx = get_keypoint_idx('right_hip_extra', self.convention)
        left_hip_idx = get_keypoint_idx('left_hip_extra', self.convention)
        # right_hip_idx = get_keypoint_idx('right_hip', self.convention)
        # left_hip_idx = get_keypoint_idx('left_hip', self.convention)
        gt_pelvis = (gt_keypoints3d[:, right_hip_idx, :] +
                     gt_keypoints3d[:, left_hip_idx, :]) / 2

        # pred_pelvis = (pred_keypoints3d[:, right_hip_idx, :] +
        #                pred_keypoints3d[:, left_hip_idx, :]) / 2

        gt_keypoints3d = gt_keypoints3d - gt_pelvis[:, None, :]
        # pred_keypoints3d = pred_keypoints3d - pred_pelvis[:, None, :]
        loss = self.loss_joints3d(pred_keypoints3d,
                                  gt_keypoints3d,
                                  reduction_override='none')

        # If has_keypoints3d is not None, then computes the losses on the
        # instances that have ground-truth keypoints3d.
        # But the zero confidence keypoints will be included in mean.
        # Otherwise, only compute the keypoints3d
        # which have positive confidence.

        # has_keypoints3d is None when the key has_keypoints3d
        # is not in the datasets
        if has_keypoints3d is None:

            valid_pos = keypoints3d_conf > 0
            if keypoints3d_conf[valid_pos].numel() == 0:
                return torch.Tensor([0]).type_as(gt_keypoints3d)
            loss = torch.sum(loss * keypoints3d_conf)
            loss /= keypoints3d_conf[valid_pos].numel()
        else:

            keypoints3d_conf = keypoints3d_conf[has_keypoints3d == 1]
            if keypoints3d_conf.shape[0] == 0:
                return torch.Tensor([0]).type_as(gt_keypoints3d)
            loss = loss[has_keypoints3d == 1]
            loss = (loss * keypoints3d_conf).mean()
        return loss

    def compute_keypoints2d_loss(
            self,
            pred_keypoints3d: torch.Tensor,
            pred_cam: torch.Tensor,
            gt_keypoints2d: torch.Tensor,
            img_res: Optional[int] = 224,
            focal_length: Optional[int] = 5000,
            has_keypoints2d: Optional[torch.Tensor] = None):
        """Compute loss for 2d keypoints."""

        assert pred_keypoints3d.shape[-1] == 4
        keypoints2d_conf = gt_keypoints2d[:, :, 2].float().unsqueeze(
            -1) * pred_keypoints3d[:, :, 3].float().unsqueeze(-1)
        keypoints2d_conf = keypoints2d_conf.repeat(1, 1, 2)
        gt_keypoints2d = gt_keypoints2d[:, :, :2].float()
        pred_keypoints3d = pred_keypoints3d[..., :3]
        pred_keypoints2d = project_points_pred_cam(pred_keypoints3d,
                                                   pred_cam,
                                                   focal_length=focal_length,
                                                   img_res=img_res)
        # Normalize keypoints to [-1,1]
        # The coordinate origin of pred_keypoints_2d is
        # the center of the input image.
        pred_keypoints2d = 2 * pred_keypoints2d / (img_res - 1) - 1
        # The coordinate origin of gt_keypoints_2d is
        # the top left corner of the input image.
        gt_keypoints2d = 2 * gt_keypoints2d / (img_res - 1) - 1

        loss = self.loss_keypoints2d(
            pred_keypoints2d,
            gt_keypoints2d,
            reduction_override='none',
        )

        if has_keypoints2d is None:
            valid_pos = keypoints2d_conf > 0
            if keypoints2d_conf[valid_pos].numel() == 0:
                return torch.Tensor([0]).type_as(gt_keypoints2d)
            loss = torch.sum(loss * keypoints2d_conf)
            loss /= keypoints2d_conf[valid_pos].numel()
        else:
            keypoints2d_conf = keypoints2d_conf[has_keypoints2d == 1]
            if keypoints2d_conf.shape[0] == 0:
                return torch.Tensor([0]).type_as(gt_keypoints2d)
            loss = loss[has_keypoints2d == 1]
            loss = (loss * keypoints2d_conf).mean()

        return loss

    def compute_joints2d_loss(self,
                              pred_keypoints3d: torch.Tensor,
                              pred_cam: torch.Tensor,
                              gt_keypoints2d: torch.Tensor,
                              img_res: Optional[int] = 224,
                              focal_length: Optional[int] = 5000,
                              has_keypoints2d: Optional[torch.Tensor] = None):
        """Compute loss for 2d keypoints."""

        assert pred_keypoints3d.shape[-1] == 4
        keypoints2d_conf = gt_keypoints2d[:, :, 2].float().unsqueeze(
            -1) * pred_keypoints3d[:, :, 3].float().unsqueeze(-1)
        keypoints2d_conf = keypoints2d_conf.repeat(1, 1, 2)
        gt_keypoints2d = gt_keypoints2d[:, :, :2].float()
        pred_keypoints3d = pred_keypoints3d[..., :3]
        pred_keypoints2d = project_points_pred_cam(pred_keypoints3d,
                                                   pred_cam,
                                                   focal_length=focal_length,
                                                   img_res=img_res)
        # Normalize keypoints to [-1,1]
        # The coordinate origin of pred_keypoints_2d is
        # the center of the input image.
        pred_keypoints2d = 2 * pred_keypoints2d / (img_res - 1) - 1
        # The coordinate origin of gt_keypoints_2d is
        # the top left corner of the input image.
        gt_keypoints2d = 2 * gt_keypoints2d / (img_res - 1) - 1

        loss = self.loss_joints2d(
            pred_keypoints2d,
            gt_keypoints2d,
            reduction_override='none',
        )

        if has_keypoints2d is None:
            valid_pos = keypoints2d_conf > 0
            if keypoints2d_conf[valid_pos].numel() == 0:
                return torch.Tensor([0]).type_as(gt_keypoints2d)
            loss = torch.sum(loss * keypoints2d_conf)
            loss /= keypoints2d_conf[valid_pos].numel()
        else:
            keypoints2d_conf = keypoints2d_conf[has_keypoints2d == 1]
            if keypoints2d_conf.shape[0] == 0:
                return torch.Tensor([0]).type_as(gt_keypoints2d)
            loss = loss[has_keypoints2d == 1]
            loss = (loss * keypoints2d_conf).mean()

        return loss

    def compute_vertex_loss(self, pred_vertices: torch.Tensor,
                            gt_vertices: torch.Tensor,
                            gt_model_joints3d: torch.Tensor,
                            has_smpl: torch.Tensor, down_scale, loss_func):
        """Compute loss for vertices."""

        gt_vertices = gt_vertices.float()
        n1 = math.log(down_scale, 4)

        right_hip_idx = get_keypoint_idx('right_hip_extra', self.convention)
        left_hip_idx = get_keypoint_idx('right_hip_extra', self.convention)
        gt_pelvis = (gt_model_joints3d[:, right_hip_idx, :] +
                     gt_model_joints3d[:, left_hip_idx, :]) / 2

        if n1 > 0:
            gt_vertices = self.mesh_sampler.downsample(gt_vertices,
                                                       n1=0,
                                                       n2=int(n1))
        gt_vertices = gt_vertices - gt_pelvis[:, None, :]
        assert gt_vertices.shape[1] == pred_vertices.shape[1]
        batch_size = gt_vertices.shape[0]
        conf = has_smpl.float().view(batch_size, -1, 1)[:, :1]
        conf = conf.repeat(1, gt_vertices.shape[1], gt_vertices.shape[2])
        loss = loss_func(pred_vertices, gt_vertices, reduction_override='none')
        valid_pos = conf > 0
        if conf[valid_pos].numel() == 0:
            return torch.Tensor([0]).type_as(gt_vertices)
        loss = torch.sum(loss * conf) / conf[valid_pos].numel()
        return loss

    def compute_camera_loss(self, cameras: torch.Tensor):
        """Compute loss for predicted camera parameters."""
        loss = self.loss_camera(cameras)
        return loss

    def compute_smpl_pose_loss(self, pred_rotmat: torch.Tensor,
                               gt_pose: torch.Tensor, has_smpl: torch.Tensor):
        """Compute loss for smpl pose."""

        conf = has_smpl.float().view(-1)
        valid_pos = conf > 0
        if conf[valid_pos].numel() == 0:
            return torch.Tensor([0]).type_as(gt_pose)
        pred_rotmat = pred_rotmat[valid_pos]
        gt_pose = gt_pose[valid_pos]
        conf = conf[valid_pos]
        gt_rotmat = batch_rodrigues(gt_pose.view(-1, 3)).view(-1, 24, 3, 3)
        loss = self.loss_smpl_pose(pred_rotmat,
                                   gt_rotmat,
                                   reduction_override='none')
        loss = loss.view(loss.shape[0], -1).mean(-1)
        loss = torch.mean(loss * conf)
        return loss

    def compute_smpl_betas_loss(self, pred_betas: torch.Tensor,
                                gt_betas: torch.Tensor,
                                has_smpl: torch.Tensor):
        """Compute loss for smpl betas."""
        # conf = has_smpl.float().view(-1)
        # valid_pos = conf > 0
        conf = has_smpl.float().view(-1)
        valid_pos = conf > 0
        if conf[valid_pos].numel() == 0:
            return torch.Tensor([0]).type_as(gt_betas)
        pred_betas = pred_betas[valid_pos]
        gt_betas = gt_betas[valid_pos]
        conf = conf[valid_pos]
        loss = self.loss_smpl_betas(pred_betas,
                                    gt_betas,
                                    reduction_override='none')
        loss = loss.view(loss.shape[0], -1).mean(-1)
        loss = torch.mean(loss * conf)
        return loss
