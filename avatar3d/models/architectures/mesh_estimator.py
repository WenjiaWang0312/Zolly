from distutils.command.build import build
import shutil
from typing import Optional, Tuple, Union, Tuple
from avatar3d.utils.torch_utils import dict2numpy
import torch
import torch.nn.functional as F
import numpy as np
from pytorch3d.structures import Meshes

# from mmhuman3d.models.backbones.builder import build_backbone
from mmhuman3d.models.necks.builder import build_neck
from mmhuman3d.models.architectures.base_architecture import BaseArchitecture

from mmhuman3d.utils.geometry import batch_rodrigues
from avatar3d.models.extractors.builder import build_extractor
from avatar3d.models.visualizers.builder import build_visualizer
from avatar3d.cameras.builder import build_cameras
from avatar3d.models.body_models.mappings import convert_kps
from avatar3d.models.body_models.mappings import get_keypoint_idx
from avatar3d.render.builder import build_renderer
from avatar3d.models.body_models.builder import build_body_model
from avatar3d.models.losses.builder import build_loss
from avatar3d.models.heads.builder import build_head
from avatar3d.models.backbones.builder import build_backbone
from avatar3d.transforms.transform3d import rotmat_to_rot6d, ee_to_rotmat, aa_to_rotmat
from avatar3d.cameras.utils import (
    estimate_transl_weakperspective, pred_cam_to_transl,
    merge_cam_to_full_transl, estimate_cam_weakperspective,
    estimate_cam_weakperspective_batch, estimate_transl_weakperspective_batch,
    project_points_focal_length, pred_cam_to_full_transl,
    project_points_pred_cam, project_points_focal_length_pixel)


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class TestMeshEstimator(BaseArchitecture):
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
        hmr_head: Optional[Union[dict, None]] = None,
        iuvd_head: Optional[Union[dict, None]] = None,
        head_keys: Union[list, tuple] = (),
        extractor_key: str = 'img',
        resolution: Union[int, Tuple[int, int]] = (224, 224),
        body_model_train: Optional[Union[dict, None]] = None,
        body_model_test: Optional[Union[dict, None]] = None,
        convention: Optional[str] = 'human_data',
        uv_renderer: dict = None,
        depth_renderer: dict = None,
        use_d_weight: bool = False,
        freeze_modules: Tuple[str] = (),
        ###
        loss_keypoints2d_prompt: Optional[Union[dict, None]] = None,
        loss_keypoints2d_prompt_cam: Optional[Union[dict, None]] = None,
        loss_keypoints2d_cliff: Optional[Union[dict, None]] = None,
        loss_keypoints2d_spec: Optional[Union[dict, None]] = None,
        loss_keypoints2d_hmr: Optional[Union[dict, None]] = None,
        ###
        loss_joints3d: Optional[Union[dict, None]] = None,
        loss_keypoints3d: Optional[Union[dict, None]] = None,
        loss_vertex: Optional[Union[dict, None]] = None,
        loss_global_orient: Optional[Union[dict, None]] = None,
        loss_smpl_pose: Optional[Union[dict, None]] = None,
        loss_body_pose: Optional[Union[dict, None]] = None,
        loss_smpl_betas: Optional[Union[dict, None]] = None,
        loss_camera: Optional[Union[dict, None]] = None,
        ##
        loss_transl_z: Optional[Union[dict, None]] = None,
        loss_iuv: Optional[Union[dict, None]] = None,
        loss_distortion_img: Optional[Union[dict, None]] = None,
        loss_image_grad_u: Optional[Union[dict, None]] = None,
        loss_image_grad_v: Optional[Union[dict, None]] = None,
        loss_wrapped_distortion: Optional[Union[dict, None]] = None,
        ###
        init_cfg: Optional[Union[list, dict, None]] = None,
        visualizer: Optional[Union[int, None]] = None,
        use_pred_transl: bool = False,
    ):
        super(TestMeshEstimator, self).__init__(init_cfg)
        self.extractor = build_extractor(extractor)
        self.extractor_key = extractor_key
        self.use_pred_transl = use_pred_transl
        self.use_d_weight = use_d_weight

        self.iuvd_head = build_head(iuvd_head)
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.hmr_head = build_head(hmr_head)
        self.head_keys = head_keys

        self.body_model_train = build_body_model(body_model_train)
        self.body_model_test = build_body_model(body_model_test)
        self.depth_renderer = build_renderer(depth_renderer)
        self.uv_renderer = build_renderer(uv_renderer)
        self.resolution = resolution

        self.convention = convention
        self.freeze_modules = freeze_modules

        self.loss_keypoints2d_prompt_cam = build_loss(
            loss_keypoints2d_prompt_cam)
        self.loss_keypoints2d_prompt = build_loss(loss_keypoints2d_prompt)
        self.loss_keypoints2d_cliff = build_loss(loss_keypoints2d_cliff)
        self.loss_keypoints2d_spec = build_loss(loss_keypoints2d_spec)
        self.loss_keypoints2d_hmr = build_loss(loss_keypoints2d_hmr)
        self.loss_keypoints3d = build_loss(loss_keypoints3d)
        self.loss_joints3d = build_loss(loss_joints3d)

        self.loss_transl_z = build_loss(loss_transl_z)
        self.loss_iuv = build_loss(loss_iuv)
        self.loss_distortion_img = build_loss(loss_distortion_img)
        self.loss_image_grad_v = build_loss(loss_image_grad_v)
        self.loss_image_grad_u = build_loss(loss_image_grad_u)
        self.loss_wrapped_distortion = build_loss(loss_wrapped_distortion)

        self.loss_global_orient = build_loss(loss_global_orient)
        self.loss_vertex = build_loss(loss_vertex)
        self.loss_smpl_pose = build_loss(loss_smpl_pose)
        self.loss_body_pose = build_loss(loss_body_pose)
        self.loss_smpl_betas = build_loss(loss_smpl_betas)
        self.loss_camera = build_loss(loss_camera)

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

        if self.iuvd_head is not None:
            predictions_iuvd = self.iuvd_head(features)
        else:
            predictions_iuvd = dict()

        predictions.update(predictions_iuvd)

        if self.hmr_head is not None:
            head_data = []

            if 'gt_z' in self.head_keys:
                data_batch['gt_z'] = data_batch['smpl_transl'][..., 2:3]
            if 'pred_z' in self.head_keys:
                data_batch['pred_z'] = predictions['pred_z']
            if 'warped_d_img' in self.head_keys:
                pred_d_img = predictions['pred_d_img']
                pred_iuv_img = predictions['pred_iuv_img']
                uv_renderer = self.uv_renderer.to(pred_d_img.device)
                mask = self.uv_renderer.mask.to(pred_d_img.device)[None, None]
                data_batch['warped_d_img'] = uv_renderer.inverse_wrap(
                    pred_iuv_img, pred_d_img) * mask
            if 'warped_pose_feat' in self.head_keys:
                pose_feat = predictions['pose_feat']
                pred_iuv_img = predictions['pred_iuv_img']
                uv_renderer = self.uv_renderer.to(pred_d_img.device)
                mask = self.uv_renderer.mask.to(pred_d_img.device)[None, None]
                data_batch['warped_pose_feat'] = uv_renderer.inverse_wrap(
                    pred_iuv_img, pose_feat) * mask
            for key in self.head_keys:
                head_data.append(data_batch[key])
            predictions_hmr = self.hmr_head(features, *head_data)
        else:
            predictions_hmr = dict()

        predictions.update(predictions_hmr)
        targets = self.prepare_targets(data_batch)

        losses = self.compute_losses(predictions, targets)
        for k, v in losses.items():
            losses[k] = v.float()

        loss, log_vars = self._parse_losses(losses)

        if self.backbone is not None:
            optimizer['backbone'].zero_grad()
        if self.neck is not None:
            optimizer['neck'].zero_grad()
        if self.hmr_head is not None:
            optimizer['hmr_head'].zero_grad()
        if self.iuvd_head is not None:
            optimizer['iuvd_head'].zero_grad()

        loss.backward()

        if self.backbone is not None:
            optimizer['backbone'].step()
        if self.neck is not None:
            optimizer['neck'].step()
        if self.hmr_head is not None:
            optimizer['hmr_head'].step()
        if self.iuvd_head is not None:
            optimizer['iuvd_head'].step()

        outputs = dict(loss=loss,
                       log_vars=log_vars,
                       num_samples=len(next(iter(data_batch.values()))))
        return outputs

    def compute_losses(self, predictions: dict, targets: dict):
        """Compute losses."""
        # from IPython import embed
        # embed()
        # torch.where(targets['has_uvd']>0)[0]

        # from avatar3d.utils.torch_utils import image_tensor2numpy
        # import cv2

        # targets['img_metas'][index]

        # im = image_tensor2numpy(targets['img'].permute(0, 2, 3, 1))
        # cv2.imwrite('im.png', im[index])
        # iuv = image_tensor2numpy(targets['iuv_img'].permute(0, 2, 3, 1))
        # cv2.imwrite('iuv.png', iuv[index])

        # d = image_tensor2numpy(targets['d_img'].permute(0, 2, 3, 1))
        # cv2.imwrite('d.png', d[index])

        gt_keypoints2d = targets['keypoints2d']

        batch_size = gt_keypoints2d.shape[0]

        if self.hmr_head is not None:

            pred_betas = predictions['pred_shape'].view(-1, 10)
            pred_pose = predictions['pred_pose']
            pred_body_pose = pred_pose.view(-1, 23, 3, 3)
            # pred_cam = predictions['pred_cam'].view(-1, 3)
            # pred_global_orient = predictions['pred_pose'].view(-1, 24, 3,
            #                                                    3)[:, :1]

            # gt_global_orient = batch_rodrigues(
            #     targets['smpl_global_orient']).view(-1, 1, 3, 3).float()

            # pred_pose N, 24, 3, 3

            gt_body_pose = targets['smpl_body_pose'].float()
            gt_betas = targets['smpl_betas'].float()
            gt_global_orient = targets['smpl_global_orient'].float()

            if self.body_model_train is not None:
                # from IPython import embed
                # embed(header='315')
                pred_output1 = self.body_model_train(
                    betas=pred_betas,
                    body_pose=pred_body_pose,
                    global_orient=aa_to_rotmat(
                        gt_global_orient
                    ),  #pred_origin_orient.clone().detach()
                    pose2rot=False,
                    num_joints=gt_keypoints2d.shape[1])
                pred_keypoints3d1 = pred_output1['joints']
                pred_vertices1 = pred_output1['vertices']

                # pred_output2 = self.body_model_train(
                #     betas=pred_betas,
                #     body_pose=aa_to_rotmat(gt_body_pose),
                #     global_orient=
                #     pred_global_orient,  #pred_origin_orient.clone().detach()
                #     pose2rot=False,
                #     num_joints=gt_keypoints2d.shape[1])
                # pred_keypoints3d2 = pred_output2['joints']
                # pred_vertices2 = pred_output2['vertices']

        # # TODO: temp. Should we multiply confs here?
        # pred_keypoints3d_mask = pred_output['joint_mask']
        # keypoints3d_mask = keypoints3d_mask * pred_keypoints3d_mask
        """Compute losses."""

        has_smpl = targets['has_smpl'].view(batch_size, -1)

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
            gt_model_joint_mask = gt_output['joint_mask']
        if 'has_keypoints3d' in targets:
            has_keypoints3d = targets['has_keypoints3d'].squeeze(-1)
        else:
            has_keypoints3d = None
        if 'has_keypoints2d' in targets:
            has_keypoints2d = targets['has_keypoints2d'].squeeze(-1)
        else:
            has_keypoints2d = None

        ################################################
        ################################################

        losses = {}

        if self.loss_keypoints2d_hmr is not None:
            gt_keypoints2d = targets['keypoints2d']

            cam = predictions['pred_cam']
            gt_focal_length = 5000

            losses['keypoints2d_loss_hmr'] = self.compute_keypoints2d_hmr_loss(
                pred_keypoints3d1[..., :3],
                cam,
                gt_keypoints2d,
                has_keypoints2d=has_keypoints2d,
                focal_length=gt_focal_length,
            )
            # + self.compute_keypoints2d_hmr_loss(
            #     pred_keypoints3d2[..., :3],
            #     cam,
            #     gt_keypoints2d,
            #     has_keypoints2d=has_keypoints2d,
            #     focal_length=gt_focal_length,
            # )
        # if self.loss_global_orient is not None:
        #     losses['global_orient_loss'] = self.compute_global_orient_loss(
        #         pred_global_orient, gt_global_orient, has_smpl)

        if self.loss_keypoints3d is not None:

            # gt_keypoints3d = gt_model_joints
            gt_keypoints3d = targets['keypoints3d']
            # gt_keypoints3d = torch.cat([
            #     gt_keypoints3d,
            #     gt_model_joint_mask.view(1, -1, 1).repeat_interleave(
            #         batch_size, 0).float()
            # ], -1)

            batch_size = gt_keypoints3d.shape[0]
            losses['keypoints3d_loss'] = self.compute_keypoints3d_loss(
                pred_keypoints3d1,
                gt_keypoints3d,
                has_keypoints3d=has_keypoints3d)
            #     + self.compute_keypoints3d_loss(pred_keypoints3d2,
            #                                       gt_keypoints3d,
            #                                       has_keypoints3d=has_keypoints3d)
        if self.loss_camera is not None:

            joints2d_conf = targets['keypoints2d'][..., 2:3]
            gt_cam = estimate_cam_weakperspective_batch(
                gt_model_joints, targets['keypoints2d'], joints2d_conf,
                joints2d_conf, 224)
            losses['camera_loss'] = self.compute_camera_loss(cam, gt_cam)

        if self.loss_vertex is not None:
            losses['vertex_loss'] = self.compute_vertex_loss(
                pred_vertices1, gt_vertices, has_smpl)
            # + self.compute_vertex_loss(
            #     pred_vertices2, gt_vertices, has_smpl)

        if self.loss_smpl_pose is not None:
            losses['smpl_pose_loss'] = self.compute_smpl_pose_loss(
                pred_pose, gt_pose.float(), has_smpl)
        if self.loss_body_pose is not None:
            losses['body_pose_loss'] = self.compute_body_pose_loss(
                pred_body_pose, gt_body_pose, has_smpl)
        if self.loss_smpl_betas is not None:
            losses['smpl_betas_loss'] = self.compute_smpl_betas_loss(
                pred_betas, gt_betas, has_smpl)

        return losses

    def gen_mesh_grid(self, img_res=224):
        h_grid = torch.linspace(0, 1, img_res).view(-1, 1).repeat(1, img_res)
        v_grid = torch.linspace(0, 1, img_res).repeat(img_res, 1)

        mesh_grid = torch.cat((v_grid.unsqueeze(2), h_grid.unsqueeze(2)),
                              dim=2)[None]
        return mesh_grid

    def prepare_targets(self, data_batch: dict):
        if self.iuvd_head is not None:
            has_smpl = data_batch['has_smpl'].view(-1)

            has_transl = data_batch['has_transl']

            has_uvd = data_batch['has_uvd']

            has_focal_length = data_batch['has_focal_length']

            gt_iuv_img = data_batch['iuv_img'].float()
            gt_d_img = data_batch['d_img'].float()
            gt_transl = data_batch['smpl_transl'].float()

            if not has_transl.all():
                has_smpl = data_batch['has_smpl'].view(-1) > 0
                no_transl = data_batch['has_transl'].view(-1) == 0
                get_transl_ids = torch.where((has_smpl * no_transl) > 0)[0]

                gt_keypoints2d = data_batch['keypoints2d']
                joints2d_conf = gt_keypoints2d[..., 2:3]
                if self.body_model_train is not None:
                    gt_body_pose = data_batch['smpl_body_pose'].float()
                    gt_betas = data_batch['smpl_betas'].float()
                    gt_global_orient = data_batch['smpl_global_orient'].float()
                    gt_output = self.body_model_train(
                        betas=gt_betas,
                        body_pose=gt_body_pose.float(),
                        global_orient=gt_global_orient,
                        num_joints=gt_keypoints2d.shape[1])
                    gt_model_joints = gt_output['joints']

                center, scale = data_batch['center'].float(
                ), data_batch['scale'][:, 0].float()
                gt_cam = estimate_cam_weakperspective_batch(
                    gt_model_joints, data_batch['keypoints2d'], joints2d_conf,
                    joints2d_conf, 224)

                gt_transl = merge_cam_to_full_transl(gt_cam, center, scale,
                                                     data_batch['ori_shape'],
                                                     gt_transl)
                gt_transl[get_transl_ids, 2:3] = torch.Tensor(
                    np.random.uniform(5, 10, size=(get_transl_ids.shape[0],
                                                   1))).to(gt_transl.device)

                random_transl_ids = torch.where(((1 - has_smpl * 1.0) *
                                                 no_transl) > 0)[0]

                xy = torch.Tensor(
                    np.random.uniform(-2,
                                      2,
                                      size=(random_transl_ids.shape[0], 2)))
                z = torch.Tensor(
                    np.random.uniform(5,
                                      10,
                                      size=(random_transl_ids.shape[0], 1)))
                gt_transl_online = torch.cat([xy, z], 1).to(gt_transl.device)
                gt_transl[random_transl_ids] = gt_transl_online
                has_transl = torch.ones_like(has_transl).float()
                has_transl[random_transl_ids] *= 0.5

            if not has_focal_length.all():
                no_f_ids = torch.where(has_focal_length == 0)[0]
                gt_focal_length = data_batch['focal_length'].view(-1)
                gt_focal_length[no_f_ids] = gt_transl[no_f_ids,
                                                      2] * gt_cam[no_f_ids, 0]
            if not has_uvd.all():
                gt_body_pose = data_batch['smpl_body_pose']

                gt_betas = data_batch['smpl_betas'].float()
                gt_keypoints2d = data_batch['keypoints2d']
                if self.body_model_train is not None:
                    gt_output = self.body_model_train(
                        betas=gt_betas,
                        body_pose=gt_body_pose.float(),
                        global_orient=gt_global_orient.float())
                    gt_vertices = gt_output['vertices']
                    gt_model_joints = gt_output['joints']
                    gt_model_joints_mask = gt_output['joint_mask']
                render_ids = torch.where(((1 - has_uvd.view(-1)) *
                                          has_smpl) > 0)[0]

                # from IPython import embed
                # embed(header='344')

                if len(render_ids):
                    uv_res = int(gt_iuv_img.shape[2])
                    down_scale = int(224 / uv_res)
                    gt_iuv_img_online, gt_d_img_online = self.render_gt_iuvd(
                        gt_vertices[render_ids],
                        gt_keypoints2d[render_ids] / down_scale,
                        gt_model_joints[render_ids],
                        gt_model_joints_mask,
                        img_res=uv_res,
                        focal_length=gt_focal_length[render_ids] * uv_res / 2)
                    gt_iuv_img[render_ids] = gt_iuv_img_online
                    gt_d_img[render_ids] = gt_d_img_online
                has_uvd = (has_uvd.view(-1) + has_smpl).bool() * 1.0
            # Image Mesh Estimator does not need extra process for ground truth

            data_batch['iuv_img'] = gt_iuv_img
            data_batch['d_img'] = gt_d_img
            data_batch['smpl_transl'] = gt_transl
        return data_batch

    def forward_test(self, **data_batch):
        """Defines the computation performed at every call when testing."""

        predictions = dict()
        for name in self.freeze_modules:
            for parameter in getattr(self, name).parameters():
                parameter.requires_grad = False

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
        device = data_batch['img'].device
        if self.neck is not None:
            features = self.neck(features)

        if 'bbox_info' not in data_batch:

            bbox_info = torch.zeros(batch_size, 3).to(device)
            data_batch['bbox_info'] = bbox_info
        if self.iuvd_head is not None:
            predictions_iuvd = self.iuvd_head(features)
        else:
            predictions_iuvd = dict()
        predictions.update(predictions_iuvd)
        if self.hmr_head is not None:
            head_data = []
            for key in self.head_keys:
                head_data.append(data_batch[key])
            predictions_hmr = self.hmr_head(features, *head_data)
        else:
            predictions_hmr = dict()

        predictions.update(predictions_hmr)
        predictions.update(extracted_data)

        pred_cam = predictions['pred_cam']

        center, scale, orig_focal_length = data_batch['center'], data_batch[
            'scale'][:, 0], data_batch['orig_focal_length'].squeeze(dim=1)

        predictions.update(pred_transl=pred_cam_to_transl(pred_cam, 5000, 224))
        predictions.update(orig_transl=pred_cam_to_full_transl(
            pred_cam, center, scale, data_batch['ori_shape'], 5000).float())
        predictions.update(pred_focal_length=5000. / 112)
        predictions.update(orig_focal_length=5000)

        pred_betas = predictions['pred_shape']
        pred_body_pose = predictions['pred_pose']
        # pred_global_orient = predictions['pred_pose'][:, :1]
        pred_global_orient = batch_rodrigues(
            data_batch['smpl_global_orient']).view(-1, 1, 3, 3).float()

        pred_output = self.body_model_test(betas=pred_betas,
                                           body_pose=pred_body_pose,
                                           global_orient=pred_global_orient,
                                           pose2rot=False)
        pred_keypoints3d = pred_output['joints']
        pred_vertices = pred_output['vertices']
        orig_wh = torch.cat(
            [data_batch['ori_shape'][:, 1], data_batch['ori_shape'][:, 0]], -1)
        origin_focal_length = predictions['orig_focal_length']
        if isinstance(origin_focal_length, torch.Tensor):
            origin_focal_length = origin_focal_length.view(-1).float()
        pred_keypoints2d = project_points_focal_length_pixel(
            pred_keypoints3d,
            translation=predictions['orig_transl'].float(),
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

        predictions.update(img=data_batch['img'],
                           pred_vertices=pred_vertices,
                           sample_idx=data_batch['sample_idx'],
                           pred_keypoints2d=pred_keypoints2d,
                           img_metas=data_batch['img_metas'])

        if self.visualizer is not None and self.vis:
            self.visualizer(predictions)
        all_preds = {}
        all_preds['keypoints_3d'] = pred_keypoints3d
        all_preds['smpl_pose'] = torch.cat(
            [pred_global_orient, pred_body_pose], 1)
        all_preds['smpl_beta'] = pred_betas
        all_preds['camera'] = pred_cam
        all_preds['vertices'] = pred_vertices
        image_path = []
        for img_meta in data_batch['img_metas']:
            image_path.append(img_meta['image_path'])
        all_preds['image_path'] = image_path
        all_preds['image_idx'] = data_batch['sample_idx']

        all_preds['pred_keypoints2d'] = predictions['pred_keypoints2d']
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

    def compute_focal_length_loss(self, pred_f, gt_f):
        return self.loss_focal_length(pred_f, gt_f)

    def compute_keypoints3d_loss(
            self,
            pred_keypoints3d: torch.Tensor,
            gt_keypoints3d: torch.Tensor,
            has_keypoints3d: Optional[torch.Tensor] = None):
        """Compute loss for 3d keypoints."""
        keypoints3d_conf = gt_keypoints3d[:, :, 3].float().unsqueeze(-1)
        keypoints3d_conf = keypoints3d_conf.repeat(1, 1, 3)
        pred_keypoints3d = pred_keypoints3d.float()
        gt_keypoints3d = gt_keypoints3d[:, :, :3].float()

        # currently, only mpi_inf_3dhp and h36m have 3d keypoints
        # both datasets have right_hip_extra and left_hip_extra
        right_hip_idx = get_keypoint_idx('right_hip_extra', self.convention)
        left_hip_idx = get_keypoint_idx('left_hip_extra', self.convention)
        # right_hip_idx = get_keypoint_idx('right_hip', self.convention)
        # left_hip_idx = get_keypoint_idx('left_hip', self.convention)
        gt_pelvis = (gt_keypoints3d[:, right_hip_idx, :] +
                     gt_keypoints3d[:, left_hip_idx, :]) / 2
        pred_pelvis = (pred_keypoints3d[:, right_hip_idx, :] +
                       pred_keypoints3d[:, left_hip_idx, :]) / 2

        gt_keypoints3d = gt_keypoints3d - gt_pelvis[:, None, :]
        pred_keypoints3d = pred_keypoints3d - pred_pelvis[:, None, :]
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
        keypoints3d_conf = gt_keypoints3d[:, :, 3].float().unsqueeze(-1)
        keypoints3d_conf = keypoints3d_conf.repeat(1, 1, 3)
        pred_keypoints3d = pred_keypoints3d.float()
        gt_keypoints3d = gt_keypoints3d[:, :, :3].float()

        # currently, only mpi_inf_3dhp and h36m have 3d keypoints
        # both datasets have right_hip_extra and left_hip_extra
        # right_hip_idx = get_keypoint_idx('right_hip_extra', self.convention)
        # left_hip_idx = get_keypoint_idx('left_hip_extra', self.convention)
        right_hip_idx = get_keypoint_idx('right_hip', self.convention)
        left_hip_idx = get_keypoint_idx('left_hip', self.convention)
        gt_pelvis = (gt_keypoints3d[:, right_hip_idx, :] +
                     gt_keypoints3d[:, left_hip_idx, :]) / 2
        pred_pelvis = (pred_keypoints3d[:, right_hip_idx, :] +
                       pred_keypoints3d[:, left_hip_idx, :]) / 2

        gt_keypoints3d = gt_keypoints3d - gt_pelvis[:, None, :]
        pred_keypoints3d = pred_keypoints3d - pred_pelvis[:, None, :]
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

    def compute_keypoints2d_prompt_loss(
            self,
            pred_keypoints3d: torch.Tensor,
            pred_transl: torch.Tensor,
            pred_focal_length: torch.Tensor,
            gt_keypoints2d: torch.Tensor,
            img_res: Optional[int] = 224,
            has_keypoints2d: Optional[torch.Tensor] = None,
            weight=None):
        """Compute loss for 2d keypoints."""
        keypoints2d_conf = gt_keypoints2d[:, :, 2].float().unsqueeze(-1)
        keypoints2d_conf = keypoints2d_conf.repeat(1, 1, 2)
        gt_keypoints2d = gt_keypoints2d[:, :, :2].float()
        pred_keypoints2d = project_points_focal_length(
            pred_keypoints3d + pred_transl.view(-1, 1, 3),
            focal_length=pred_focal_length,
            img_res=img_res)
        #  pelvis_idx + 1]
        # Normalize keypoints to [-1,1]
        # The coordinate origin of pred_keypoints_2d is
        # the center of the input image.
        pred_keypoints2d = 2 * pred_keypoints2d / (img_res - 1) - 1
        # The coordinate origin of gt_keypoints_2d is
        # the top left corner of the input image.
        gt_keypoints2d = 2 * gt_keypoints2d / (img_res - 1) - 1
        loss = self.loss_keypoints2d_prompt(pred_keypoints2d,
                                            gt_keypoints2d[..., :2],
                                            reduction_override='none')
        # If has_keypoints2d is not None, then computes the losses on the
        # instances that have ground-truth keypoints2d.
        # But the zero confidence keypoints will be included in mean.
        # Otherwise, only compute the keypoints2d
        # which have positive confidence.
        # has_keypoints2d is None when the key has_keypoints2d
        # is not in the datasets
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

    def compute_keypoints2d_perspective_loss(
            self,
            pred_keypoints3d: torch.Tensor,
            pred_transl: torch.Tensor,
            gt_keypoints2d: torch.Tensor,
            camera_center: torch.Tensor,
            focal_length: torch.Tensor,
            img_res: Optional[int] = 224,
            has_keypoints2d: Optional[torch.Tensor] = None,
            loss_func=None):
        """Compute loss for 2d keypoints."""
        keypoints2d_conf = gt_keypoints2d[:, :, 2].float().unsqueeze(-1)
        keypoints2d_conf = keypoints2d_conf.repeat(1, 1, 2)
        gt_keypoints2d = gt_keypoints2d[:, :, :2].float()

        pred_keypoints2d = project_points_focal_length_pixel(
            pred_keypoints3d,
            translation=pred_transl,
            focal_length=focal_length,
            camera_center=camera_center,
            img_res=None)

        # trans @ pred_keypoints2d2

        # pred_keypoints2d = affine_transform(pred_keypoints2d, trans)

        # The coordinate origin of pred_keypoints_2d is
        # the center of the input image.
        pred_keypoints2d = 2 * pred_keypoints2d / (img_res.view(-1, 1, 2) -
                                                   1) - 1
        # The coordinate origin of gt_keypoints_2d is
        # the top left corner of the input image.
        gt_keypoints2d = 2 * gt_keypoints2d / (img_res.view(-1, 1, 2) - 1) - 1
        loss = loss_func(pred_keypoints2d,
                         gt_keypoints2d,
                         reduction_override='none')

        # If has_keypoints2d is not None, then computes the losses on the
        # instances that have ground-truth keypoints2d.
        # But the zero confidence keypoints will be included in mean.
        # Otherwise, only compute the keypoints2d
        # which have positive confidence.
        # has_keypoints2d is None when the key has_keypoints2d
        # is not in the datasets

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

    def compute_keypoints2d_hmr_loss(
            self,
            pred_keypoints3d: torch.Tensor,
            pred_cam: torch.Tensor,
            gt_keypoints2d: torch.Tensor,
            img_res: Optional[int] = 224,
            focal_length: Optional[int] = 5000,
            has_keypoints2d: Optional[torch.Tensor] = None,
            weight=None):
        """Compute loss for 2d keypoints."""
        keypoints2d_conf = gt_keypoints2d[:, :, 2].float().unsqueeze(-1)
        keypoints2d_conf = keypoints2d_conf.repeat(1, 1, 2)
        gt_keypoints2d = gt_keypoints2d[:, :, :2].float()
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

        loss = self.loss_keypoints2d_hmr(
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
                            gt_vertices: torch.Tensor, has_smpl: torch.Tensor):
        """Compute loss for vertices."""
        gt_vertices = gt_vertices.float()
        batch_size = gt_vertices.shape[0]
        conf = has_smpl.float().view(batch_size, -1, 1)[:, :1]
        conf = conf.repeat(1, gt_vertices.shape[1], gt_vertices.shape[2])
        loss = self.loss_vertex(pred_vertices,
                                gt_vertices,
                                reduction_override='none')
        valid_pos = conf > 0
        if conf[valid_pos].numel() == 0:
            return torch.Tensor([0]).type_as(gt_vertices)
        loss = torch.sum(loss * conf) / conf[valid_pos].numel()
        return loss

    def compute_global_orient_loss(self, pred_orient, gt_orient, has_smpl):
        gt_rotmat = batch_rodrigues(gt_orient.view(-1, 3)).view(-1, 1, 3, 3)
        batch_size = gt_rotmat.shape[0]

        conf = has_smpl.float().view(batch_size, -1, 1, 1)

        # valid_pos = conf > 0

        if conf.max() == 0:
            return torch.Tensor([0]).type_as(gt_orient)
        loss = self.loss_global_orient(pred_orient,
                                       gt_rotmat,
                                       reduction_override='none')
        loss = loss * conf
        loss = loss.view(loss.shape[0], -1).mean(-1)
        return loss

    def compute_body_pose_loss(self, pred_rotmat: torch.Tensor,
                               gt_pose: torch.Tensor, has_smpl: torch.Tensor):
        """Compute loss for smpl pose."""
        conf = has_smpl.float().view(-1)
        valid_pos = conf > 0
        if conf[valid_pos].numel() == 0:
            return torch.Tensor([0]).type_as(gt_pose)
        pred_rotmat = pred_rotmat[valid_pos]
        gt_pose = gt_pose[valid_pos]
        conf = conf[valid_pos]
        gt_rotmat = batch_rodrigues(gt_pose.view(-1, 3)).view(-1, 23, 3, 3)
        loss = self.loss_body_pose(pred_rotmat,
                                   gt_rotmat,
                                   reduction_override='none')
        loss = loss.view(loss.shape[0], -1).mean(-1)
        loss = torch.mean(loss * conf)
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

    def compute_camera_loss(
        self,
        pred_cam: torch.Tensor,
        gt_cam: torch.Tensor,
    ):
        """Compute loss for predicted camera parameters."""
        loss = self.loss_camera(pred_cam, gt_cam)
        return loss

    def compute_iuv_loss(self, pred_iuv, gt_iuv, has_iuv):
        batch_size = gt_iuv.shape[0]
        valid = has_iuv.float().view(batch_size) > 0
        loss = self.loss_iuv(pred_iuv[valid], gt_iuv[valid])
        return loss

    def compute_distortion_loss(self, pred_distortion, gt_distortion, has_d):
        batch_size = has_d.shape[0]
        valid = has_d.float().view(batch_size) > 0
        loss = self.loss_distortion_img(pred_distortion[valid],
                                        gt_distortion[valid])
        return loss

    def compute_image_grad_u_loss(self, pred_u, gt_u, has_uvd):
        batch_size = has_uvd.shape[0]
        valid = has_uvd.float().view(batch_size) > 0
        return self.loss_image_grad_u(pred_u[valid], gt_u[valid])

    def compute_image_grad_v_loss(self, pred_v, gt_v, has_uvd):
        batch_size = has_uvd.shape[0]
        valid = has_uvd.float().view(batch_size) > 0
        return self.loss_image_grad_v(pred_v[valid], gt_v[valid])

    def compute_transl_loss(self, pred_transl, gt_transl, has_transl):
        batch_size = gt_transl.shape[0]
        conf = has_transl.float().view(batch_size, 1)
        loss = self.loss_transl_z(pred_transl, gt_transl, weight=conf)
        return loss

    def compute_wrapped_distortion_loss(self, pred_distortion, gt_distortion,
                                        has_uvd):
        batch_size = has_uvd.shape[0]
        valid = has_uvd.float().view(batch_size) > 0
        return self.loss_wrapped_distortion(pred_distortion[valid],
                                            gt_distortion[valid])

    def get_focal_length(self,
                         pred_transl: torch.Tensor,
                         gt_keypoints2d: torch.Tensor,
                         gt_model_joints: torch.Tensor,
                         gt_model_joints_mask: torch.Tensor,
                         img_res: Optional[int] = 224):
        """Compute loss for part segmentations."""
        joints2d_conf = gt_keypoints2d[..., 2:]
        conf = gt_model_joints_mask * joints2d_conf
        gt_cam = estimate_cam_weakperspective_batch(
            gt_model_joints,
            gt_keypoints2d,
            conf,
            conf,
            img_size=img_res,
        )
        focal_length = gt_cam[:, 0] * pred_transl[:, 2]
        return focal_length

    def render_gt_iuvd(self,
                       gt_vertices: torch.Tensor,
                       gt_keypoints2d: torch.Tensor,
                       gt_model_joints: torch.Tensor,
                       gt_model_joints_mask: torch.Tensor,
                       img_res: Optional[int] = 224,
                       focal_length: Optional[int] = 1000):
        """Compute loss for part segmentations."""
        device = gt_keypoints2d.device
        uv_renderer = self.uv_renderer.to(device)
        depth_renderer = self.depth_renderer.to(device)

        batch_size = gt_keypoints2d.shape[0]

        joints2d_conf = gt_keypoints2d[..., 2:]
        conf = gt_model_joints_mask.view(1, -1, 1) * joints2d_conf
        gt_transl = estimate_transl_weakperspective_batch(
            gt_model_joints,
            gt_keypoints2d,
            conf,
            conf,
            focal_length=focal_length,
            img_size=img_res,
        )

        gt_transl = gt_transl.unsqueeze(1)
        K = torch.eye(3)[None].repeat_interleave(batch_size, 0)
        K[:, 0, 0] = focal_length
        K[:, 1, 1] = focal_length
        K[:, 0, 2] = img_res / 2.
        K[:, 1, 2] = img_res / 2.
        cameras = build_cameras(
            dict(type='perspective',
                 image_size=(img_res, img_res),
                 in_ndc=False,
                 K=K,
                 convention='opencv')).to(device)

        mesh = Meshes(verts=gt_vertices + gt_transl,
                      faces=self.body_model_train.faces_tensor[None].repeat(
                          batch_size, 1, 1)).to(device)
        uv_renderer = uv_renderer.to(device)
        gt_iuv = uv_renderer(mesh, cameras)
        gt_iuv = gt_iuv.permute(0, 3, 1, 2)
        gt_depth = depth_renderer(mesh, cameras)[..., :1]

        mask = (gt_depth > 0).float()
        gt_depth = gt_depth * mask + 1 * (1 - mask)

        gt_distortion = gt_transl[..., 2].view(batch_size, 1, 1, 1) / gt_depth
        gt_distortion = gt_distortion * mask
        gt_distortion = gt_distortion.permute(0, 3, 1, 2)

        return gt_iuv, gt_distortion
