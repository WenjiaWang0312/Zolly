from distutils.command.build import build
import shutil
from typing import Optional, Tuple, Union, Tuple
from avatar3d.utils.torch_utils import dict2numpy
from mmhuman3d.models.utils import FitsDict
import torch
import torch.nn.functional as F
import numpy as np
from pytorch3d.structures import Meshes

# from mmhuman3d.models.backbones.builder import build_backbone

from mmhuman3d.models.architectures.base_architecture import BaseArchitecture
from mmhuman3d.core.conventions.segmentation import body_segmentation
from mmhuman3d.core.visualization import visualize_smpl
from mmhuman3d.models.registrants.builder import build_registrant
from mmhuman3d.utils.geometry import batch_rodrigues, estimate_translation
from avatar3d.models.necks.builder import build_neck
from avatar3d.models.extractors.builder import build_extractor
from avatar3d.models.visualizers.builder import build_visualizer
from avatar3d.cameras.builder import build_cameras
from avatar3d.render.builder import build_textures
from avatar3d.models.body_models.mappings import convert_kps
from avatar3d.models.body_models.mappings import get_keypoint_idx
from avatar3d.render.builder import build_renderer
from avatar3d.models.body_models.builder import build_body_model
from avatar3d.models.losses.builder import build_loss
from avatar3d.models.heads.builder import build_head
from avatar3d.models.backbones.builder import build_backbone
from avatar3d.transforms.transform3d import rotmat_to_rot6d, ee_to_rotmat, rotmat_to_aa
from avatar3d.cameras.utils import (
    pred_cam_to_transl, merge_cam_to_full_transl,
    estimate_cam_weakperspective_batch, estimate_transl_weakperspective_batch,
    project_points_focal_length, pred_cam_to_full_transl,
    full_transl_to_pred_cam, project_points_pred_cam,
    project_points_focal_length_pixel, rotate_smpl_cam)

from avatar3d.runners.builder import build_runner


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


class PersepectiveMeshEstimator(BaseArchitecture):
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
        resolution: int = 224,
        body_model_train: Optional[Union[dict, None]] = None,
        body_model_test: Optional[Union[dict, None]] = None,
        convention: Optional[str] = 'human_data',
        uv_renderer: dict = None,
        depth_renderer: dict = None,
        use_d_weight: bool = False,
        f_ablation: bool = False,
        detach_ablation: bool = False,
        freeze_modules: Tuple[str] = (),
        ###
        registration: Optional[Union[dict, None]] = None,
        loss_keypoints2d_prompt: Optional[Union[dict, None]] = None,
        loss_joints2d_prompt: Optional[Union[dict, None]] = None,
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
        loss_segm_mask: Optional[Union[dict, None]] = None,
        ##
        loss_transl_z: Optional[Union[dict, None]] = None,
        loss_iuv: Optional[Union[dict, None]] = None,
        loss_distortion_img: Optional[Union[dict, None]] = None,
        loss_image_grad_u: Optional[Union[dict, None]] = None,
        loss_image_grad_v: Optional[Union[dict, None]] = None,
        loss_wrapped_distortion: Optional[Union[dict, None]] = None,
        ###
        full_uvd: bool = True,
        init_cfg: Optional[Union[list, dict, None]] = None,
        visualizer: Optional[Union[int, None]] = None,
        use_pred_transl: bool = False,
    ):
        super(PersepectiveMeshEstimator, self).__init__(init_cfg)
        self.extractor = build_extractor(extractor)
        self.extractor_key = extractor_key
        self.use_pred_transl = use_pred_transl
        self.use_d_weight = use_d_weight
        self.f_ablation = f_ablation

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

        self.registration = registration
        if registration is not None:
            self.fits_dict = FitsDict(fits='static')
            self.registration_mode = self.registration['mode']
            self.registrant = build_registrant(registration['registrant'])
        else:
            self.registrant = None

        self.loss_joints2d_prompt = build_loss(loss_joints2d_prompt)
        self.loss_keypoints2d_prompt = build_loss(loss_keypoints2d_prompt)
        self.loss_keypoints2d_cliff = build_loss(loss_keypoints2d_cliff)
        self.loss_keypoints2d_spec = build_loss(loss_keypoints2d_spec)
        self.loss_keypoints2d_hmr = build_loss(loss_keypoints2d_hmr)
        self.loss_keypoints3d = build_loss(loss_keypoints3d)
        self.loss_joints3d = build_loss(loss_joints3d)
        self.detach_ablation = detach_ablation
        self.loss_transl_z = build_loss(loss_transl_z)
        self.loss_iuv = build_loss(loss_iuv)
        self.loss_distortion_img = build_loss(loss_distortion_img)
        self.loss_image_grad_v = build_loss(loss_image_grad_v)
        self.loss_image_grad_u = build_loss(loss_image_grad_u)
        self.loss_wrapped_distortion = build_loss(loss_wrapped_distortion)

        self.loss_segm_mask = build_loss(loss_segm_mask)

        self.loss_global_orient = build_loss(loss_global_orient)
        self.loss_vertex = build_loss(loss_vertex)
        self.loss_smpl_pose = build_loss(loss_smpl_pose)
        self.loss_body_pose = build_loss(loss_body_pose)
        self.loss_smpl_betas = build_loss(loss_smpl_betas)
        self.loss_camera = build_loss(loss_camera)
        self.full_uvd = full_uvd
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
            if 'fpn_feat' in self.head_keys:
                data_batch['fpn_feat'] = predictions['fpn_feat']
            if 'distortion_feat' in self.head_keys:
                data_batch['distortion_feat'] = predictions['distortion_feat']
            for key in self.head_keys:
                head_data.append(data_batch[key])
            predictions_hmr = self.hmr_head(features, *head_data)
        else:
            predictions_hmr = dict()

        predictions.update(predictions_hmr)
        targets = self.prepare_targets(data_batch)

        if self.registration is not None:
            targets = self.run_registration(predictions, targets)

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

        # import cv2
        # from avatar3d.utils.torch_utils import image_tensor2numpy
        # im = image_tensor2numpy(targets['img'][index].permute(1, 2, 0))
        # cv2.imwrite(f'{index}_im.png', im)
        # im = image_tensor2numpy(predictions['pred_iuv_img'][index].permute(1, 2, 0))
        # cv2.imwrite(f'{index}_pred_iuv.png', im)
        # im = image_tensor2numpy(targets['iuv_img'][index].permute(1, 2, 0))
        # cv2.imwrite(f'{index}_iuv.png', im)

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
        gt_keypoints2d = targets['keypoints2d']

        batch_size = gt_keypoints2d.shape[0]

        if self.hmr_head is not None:

            pred_betas = predictions['pred_shape'].view(-1, 10)
            pred_pose = predictions['pred_pose']
            pred_body_pose = pred_pose.view(-1, 24, 3, 3)[:, 1:]
            # pred_cam = predictions['pred_cam'].view(-1, 3)
            pred_global_orient = predictions['pred_pose'].view(-1, 24, 3,
                                                               3)[:, :1]

            # gt_global_orient = batch_rodrigues(
            #     targets['smpl_global_orient']).view(-1, 1, 3, 3).float()

            # pred_pose N, 24, 3, 3
            if self.body_model_train is not None:
                pred_output = self.body_model_train(
                    betas=pred_betas,
                    body_pose=pred_body_pose,
                    global_orient=
                    pred_global_orient,  #pred_origin_orient.clone().detach()
                    pose2rot=False,
                    num_joints=gt_keypoints2d.shape[1])
                pred_keypoints3d = pred_output['joints']
                pred_vertices = pred_output['vertices']

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

        pred_cam = predictions['pred_cam']
        ################################################
        ################################################

        losses = {}

        if self.loss_keypoints2d_spec is not None:

            gt_keypoints2d = targets['keypoints2d']
            center, scale = targets['center'], targets['scale'][:, 0]

            pred_focal_length = 1. / torch.tan(targets['cam_vfov'] / 2)

            pred_focal_length = torch.clip(pred_focal_length, 0.5, 100)

            pred_cam = predictions['pred_cam']

            img_h = targets['ori_shape'][:, 0]
            img_w = targets['ori_shape'][:, 1]

            orig_focal_length = pred_focal_length * img_h / 2
            pred_transl = pred_cam_to_full_transl(pred_cam, center, scale,
                                                  targets['ori_shape'],
                                                  orig_focal_length).float()

            gt_keypoints2d = targets['keypoints2d']

            origin_keypoints2d = gt_keypoints2d.clone()[..., :2]

            origin_keypoints2d = (
                origin_keypoints2d - self.resolution / 2
            ) / self.resolution * scale[:, None, None] + center.view(-1, 1, 2)
            origin_keypoints2d = torch.cat(
                [origin_keypoints2d, targets['keypoints2d'][..., 2:3]], -1)

            camera_center = torch.cat(
                [img_w.view(-1, 1), img_h.view(-1, 1)], -1) / 2

            losses[
                'keypoints2d_loss_spec'] = self.compute_keypoints2d_perspective_loss(
                    pred_keypoints3d,
                    # gt_model_joints,
                    pred_transl,
                    # gt_transl,
                    origin_keypoints2d,
                    camera_center,
                    orig_focal_length,
                    img_res=targets['ori_shape'],
                    has_keypoints2d=has_keypoints2d,
                    loss_func=self.loss_keypoints2d_spec)

        if self.loss_keypoints2d_prompt is not None:

            center, scale = targets['center'], targets['scale'][:, 0]
            gt_keypoints2d = targets['keypoints2d']
            origin_keypoints2d = gt_keypoints2d.clone()[..., :2]
            origin_keypoints2d = (
                origin_keypoints2d - self.resolution / 2
            ) / self.resolution * scale[:, None, None] + center.view(-1, 1, 2)
            origin_keypoints2d = torch.cat(
                [origin_keypoints2d, targets['keypoints2d'][..., 2:3]], -1)
            conf = targets['keypoints2d'][..., 2:3]
            gt_keypoints2d = torch.cat([gt_keypoints2d, conf], -1)

            if self.full_uvd:
                has_transl = targets['has_transl'].view(-1)
            else:
                has_transl = targets['is_distorted'].view(
                    -1) * targets['has_transl'].view(-1)
            #
            no_transl_ids = torch.where(has_transl == 0)[0]
            # has_transl_ids = torch.where(has_transl > 0)[0]

            target_focal_length_ndc = targets['orig_focal_length'].view(
                -1) * 2 / scale
            target_focal_length_ndc[no_transl_ids] = 1000 / (self.resolution /
                                                             2)

            orig_focal_length = target_focal_length_ndc.view(-1) * scale / 2

            target_transl = targets['smpl_transl'].float()

            joints2d_conf = targets['keypoints2d'][..., 2:3]
            gt_cam_ = estimate_cam_weakperspective_batch(
                gt_model_joints[no_transl_ids],
                targets['keypoints2d'][no_transl_ids],
                joints2d_conf[no_transl_ids], joints2d_conf[no_transl_ids],
                self.resolution)
            computed_transl = pred_cam_to_full_transl(
                gt_cam_, center[no_transl_ids], scale[no_transl_ids],
                targets['ori_shape'][no_transl_ids],
                orig_focal_length[no_transl_ids])

            target_transl[no_transl_ids, :] = computed_transl.float()

            camera_center = torch.cat([
                targets['ori_shape'][:, 1].view(-1, 1),
                targets['ori_shape'][:, 0].view(-1, 1)
            ], -1) / 2

            has_K_ids = torch.where(targets['has_K'] > 0)[0]
            camera_center[has_K_ids] = targets['K'][:, :2, 2][has_K_ids]

            #==================================================#
            # pred_keypoints2d = project_points_focal_length_pixel(
            #     gt_model_joints,
            #     translation=target_transl,
            #     focal_length=orig_focal_length,
            #     camera_center=camera_center,
            #     img_res=None)
            # from avatar3d.utils.demo_utils import draw_skeletons_image
            # import cv2

            # from avatar3d.utils.torch_utils import image_tensor2numpy
            # origin_keypoints2d_ = convert_kps(origin_keypoints2d.float(), 'smpl_54', 'h36m')[0]
            # pred_keypoints2d_ = convert_kps(pred_keypoints2d.float(), 'smpl_54', 'h36m')[0]

            # for index in range(4):
            #     im = cv2.imread(targets['img_metas'][index]['image_path'])
            #     im = draw_skeletons_image(
            #         pred_keypoints2d_.detach().cpu().numpy()[index],
            #         im,
            #         convention='h36m', palette=[255,0,0])
            #     cv2.imwrite(f'{index}_pred.png', im)

            #     # im = cv2.imread(targets['img_metas'][index]['image_path'])
            #     im = cv2.imread(targets['img_metas'][index]['image_path'])
            #     im = draw_skeletons_image(
            #         origin_keypoints2d_.detach().cpu().numpy()[index],
            #         im,
            #         convention='h36m',palette=[0,255,255])
            #     cv2.imwrite(f'{index}_gt.png', im)
            #==================================================#
            # from IPython import embed
            # embed()
            # import cv2
            # from avatar3d.utils.torch_utils import image_tensor2numpy
            # from avatar3d.uti
            # im = cv2.imread(targets['img_metas'][index]['image_path'])

            losses[
                'keypoints2d_loss_prompt'] = self.compute_keypoints2d_perspective_loss(
                    pred_keypoints3d,  # pred_keypoints3d,
                    target_transl,  # pred_transl,
                    origin_keypoints2d,
                    camera_center,
                    orig_focal_length,
                    img_res=targets['ori_shape'],
                    has_keypoints2d=has_keypoints2d,
                    loss_func=self.loss_keypoints2d_prompt)

        if self.loss_joints2d_prompt is not None:
            center, scale = targets['center'], targets['scale'][:, 0]

            # has_transl = targets['is_distorted']
            has_transl = targets['has_transl']
            no_transl_ids = torch.where(has_transl == 0)[0]

            target_focal_length_ndc = targets['orig_focal_length'].view(
                -1) * 2 / scale
            target_focal_length_ndc[no_transl_ids] = 1000 / (self.resolution /
                                                             2)

            orig_focal_length = target_focal_length_ndc.view(-1) * scale / 2

            target_transl = targets['smpl_transl'].float()

            joints2d_conf = targets['keypoints2d'][..., 2:3]
            gt_cam_ = estimate_cam_weakperspective_batch(
                gt_model_joints[no_transl_ids],
                targets['keypoints2d'][no_transl_ids],
                joints2d_conf[no_transl_ids], joints2d_conf[no_transl_ids],
                self.resolution)
            computed_transl = pred_cam_to_full_transl(
                gt_cam_, center[no_transl_ids], scale[no_transl_ids],
                targets['ori_shape'][no_transl_ids],
                orig_focal_length[no_transl_ids])

            target_transl[no_transl_ids, :] = computed_transl.float()

            camera_center = torch.cat([
                targets['ori_shape'][:, 1].view(-1, 1),
                targets['ori_shape'][:, 0].view(-1, 1)
            ], -1) / 2

            has_K_ids = torch.where(targets['has_K'] > 0)[0]
            camera_center[has_K_ids] = targets['K'][:, :2, 2][has_K_ids]

            #==================================================#
            origin_keypoints2d = project_points_focal_length_pixel(
                gt_model_joints,
                translation=target_transl,
                focal_length=orig_focal_length,
                camera_center=camera_center,
                img_res=None)
            conf = targets['keypoints2d'][..., 2:3]
            origin_keypoints2d = torch.cat([origin_keypoints2d, conf], -1)
            # from avatar3d.utils.demo_utils import draw_skeletons_image
            # import cv2

            # from avatar3d.utils.torch_utils import image_tensor2numpy
            # origin_keypoints2d_ = convert_kps(origin_keypoints2d.float(), 'smpl_54', 'h36m')[0]
            # pred_keypoints2d_ = convert_kps(pred_keypoints2d.float(), 'smpl_54', 'h36m')[0]
            # for index in range(128):
            #     im = cv2.imread(targets['img_metas'][index]['image_path'])
            #     im = draw_skeletons_image(
            #         pred_keypoints2d_.detach().cpu().numpy()[index],
            #         im,
            #         convention='h36m', palette=[255,0,0])
            #     # cv2.imwrite(f'{index}_pred.png', im)

            #     # im = cv2.imread(targets['img_metas'][index]['image_path'])
            #     im = draw_skeletons_image(
            #         origin_keypoints2d_.detach().cpu().numpy()[index],
            #         im,
            #         convention='h36m',palette=[0,255,255])
            #     cv2.imwrite(f'{index}_gt.png', im)
            #==================================================#

            losses[
                'joints2d_loss_prompt'] = self.compute_keypoints2d_perspective_loss(
                    pred_keypoints3d,  # pred_keypoints3d,
                    target_transl,  # pred_transl,
                    origin_keypoints2d,
                    camera_center,
                    orig_focal_length,
                    img_res=targets['ori_shape'],
                    has_keypoints2d=has_keypoints2d,
                    loss_func=self.loss_joints2d_prompt)

        if self.loss_keypoints2d_hmr is not None:

            gt_keypoints2d = targets['keypoints2d']

            pred_cam = predictions['pred_cam']
            pred_keypoints3d = pred_keypoints3d[..., :3]
            gt_focal_length = 5000

            kp2d_weight = None

            if self.f_ablation:
                gt_focal_length = torch.ones_like(
                    targets['orig_focal_length'].float()).view(-1) * 5000
                scale = targets['scale'][:, 0].float()
                has_f_ids = torch.where(targets['has_focal_length'] > 0)[0]
                has_K_ids = torch.where(targets['has_K'] > 0)[0]
                gt_focal_length[has_f_ids] = targets['orig_focal_length'].view(
                    -1)[has_f_ids] * self.resolution / scale[has_f_ids]
                gt_focal_length[has_K_ids] = targets['K'][
                    has_K_ids, 0,
                    0].float() * self.resolution / scale[has_f_ids]

            if self.use_d_weight:

                # transl = targets['smpl_transl'].float().view(-1, 1, 3)
                # gt_joint_distortion = transl[..., 2] / (gt_model_joints +
                #                                         transl)[..., 2]

                # kp2d_weight = gt_joint_distortion.view(batch_size, -1, 1)
                # kp2d_weight = 1 / torch.clip(kp2d_weight, 0.5, 10)

                # ignore_joints = [
                #     'left_ankle',
                #     'right_ankle',
                #     'head',
                #     'left_wrist',
                #     'right_wrist',
                #     'left_hand',
                #     'right_hand',
                # ]
                # distorted_ids = torch.where(
                #     targets['is_distorted'].view(-1) > 0)[0]
                # non_distorted_ids = torch.where(
                #     targets['is_distorted'].view(-1) == 0)[0]
                # for joint_name in ignore_joints:
                #     index = get_keypoint_idx(joint_name,
                #                              self.convention,
                #                              approximate=True)
                #     kp2d_weight[distorted_ids, index] = 0
                uv_res = targets['d_img'].shape[-1]
                down_scale = self.resolution / uv_res
                has_uvd_ids = torch.where(targets['has_uvd'] > 0)[0]
                kp2d_weight = torch.ones_like(pred_keypoints3d)[..., :1]
                for idx in has_uvd_ids:
                    for kp_idx in range(gt_keypoints2d.shape[1]):
                        x = int(gt_keypoints2d[idx, kp_idx, 0] / down_scale)
                        y = int(gt_keypoints2d[idx, kp_idx, 1] / down_scale)
                        if 0 < x < uv_res and 0 < y < uv_res:
                            kp2d_weight[idx, kp_idx] = targets['d_img'][idx,
                                                                        0][y,
                                                                           x]
                # kp2d_weight = 1 / torch.clip(kp2d_weight, 0.5, 10)
                # kp2d_weight[non_distorted_ids] = 1
            # from avatar3d.utils.demo_utils import draw_skeletons_image
            # im2 = draw_skeletons_image(gt_keypoints2d[1:2].detach().cpu().numpy(), im[:,:, None].repeat(3, -1), convention='smpl_54')
            # has_transl = targets['has_transl']
            # no_transl_ids = torch.where(has_transl == 0)[0]
            # gt_focal_length[no_transl_ids] = 1000
            # gt_focal_length = gt_focal_length.view(-1)
            # pred_keypoints2d = project_points_pred_cam(pred_keypoints3d,
            #                                         pred_cam,
            #                                         focal_length=5000,
            #                                         img_res=self.resolution)

            if self.loss_keypoints2d_prompt is not None:
                # has_uvd_ids = torch.where(targets['has_uvd'] > 0)[0]
                # distortion_kp2d = targets['distortion_kp2d']
                # distortion_kp2d[no_uvd_ids] = 1.
                # gt_keypoints2d[...,
                #                2:3] = gt_keypoints2d[...,
                #                                      2:3] / distortion_kp2d
                if not self.detach_ablation:
                    pred_keypoints3d = pred_output['joints'].clone().detach()
            losses['keypoints2d_loss_hmr'] = self.compute_keypoints2d_hmr_loss(
                pred_keypoints3d,
                pred_cam,
                gt_keypoints2d,
                has_keypoints2d=has_keypoints2d,
                focal_length=gt_focal_length,
                img_res=self.resolution,
                weight=kp2d_weight)

        if self.loss_global_orient is not None:
            losses['global_orient_loss'] = self.compute_global_orient_loss(
                pred_global_orient, gt_global_orient, has_smpl)

        if self.loss_keypoints3d is not None:

            gt_keypoints3d = targets['keypoints3d']
            pred_keypoints3d = pred_output['joints']

            batch_size = gt_keypoints3d.shape[0]
            losses['keypoints3d_loss'] = self.compute_keypoints3d_loss(
                pred_keypoints3d,
                gt_keypoints3d,
                has_keypoints3d=has_keypoints3d)

        if self.loss_camera is not None:
            losses['camera_loss'] = self.compute_camera_loss(pred_cam)
        if self.loss_joints3d is not None:

            pred_keypoints3d = pred_output['joints']
            gt_model_joints3d = torch.cat([
                gt_model_joints,
                gt_model_joint_mask.view(1, -1, 1).repeat_interleave(
                    batch_size, 0).float()
            ], -1)
            losses['joints3d_loss'] = self.compute_joints3d_loss(
                pred_keypoints3d, gt_model_joints3d, has_keypoints3d=None)

        if self.loss_vertex is not None:
            losses['vertex_loss'] = self.compute_vertex_loss(
                pred_vertices, gt_vertices, has_smpl)

        if self.loss_smpl_pose is not None:
            losses['smpl_pose_loss'] = self.compute_smpl_pose_loss(
                pred_pose, gt_pose.float(), has_smpl)
        if self.loss_body_pose is not None:
            losses['body_pose_loss'] = self.compute_body_pose_loss(
                pred_body_pose, gt_body_pose, has_smpl)
        if self.loss_smpl_betas is not None:
            losses['smpl_betas_loss'] = self.compute_smpl_betas_loss(
                pred_betas, gt_betas, has_smpl)

        if self.loss_iuv is not None:
            pred_iuv_img = predictions['pred_iuv_img']
            gt_iuv_img = targets['iuv_img']
            has_uvd = targets['has_uvd']
            losses['loss_iuv'] = self.compute_iuv_loss(pred_iuv_img,
                                                       gt_iuv_img, has_uvd)

        if self.loss_distortion_img is not None:
            pred_d_img = predictions['pred_d_img']
            gt_d_img = targets['d_img']
            has_uvd = targets['has_uvd']
            losses['loss_distortion_img'] = self.compute_distortion_loss(
                pred_d_img, gt_d_img, has_uvd)

        if self.loss_image_grad_u is not None:
            pred_iuv_img = predictions['pred_iuv_img']
            gt_iuv_img = targets['iuv_img']
            has_uvd = targets['has_uvd']
            losses['loss_image_grad_u'] = self.compute_image_grad_u_loss(
                pred_iuv_img[:, 1:2], gt_iuv_img[:, 1:2], has_uvd)
        if self.loss_image_grad_v is not None:
            losses['loss_image_grad_v'] = self.compute_image_grad_v_loss(
                pred_iuv_img[:, 2:3], gt_iuv_img[:, 2:3], has_uvd)

        if self.loss_wrapped_distortion is not None:
            pred_iuv_img = predictions['pred_iuv_img']
            pred_d_img = predictions['pred_d_img']
            gt_d_img = targets['d_img']
            gt_iuv_img = targets['iuv_img']
            has_uvd = targets['has_uvd']
            mask = self.uv_renderer.mask.to(pred_d_img.device)[None, None]
            uv_renderer = self.uv_renderer.to(pred_d_img.device)
            pred_distortion_map = uv_renderer.inverse_wrap(
                pred_iuv_img, pred_d_img) * mask
            gt_distortion_map = uv_renderer.inverse_wrap(gt_iuv_img,
                                                         gt_d_img) * mask
            losses[
                'loss_wrapped_distortion'] = self.compute_wrapped_distortion_loss(
                    pred_distortion_map, gt_distortion_map, has_uvd)

        if self.loss_transl_z is not None:

            has_transl = targets['has_transl']
            gt_transl = targets['smpl_transl']
            transl_weight = 1. / gt_transl[:, 2] * has_transl.view(-1)
            pred_z = predictions['pred_z']
            losses['transl_loss'] = self.compute_transl_loss(
                pred_z, gt_transl[..., 2:3], transl_weight)
        if self.loss_segm_mask is not None:

            center = targets['center'].float()
            scale = targets['scale'].float()[:, 0]
            orig_focal_length = targets['orig_focal_length'].view(-1).float()
            joints2d_conf = targets['keypoints2d'][..., 2:3]

            gt_transl = targets['smpl_transl']
            no_transl_ids = torch.where(targets['has_transl'] == 0)[0]
            gt_cam_ = estimate_cam_weakperspective_batch(
                gt_model_joints[no_transl_ids],
                targets['keypoints2d'][no_transl_ids],
                joints2d_conf[no_transl_ids], joints2d_conf[no_transl_ids],
                self.resolution)
            computed_transl = pred_cam_to_full_transl(
                gt_cam_, center[no_transl_ids], scale[no_transl_ids],
                targets['ori_shape'][no_transl_ids], 5000)

            focal_length_ndc = orig_focal_length * 2 / scale
            focal_length_ndc[no_transl_ids] = 5000. / (self.resolution / 2)
            gt_transl[no_transl_ids, :] = computed_transl.float()

            if 'pred_segm_mask' in predictions:
                pred_segm_mask = predictions['pred_segm_mask']
            seg_res = pred_segm_mask.shape[-1]

            px = targets['ori_shape'][:, 1] / 2
            py = targets['ori_shape'][:, 0] / 2
            gt_seg_mask = self.render_segmask(
                gt_vertices,
                gt_transl,
                center,
                scale,
                focal_length_ndc=focal_length_ndc,
                px=px,
                py=py,
                img_res=seg_res)

            # from avatar3d.utils.torch_utils import image_tensor2numpy
            # import cv2

            # im = image_tensor2numpy(gt_seg_mask[index, 0])
            # cv2.imwrite(f'{index}_seg.png', im)
            # im = image_tensor2numpy(targets['img'][index].permute(1, 2, 0))
            # cv2.imwrite(f'{index}.png', im)

            losses['loss_segm_mask'] = self.compute_part_segmentation_loss(
                pred_segm_mask,
                gt_seg_mask.long().squeeze(1), has_smpl.view(-1))
        return losses

    def prepare_targets(self, data_batch: dict):

        if self.loss_distortion_img is not None or self.loss_iuv is not None:
            is_distorted = data_batch['is_distorted'].view(-1)
            has_uvd = data_batch['has_uvd'].view(-1)
            batch_size = has_uvd.shape[0]
            device = has_uvd.device
            has_smpl = data_batch['has_smpl'].view(-1)
            has_transl = data_batch['has_transl'].view(-1)
            # has_focal_length = data_batch['has_focal_length'].view(-1)
            has_kp3d = data_batch['has_keypoints3d'].view(-1)

            level_a_ids = torch.where(is_distorted > 0)[0]
            level_b_ids = torch.where((has_transl * (1 - is_distorted)) > 0)[0]
            level_c_ids = torch.where((has_smpl * (1 - has_transl)) > 0)[0]
            level_d_ids = torch.where((has_kp3d * (1 - has_smpl)) > 0)[0]
            # level_e_ids = torch.where(has_kp3d == 0)[0]

            # for level a, b, render uvd by gt_transl - pred_cam
            # for level c, render uvd by compute gt_cam from kp2d and model joints, transl use random z and corresponding focal length
            # for level d, get transl use random z and corresponding focal length
            uv_res = self.uv_renderer.resolution[0]

            gt_iuv_img = torch.zeros(batch_size, 3, uv_res, uv_res).to(device)
            gt_d_img = torch.zeros(batch_size, 1, uv_res, uv_res).to(device)
            ori_shape = data_batch['ori_shape']
            gt_transl = data_batch['smpl_transl'].float()
            orig_focal_length = data_batch['orig_focal_length'].float().view(
                -1)
            center, scale = data_batch['center'].float(
            ), data_batch['scale'][:, 0].float()
            gt_keypoints2d = data_batch['keypoints2d']
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
            if len(level_c_ids) + len(level_d_ids):

                joints2d_conf = gt_keypoints2d[..., 2:3]

                gt_kp3d = data_batch['keypoints3d'].clone()

                right_hip_idx = get_keypoint_idx('right_hip_extra',
                                                 self.convention)
                left_hip_idx = get_keypoint_idx('left_hip_extra',
                                                self.convention)

                gt_pelvis = (gt_kp3d[:, right_hip_idx, :] +
                             gt_kp3d[:, left_hip_idx, :]) / 2

                gt_kp3d = gt_kp3d - gt_pelvis[:, None, :]

                gt_model_joints[level_d_ids] = gt_kp3d[level_d_ids,
                                                       ..., :3].float()

                level_cd_ids = torch.cat([level_c_ids, level_d_ids])
                gt_cam_cd = estimate_cam_weakperspective_batch(
                    gt_model_joints[level_cd_ids],
                    data_batch['keypoints2d'][level_cd_ids],
                    joints2d_conf[level_cd_ids], joints2d_conf[level_cd_ids],
                    self.resolution)

                computed_transl = pred_cam_to_full_transl(
                    gt_cam_cd, center[level_cd_ids], scale[level_cd_ids],
                    data_batch['ori_shape'][level_cd_ids],
                    orig_focal_length[level_cd_ids])
                gt_transl[level_cd_ids] = computed_transl
                # has_transl[level_cd_ids] = 1

            if len(level_b_ids) + len(level_a_ids):
                if self.full_uvd:
                    render_ids = torch.cat([level_a_ids, level_b_ids])
                else:
                    render_ids = level_a_ids
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

                K = data_batch['K'].float()

                if len(level_a_ids):
                    gt_iuv_img_a, gt_d_img_a = self.render_gt_iuvd_real(
                        gt_vertices[level_a_ids],
                        gt_transl[level_a_ids],
                        img_res=uv_res,
                        center=center[level_a_ids],
                        scale=scale[level_a_ids],
                        orig_K=K[level_a_ids])

                    gt_iuv_img[level_a_ids] = gt_iuv_img_a
                    gt_d_img[level_a_ids] = gt_d_img_a

                px = K[:, 0, 2]
                py = K[:, 1, 2]

                gt_cam = full_transl_to_pred_cam(gt_transl, center, scale,
                                                 ori_shape, orig_focal_length,
                                                 px, py)

                angle = []
                for j in range(batch_size):
                    angle.append(-torch.Tensor(
                        [float(data_batch['img_metas'][j]['rotation'])]))
                angle = torch.cat(angle).to(device)
                gt_cam_new = rotate_smpl_cam(gt_cam, angle, gt_model_joints)

                gt_transl_b = pred_cam_to_full_transl(gt_cam_new, center,
                                                      scale, ori_shape,
                                                      orig_focal_length, px,
                                                      py)
                gt_transl[level_b_ids] = gt_transl_b[level_b_ids]

                if self.full_uvd:
                    if len(level_b_ids):
                        gt_iuv_img_b, gt_d_img_b = self.render_gt_iuvd_cam(
                            gt_vertices[level_b_ids], gt_cam_new[level_b_ids],
                            uv_res, orig_focal_length[level_b_ids] /
                            scale[level_b_ids] * uv_res)

                        gt_iuv_img[level_b_ids] = gt_iuv_img_b
                        gt_d_img[level_b_ids] = gt_d_img_b

                has_uvd[render_ids] = 1

                # gt_cam[level_b_ids] = gt_cam_new
                #==================================================
                # import kornia

                # gt_cam_ = estimate_cam_weakperspective_batch(
                #     gt_model_joints[index, None],
                #     data_batch['keypoints2d'][index, None],
                #     joints2d_conf[index, None], joints2d_conf[index,
                #                                               None], self.resolution)

                # transl = pred_cam_to_transl(gt_cam_, 5000, self.resolution)

                # K = data_batch['K'][index, None]
                # cameras_real = build_cameras(
                #     dict(type='perspective',
                #          in_ndc=False,
                #          K=K,
                #          resolution=data_batch['ori_shape'][index, None],
                #          convention='opencv')).to(device)
                # smpl_origin_orient = data_batch['smpl_origin_orient'].float()
                # origin_output = self.body_model_train(
                #     betas=gt_betas,
                #     body_pose=gt_body_pose.float(),
                #     global_orient=smpl_origin_orient.float())
                # origin_vertices = origin_output['vertices']

                # zero_output = self.body_model_train(
                #     betas=gt_betas,
                #     body_pose=gt_body_pose.float(),
                #     global_orient=smpl_origin_orient.float() * 0)
                # zero_vertices = zero_output['vertices']

                # gt_verts2d = cameras_real.transform_points_screen(
                #     origin_vertices[index, None] +
                #     data_batch['smpl_transl'][index].view(-1, 1, 3))
                # trans = data_batch['trans']
                # gt_verts2d[..., 2] = 1
                # gt_verts2d = torch.einsum('bij,bkj->bki', trans[index,
                #                                                 None].float(),
                #                           gt_verts2d)

                # from avatar3d.utils.torch_utils import image_tensor2numpy
                # import cv2
                # im = data_batch['img'][index].permute(1, 2, 0)
                # im = image_tensor2numpy(im)
                # gt_verts2d_ = torch.clip(gt_verts2d.long(), 0,
                #                          223).detach().cpu().numpy()
                # im[gt_verts2d_[..., 1], gt_verts2d_[..., 0]] = 255
                # cv2.imwrite(f'{index}_.png', im)

                # verts3d = zero_vertices[index, None]
                # ones = torch.ones([verts3d.shape[0], verts3d.shape[1], 1],
                #                   device=verts3d.device)
                # verts_homo = torch.cat((verts3d, ones), 2)

                # K_ = torch.eye(3, 3)[None].to(device)
                # K_[:, 0, 0] = K_[:, 1, 1] = 5000
                # K_[:, 0, 2] = K_[:, 1, 2] = (self.resolution/2)
                # RT_ = kornia.geometry.solve_pnp_dlt(verts3d+transl[None], gt_verts2d, K_)
                # RT = torch.cat([
                #     RT_,
                #     torch.Tensor([0, 0, 0, 1]).view(1, 1, 4).to(RT_.device)
                # ], 1)

                # cameras = build_cameras(
                #     dict(type='perspective',
                #          convention='opencv',
                #          in_ndc=False,
                #          image_size=(self.resolution, self.resolution),
                #          K=K_,
                #          device=device))
                # new_points2d = cameras.transform_points_screen(
                #     (RT @ (verts_homo[0].view(-1, 4, 1)))[:, :3,
                #                                           0][None])[..., :2]
                # im = data_batch['img'][index].permute(1, 2, 0)
                # im = image_tensor2numpy(im)
                # new_points2d_ = torch.clip(new_points2d.long(), 0,
                #                            223).detach().cpu().numpy()
                # im[new_points2d_[..., 1], new_points2d_[..., 0]] = 255
                # cv2.imwrite(f'{index}_new.png', im)

                #==================================================#
                # from avatar3d.utils.torch_utils import image_tensor2numpy
                # import cv2

                # for i,index in enumerate(level_a_ids):
                #     im = image_tensor2numpy(data_batch['img'][index].permute(
                #         1, 2, 0))
                #     cv2.imwrite(f'level_a/{index}.png', im)
                #     im = image_tensor2numpy(gt_iuv_img_a[i].permute(1, 2, 0))
                #     cv2.imwrite(f'level_a/{index}_iuv.png', im)
                # for i,index in enumerate(level_b_ids):
                #     im = image_tensor2numpy(data_batch['img'][index].permute(
                #         1, 2, 0))
                #     cv2.imwrite(f'level_b/{index}.png', im)
                #     im = image_tensor2numpy(gt_iuv_img_b[i].permute(1, 2, 0))
                #     cv2.imwrite(f'level_b/{index}_iuv.png', im)
                #==================================================#
                # transl = pred_cam_to_transl(gt_cam_new,
                #                             orig_focal_length / scale * self.resolution,
                #                             self.resolution)

                # K_ = torch.eye(3)[None].repeat_interleave(batch_size, 0)
                # K_[:, 0, 0] = orig_focal_length / scale * self.resolution
                # K_[:, 1, 1] = orig_focal_length / scale * self.resolution
                # K_[:, 0, 2] = self.resolution / 2.
                # K_[:, 1, 2] = self.resolution / 2.

                # cameras = build_cameras(
                #     dict(type='perspective',
                #          image_size=(self.resolution, self.resolution),
                #          in_ndc=False,
                #          K=K_,
                #          convention='opencv')).to(device)

                # cameras = build_cameras(
                #     dict(type='perspective',
                #          image_size=ori_shape,
                #          in_ndc=False,
                #          K=K,
                #  convention='opencv')).to(device)

                # origin_keypoints2d = gt_keypoints2d.clone()[..., :2]
                # origin_keypoints2d = (
                #     origin_keypoints2d -
                #     (self.resolution/2)) / self.resolution * scale[:, None, None] + center.view(-1, 1, 2)
                # origin_keypoints2d = torch.cat(
                #     [origin_keypoints2d, data_batch['keypoints2d'][..., 2:3]],
                #     -1)

                # from avatar3d.utils.visualize_smpl import vis_smpl
                # import cv2
                # from avatar3d.utils.demo_utils import draw_skeletons_image
                # from avatar3d.utils.torch_utils import image_tensor2numpy
                # for index in level_b_ids:
                #     index = int(index)
                #     im = image_tensor2numpy(data_batch['img'][index].permute(
                #         1, 2, 0))
                #     im =draw_skeletons_image(
                #         gt_keypoints2d.cpu().numpy()[index],
                #         im,
                #         convention='smpl_54')
                #     cv2.imwrite(f'{index}_kp2d.png',im)
                # # transl_[..., :2] = 0
                # vis_smpl(verts=gt_vertices[index][None] +
                #         transl[index].view(1, 1, 3),
                #         cameras=cameras[index],
                #         image_array=im[None],
                #         device=gt_transl[index].view(1, 1, 3).device,
                #         body_model=self.body_model_train,
                #         output_path=f'{index}_pred.png',
                #         return_tensor=False,
                #         alpha=0.9,
                #         overwrite=True,
                #         no_grad=True)

                # im_orig = cv2.imread(
                #     data_batch['img_metas'][index]['image_path'])
                # im_orig =draw_skeletons_image(
                #     origin_keypoints2d.cpu().numpy()[index],
                #     im_orig,
                #     convention='smpl_54')
                # vis_smpl(verts=gt_vertices[index][None] +
                #          gt_transl_b[index].view(1, 1, 3),
                #          cameras=cameras[index],
                #          image_array=im_orig[None],
                #          device=transl.device,
                #          body_model=self.body_model_train,
                #          output_path=f'{index}_orig.png',
                #          return_tensor=False,
                #          alpha=0.9,
                #          overwrite=True,
                #          no_grad=True)

                # from avatar3d.utils.torch_utils import image_tensor2numpy
                # import cv2
                # for index in level_b_ids:
                #     cv2.imwrite(
                #         f'{index}_img.png',
                #         image_tensor2numpy(data_batch['img'][index].permute(
                #             1, 2, 0)))
                #     iuv = image_tensor2numpy(gt_iuv_img[index].permute(
                #         1, 2, 0))
                #     iuv = cv2.resize(iuv, (self.resolution, self.resolution))
                #     cv2.imwrite(f'{index}_iuv.png', iuv)
                #==================================================#
            data_batch['iuv_img'] = gt_iuv_img
            data_batch['d_img'] = gt_d_img
            data_batch['smpl_transl'] = gt_transl
            data_batch['orig_focal_length'] = orig_focal_length

            data_batch['has_uvd'] = has_uvd

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
            extracted_data = self.extractor(data_batch[self.extractor_key])
        else:
            extracted_data = dict()

        data_batch.update(extracted_data)
        batch_size = data_batch['img'].shape[0]

        features = self.backbone(data_batch['img'])
        device = data_batch['img'].device
        if self.neck is not None:
            features = self.neck(features)

        if self.iuvd_head is not None:
            predictions_iuvd = self.iuvd_head(features)
        else:
            predictions_iuvd = dict()
        predictions.update(predictions_iuvd)
        if self.hmr_head is not None:
            head_data = []
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
            if 'fpn_feat' in self.head_keys:
                data_batch['fpn_feat'] = predictions['fpn_feat']
            if 'distortion_feat' in self.head_keys:
                data_batch['distortion_feat'] = predictions['distortion_feat']
            for key in self.head_keys:
                head_data.append(data_batch[key])
            predictions_hmr = self.hmr_head(features, *head_data)
        else:
            predictions_hmr = dict()

        predictions.update(predictions_hmr)
        predictions.update(extracted_data)

        pred_cam = predictions['pred_cam']

        center, scale, ori_shape = data_batch['center'].float(
        ), data_batch['scale'][:, 0].float(), data_batch['ori_shape'].float()

        if self.loss_keypoints2d_cliff is not None:
            H, W = ori_shape.unbind(-1)
            orig_focal_length = torch.sqrt(H**2 + W**2).view(-1)
            pred_focal_length = orig_focal_length / (scale / 2)
            predictions.update(pred_focal_length=pred_focal_length)
            predictions.update(orig_focal_length=orig_focal_length)
            predictions.update(pred_transl=pred_cam_to_transl(
                pred_cam, orig_focal_length, scale).float())
            predictions.update(full_transl=pred_cam_to_full_transl(
                pred_cam, center, scale, data_batch['ori_shape'],
                orig_focal_length).float())
        elif self.loss_keypoints2d_spec is not None:

            pred_focal_length = 1. / torch.tan(predictions['cam_vfov'] / 2)
            orig_focal_length = pred_focal_length * data_batch[
                'ori_shape'][:, 0] / 2
            predictions.update(pred_focal_length=pred_focal_length)
            predictions.update(orig_focal_length=orig_focal_length)

            predictions.update(pred_transl=pred_cam_to_transl(
                pred_cam, pred_focal_length *
                (self.resolution / 2), self.resolution))
            predictions.update(full_transl=pred_cam_to_full_transl(
                pred_cam, center, scale, data_batch['ori_shape'],
                orig_focal_length).float())
        elif self.loss_keypoints2d_prompt is not None:
            if 'pred_z' in predictions:
                pred_focal_length = predictions['pred_z'].view(
                    -1) * pred_cam[:, 0]
            elif 'pred_transl' in predictions:
                pred_focal_length = predictions['pred_transl'][:, 2].view(
                    -1) * pred_cam[:, 0]
            predictions.update(pred_transl=pred_cam_to_transl(
                pred_cam, pred_focal_length *
                (self.resolution / 2), self.resolution).float())
            if 'pred_transl' not in predictions:
                tmp_transl = torch.cat([
                    torch.zeros(batch_size, 2).to(device),
                    predictions['pred_z']
                ], -1)

            else:
                tmp_transl = predictions['pred_transl']
            predictions.update(full_transl=merge_cam_to_full_transl(
                pred_cam, center, scale, data_batch['ori_shape'], tmp_transl))

            predictions.update(pred_focal_length=pred_focal_length)
            predictions.update(orig_focal_length=pred_focal_length * scale / 2)
        else:
            pred_focal_length = 5000 / (self.resolution / 2)
            predictions.update(pred_transl=pred_cam_to_transl(
                pred_cam, 5000, self.resolution))
            predictions.update(full_transl=pred_cam_to_full_transl(
                pred_cam, center, scale, data_batch['ori_shape'], 5000 /
                self.resolution * scale).float())
            predictions.update(pred_focal_length=pred_focal_length)
            predictions.update(orig_focal_length=pred_focal_length * scale / 2)

        pred_betas = predictions['pred_shape']
        pred_pose = predictions['pred_pose']
        pred_global_orient = predictions['pred_pose'][:, :1]

        pred_output = self.body_model_test(betas=pred_betas,
                                           body_pose=pred_pose[:, 1:],
                                           global_orient=pred_global_orient,
                                           pose2rot=False)
        pred_keypoints3d = pred_output['joints']
        pred_vertices = pred_output['vertices']

        # from pytorch3d.structures import Meshes
        # from mmhuman3d.utils.mesh_utils import save_meshes_as_objs

        # gt_body_pose = data_batch['smpl_body_pose'].float()
        # gt_betas = data_batch['smpl_betas'].float()
        # gt_global_orient = data_batch['smpl_global_orient'].float()
        # gt_output = self.body_model_train(
        #     betas=gt_betas,
        #     body_pose=gt_body_pose.float(),
        #     global_orient=gt_global_orient)
        # gt_vertices = gt_output['vertices']

        # mesh = Meshes(gt_vertices[index:index+1],
        #               faces=self.body_model_test.faces_tensor[None])
        # save_meshes_as_objs('0.obj', mesh)

        # mesh = Meshes(pred_vertices[index:index+1],
        #               faces=self.body_model_test.faces_tensor[None])
        # save_meshes_as_objs('1.obj', mesh)

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

            origin_keypoints2d = (
                origin_keypoints2d - (self.resolution / 2)
            ) / self.resolution * scale[:, None, None] + center.view(-1, 1, 2)
            origin_keypoints2d = torch.cat([
                origin_keypoints2d / orig_wh.view(batch_size, 1, 2),
                gt_keypoints2d[..., 2:3]
            ], -1)

            predictions.update(origin_keypoints2d=origin_keypoints2d)

        predictions.update(img=data_batch['img'],
                           pred_vertices=pred_vertices,
                           sample_idx=data_batch['sample_idx'],
                           ori_shape=data_batch['ori_shape'],
                           pred_keypoints2d=pred_keypoints2d,
                           img_metas=data_batch['img_metas'])

        if self.miou or self.pmiou:
            test_res = self.resolution
            pred_focal_length = predictions['pred_focal_length']
            K = data_batch['K']
            gt_transl = data_batch['smpl_transl']
            ori_shape = data_batch['ori_shape']
            gt_focal_length = data_batch['orig_focal_length'].float().view(-1)
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
                # from IPython import embed
                # embed()

                gt_mask = self.render_segmask(
                    vertices=gt_vertices,
                    transl=gt_transl,
                    center=center,
                    scale=scale,
                    focal_length_ndc=gt_focal_length / scale * 2,
                    px=px,
                    py=py,
                    img_res=test_res)

                pred_mask = self.render_segmask(
                    vertices=pred_vertices,
                    transl=predictions['full_transl'],
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
                    focal_length_ndc=gt_focal_length / scale * 2,
                    px=px,
                    py=py,
                    img_res=test_res)

                pred_mask = self.render_segmask(
                    vertices=pred_vertices,
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
        if 'orig_focal_length' in data_batch:
            predictions['gt_focal_length'] = data_batch['orig_focal_length']
        else:
            predictions['gt_focal_length'] = orig_focal_length * 0

        if 'smpl_transl' in data_batch:
            predictions['smpl_transl'] = data_batch['smpl_transl']
        else:
            predictions['smpl_transl'] = predictions['full_transl'] * 0

        predictions['batch_pmiou'] = batch_piou * 100
        if self.visualizer is not None and self.vis:
            self.visualizer(predictions)

        all_preds = {}
        all_preds['batch_miou'] = batch_iou * 100
        all_preds['batch_pmiou'] = batch_piou * 100
        all_preds['keypoints_3d'] = pred_keypoints3d
        all_preds['smpl_pose'] = pred_pose
        all_preds['smpl_beta'] = pred_betas
        all_preds['camera'] = pred_cam
        all_preds['vertices'] = pred_vertices
        image_path = []
        for img_meta in data_batch['img_metas']:
            image_path.append(img_meta['image_path'])
        all_preds['image_path'] = image_path
        all_preds['image_idx'] = data_batch['sample_idx']

        all_preds['pred_keypoints2d'] = predictions['pred_keypoints2d']
        if 'origin_keypoints2d' in predictions:
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
        batch_size = keypoints2d_conf.shape[0]
        if pred_keypoints3d.shape[-1] >= 4:
            kp3d_conf = pred_keypoints3d[..., 3:4]
        else:
            kp3d_conf = None
        if kp3d_conf is not None:
            keypoints2d_conf = keypoints2d_conf.repeat(
                1, 1, 2) * kp3d_conf.view(batch_size, -1, 1)
        else:
            keypoints2d_conf = keypoints2d_conf.repeat(1, 1, 2)
        gt_keypoints2d = gt_keypoints2d[:, :, :2].float()
        pred_keypoints3d = pred_keypoints3d[:, :, :3].float()

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
            keypoints3d_conf: Optional[torch.Tensor] = None,
            weight=None,
            loss_func=None):
        """Compute loss for 2d keypoints."""

        pred_keypoints3d = pred_keypoints3d[..., :3]
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
        if loss_func is None:
            loss_func = self.loss_keypoints2d_hmr
        loss = loss_func(
            pred_keypoints2d,
            gt_keypoints2d,
            reduction_override='none',
        )
        if weight is not None:
            keypoints2d_conf *= weight
        if keypoints3d_conf is not None:
            keypoints2d_conf *= keypoints3d_conf
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

    def compute_camera_loss(self, cameras: torch.Tensor):
        """Compute loss for predicted camera parameters."""
        loss = self.loss_camera(cameras)
        return loss

    def compute_iuv_loss(self, pred_iuv, gt_iuv, has_iuv):
        batch_size = gt_iuv.shape[0]
        valid = has_iuv.float().view(batch_size) > 0
        loss = self.loss_iuv(pred_iuv[valid], gt_iuv[valid])
        return loss

    def compute_part_segmentation_loss(self, pred_segmask, gt_segmask,
                                       has_smpl):
        batch_size = pred_segmask.shape[0]
        valid = has_smpl.float().view(batch_size) > 0
        loss = self.loss_segm_mask(pred_segmask[valid], gt_segmask[valid])
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

    def render_gt_iuvd_cam(self,
                           gt_vertices: torch.Tensor,
                           gt_cam: torch.Tensor,
                           img_res: Optional[int] = 224,
                           focal_length: Optional[int] = 1000):
        """Compute loss for part segmentations."""
        device = gt_vertices.device
        uv_renderer = self.uv_renderer.to(device)
        depth_renderer = self.depth_renderer.to(device)

        batch_size = gt_vertices.shape[0]

        gt_transl = pred_cam_to_transl(
            pred_camera=gt_cam,
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

    def render_gt_iuvd_real(
        self,
        gt_vertices: torch.Tensor,
        gt_transl: torch.Tensor,
        center: torch.Tensor,
        scale: torch.Tensor,
        orig_K: torch.Tensor,
        img_res: Optional[int] = 224,
    ):
        """Compute loss for part segmentations."""
        device = gt_vertices.device
        uv_renderer = self.uv_renderer.to(device)
        depth_renderer = self.depth_renderer.to(device)

        batch_size = gt_vertices.shape[0]

        gt_transl = gt_transl.unsqueeze(1)
        K = torch.eye(3)[None].repeat_interleave(batch_size, 0)

        orig_focal_length = orig_K[:, 0, 0]
        K[:, 0, 0] = orig_focal_length / scale * img_res
        K[:, 1, 1] = orig_focal_length / scale * img_res
        cx, cy = center.unbind(-1)
        ori_cx = orig_K[:, 0, 2]
        ori_cy = orig_K[:, 1, 2]
        K[:, 0, 2] = img_res / 2. - img_res * (cx - ori_cx) / scale
        K[:, 1, 2] = img_res / 2. - img_res * (cy - ori_cy) / scale
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

    def render_segmask(
        self,
        vertices: torch.Tensor,
        transl: torch.Tensor,
        center: torch.Tensor,
        scale: torch.Tensor,
        focal_length_ndc: torch.Tensor,
        px: torch.Tensor,
        py: torch.Tensor,
        img_res: Optional[int] = 224,
    ):
        """Compute loss for part segmentations."""
        device = vertices.device
        seg_renderer = build_renderer(
            dict(type='segmentation', resolution=img_res, num_class=24))
        seg_renderer = seg_renderer.to(device)

        batch_size = vertices.shape[0]

        transl = transl.unsqueeze(1)
        K = torch.eye(3)[None].repeat_interleave(batch_size, 0)

        K[:, 0, 0] = focal_length_ndc * img_res / 2
        K[:, 1, 1] = focal_length_ndc * img_res / 2
        cx, cy = center.unbind(-1)

        K[:, 0, 2] = img_res / 2. - img_res * (cx - px) / scale
        K[:, 1, 2] = img_res / 2. - img_res * (cy - py) / scale
        cameras = build_cameras(
            dict(type='perspective',
                 image_size=(img_res, img_res),
                 in_ndc=False,
                 K=K,
                 convention='opencv')).to(device)

        mesh = Meshes(verts=vertices + transl,
                      faces=self.body_model_train.faces_tensor[None].repeat(
                          batch_size, 1, 1)).to(device)

        colors = torch.zeros_like(vertices)
        body_segger = body_segmentation('smpl')
        for i, k in enumerate(body_segger.keys()):
            colors[:, body_segger[k]] = i + 1
        mesh.textures = build_textures(
            dict(type='TexturesNearest', verts_features=colors))

        segmask = seg_renderer(mesh, cameras)
        segmask = segmask.permute(0, 3, 1, 2)
        return segmask

    def render_mask(
        self,
        vertices: torch.Tensor,
        transl: torch.Tensor,
        center: torch.Tensor,
        scale: torch.Tensor,
        focal_length_ndc: torch.Tensor,
        px: torch.Tensor,
        py: torch.Tensor,
        img_res: Optional[int] = 224,
    ):
        """Compute loss for part segmentations."""
        device = vertices.device

        sihouette_renderer = build_renderer(
            dict(type='Silhouette', resolution=img_res))
        sihouette_renderer = sihouette_renderer.to(device)

        batch_size = vertices.shape[0]

        transl = transl.unsqueeze(1)
        K = torch.eye(3)[None].repeat_interleave(batch_size, 0)

        K[:, 0, 0] = focal_length_ndc * img_res / 2
        K[:, 1, 1] = focal_length_ndc * img_res / 2
        cx, cy = center.unbind(-1)

        K[:, 0, 2] = img_res / 2. - img_res * (cx - px) / scale
        K[:, 1, 2] = img_res / 2. - img_res * (cy - py) / scale
        cameras = build_cameras(
            dict(type='perspective',
                 image_size=(img_res, img_res),
                 in_ndc=False,
                 K=K,
                 convention='opencv')).to(device)

        mesh = Meshes(verts=vertices + transl,
                      faces=self.body_model_train.faces_tensor[None].repeat(
                          batch_size, 1, 1)).to(device)
        colors = torch.ones_like(vertices)
        mesh.textures = build_textures(
            dict(type='TexturesVertex', verts_features=colors))
        mask = sihouette_renderer(mesh, cameras)[..., 3:] > 0
        return mask

    def run_registration(
            self,
            predictions: dict,
            targets: dict,
            threshold: Optional[float] = 10.0,
            focal_length: Optional[float] = 5000.0,
            img_res: Optional[Union[Tuple[int], int]] = 224) -> dict:
        """Run registration on 2D keypoinst in predictions to obtain SMPL
        parameters as pseudo ground truth.

        Args:
            predictions (dict): predicted SMPL parameters are used for
                initialization.
            targets (dict): existing ground truths with 2D keypoints
            threshold (float, optional): the threshold to update fits
                dictionary. Default: 10.0.
            focal_length (tuple(int) | int, optional): camera focal_length
            img_res (int, optional): image resolution

        Returns:
            targets: contains additional SMPL parameters
        """

        img_metas = targets['img_metas']
        dataset_name = [meta['dataset_name'] for meta in img_metas
                        ]  # name of the dataset the image comes from

        indices = targets['sample_idx'].squeeze()
        is_flipped = targets['is_flipped'].squeeze().bool(
        )  # flag that indicates whether image was flipped
        # during data augmentation
        rot_angle = targets['rotation'].squeeze(
        )  # rotation angle used for data augmentation Q
        gt_betas = targets['smpl_betas'].float()
        gt_global_orient = targets['smpl_global_orient'].float()
        gt_pose = targets['smpl_body_pose'].float().view(-1, 69)

        pred_rotmat = predictions['pred_pose'].detach().clone()
        pred_betas = predictions['pred_shape'].detach().clone()
        pred_cam = predictions['pred_cam'].detach().clone()
        pred_cam_t = torch.stack([
            pred_cam[:, 1], pred_cam[:, 2], 2 * focal_length /
            (img_res * pred_cam[:, 0] + 1e-9)
        ],
                                 dim=-1)

        gt_keypoints_2d = targets['keypoints2d'].float()
        # num_keypoints = gt_keypoints_2d.shape[1]

        has_smpl = targets['has_smpl'].view(
            -1).bool()  # flag that indicates whether SMPL parameters are valid
        batch_size = has_smpl.shape[0]
        device = has_smpl.device

        # Get GT vertices and model joints
        # Note that gt_model_joints is different from gt_joints as
        # it comes from SMPL
        gt_out = self.body_model_train(betas=gt_betas,
                                       body_pose=gt_pose,
                                       global_orient=gt_global_orient)
        # TODO: support more convention
        # assert num_keypoints == 49
        gt_model_joints = gt_out['joints']
        gt_vertices = gt_out['vertices']

        # Get current best fits from the dictionary
        opt_pose, opt_betas = self.fits_dict[(dataset_name, indices.cpu(),
                                              rot_angle.cpu(),
                                              is_flipped.cpu())]

        opt_pose = opt_pose.to(device)
        opt_betas = opt_betas.to(device)
        opt_output = self.body_model_train(betas=opt_betas,
                                           body_pose=opt_pose[:, 3:],
                                           global_orient=opt_pose[:, :3])
        opt_joints = opt_output['joints']
        opt_vertices = opt_output['vertices']

        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        # Estimate camera translation given the model joints and 2D keypoints
        # by minimizing a weighted least squares loss
        gt_cam_t = estimate_translation(gt_model_joints,
                                        gt_keypoints_2d_orig,
                                        focal_length=focal_length,
                                        img_size=img_res)

        opt_cam_t = estimate_translation(opt_joints,
                                         gt_keypoints_2d_orig,
                                         focal_length=focal_length,
                                         img_size=img_res)

        with torch.no_grad():
            loss_dict = self.registrant.evaluate(
                global_orient=opt_pose[:, :3],
                body_pose=opt_pose[:, 3:],
                betas=opt_betas,
                transl=opt_cam_t,
                keypoints2d=gt_keypoints_2d_orig[:, :, :2],
                keypoints2d_conf=gt_keypoints_2d_orig[:, :, 2],
                reduction_override='none')
        opt_joint_loss = loss_dict['keypoint2d_loss'].sum(dim=-1).sum(dim=-1)

        if self.registration_mode == 'in_the_loop':
            # Convert predicted rotation matrices to axis-angle
            pred_rotmat_hom = torch.cat([
                pred_rotmat.detach().view(-1, 3, 3).detach(),
                torch.tensor([0, 0, 1], dtype=torch.float32,
                             device=device).view(1, 3, 1).expand(
                                 batch_size * 24, -1, -1)
            ],
                                        dim=-1)
            pred_pose = rotmat_to_aa(pred_rotmat_hom).contiguous().view(
                batch_size, -1)
            # tgm.rotation_matrix_to_angle_axis returns NaN for 0 rotation,
            # so manually hack it
            pred_pose[torch.isnan(pred_pose)] = 0.0

            registrant_output = self.registrant(
                keypoints2d=gt_keypoints_2d_orig[:, :, :2],
                keypoints2d_conf=gt_keypoints_2d_orig[:, :, 2],
                init_global_orient=pred_pose[:, :3],
                init_transl=pred_cam_t,
                init_body_pose=pred_pose[:, 3:],
                init_betas=pred_betas,
                return_joints=True,
                return_verts=True,
                return_losses=True)
            new_opt_vertices = registrant_output[
                'vertices'] - pred_cam_t.unsqueeze(1)
            new_opt_joints = registrant_output[
                'joints'] - pred_cam_t.unsqueeze(1)

            new_opt_global_orient = registrant_output['global_orient']
            new_opt_body_pose = registrant_output['body_pose']
            new_opt_pose = torch.cat(
                [new_opt_global_orient, new_opt_body_pose], dim=1)

            new_opt_betas = registrant_output['betas']
            new_opt_cam_t = registrant_output['transl']
            new_opt_joint_loss = registrant_output['keypoint2d_loss'].sum(
                dim=-1).sum(dim=-1)

            # Will update the dictionary for the examples where the new loss
            # is less than the current one
            update = (new_opt_joint_loss < opt_joint_loss)

            opt_joint_loss[update] = new_opt_joint_loss[update]
            opt_vertices[update, :] = new_opt_vertices[update, :]
            opt_joints[update, :] = new_opt_joints[update, :]
            opt_pose[update, :] = new_opt_pose[update, :]
            opt_betas[update, :] = new_opt_betas[update, :]
            opt_cam_t[update, :] = new_opt_cam_t[update, :]

            self.fits_dict[(dataset_name, indices.cpu(), rot_angle.cpu(),
                            is_flipped.cpu(),
                            update.cpu())] = (opt_pose.cpu(), opt_betas.cpu())

        # Replace extreme betas with zero betas
        opt_betas[(opt_betas.abs() > 3).any(dim=-1)] = 0.

        # Replace the optimized parameters with the ground truth parameters,
        # if available
        opt_vertices[has_smpl, :, :] = gt_vertices[has_smpl, :, :]
        opt_cam_t[has_smpl, :] = gt_cam_t[has_smpl, :]
        opt_joints[has_smpl, :, :] = gt_model_joints[has_smpl, :, :]
        opt_pose[has_smpl, 3:] = gt_pose[has_smpl, :]
        opt_pose[has_smpl, :3] = gt_global_orient[has_smpl, :]
        opt_betas[has_smpl, :] = gt_betas[has_smpl, :]

        # Assert whether a fit is valid by comparing the joint loss with
        # the threshold
        valid_fit = (opt_joint_loss < threshold).to(device)
        valid_fit = valid_fit | has_smpl
        targets['valid_fit'] = valid_fit

        targets['opt_vertices'] = opt_vertices
        targets['opt_cam_t'] = opt_cam_t
        targets['opt_joints'] = opt_joints
        targets['opt_pose'] = opt_pose
        targets['opt_betas'] = opt_betas

        return targets
