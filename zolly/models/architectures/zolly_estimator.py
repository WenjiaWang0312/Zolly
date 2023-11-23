import torch
import math
from typing import Optional, Tuple, Union
import numpy as np
from pytorch3d.structures import Meshes

from zolly.models.necks.builder import build_neck

from mmhuman3d.utils.geometry import batch_rodrigues

from mmhuman3d.core.conventions.segmentation import body_segmentation
from mmhuman3d.models.architectures.base_architecture import BaseArchitecture
from zolly.utils.torch_utils import dict2numpy
from zolly.structures.meshes.utils import MeshSampler
from zolly.cameras.builder import build_cameras
from zolly.models.visualizers.builder import build_visualizer
from zolly.models.body_models.mappings import get_keypoint_idx, convert_kps
from zolly.models.body_models.builder import build_body_model
from zolly.models.losses.builder import build_loss
from zolly.models.heads.builder import build_head
from zolly.render.builder import build_renderer
from zolly.models.backbones.builder import build_backbone
from zolly.transforms.transform2d.transform2d import warp_feature
from zolly.cameras.utils import (pred_cam_to_transl, merge_cam_to_full_transl,
                                 estimate_cam_weakperspective_batch,
                                 pred_cam_to_full_transl,
                                 pred_transl_to_pred_cam,
                                 project_points_pred_cam,
                                 project_points_focal_length_pixel)
from zolly.render.explicit.pytorch3d_wrapper.textures.builder import (
    build_textures)
from zolly.cameras.utils import rotate_smpl_cam, full_transl_to_pred_cam


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


class ZollyEstimator(BaseArchitecture):
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
        backbone: Optional[Union[dict, None]] = None,
        neck: Optional[Union[dict, None]] = None,
        verts_head: Optional[Union[dict, None]] = None,
        iuvd_head: Optional[Union[dict, None]] = None,
        uv_renderer: Optional[Union[dict, None]] = None,
        depth_renderer: Optional[Union[dict, None]] = None,
        head_keys: Union[list, tuple] = (),
        full_uvd: bool = False,
        pred_kp3d: bool = True,
        test_joints3d: bool = True,
        resolution: int = 224,
        body_model_train: Optional[Union[dict, None]] = None,
        body_model_test: Optional[Union[dict, None]] = None,
        mesh_sampler: Optional[Union[dict, None]] = None,
        convention: Optional[str] = 'human_data',
        convention_pred: Optional[str] = 'h36m_eval_j14_smpl',
        freeze_modules: Tuple[str] = (),
        use_deco_warp: bool = False,
        use_d_weight: bool = False,
        uv_res: int = 56,
        ##
        loss_keypoints2d_zolly: Optional[Union[dict, None]] = None,
        loss_keypoints2d_hmr: Optional[Union[dict, None]] = None,
        loss_keypoints2d: Optional[Union[dict, None]] = None,
        loss_joints2d: Optional[Union[dict, None]] = None,
        loss_joints2d_hmr: Optional[Union[dict, None]] = None,
        loss_joints2d_zolly: Optional[Union[dict, None]] = None,
        loss_keypoints3d: Optional[Union[dict, None]] = None,
        loss_joints3d: Optional[Union[dict, None]] = None,
        loss_vertex: Optional[Union[dict, None]] = None,
        loss_vertex_sub1: Optional[Union[dict, None]] = None,
        loss_vertex_sub2: Optional[Union[dict, None]] = None,
        loss_camera: Optional[Union[dict, None]] = None,
        ##
        loss_vertex2d: Optional[Union[dict, None]] = None,
        loss_vertex_distortion: Optional[Union[dict, None]] = None,
        ##
        loss_transl_z: Optional[Union[dict, None]] = None,
        loss_iuv: Optional[Union[dict, None]] = None,
        loss_distortion_img: Optional[Union[dict, None]] = None,
        loss_image_grad_u: Optional[Union[dict, None]] = None,
        loss_image_grad_v: Optional[Union[dict, None]] = None,
        loss_wrapped_distortion: Optional[Union[dict, None]] = None,
        ###
        loss_mesh_smooth: Optional[Union[dict, None]] = None,
        loss_mesh_normal: Optional[Union[dict, None]] = None,
        loss_mesh_edge: Optional[Union[dict, None]] = None,
        loss_smpl_pose: Optional[Union[dict, None]] = None,
        loss_smpl_betas: Optional[Union[dict, None]] = None,
        init_cfg: Optional[Union[list, dict, None]] = None,
        visualizer: Optional[Union[int, None]] = None,
    ):
        super(ZollyEstimator, self).__init__(init_cfg)

        self.freeze_modules = freeze_modules

        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.verts_head = build_head(verts_head)
        self.iuvd_head = build_head(iuvd_head)
        self.head_keys = head_keys
        self.depth_renderer = build_renderer(depth_renderer)
        self.uv_renderer = build_renderer(uv_renderer)
        self.uv_res = uv_res
        self.use_d_weight = use_d_weight

        self.body_model_train = build_body_model(body_model_train)
        self.body_model_test = build_body_model(body_model_test)
        self.mesh_sampler = MeshSampler(**mesh_sampler)
        self.resolution = resolution

        self.pred_kp3d = pred_kp3d
        self.test_joints3d = test_joints3d
        self.convention = convention
        self.convention_pred = convention_pred
        self.use_deco_warp = use_deco_warp
        self.loss_keypoints2d_hmr = build_loss(loss_keypoints2d_hmr)
        self.loss_keypoints2d_zolly = build_loss(loss_keypoints2d_zolly)
        self.loss_keypoints2d = build_loss(loss_keypoints2d)
        self.loss_joints2d_hmr = build_loss(loss_joints2d_hmr)
        self.loss_joints2d_zolly = build_loss(loss_joints2d_zolly)
        self.loss_joints2d = build_loss(loss_joints2d)
        self.loss_keypoints3d = build_loss(loss_keypoints3d)
        self.loss_joints3d = build_loss(loss_joints3d)

        self.loss_vertex2d = build_loss(loss_vertex2d)
        self.loss_vertex_distortion = build_loss(loss_vertex_distortion)

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
        self.loss_mesh_smooth = build_loss(loss_mesh_smooth)
        self.loss_mesh_normal = build_loss(loss_mesh_normal)
        self.loss_mesh_edge = build_loss(loss_mesh_edge)

        self.full_uvd = full_uvd

        self.visualizer = build_visualizer(visualizer)

        set_requires_grad(self.body_model_train, False)
        set_requires_grad(self.body_model_test, False)

        verts_uv = torch.zeros(6890, 2)
        verts_uv[self.body_model_train.
                 faces_tensor] = self.uv_renderer.face_uv_coord
        self.vertex_uv_sub2 = self.mesh_sampler.downsample(verts_uv,
                                                           n1=0,
                                                           n2=2).view(-1, 2)

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

        targets = self.prepare_targets(data_batch)

        features = self.backbone(data_batch['img'])

        if self.neck is not None:
            features = self.neck(features)

        if self.iuvd_head is not None:
            predictions_iuvd = self.iuvd_head(features)
        else:
            predictions_iuvd = dict()

        predictions.update(predictions_iuvd)

        if self.verts_head is not None:
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

                if self.use_deco_warp:
                    data_batch['warped_pose_feat'] = warp_feature(
                        pred_iuv_img, pose_feat, self.uv_res)
                else:
                    uv_renderer = self.uv_renderer.to(pred_d_img.device)
                    mask = self.uv_renderer.mask.to(pred_d_img.device)[None,
                                                                       None]
                    data_batch['warped_pose_feat'] = uv_renderer.inverse_wrap(
                        pred_iuv_img, pose_feat) * mask
            if 'warped_grid_feat' in self.head_keys:
                grid_feat = features[0]
                pred_iuv_img = predictions['pred_iuv_img']

                if self.use_deco_warp:
                    data_batch['warped_grid_feat'] = warp_feature(
                        pred_iuv_img, grid_feat, self.uv_res)
                else:
                    uv_renderer = self.uv_renderer.to(pred_d_img.device)
                    mask = self.uv_renderer.mask.to(pred_d_img.device)[None,
                                                                       None]
                    data_batch['warped_grid_feat'] = uv_renderer.inverse_wrap(
                        pred_iuv_img, grid_feat) * mask
            if 'pose_feat' in self.head_keys:
                data_batch['pose_feat'] = predictions['pose_feat']
            if 'vertex_uv' in self.head_keys:
                data_batch['vertex_uv'] = self.vertex_uv_sub2
            if 'pred_d_img' in self.head_keys:
                data_batch['pred_d_img'] = predictions['pred_d_img']
            for key in self.head_keys:
                head_data.append(data_batch[key])
            predictions_verts = self.verts_head(features, *head_data)
        else:
            predictions_verts = dict()

        predictions.update(predictions_verts)

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
        if self.iuvd_head is not None:
            optimizer['iuvd_head'].zero_grad()

        loss.backward()

        if self.backbone is not None:
            optimizer['backbone'].step()
        if self.neck is not None:
            optimizer['neck'].step()
        if self.verts_head is not None:
            optimizer['verts_head'].step()
        if self.iuvd_head is not None:
            optimizer['iuvd_head'].step()

        outputs = dict(loss=loss,
                       log_vars=log_vars,
                       num_samples=len(next(iter(data_batch.values()))))
        return outputs

    def prepare_targets(self, data_batch: dict):

        if self.loss_distortion_img is not None or self.loss_iuv is not None:
            is_distorted = data_batch['is_distorted'].view(-1)
            batch_size = is_distorted.shape[0]
            device = is_distorted.device
            has_smpl = data_batch['has_smpl'].view(-1)
            has_uvd = torch.zeros(batch_size).to(device)
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
            ori_focal_length = data_batch['ori_focal_length'].float().view(-1)
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
                    ori_focal_length[level_cd_ids])
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
                                                 ori_shape, ori_focal_length,
                                                 px, py)

                angle = []
                for j in range(batch_size):
                    angle.append(-torch.Tensor(
                        [float(data_batch['img_metas'][j]['rotation'])]))
                angle = torch.cat(angle).to(device)
                gt_cam_new = rotate_smpl_cam(gt_cam, angle, gt_model_joints)

                gt_transl_b = pred_cam_to_full_transl(gt_cam_new, center,
                                                      scale, ori_shape,
                                                      ori_focal_length, px, py)
                gt_transl[level_b_ids] = gt_transl_b[level_b_ids]

                if self.full_uvd:
                    if len(level_b_ids):
                        gt_iuv_img_b, gt_d_img_b = self.render_gt_iuvd_cam(
                            gt_vertices[level_b_ids], gt_cam_new[level_b_ids],
                            uv_res, ori_focal_length[level_b_ids] /
                            scale[level_b_ids] * uv_res)

                        gt_iuv_img[level_b_ids] = gt_iuv_img_b
                        gt_d_img[level_b_ids] = gt_d_img_b

                has_uvd[render_ids] = 1

            data_batch['iuv_img'] = gt_iuv_img
            data_batch['d_img'] = gt_d_img
            data_batch['smpl_transl'] = gt_transl
            data_batch['ori_focal_length'] = ori_focal_length

            data_batch['has_uvd'] = has_uvd

        has_smpl = data_batch['has_smpl'].view(-1)
        device = has_smpl.device
        batch_size = has_smpl.shape[0]

        gt_body_pose = data_batch['smpl_body_pose']
        gt_global_orient = data_batch['smpl_global_orient']
        smpl_origin_orient = data_batch['smpl_origin_orient']

        gt_betas = data_batch['smpl_betas'].float()

        if self.body_model_train is not None:
            gt_output = self.body_model_train(
                betas=gt_betas,
                body_pose=gt_body_pose.float(),
                global_orient=gt_global_orient.float())
            gt_vertices = gt_output['vertices']
            gt_model_joints = gt_output['joints']
        joints2d_conf = data_batch['keypoints2d'][..., 2:3]
        gt_cam = estimate_cam_weakperspective_batch(gt_model_joints,
                                                    data_batch['keypoints2d'],
                                                    joints2d_conf,
                                                    joints2d_conf,
                                                    self.resolution)

        has_transl_ids = torch.where(data_batch['has_transl'] > 0)[0]

        meshes = Meshes(
            verts=gt_vertices,
            faces=self.body_model_train.faces_tensor[None].repeat_interleave(
                batch_size, 0))

        K = torch.eye(3, 3)[None].to(device).repeat_interleave(batch_size, 0)
        K[:, 0, 0] = 5000 / (self.resolution / 2)
        K[:, 1, 1] = 5000 / (self.resolution / 2)

        cameras = build_cameras(
            dict(type='perspective',
                 in_ndc=True,
                 K=K,
                 resolution=data_batch['ori_shape'],
                 convention='opencv')).to(device)

        normals = cameras.compute_normal_of_meshes(meshes)

        gt_vertices_sub2 = self.mesh_sampler.downsample(gt_vertices,
                                                        n1=0,
                                                        n2=2)

        gt_verts2d = project_points_pred_cam(gt_vertices_sub2,
                                             gt_cam,
                                             focal_length=5000,
                                             img_res=self.resolution)
        transl = pred_cam_to_transl(gt_cam, 5000, self.resolution).float()

        # gt_verts2d_full = project_points_pred_cam(gt_vertices,
        #                                           gt_cam,
        #                                           focal_length=5000,
        #                                           img_res=self.resolution)

        # gt_verts2d_cache = gt_verts2d.clone()
        if len(has_transl_ids) > 0:
            trans = data_batch['trans']
            if self.body_model_train is not None:
                origin_output = self.body_model_train(
                    betas=gt_betas,
                    body_pose=gt_body_pose.float(),
                    global_orient=smpl_origin_orient.float())
                origin_vertices = origin_output['vertices']

            origin_vertices_sub2 = self.mesh_sampler.downsample(
                origin_vertices, n1=0, n2=2)

            K = data_batch['K'][has_transl_ids]
            # from zolly.cameras.convert_convention import convert_ndc_to_screen
            cameras_real = build_cameras(
                dict(type='perspective',
                     in_ndc=False,
                     K=K,
                     resolution=data_batch['ori_shape'][has_transl_ids],
                     convention='opencv')).to(device)
            _gt_verts2d = cameras_real.transform_points_screen(
                origin_vertices_sub2[has_transl_ids] +
                data_batch['smpl_transl'][has_transl_ids].view(-1, 1, 3))
            # _gt_verts2d_full = cameras_real.transform_points_screen(
            #     origin_vertices[has_transl_ids] +
            #     data_batch['smpl_transl'][has_transl_ids].view(-1, 1, 3))
            normals_real = cameras_real.compute_normal_of_meshes(
                meshes[has_transl_ids])
            normals[has_transl_ids] = normals_real
            # trans @ pred_keypoints2d2
            _gt_verts2d[..., 2] = 1
            # _gt_verts2d_full[..., 2] = 1
            gt_verts2d[has_transl_ids] = torch.einsum(
                'bij,bkj->bki', trans[has_transl_ids].float(), _gt_verts2d)
            transl[has_transl_ids] = data_batch['smpl_transl'].float(
            )[has_transl_ids]

            # gt_verts2d_full[has_transl_ids] = torch.einsum(
            #     'bij,bkj->bki', trans[has_transl_ids].float(),
            #     _gt_verts2d_full)
        # gt_pelvis = gt_model_joints[:, 0:1]
        transl = transl.float().view(-1, 1, 3)
        gt_vertex_distortion = transl[...,
                                      2] / (gt_vertices_sub2 + transl)[..., 2]

        gt_joint_distortion = transl[..., 2] / (gt_model_joints + transl)[...,
                                                                          2]

        data_batch['has_vertex2d'] = data_batch['has_transl']
        data_batch['has_distortion'] = data_batch['has_transl']
        data_batch['gt_vertex_distortion'] = gt_vertex_distortion.float()
        data_batch['gt_joint_distortion'] = gt_joint_distortion.float()
        data_batch['gt_vertex2d'] = gt_verts2d
        data_batch['gt_normals'] = normals

        return data_batch

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
                pred_vertices_sub1 = predictions['pred_vertices_sub1']
                pred_vertices_sub2 = predictions['pred_vertices_sub2']

                if 'pred_vertices' in predictions:
                    pred_vertices = predictions['pred_vertices']
                else:
                    pred_vertices = self.mesh_sampler.upsample(
                        pred_vertices_sub1, n1=1, n2=0)

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

        right_hip_idx = get_keypoint_idx('right_hip_extra',
                                         self.body_model_train.keypoint_dst)
        left_hip_idx = get_keypoint_idx('left_hip_extra',
                                        self.body_model_train.keypoint_dst)
        target_pelvis = (gt_model_joints[:, right_hip_idx, :] +
                         gt_model_joints[:, left_hip_idx, :]) / 2
        ################################################
        ################################################

        losses = {}
        if self.loss_keypoints2d is not None:
            gt_keypoints2d = targets['keypoints2d']

            pred_cam = predictions['pred_cam']
            gt_focal_length = 5000
            if self.pred_kp3d:
                pred_keypoints3d_ = pred_keypoints3d
            else:
                conf = torch.ones_like(pred_model_joints)[..., :1]
                pred_keypoints3d_ = torch.cat([pred_model_joints, conf], -1)
            losses['keypoints2d_loss'] = self.compute_keypoints2d_loss(
                pred_keypoints3d_,
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

        if self.loss_joints3d is not None:

            batch_size = gt_keypoints3d.shape[0]
            conf = torch.ones_like(pred_model_joints)[..., :1]
            if self.pred_kp3d:
                pred_keypoints3d_ = pred_keypoints3d
            else:
                conf = torch.ones_like(pred_model_joints)[..., :1]
                pred_keypoints3d_ = torch.cat([pred_model_joints, conf], -1)
            losses['joints3d_loss'] = self.compute_joints3d_loss(
                pred_keypoints3d_,
                torch.cat([gt_model_joints, conf], -1),
                has_keypoints3d=has_keypoints3d)

        if self.loss_keypoints3d is not None:

            batch_size = gt_keypoints3d.shape[0]
            if self.pred_kp3d:
                pred_keypoints3d_ = pred_keypoints3d
            else:
                pred_keypoints3d_ = pred_model_joints
            losses['keypoints3d_loss'] = self.compute_keypoints3d_loss(
                pred_keypoints3d_,
                torch.cat([
                    gt_model_joints,
                    torch.ones_like(gt_model_joints)[..., :1]
                ], -1),
                has_keypoints3d=has_keypoints3d)

        if self.loss_camera is not None:
            pred_cam = predictions['pred_cam']
            losses['camera_loss'] = self.compute_camera_loss(pred_cam)

        if self.loss_keypoints2d_zolly is not None:

            center, scale = targets['center'], targets['scale'][:, 0]
            gt_keypoints2d = targets['keypoints2d']
            orig_keypoints2d = gt_keypoints2d.clone()[..., :2]
            orig_keypoints2d = (
                orig_keypoints2d - (self.resolution / 2)
            ) / self.resolution * scale[:, None, None] + center.view(-1, 1, 2)
            orig_keypoints2d = torch.cat(
                [orig_keypoints2d, targets['keypoints2d'][..., 2:3]], -1)
            conf = targets['keypoints2d'][..., 2:3]
            gt_keypoints2d = torch.cat([gt_keypoints2d, conf], -1)

            # has_transl = targets['is_distorted'].view(
            #     -1) * targets['has_transl'].view(-1)
            has_transl = targets['has_transl'].view(-1)
            no_transl_ids = torch.where(has_transl == 0)[0]
            # has_transl_ids = torch.where(has_transl > 0)[0]

            target_focal_length_ndc = targets['ori_focal_length'].view(
                -1) * 2 / scale
            target_focal_length_ndc[no_transl_ids] = 1000 / (self.resolution /
                                                             2)

            ori_focal_length = target_focal_length_ndc.view(-1) * scale / 2

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
                ori_focal_length[no_transl_ids])

            target_transl[no_transl_ids, :] = computed_transl.float()

            camera_center = torch.cat([
                targets['ori_shape'][:, 1].view(-1, 1),
                targets['ori_shape'][:, 0].view(-1, 1)
            ], -1) / 2

            has_K_ids = torch.where(targets['has_K'] > 0)[0]
            camera_center[has_K_ids] = targets['K'][:, :2, 2][has_K_ids]

            #==================================================#
            if self.pred_kp3d:
                pred_keypoints3d_ = pred_keypoints3d[
                    ..., :3] + target_pelvis[:, None]
                pred_keypoints3d_ = torch.cat(
                    [pred_keypoints3d_, pred_keypoints3d[..., 3:4]], -1)
            else:
                conf = torch.ones_like(pred_model_joints)[..., :1]
                pred_keypoints3d_ = pred_model_joints[
                    ..., :3] + target_pelvis[:, None]
                pred_keypoints3d_ = torch.cat([pred_keypoints3d_, conf], -1)

            losses[
                'keypoints2d_loss_zolly'] = self.compute_keypoints2d_perspective_loss(
                    pred_keypoints3d_,  # pred_keypoints3d,
                    target_transl,  # pred_transl,
                    orig_keypoints2d,
                    camera_center,
                    ori_focal_length,
                    img_res=targets['ori_shape'],
                    has_keypoints2d=has_keypoints2d,
                    loss_func=self.loss_keypoints2d_zolly)

        if self.loss_joints2d_zolly is not None:

            center, scale = targets['center'], targets['scale'][:, 0]
            gt_keypoints2d = targets['keypoints2d']
            orig_keypoints2d = gt_keypoints2d.clone()[..., :2]
            orig_keypoints2d = (
                orig_keypoints2d - (self.resolution / 2)
            ) / self.resolution * scale[:, None, None] + center.view(-1, 1, 2)
            orig_keypoints2d = torch.cat(
                [orig_keypoints2d, targets['keypoints2d'][..., 2:3]], -1)
            conf = targets['keypoints2d'][..., 2:3]
            gt_keypoints2d = torch.cat([gt_keypoints2d, conf], -1)

            # has_transl = targets['is_distorted']
            has_transl = targets['has_transl']
            no_transl_ids = torch.where(has_transl == 0)[0]

            target_focal_length_ndc = targets['ori_focal_length'].view(
                -1) * 2 / scale
            target_focal_length_ndc[no_transl_ids] = 1000 / (self.resolution /
                                                             2)

            ori_focal_length = target_focal_length_ndc.view(-1) * scale / 2

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
                ori_focal_length[no_transl_ids])

            target_transl[no_transl_ids, :] = computed_transl.float()

            camera_center = torch.cat([
                targets['ori_shape'][:, 1].view(-1, 1),
                targets['ori_shape'][:, 0].view(-1, 1)
            ], -1) / 2

            has_K_ids = torch.where(targets['has_K'] > 0)[0]
            camera_center[has_K_ids] = targets['K'][:, :2, 2][has_K_ids]

            #==================================================#
            if self.pred_kp3d:
                pred_keypoints3d_ = pred_keypoints3d[
                    ..., :3] + target_pelvis[:, None]
                pred_keypoints3d_ = torch.cat(
                    [pred_keypoints3d_, pred_keypoints3d[..., 3:4]], -1)
            else:
                conf = torch.ones_like(pred_model_joints)[..., :1]
                pred_keypoints3d_ = pred_model_joints[
                    ..., :3] + target_pelvis[:, None]
                pred_keypoints3d_ = torch.cat([pred_keypoints3d_, conf], -1)

            losses[
                'joints2d_loss_zolly'] = self.compute_keypoints2d_perspective_loss(
                    pred_keypoints3d_,  # pred_keypoints3d,
                    target_transl,  # pred_transl,
                    orig_keypoints2d,
                    camera_center,
                    ori_focal_length,
                    img_res=targets['ori_shape'],
                    has_keypoints2d=has_keypoints2d,
                    loss_func=self.loss_joints2d_zolly)

        if self.loss_joints2d_hmr is not None:

            gt_keypoints2d = targets['keypoints2d']

            pred_cam = predictions['pred_cam']

            gt_focal_length = 5000

            kp2d_weight = None
            keypoints3d_conf = None

            if self.use_d_weight:
                non_distorted_ids = torch.where(
                    targets['is_distorted'].view(-1) == 0)[0]
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
                kp2d_weight = 1 / torch.clip(kp2d_weight, 0.5, 10)
                kp2d_weight[non_distorted_ids] = 1

            if (self.loss_keypoints2d_zolly
                    is not None) or (self.loss_joints2d_zolly is not None):
                pred_model_joints = pred_model_joints.clone().detach()
            pred_keypoints3d_ = pred_model_joints[..., :3]
            losses['joints2d_loss_hmr'] = self.compute_keypoints2d_hmr_loss(
                pred_keypoints3d_,
                pred_cam,
                gt_keypoints2d,
                has_keypoints2d=has_keypoints2d,
                focal_length=gt_focal_length,
                keypoints3d_conf=keypoints3d_conf,
                weight=kp2d_weight,
                loss_func=self.loss_joints2d_hmr)

        if self.loss_keypoints2d_hmr is not None:

            gt_keypoints2d = targets['keypoints2d']

            pred_cam = predictions['pred_cam']

            gt_focal_length = 5000

            kp2d_weight = None

            if self.use_d_weight:
                non_distorted_ids = torch.where(
                    targets['is_distorted'].view(-1) == 0)[0]
                uv_res = targets['d_img'].shape[-1]
                down_scale = self.resolution / uv_res
                has_uvd_ids = torch.where(targets['has_uvd'] > 0)[0]
                kp2d_weight = torch.ones_like(gt_keypoints2d)[..., :1]
                for idx in has_uvd_ids:
                    for kp_idx in range(gt_keypoints2d.shape[1]):
                        x = int(gt_keypoints2d[idx, kp_idx, 0] / down_scale)
                        y = int(gt_keypoints2d[idx, kp_idx, 1] / down_scale)
                        if 0 < x < uv_res and 0 < y < uv_res:
                            kp2d_weight[idx, kp_idx] = targets['d_img'][idx,
                                                                        0][y,
                                                                           x]
                kp2d_weight = 1 / torch.clip(kp2d_weight, 0.5, 10)
                kp2d_weight[non_distorted_ids] = 1

            if self.pred_kp3d:
                keypoints3d_conf = pred_keypoints3d[..., 3:4]
                pred_keypoints3d_ = pred_keypoints3d[..., :3]
            else:
                keypoints3d_conf = torch.ones_like(pred_model_joints)[..., :1]
                pred_keypoints3d_ = pred_model_joints[..., :3]

            if (self.loss_keypoints2d_zolly
                    is not None) or (self.loss_joints2d_zolly is not None):

                pred_keypoints3d_ = pred_keypoints3d_.clone().detach()

            losses['keypoints2d_loss_hmr'] = self.compute_keypoints2d_hmr_loss(
                pred_keypoints3d_[..., :3],
                pred_cam,
                gt_keypoints2d,
                has_keypoints2d=has_keypoints2d,
                focal_length=gt_focal_length,
                keypoints3d_conf=keypoints3d_conf,
                weight=kp2d_weight,
                loss_func=self.loss_keypoints2d_hmr)

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

        if self.loss_vertex2d is not None:
            gt_vertex2d = targets['gt_vertex2d']
            pred_vertex2d = predictions['pred_vertex2d']
            has_vertex2d = targets['has_vertex2d'].view(-1)
            losses['vertex2d_loss'] = self.compute_vertex2d_loss(
                pred_vertex2d, gt_vertex2d, has_vertex2d) * 1e-2

        if self.loss_vertex_distortion is not None:

            gt_vertex_distortion = targets['gt_vertex_distortion']
            pred_vertex_distortion = predictions['pred_vertex_distortion']
            has_distortion = targets['has_distortion'].view(-1)
            losses[
                'vertex_distortion_loss'] = self.compute_vertex_distortion_loss(
                    pred_vertex_distortion, gt_vertex_distortion,
                    has_distortion)

        if self.loss_mesh_smooth is not None:
            meshes = Meshes(
                pred_vertices,
                self.body_model_train.faces_tensor[None].repeat_interleave(
                    batch_size, 0))
            losses['mesh_smooth_loss'] = self.loss_mesh_smooth(meshes)

        if self.loss_mesh_normal is not None:
            meshes = Meshes(
                pred_vertices,
                self.body_model_train.faces_tensor[None].repeat_interleave(
                    batch_size, 0))
            losses['mesh_normal_loss'] = self.loss_mesh_normal(meshes)

        if self.loss_mesh_edge is not None:
            meshes = Meshes(
                pred_vertices,
                self.body_model_train.faces_tensor[None].repeat_interleave(
                    batch_size, 0))
            losses['mesh_edge_loss'] = self.loss_mesh_edges(meshes)

        return losses

    def forward_test(self, **data_batch):
        """Defines the computation performed at every call when testing."""

        for name in self.freeze_modules:
            for parameter in getattr(self, name).parameters():
                parameter.requires_grad = False
        predictions = dict()

        batch_size = data_batch['img'].shape[0]
        device = data_batch['img'].device

        features = self.backbone(data_batch['img'])

        if self.neck is not None:
            features = self.neck(features)

        if self.iuvd_head is not None:
            predictions_iuvd = self.iuvd_head(features)
        else:
            predictions_iuvd = dict()
        predictions.update(predictions_iuvd)

        if self.verts_head is not None:
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
                if self.use_deco_warp:
                    data_batch['warped_pose_feat'] = warp_feature(
                        pred_iuv_img, pose_feat, self.uv_res)
                else:
                    uv_renderer = self.uv_renderer.to(pred_d_img.device)
                    mask = self.uv_renderer.mask.to(pred_d_img.device)[None,
                                                                       None]
                    data_batch['warped_pose_feat'] = uv_renderer.inverse_wrap(
                        pred_iuv_img, pose_feat) * mask
            if 'warped_grid_feat' in self.head_keys:
                grid_feat = features[0]
                pred_iuv_img = predictions['pred_iuv_img']

                if self.use_deco_warp:
                    data_batch['warped_grid_feat'] = warp_feature(
                        pred_iuv_img, grid_feat, self.uv_res)
                else:
                    uv_renderer = self.uv_renderer.to(pred_d_img.device)
                    mask = self.uv_renderer.mask.to(pred_d_img.device)[None,
                                                                       None]
                    data_batch['warped_grid_feat'] = uv_renderer.inverse_wrap(
                        pred_iuv_img, grid_feat) * mask
            if 'pose_feat' in self.head_keys:
                data_batch['pose_feat'] = predictions['pose_feat']
            if 'vertex_uv' in self.head_keys:
                data_batch['vertex_uv'] = self.vertex_uv_sub2
            if 'pred_d_img' in self.head_keys:
                data_batch['pred_d_img'] = predictions['pred_d_img']
            for key in self.head_keys:
                head_data.append(data_batch[key])
            predictions_verts = self.verts_head(features, *head_data)
        else:
            predictions_verts = dict()

        predictions.update(predictions_verts)

        pred_cam = predictions['pred_cam']

        center, scale = data_batch['center'], data_batch['scale'][:, 0]

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

        if self.loss_keypoints2d_hmr is not None:
            if self.pred_kp3d:
                pred_pelvis_offset = (
                    predictions['pred_keypoints3d'][:, right_hip_idx, :] +
                    predictions['pred_keypoints3d'][:, left_hip_idx, :]) / 2
            else:
                pred_pelvis_offset = pred_pelvis
        elif self.loss_joints2d_hmr is not None:
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
        pred_transl = pred_cam_to_transl(pred_cam, 5000, self.resolution)
        pred_transl = pred_pelvis_offset + pred_transl
        # pred_transl = pred_pelvis + pred_transl
        pred_cam = pred_transl_to_pred_cam(pred_transl, 5000, self.resolution)

        if self.loss_keypoints2d_zolly or self.loss_joints2d_zolly is not None:
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
            predictions.update(ori_focal_length=pred_focal_length * scale / 2)
        else:
            pred_focal_length = 5000 / (self.resolution / 2)
            predictions.update(pred_transl=pred_cam_to_transl(
                pred_cam, 5000, self.resolution))
            predictions.update(full_transl=pred_cam_to_full_transl(
                pred_cam, center, scale, data_batch['ori_shape'], 5000 /
                self.resolution * scale).float())
            predictions.update(pred_focal_length=pred_focal_length)
            predictions.update(ori_focal_length=pred_focal_length * scale / 2)

        orig_wh = torch.cat(
            [data_batch['ori_shape'][:, 1], data_batch['ori_shape'][:, 0]], -1)
        origin_focal_length = predictions['ori_focal_length']
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
            orig_keypoints2d = gt_keypoints2d.clone()[..., :2]

            orig_keypoints2d = (
                orig_keypoints2d - (self.resolution / 2)
            ) / self.resolution * scale[:, None, None] + center.view(-1, 1, 2)
            orig_keypoints2d = torch.cat([
                orig_keypoints2d / orig_wh.view(batch_size, 1, 2),
                gt_keypoints2d[..., 2:3]
            ], -1)

            predictions.update(orig_keypoints2d=orig_keypoints2d)

        if self.loss_keypoints2d_hmr is not None:
            if self.pred_kp3d:
                pred_vertices_vis = pred_vertices - pred_pelvis[:,
                                                                None] - pred_pelvis_offset[:,
                                                                                           None]
            else:
                pred_vertices_vis = pred_vertices - pred_pelvis[:, None]
        elif self.loss_joints2d_hmr is not None:
            pred_vertices_vis = pred_vertices - pred_pelvis[:, None]

        predictions.update(img=data_batch['img'],
                           pred_vertices=pred_vertices_vis,
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
            ori_focal_length = data_batch['ori_focal_length'].float().view(-1)
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
                    focal_length_ndc=ori_focal_length.float() / scale.float() *
                    2,
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
                    focal_length_ndc=ori_focal_length.float() / scale.float() *
                    2,
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

        if 'ori_focal_length' in data_batch:
            predictions['gt_focal_length'] = data_batch['ori_focal_length']
        else:
            predictions['gt_focal_length'] = ori_focal_length * 0

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

        if 'orig_keypoints2d' in predictions:
            all_preds['orig_keypoints2d'] = predictions['orig_keypoints2d']
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

    def compute_distortion_loss(self, pred_distortion, gt_distortion, has_d):
        batch_size = has_d.shape[0]
        valid = has_d.float().view(batch_size) > 0
        loss = self.loss_distortion_img(pred_distortion[valid],
                                        gt_distortion[valid])
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

    def compute_iuv_loss(self, pred_iuv, gt_iuv, has_iuv):
        batch_size = gt_iuv.shape[0]
        valid = has_iuv.float().view(batch_size) > 0
        loss = self.loss_iuv(pred_iuv[valid], gt_iuv[valid])
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

    def compute_image_grad_u_loss(self, pred_u, gt_u, has_uvd):
        batch_size = has_uvd.shape[0]
        valid = has_uvd.float().view(batch_size) > 0
        return self.loss_image_grad_u(pred_u[valid], gt_u[valid])

    def compute_image_grad_v_loss(self, pred_v, gt_v, has_uvd):
        batch_size = has_uvd.shape[0]
        valid = has_uvd.float().view(batch_size) > 0
        return self.loss_image_grad_v(pred_v[valid], gt_v[valid])

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

    def compute_vertex2d_loss(self, pred_vertex2d: torch.Tensor,
                              gt_vertex2d: torch.Tensor,
                              has_vertex2d: torch.Tensor):
        """Compute loss for predicted camera parameters."""
        valid = has_vertex2d.view(-1) == 1
        target_weights = torch.ones_like(gt_vertex2d[valid])
        loss = self.loss_vertex2d(pred_vertex2d[valid],
                                  gt_vertex2d[valid] / self.resolution,
                                  target_weights).mean()
        return loss

    def compute_vertex_distortion_loss(self, pred_distortion: torch.Tensor,
                                       gt_distortion: torch.Tensor,
                                       has_distortion: torch.Tensor):
        """Compute loss for predicted camera parameters."""
        valid = has_distortion.view(-1) == 1
        loss = self.loss_vertex_distortion(pred_distortion[valid],
                                           gt_distortion[valid])
        return loss

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

        ori_focal_length = orig_K[:, 0, 0]
        K[:, 0, 0] = ori_focal_length / scale * img_res
        K[:, 1, 1] = ori_focal_length / scale * img_res
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
