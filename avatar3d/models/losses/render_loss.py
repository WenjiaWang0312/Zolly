import torch
import torch.nn.functional as F
from mmhuman3d.models.losses.mse_loss import MSELoss
from avatar3d.cameras import NewCamerasBase
from avatar3d.render.builder import build_renderer


class SilhouetteMSELoss(MSELoss):

    def __init__(self, reduction='mean', loss_weight=1, rasterizer=None):
        super().__init__(reduction, loss_weight)
        if rasterizer is not None:
            self.rasterizer = rasterizer

    def forward(self,
                mesh_pred,
                camera_pred,
                mask_target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        resolution = mask_target.shape[1:3]
        renderer = build_renderer(
            dict(type='silhouette',
                 rasterizer=getattr(
                     self, 'rasterizer',
                     dict(bin_size=0,
                          blur_radius=2e-5,
                          faces_per_pixel=50,
                          perspective_correct=False)),
                 resolution=resolution))
        renderer = renderer.to(mesh_pred.device)
        mask_pred = renderer(meshes=mesh_pred, cameras=camera_pred)[..., 3]
        return super().forward(mask_pred, mask_target, weight, avg_factor,
                               reduction_override)


class SilhouetteRegLoss(MSELoss):

    def __init__(self, reduction='mean', loss_weight=1, rasterizer=None):
        super().__init__(reduction, loss_weight)
        if rasterizer is not None:
            self.rasterizer = rasterizer

    def forward(self,
                mesh_pred,
                camera_pred,
                mask_target,
                weight=None,
                reduction_override=None):
        resolution = mask_target.shape[1:3]
        renderer = build_renderer(
            dict(type='silhouette',
                 rasterizer=getattr(
                     self, 'rasterizer',
                     dict(bin_size=0,
                          blur_radius=2e-5,
                          faces_per_pixel=50,
                          perspective_correct=False)),
                 resolution=resolution))
        renderer = renderer.to(mesh_pred.device)
        mask_pred = renderer(meshes=mesh_pred, cameras=camera_pred)[..., 3]

        loss = (1 - mask_target) * mask_pred**2

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override
                     if reduction_override else self.reduction)
        loss_weight = (weight if weight is not None else self.loss_weight)

        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

        loss *= loss_weight
        return loss


class FlowWarpingLoss(MSELoss):

    def __init__(self, reduction='mean', loss_weight=1):
        super().__init__(reduction, loss_weight)

    def forward(
            self,
            mesh_src: torch.Tensor,
            mesh_dst: torch.Tensor,
            camera_src: NewCamerasBase,
            camera_dst: NewCamerasBase,
            image_src: torch.Tensor,  # N, H, W, C
            image_dst: torch.Tensor,  # N, H, W, C
            weight=None,
            avg_factor=None,
            reduction_override=None):
        resolution = image_src.shape[1:3]
        renderer = build_renderer(
            dict(type='opticalflow',
                 resolution=resolution,
                 device=mesh_src.device))

        scene_flow = renderer(mesh_src, mesh_dst, camera_src,
                              camera_dst)  # N, H, W, 2
        flow = scene_flow[..., :2]
        valid_mask = scene_flow[..., 4:]
        warpped_image_src = F.grid_sample(image_src.permute(0, 3, 1, 2),
                                          flow,
                                          align_corners=False).permute(
                                              0, 2, 3, 1)
        loss = super().forward(warpped_image_src, image_dst, weight,
                               avg_factor, reduction_override) * valid_mask
        return loss
