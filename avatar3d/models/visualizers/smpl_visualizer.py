import cv2
import os
import torch
import torch.nn as nn
import numpy as np

from typing import Union
from avatar3d.transforms.transform3d import ee_to_rotmat
from avatar3d.utils.visualize_smpl import vis_smpl
from avatar3d.utils.torch_utils import image_tensor2numpy, convert_RGB_BGR
from avatar3d.cameras.builder import build_cameras
from avatar3d.models.body_models.builder import build_body_model


class SmplVisualizer:

    def __init__(self,
                 body_model: Union[dict, nn.Module] = None,
                 num_batch: int = -1,
                 random: bool = False,
                 write_f: bool = False,
                 num_per_batch: int = -1,
                 demo_root: str = '',
                 full_image: bool = False,
                 full_image_batch: bool = False,
                 side_view: bool = False,
                 stack: bool = True,
                 pmiou=None,
                 alpha=1.0,
                 z=None,
                 sample_ids=None,
                 color: Union[list, str, tuple] = 'white',
                 resolution: Union[list, tuple, int] = None) -> None:
        if isinstance(body_model, dict):
            self.body_model = build_body_model(body_model)
        self.num_batch = num_batch
        self.write_f = write_f
        self.full_image = full_image
        self.num_per_batch = num_per_batch
        self.demo_root = demo_root
        self.resolution = resolution
        self.batch_idx = 0
        self.color = color
        self.alpha = alpha
        self.sample_ids = sample_ids
        self.random = random
        self.side_view = side_view
        self.full_image_batch = full_image_batch
        self.stack_images = stack
        self.pmiou = pmiou
        self.z = z

    def __call__(self, data_batch):
        os.makedirs(self.demo_root, exist_ok=True)
        if self.batch_idx >= self.num_batch and self.num_batch > 0:
            return

        img = data_batch['img']
        batch_size = img.shape[0]
        sample_idx = data_batch['sample_idx'].view(-1).tolist()

        flag = False

        if self.sample_ids is not None:
            indexes_tmp = []
            for idx in sample_idx:
                if int(idx) in self.sample_ids:
                    flag = True
                    indexes_tmp.append(int(sample_idx.index(idx)))
        else:
            flag = True
        if not flag:
            return

        device = img.device
        if self.resolution is None:
            if not self.full_image:
                resolution = img.shape[2:4]
            else:
                resolution = None
        else:
            resolution = self.resolution if isinstance(
                self.resolution, (tuple, list)) else (int(self.resolution),
                                                      int(self.resolution))
        pred_vertices = data_batch['pred_vertices']
        if self.pmiou is not None:
            indexes = torch.where(
                data_batch['batch_pmiou'] > self.pmiou)[0].tolist()
        elif self.z is not None:
            indexes = torch.where(
                data_batch['smpl_transl'][..., 2] < self.z)[0].tolist()
        elif self.sample_ids is not None:
            indexes = indexes_tmp

        else:

            if self.random:
                indexes = np.random.randint(
                    0, batch_size, size=(self.num_per_batch)).tolist(
                    ) if self.num_per_batch > 0 else list(range(batch_size))
            else:
                indexes = list(range(min(
                    batch_size,
                    self.num_per_batch))) if self.num_per_batch > 0 else list(
                        range(batch_size))
        if len(indexes) == 0:
            return

        # pred_cam = data_batch['pred_cam']
        K = torch.eye(3, 3)[None].repeat_interleave(batch_size, 0)
        # s = pred_cam[:, 0]
        # tx = pred_cam[:, 1]
        # ty = pred_cam[:, 2]
        if self.full_image:
            transl = data_batch['full_transl'].float()
            K[:, 0, 0] = data_batch['orig_focal_length']
            K[:, 1, 1] = data_batch['orig_focal_length']
            K[:, 0, 2] = data_batch['ori_shape'][:, 1] / 2
            K[:, 1, 2] = data_batch['ori_shape'][:, 0] / 2
            cameras = build_cameras(
                dict(type='perspective',
                     in_ndc=False,
                     K=K,
                     resolution=data_batch['ori_shape'].view(-1, 2),
                     convention='opencv')).to(device)
        else:
            transl = data_batch['pred_transl'].float()
            K[:, 0, 0] = data_batch['pred_focal_length'] * resolution[0] / 2
            K[:, 1, 1] = data_batch['pred_focal_length'] * resolution[0] / 2
            K[:, 0, 2] = resolution[1] / 2
            K[:, 1, 2] = resolution[0] / 2

            cameras = build_cameras(
                dict(type='perspective',
                     in_ndc=False,
                     K=K,
                     resolution=resolution,
                     convention='opencv')).to(device)
        if ((not self.full_image_batch) and (self.full_image)):
            by_frame = True
        else:
            by_frame = False

        if not self.full_image:
            images = data_batch['img'][indexes].permute(0, 2, 3, 1)
            images = (images - images.min()) / (images.max() -
                                                images.min()) * 255
            images = image_tensor2numpy(convert_RGB_BGR(images))
        else:
            images = []
            for i in (indexes):
                im = cv2.imread(data_batch['img_metas'][int(i)]['image_path'])
                if not by_frame:
                    im = cv2.resize(im, (resolution, resolution),
                                    cv2.INTER_CUBIC)
                images.append(im)
            if not by_frame:
                images = np.stack(images, 0)
        body_model = self.body_model.to(device)

        if by_frame:
            rendered_image = []
            for i, image_id in enumerate(indexes):
                im = vis_smpl(verts=pred_vertices[image_id][None] +
                              transl[image_id].view(1, 1, 3),
                              cameras=cameras[[image_id]],
                              device=device,
                              body_model=body_model,
                              image_array=images[i][None],
                              return_tensor=True,
                              alpha=self.alpha,
                              palette=self.color,
                              no_grad=True)[0]
                rendered_image.append(image_tensor2numpy(im))
        else:
            rendered_image = vis_smpl(verts=pred_vertices[indexes] +
                                      transl[indexes].view(len(indexes), 1, 3),
                                      cameras=cameras[indexes],
                                      device=device,
                                      body_model=body_model,
                                      image_array=images,
                                      return_tensor=True,
                                      palette=self.color,
                                      alpha=self.alpha,
                                      resolution=resolution,
                                      no_grad=True)
            rendered_image = image_tensor2numpy(rendered_image)

        if self.side_view:
            side_pelvis = pred_vertices.mean(1).view(-1, 1, 3)
            rotation_matrix = ee_to_rotmat(
                torch.rad2deg(torch.Tensor([0, 90, 0]).view(-1, 3)).to(device))

            rotated_vertices = torch.bmm(
                rotation_matrix.repeat_interleave(batch_size, 0),
                (pred_vertices - side_pelvis).permute(0, 2, 1)).permute(
                    0, 2, 1) + side_pelvis
            if by_frame:
                side_views = []
                for _i in indexes:
                    side_views.append(
                        vis_smpl(
                            verts=rotated_vertices[[_i]] +
                            transl[[_i]].view(1, 1, 3),
                            cameras=cameras[[_i]],
                            device=device,
                            body_model=body_model,
                            #  image_array=images[int(_i)][None],
                            return_tensor=True,
                            alpha=self.alpha,
                            palette=self.color,
                            no_grad=True)[0])
            else:
                side_views = vis_smpl(
                    verts=rotated_vertices[indexes] +
                    transl[indexes].view(len(indexes), 1, 3),
                    cameras=cameras[indexes],
                    device=device,
                    body_model=body_model,
                    #   image_array=images,
                    alpha=self.alpha,
                    return_tensor=True,
                    palette=self.color,
                    no_grad=True)
        if self.write_f:
            gt_f = data_batch['gt_focal_length']
            pred_f = data_batch['orig_focal_length']
            gt_z = data_batch['smpl_transl'][:, 2]
            pred_z = data_batch['full_transl'][:, 2]
        if self.stack_images:
            for i, image_id in enumerate(indexes):
                if self.side_view:
                    demo_image = np.concatenate(
                        [images[i], rendered_image[i], side_views[i]], 1)
                else:
                    demo_image = np.concatenate([images[i], rendered_image[i]],
                                                1)
                if self.write_f:
                    cv2.imwrite(
                        f'{self.demo_root}/{int(sample_idx[image_id])}'
                        f'_{int(gt_f[image_id])}-{float(gt_z[image_id]):.2f}_{int(pred_f[image_id])}'
                        f'-{float(pred_z[image_id]):.2f}_stack.png',
                        demo_image)
                else:
                    cv2.imwrite(
                        f'{self.demo_root}/{int(sample_idx[image_id])}_stack.png',
                        demo_image)
        else:
            for i, image_id in enumerate(indexes):
                if self.write_f:
                    cv2.imwrite(
                        f'{self.demo_root}/{int(sample_idx[image_id])}'
                        f'_{int(gt_f[image_id])}-{float(gt_z[image_id]):.2f}_{int(pred_f[image_id])}'
                        f'-{float(pred_z[image_id]):.2f}_pred.png',
                        rendered_image[i])
                else:
                    cv2.imwrite(
                        f'{self.demo_root}/{int(sample_idx[image_id])}_pred.png',
                        rendered_image[i])
                cv2.imwrite(
                    f'{self.demo_root}/{int(sample_idx[image_id])}_rgb.png',
                    images[i])
                if self.side_view:
                    cv2.imwrite(
                        f'{self.demo_root}/{int(sample_idx[image_id])}_side.png',
                        side_views[i])
        self.batch_idx += 1
