import torch
from avatar3d.models import BaseModule
from typing import Optional, Union, Tuple

from avatar3d.models.backbones.builder import build_backbone
from avatar3d.models.heads.builder import build_head
from avatar3d.transforms.transform3d import ee_to_rotmat
import torch.nn as nn
from avatar3d.models.backbones.resnet_spec import resnet50
from avatar3d.utils.spec_utils import convert_preds_to_angles


class SpecCamExtractor(BaseModule):

    def __init__(self,
                 no_rot=False,
                 num_fc_layers=1,
                 num_fc_channels=1024,
                 num_out_channels=256,
                 init_cfg=None):

        super(SpecCamExtractor, self).__init__(init_cfg)
        self.backbone = resnet50()
        self.no_rot = no_rot
        self.num_out_channels = num_out_channels
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        out_channels = 2048

        assert num_fc_layers > 0, 'Number of FC layers should be more than 0'
        if num_fc_layers == 1:
            self.fc_vfov = nn.Linear(out_channels, num_out_channels)
            self.fc_pitch = nn.Linear(out_channels, num_out_channels)
            self.fc_roll = nn.Linear(out_channels, num_out_channels)

            nn.init.normal_(self.fc_vfov.weight, mean=0, std=0.01)
            nn.init.constant_(self.fc_vfov.bias, 0)

            nn.init.normal_(self.fc_pitch.weight, mean=0, std=0.01)
            nn.init.constant_(self.fc_pitch.bias, 0)

            nn.init.normal_(self.fc_roll.weight, mean=0, std=0.01)
            nn.init.constant_(self.fc_roll.bias, 0)

        else:
            self.fc_vfov = self._get_fc_layers(num_fc_layers, num_fc_channels,
                                               out_channels)
            self.fc_pitch = self._get_fc_layers(num_fc_layers, num_fc_channels,
                                                out_channels)
            self.fc_roll = self._get_fc_layers(num_fc_layers, num_fc_channels,
                                               out_channels)

    def _get_fc_layers(self, num_layers, num_channels, inp_channels):
        modules = []

        for i in range(num_layers):
            if i == 0:
                modules.append(nn.Linear(inp_channels, num_channels))
            elif i == num_layers - 1:
                modules.append(nn.Linear(num_channels, self.num_out_channels))
            else:
                modules.append(nn.Linear(num_channels, num_channels))

        return nn.Sequential(*modules)

    def forward(self, images):
        x = self.backbone(images)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        batch_size = x.shape[0]
        pred_vfov = self.fc_vfov(x)
        pred_pitch = self.fc_pitch(x)
        pred_roll = self.fc_roll(x)
        pred_vfov, pred_pitch, pred_roll = convert_preds_to_angles(
            pred_vfov,
            pred_pitch,
            pred_roll,
            loss_type='softargmax_l2',
        )

        if self.no_rot:
            cam_rotmat = torch.eye(3,
                                   3)[None].repeat_interleave(batch_size,
                                                              0).to(x.device)
        else:
            cam_rotmat = ee_to_rotmat(
                torch.cat(
                    [pred_pitch,
                     torch.zeros_like(pred_pitch), pred_roll],
                    -1).view(batch_size, 3))

        preds = {'cam_rotmat': cam_rotmat, 'cam_vfov': pred_vfov}
        return preds
