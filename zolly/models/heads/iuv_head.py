import torch
import torch.nn as nn
from zolly.models import BaseModule
from zolly.models.layers.conv import ConvBottleNeck


class IUVDHead(BaseModule):

    def __init__(
        self,
        norm_type='BN',
        in_channels=256,
        init_cfg=None,
    ):
        super(IUVDHead, self).__init__(init_cfg=init_cfg)
        nl_layer = nn.ReLU(inplace=True)

        self.conv = nn.Sequential(
            ConvBottleNeck(in_channels=in_channels,
                           out_channels=in_channels,
                           nl_layer=nl_layer,
                           norm_type=norm_type))

        self.mask_dec = nn.Sequential(
            ConvBottleNeck(in_channels=in_channels,
                           out_channels=32,
                           nl_layer=nl_layer,
                           norm_type=norm_type), nn.Conv2d(32,
                                                           1,
                                                           kernel_size=1),
            nn.Sigmoid())

        self.uv_dec = nn.Sequential(
            ConvBottleNeck(in_channels=in_channels,
                           out_channels=32,
                           nl_layer=nl_layer,
                           norm_type=norm_type), nn.Conv2d(32,
                                                           2,
                                                           kernel_size=1),
            nn.Sigmoid())

        self.distortion_dec = nn.Sequential(
            ConvBottleNeck(in_channels=in_channels,
                           out_channels=32,
                           nl_layer=nl_layer,
                           norm_type=norm_type),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(
        self,
        x,
    ):
        # B, 256, 56, 56
        x = self.conv(x)  # B, 256, 56, 56
        uv_image = self.uv_dec(x)  # B, 2, 56, 56
        mask = self.mask_dec(x)  # B, 1, 56, 56
        iuv_img = torch.cat((mask, uv_image), dim=1)
        distortion_img = self.distortion_dec(x)  # B, 1, 56, 56

        output = {
            'pred_iuv_img': iuv_img,  # B, 3, 56, 56
            'pred_d_img': distortion_img,  # B, 1, 56, 56
        }
        return output
