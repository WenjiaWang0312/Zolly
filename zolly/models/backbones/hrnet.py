import torch
import torch.nn as nn
import torch.nn.functional as F
from mmhuman3d.models.backbones.hrnet import (PoseHighResolutionNet as HRNet,
                                              PoseHighResolutionNetExpose as
                                              HRNetExpose)


class PoseHighResolutionNet(HRNet):

    def __init__(self, init_cfg=None, out_indices=[0, 1, 2, 3], **kwargs):
        self.out_indices = out_indices
        if init_cfg is not None:
            if init_cfg.get('checkpoint', None) is None:
                init_cfg = None
        super().__init__(init_cfg=init_cfg, **kwargs)

    def forward(self, x):
        res = super(PoseHighResolutionNet, self).forward(x)
        if self.extra['return_list']:
            outs = []
            for i in self.out_indices:
                outs.append(res[i])
            return tuple(outs)
        else:
            return res


class PoseHighResolutionNetExpose(HRNetExpose):

    def __init__(self, init_cfg=None, mean=False, **kwargs):
        if init_cfg is not None:
            if init_cfg.get('checkpoint', None) is None:
                init_cfg = None
        self.mean = mean
        super().__init__(init_cfg=init_cfg, **kwargs)

    def forward(self, x):
        """Forward function."""
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['num_branches']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['num_branches']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['num_branches']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x3 = self.subsample_3(x_list[1])
        x2 = self.subsample_2(x_list[2])
        x1 = x_list[3]
        xf = self.conv_layers(torch.cat([x3, x2, x1], dim=1))
        if self.mean:
            xf = xf.mean(dim=(2, 3))
            xf = xf.view(xf.size(0), -1)
        return xf


class PoseHighResolutionNetGraphormer(PoseHighResolutionNetExpose):

    def __init__(self, init_cfg=None, mean=False, **kwargs):
        super().__init__(init_cfg, mean, **kwargs)
        self.final_layer = nn.Sequential(
            nn.Conv2d(in_channels=2048,
                      out_channels=2048,
                      kernel_size=1,
                      stride=1,
                      padding=0), nn.BatchNorm2d(2048, momentum=0.1),
            nn.ReLU(inplace=True))

    def forward(self, x):
        y = super().forward(x)
        yy = self.final_layer(y)

        if torch._C._get_tracing_state():
            yy = yy.flatten(start_dim=2).mean(dim=2)
        else:
            yy = F.avg_pool2d(yy,
                              kernel_size=yy.size()[2:]).view(yy.size(0), -1)

        return yy, y
