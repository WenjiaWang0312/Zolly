from mmhuman3d.models.backbones.builder import BACKBONES
from .resnet import ResNet
from .hrnet import (PoseHighResolutionNet, PoseHighResolutionNetExpose,
                    PoseHighResolutionNetGraphormer)

BACKBONES.register_module(name='ResNet', module=ResNet, force=True)
BACKBONES.register_module(name='PoseHighResolutionNet',
                          module=PoseHighResolutionNet,
                          force=True)
BACKBONES.register_module(name='PoseHighResolutionNetExpose',
                          module=PoseHighResolutionNetExpose,
                          force=True)

BACKBONES.register_module(name='PoseHighResolutionNetGraphormer',
                          module=PoseHighResolutionNetGraphormer,
                          force=True)


def build_backbone(cfg):
    """Build head."""
    if cfg is None:
        return None
    return BACKBONES.build(cfg)
