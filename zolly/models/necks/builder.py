# Copyright (c) OpenMMLab. All rights reserved.

from mmcv.utils import Registry
from zolly.models.layers.fpn import DenseFPN
from zolly.models.layers.conv import Conv1x1

NECKS = Registry('necks')

NECKS.register_module(name='DenseFPN', module=DenseFPN)
NECKS.register_module(name='Conv1x1', module=Conv1x1)


def build_neck(cfg):
    """Build neck."""
    if cfg is None:
        return None
    return NECKS.build(cfg)
