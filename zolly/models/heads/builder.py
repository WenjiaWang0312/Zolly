from mmhuman3d.models.heads.builder import HEADS
from .graphcmr_head import CMRHead
from .graphormer_head import GraphormerHead
from .fastmetro_head import FastMetroHead
from .hmr_head import HMRBodyHead
from .zolly_head import ZollyHead
from .iuv_head import IUVDHead

HEADS.register_module(name='IUVDHead', module=IUVDHead)
HEADS.register_module(name='ZollyHead', module=ZollyHead)
HEADS.register_module(name='HMRBodyHead', module=HMRBodyHead)
HEADS.register_module(name='FastMetroHead', module=FastMetroHead)
HEADS.register_module(name='GraphormerHead', module=GraphormerHead)
HEADS.register_module(name='CMRHead', module=CMRHead)


def build_head(cfg):
    """Build head."""
    if cfg is None:
        return None
    return HEADS.build(cfg)
