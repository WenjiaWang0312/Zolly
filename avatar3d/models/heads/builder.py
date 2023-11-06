from mmhuman3d.models.heads.builder import HEADS
from .iuv_head import IUVDHead, IUVDFHead, IUVHead, IUVDFHead_old, IUVDFHead2, IUVDFHead_neck
from .graphcmr_head import CMRHead
from .cliff_head import CliffHead
from .spec_head import SPECHead
from .graphormer_head import GraphormerHead, GraphormerHead_orig
from .fastmetro_head import FastMetroHead
from .hmr_head import HMRBodyHead
from .zollyp_head import ZollyPHead
from .heatmap_z_head import HeatmapZHead, HeatmapJ3DHead
from .heatmap_head import HeatmapHead
from .zoex_ik_head import ZoeXIKHead, BodyIKHead, HandIKHead, NIKIHead
from .ndco_head2 import NDCHeatmapHead
from .ndco_part_head import NDCHeatmapPartHead

HEADS.register_module(name='NDCHeatmapHead', module=NDCHeatmapHead)
HEADS.register_module(name='NDCHeatmapPartHead', module=NDCHeatmapPartHead)
HEADS.register_module(name='NIKIHead', module=NIKIHead)
HEADS.register_module(name='ZoeXIKHead', module=ZoeXIKHead)
HEADS.register_module(name='BodyIKHead', module=BodyIKHead)
HEADS.register_module(name='HandIKHead', module=HandIKHead)

HEADS.register_module(name='HeatmapJ3DHead', module=HeatmapJ3DHead)
HEADS.register_module(name='HeatmapHead', module=HeatmapHead)
HEADS.register_module(name='HeatmapZHead', module=HeatmapZHead)
HEADS.register_module(name='ZollyPHead', module=ZollyPHead)
HEADS.register_module(name='IUVDFHead_neck', module=IUVDFHead_neck)
HEADS.register_module(name='IUVDFHead2', module=IUVDFHead2)
HEADS.register_module(name='IUVDFHead_old', module=IUVDFHead_old)
HEADS.register_module(name='GraphormerHead_orig', module=GraphormerHead_orig)
HEADS.register_module(name='IUVHead', module=IUVHead)
HEADS.register_module(name='HMRBodyHead', module=HMRBodyHead)
HEADS.register_module(name='FastMetroHead', module=FastMetroHead)
HEADS.register_module(name='GraphormerHead', module=GraphormerHead)
HEADS.register_module(name='IUVDFHead', module=IUVDFHead)

HEADS.register_module(name='SPECHead', module=SPECHead)
HEADS.register_module(name='CliffHead', module=CliffHead, force=True)
HEADS.register_module(name='IUVDHead', module=IUVDHead)
HEADS.register_module(name='CMRHead', module=CMRHead)


def build_head(cfg):
    """Build head."""
    if cfg is None:
        return None
    return HEADS.build(cfg)
