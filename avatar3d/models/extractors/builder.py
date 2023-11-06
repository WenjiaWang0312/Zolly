from mmcv.utils import Registry
# from .spec_camcalib import SpecCamExtractor
EXTRACTOR = Registry('visualizer')

# EXTRACTOR.register_module(name=['SpecCamExtractor', 'spec_cam'],
#                           module=SpecCamExtractor)


def build_extractor(cfg):
    """Build head."""
    if cfg is None:
        return None
    return EXTRACTOR.build(cfg)
