from mmcv.utils import Registry
from .smpl_visualizer import SmplVisualizer

VISUALIZER = Registry('visualizer')
VISUALIZER.register_module(name=['SmplVisualizer', 'smpl_visualizer'],
                           module=SmplVisualizer)


def build_visualizer(cfg):
    """Build head."""
    if cfg is None:
        return None
    return VISUALIZER.build(cfg)
