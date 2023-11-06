from mmcv.utils import Registry

RUNNERS = Registry('runners')

from .fitter import SMPLify, CycleSMPLify, SMPLifyH_focal

RUNNERS.register_module(name=['smplify', 'SMPLify'], module=SMPLify)
RUNNERS.register_module(name=['SMPLifyH_focal', 'smplifyH_focal'],
                        module=SMPLifyH_focal)
RUNNERS.register_module(
    name=['cycle-smplify', 'smplify-cycle', 'CycleSMPLify'],
    module=CycleSMPLify)


def build_runner(cfg):
    """Build loss."""
    if cfg is None:
        return None
    return RUNNERS.build(cfg)
