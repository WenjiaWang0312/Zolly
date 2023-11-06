from mmcv.utils import Registry
from . import MANO, SMPL, SMPLH, FLAME, SMPLX, SMPL_D, MANO_LEFT, MANO_RIGHT, SMPLH_D

BODY_MODELS = Registry('body_models')
BODY_MODELS.register_module(name=['SMPL', 'smpl'], module=SMPL)
BODY_MODELS.register_module(name=['MANO', 'mano'], module=MANO)
BODY_MODELS.register_module(name=['SMPLH', 'smplh'], module=SMPLH)
BODY_MODELS.register_module(name=['SMPLX', 'smplx'], module=SMPLX)
BODY_MODELS.register_module(name=['FLAME', 'flame'], module=FLAME)
BODY_MODELS.register_module(name=['SMPL_D', 'smpl_d'], module=SMPL_D)
BODY_MODELS.register_module(name=['SMPLH_D', 'smplh_d'], module=SMPLH_D)
BODY_MODELS.register_module(name=['MANO_LEFT', 'mano_left', 'mano_l'],
                            module=MANO_LEFT)
BODY_MODELS.register_module(name=['MANO_RIGHT', 'mano_right', 'mano_r'],
                            module=MANO_RIGHT)


def build_body_model(cfg):
    """Build body_models."""
    if cfg is None:
        return None
    return BODY_MODELS.build(cfg)
