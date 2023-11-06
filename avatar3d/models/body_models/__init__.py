from .mano import MANO, MANO_LEFT, MANO_RIGHT
from .smpl import SMPL
from .smpl_d import SMPL_D, SMPLH_D
from .smplh import SMPLH
from .smpl_x import SMPLX
from .flame import FLAME

__all__ = [
    'MANO', 'SMPL', 'SMPLH', 'SMPLX', 'FLAME', 'SMPLD', 'MANO_LEFT',
    'MANO_RIGHT'
]


def get_model_class(key):
    key = key.lower().replace('-', '')
    model_class = dict(smpl=SMPL,
                       mano=MANO,
                       flame=FLAME,
                       smplh=SMPLH,
                       smplx=SMPLX,
                       smpl_d=SMPL_D,
                       smplh_d=SMPLH_D,
                       mano_left=MANO_LEFT,
                       mano_right=MANO_RIGHT)
    return model_class[key]
