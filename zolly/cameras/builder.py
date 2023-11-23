from .pytorch3d_wrapper.builder import build_cameras as build_cameras_t3d


def build_cameras(cfg):
    backend = cfg.pop('backend', 'pytorch3d')
    assert backend in ['pytorch3d', 'pyrender', 'pyopengl']
    if backend == 'pytorch3d':
        return build_cameras_t3d(cfg)
