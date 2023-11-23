from .explicit import (
    build_lights,
    build_shader,
    build_textures,
    build_renderer as build_renderer_pytorch3d,
)


def build_renderer(cfg):
    if cfg:
        backend = cfg.pop('backend', 'pytorch3d')
    else:
        return None
    if backend == 'pytorch3d':
        return build_renderer_pytorch3d(cfg)
    # elif backend == 'pyrender':
    #     return build_renderer_pyrender(cfg)
    # elif backend == 'opengl':
    #     return build_renderer_opengl(cfg)


__all__ = ['build_renderer', 'build_lights', 'build_shader', 'build_textures']
