from mmhuman3d.models.architectures.builder import ARCHITECTURES
from .mesh_estimator import TestMeshEstimator
from .zolly_estimator import ZollyEstimator

ARCHITECTURES.register_module(name=['ZollyEstimator'], module=ZollyEstimator)
ARCHITECTURES.register_module(name=['TestMeshEstimator'],
                              module=TestMeshEstimator)


def build_architecture(cfg):
    """Build framework."""
    return ARCHITECTURES.build(cfg)
