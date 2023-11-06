from mmhuman3d.models.architectures.builder import ARCHITECTURES
from .perspectivte_mesh_estimator import PersepectiveMeshEstimator
from .perspectivte_mesh_estimator1 import PersepectiveMeshEstimator1
from .verts_estimator import VertsEstimator
from .mesh_estimator import TestMeshEstimator
from .zoex_estimator import ZoeXEstimator
from .niki_estimator import NIKIEstimator

ARCHITECTURES.register_module(name=['NIKIEstimator'], module=NIKIEstimator)
ARCHITECTURES.register_module(name=['ZoeXEstimator'], module=ZoeXEstimator)

ARCHITECTURES.register_module(name=['TestMeshEstimator'],
                              module=TestMeshEstimator)
ARCHITECTURES.register_module(name=['VertsEstimator'], module=VertsEstimator)
ARCHITECTURES.register_module(name=['PersepectiveMeshEstimator'],
                              module=PersepectiveMeshEstimator)
ARCHITECTURES.register_module(name=['PersepectiveMeshEstimator1'],
                              module=PersepectiveMeshEstimator1)


def build_architecture(cfg):
    """Build framework."""
    return ARCHITECTURES.build(cfg)
