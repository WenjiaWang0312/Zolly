from mmhuman3d.models.losses.builder import LOSSES
from zolly.models.losses.mse_loss import KeypointMSELoss, ImageGradMSELoss
from zolly.models.losses.smooth_loss import LimbSmoothLoss
from zolly.models.losses.prior_loss import (LimbLengthLoss, HingeLoss,
                                            LimbDirectionLoss)
from zolly.models.losses.render_loss import SilhouetteMSELoss, FlowWarpingLoss, SilhouetteRegLoss
from zolly.models.losses.mesh_loss import LaplacianLoss, ChamferLoss, MeshEdgeLoss, NormalConsistencyLoss
from zolly.models.losses.regression_loss import RLELoss, RLELossLight
from zolly.models.losses.norm_loss import NormLoss, MMDLoss
try:
    from mmseg.models.losses import CrossEntropyLoss

    LOSSES.register_module(name=['CrossEntropyLoss', 'cross_entropy_loss'],
                           module=CrossEntropyLoss,
                           force=True)
except ImportError:
    pass

LOSSES.register_module(name=['NormLoss', 'norm_loss'], module=NormLoss)
LOSSES.register_module(name=['MMDLoss', 'mmd_loss'], module=MMDLoss)
LOSSES.register_module(name=['RLELossLight'], module=RLELossLight, force=True)
LOSSES.register_module(name=['RLELoss'], module=RLELoss, force=True)
LOSSES.register_module(name=['KeypointMSELoss', 'keypoint_mse'],
                       module=KeypointMSELoss,
                       force=True)
LOSSES.register_module(name=['ImageGradMSELoss', 'image_grad_mse'],
                       module=ImageGradMSELoss,
                       force=True)
LOSSES.register_module(name=['LimbSmoothLoss', 'limb_smooth'],
                       module=LimbSmoothLoss)
LOSSES.register_module(name=['LimbLengthLoss', 'limb_length'],
                       module=LimbLengthLoss,
                       force=True)
LOSSES.register_module(name=['LimbDirectionLoss', 'limb_dirction'],
                       module=LimbDirectionLoss,
                       force=True)
LOSSES.register_module(name=[
    'silhouette', 'silhouette_mse', 'SilhouetteMSE', 'SilhouetteMSELoss'
],
                       module=SilhouetteMSELoss)

LOSSES.register_module(
    name=['flow', 'flow_warping', 'FlowWarping', 'FlowWarpingLoss'],
    module=FlowWarpingLoss)

LOSSES.register_module(name=['laplacian', 'LaplacianLoss', 'laplacian_loss'],
                       module=LaplacianLoss)
LOSSES.register_module(name=['ChamferLoss', 'chamfer', 'chamfer_loss'],
                       module=ChamferLoss)
LOSSES.register_module(name=['MeshEdgeLoss', 'mesh_edge', 'MeshEdge'],
                       module=MeshEdgeLoss)
LOSSES.register_module(name=['NormalConsistency', 'normal_consistency'],
                       module=NormalConsistencyLoss)
LOSSES.register_module(name=['HingeLoss', 'hinge_loss'], module=HingeLoss)
LOSSES.register_module(name=['SilhouetteRegLoss', 'silhouette_reg'],
                       module=SilhouetteRegLoss)


def build_loss(cfg):
    """Build losses."""
    if cfg is None:
        return None
    return LOSSES.build(cfg)
