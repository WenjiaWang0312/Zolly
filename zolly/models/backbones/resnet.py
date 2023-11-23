from mmhuman3d.models.backbones.resnet import ResNet as _ResNet


class ResNet(_ResNet):

    def __init__(self, init_cfg=None, **kwargs):
        if init_cfg is not None:
            if init_cfg.get('checkpoint', None) is None:
                init_cfg = None
        super().__init__(init_cfg=init_cfg, **kwargs)
