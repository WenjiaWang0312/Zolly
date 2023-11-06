from mmcv.runner import BaseModule as _BaseModule


class BaseModule(_BaseModule):

    def __init__(
        self,
        init_cfg=None,
    ):
        if init_cfg is not None:
            if init_cfg.get('checkpoint', None) is None:
                init_cfg = None
        super().__init__(init_cfg=init_cfg)
