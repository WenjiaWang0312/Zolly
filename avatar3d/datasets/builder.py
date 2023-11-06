from .image_dataset import ImageDataset
from .render_dataset import DistortionMapDataset
from .uvd_dataset import UVDDataset
from .video_dataset import VideoDataset
from .cycle_video_dataset import CycleVideoDataset
from .human_image_dataset import HumanImageDataset
from .seg_dataset import SegDataset
from mmhuman3d.data.datasets.builder import build_dataset, DATASETS, build_dataloader
from .demo_dataset import DemoDataset
from .smplx_image_dataset import HumanImageSMPLXDataset

DATASETS.register_module(name=['HumanImageSMPLXDataset'],
                         module=HumanImageSMPLXDataset,
                         force=True)
DATASETS.register_module(name=['DemoDataset'], module=DemoDataset)
DATASETS.register_module(name=['SegDataset'], module=SegDataset)
DATASETS.register_module(name=['ImageDataset'], module=ImageDataset)
DATASETS.register_module(name=['HumanImageDataset'],
                         module=HumanImageDataset,
                         force=True)
DATASETS.register_module(name=['DistortionMapDataset'],
                         module=DistortionMapDataset)
DATASETS.register_module(name=['VideoDataset'], module=VideoDataset)
DATASETS.register_module(name=['CycleVideoDataset'], module=CycleVideoDataset)
DATASETS.register_module(name=['UVDDataset'], module=UVDDataset)

__all__ = ['build_dataset', 'build_dataloader']
