from .image_dataset import ImageDataset
from .human_image_dataset import HumanImageDataset
from mmhuman3d.data.datasets.builder import build_dataset, DATASETS, build_dataloader
from .demo_dataset import DemoDataset

DATASETS.register_module(name=['DemoDataset'], module=DemoDataset)
DATASETS.register_module(name=['ImageDataset'], module=ImageDataset)
DATASETS.register_module(name=['HumanImageDataset'],
                         module=HumanImageDataset,
                         force=True)

__all__ = ['build_dataset', 'build_dataloader']
