import glob
import os
from typing import Optional
from avatar3d.datasets.pipelines.compose import Compose
from mmhuman3d.data.datasets.base_dataset import BaseDataset


class SegDataset(BaseDataset):

    def __init__(
        self,
        data_prefix: str,
        pipeline: list,
        dataset_name: str,
        test_mode: Optional[bool] = False,
    ):
        if dataset_name is not None:
            self.dataset_name = dataset_name

        self.data_prefix = data_prefix
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        self.mask_paths = glob.glob(f'{data_prefix}/*_uvd.png')
        self.image_paths = [
            mask_path.replace('_uvd', '').replace('labels', 'test')
            for mask_path in self.mask_paths
        ]

    def load_annotations(self):
        pass

    def prepare_raw_data(self, idx: int):
        """Get item from self.human_data."""
        info = {}
        info['img_prefix'] = None
        info['image_path'] = self.image_paths[idx]
        info['mask_path'] = self.mask_paths[idx]
        info['has_mask'] = os.path.exists(self.mask_paths[idx])
        return info

    def __len__(self):
        return len(self.mask_paths)

    def prepare_data(self, idx: int):
        """Generate and transform data."""
        info = self.prepare_raw_data(idx)
        return self.pipeline(info)
