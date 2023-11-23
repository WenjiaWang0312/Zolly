import glob
from abc import ABCMeta
import numpy as np
import cv2
from mmhuman3d.data.datasets.base_dataset import BaseDataset
from zolly.datasets.pipelines.compose import Compose


class DemoDataset(BaseDataset, metaclass=ABCMeta):

    def __init__(self,
                 data_prefix: str,
                 pipeline: list,
                 ext: str = '.png',
                 dataset_name: str = 'demo',
                 test_mode: bool = True,
                 **kwargs):
        self.image_paths = glob.glob(f'{data_prefix}/*{ext}')
        self.image_paths.sort()
        self.dataset_name = dataset_name
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode

    def __len__(self):
        return len(self.image_paths)

    def load_annotations(self, **kwargs):
        pass

    def prepare_raw_data(self, idx):
        sample_idx = idx
        # idx = int(self.valid_index[idx])

        info = {}
        info['image_path'] = self.image_paths[idx]
        info['img_prefix'] = None
        info['img'] = cv2.imread(self.image_paths[idx])
        h, w, _ = info['img'].shape
        info['scale'] = np.array([max(h, w), max(h, w)])

        info['ori_shape'] = (h, w)
        info['center'] = np.array([w / 2, h / 2])

        info['dataset_name'] = self.dataset_name
        info['sample_idx'] = sample_idx
        info['has_focal_length'] = 0
        info['has_transl'] = 0
        info['has_K'] = 0
        info['K'] = np.eye(3, 3).astype(np.float32)
        return info

    def evaluate(self, ):
        raise NotImplementedError

    def prepare_data(self, idx: int):
        """Generate and transform data."""
        info = self.prepare_raw_data(idx)
        return self.pipeline(info)
