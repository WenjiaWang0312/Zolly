from random import random
from .video_dataset import VideoDataset


class CycleVideoDataset(VideoDataset):

    def __init__(self,
                 dataset_name: str = 'people_snapshot',
                 convention: str = 'smpl_45',
                 root: str = '',
                 model_type: str = 'smpl',
                 random=False,
                 interval=1,
                 seqlen=1,
                 read_cache=False,
                 parameter_config=dict()) -> None:
        super(CycleVideoDataset,
              self).__init__(dataset_name, convention, root, model_type,
                             seqlen, read_cache, parameter_config)
        self.random = random
        self.interval = interval

    def select_index(self, index):
        if self.random:
            dst_index = max(
                0,
                min(
                    random.randint(index - self.interval,
                                   index + self.interval), self.__len__()))
        else:
            dst_index = max(0, min(index + self.interval, self.__len__()))
        return dst_index

    def __getitem__(self, index):
        src_index, dst_index = self.select_index(index)
        src_data = super().__getitem__(src_index)
        dst_data = super().__getitem__(dst_index)
        return dict(src=src_data, dst=dst_data)
