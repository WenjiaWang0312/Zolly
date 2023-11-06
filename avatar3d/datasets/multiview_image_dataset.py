from .image_dataset import ImageDataset


class MultiViewImageDataset(ImageDataset):

    def __init__(self,
                 dataset_name: str = 'people_snapshot',
                 convention='',
                 model_type='smpl',
                 root: str = '',
                 read_cache=True,
                 parameter_config=...) -> None:
        super().__init__(dataset_name, convention, model_type, root,
                         read_cache, parameter_config)
