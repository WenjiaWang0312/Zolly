import torch
from avatar3d.datasets.image_dataset import ImageDataset
from avatar3d.utils.frame_utils import images_to_array_cv2


class VideoDataset(ImageDataset):

    def __init__(
        self,
        dataset_name: str = 'people_snapshot',
        convention: str = 'smpl_45',
        root: str = '',
        model_type='smpl',
        seqlen: int = 1,
        read_cache: bool = False,
        parameter_config: dict = dict()) -> None:
        super(VideoDataset, self).__init__(
            dataset_name=dataset_name,
            convention=convention,
            root=root,
            model_type=model_type,
            read_cache=read_cache,
            parameter_config=parameter_config,
        )
        self.seqlen = seqlen

    def __len__(self):
        return super().__len__() - self.seqlen + 1

    def __getitem__(self, index):
        data = {}

        if self.read_cache:
            if (self.valid_keys is not None
                    and 'image' in self.valid_keys) or self.valid_keys is None:
                data['image'] = self.images[index:index + self.seqlen]
            if (self.valid_keys is not None
                    and 'mask' in self.valid_keys) or self.valid_keys is None:
                data['mask'] = self.masks[index:index + self.seqlen]

        else:
            if self.image_list:
                if (self.valid_keys is not None and 'image'
                        in self.valid_keys) or self.valid_keys is None:
                    data['image'] = torch.from_numpy(
                        images_to_array_cv2(
                            self.image_list[index:index +
                                            self.seqlen])).float()
            if self.mask_list:
                if (self.valid_keys is not None and 'mask'
                        in self.valid_keys) or self.valid_keys is None:
                    data['mask'] = torch.from_numpy(
                        images_to_array_cv2(
                            self.mask_list[index:index +
                                           self.seqlen])).float() / 1.0

        if self.kp2d is not None:
            if (self.valid_keys is not None
                    and 'kp2d' in self.valid_keys) or self.valid_keys is None:
                data['kp2d'] = torch.from_numpy(self.kp2d[index])

        if self.kp3d is not None:
            if (self.valid_keys is not None
                    and 'kp3d' in self.valid_keys) or self.valid_keys is None:
                data['kp3d'] = torch.from_numpy(self.kp3d[index])

        data['frame_id'] = list(range(index, index + self.seqlen))
        return data
