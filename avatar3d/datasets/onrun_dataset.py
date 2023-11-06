from torch.utils.data import Dataset


class OnRunDataset(Dataset):

    def __init__(self,
                 parameters: dict = dict(),
                 targets: dict = dict(),
                 resolution: int = 224,
                 convention: str = 'smpl',
                 batch_size: int = 1) -> None:
        super().__init__()
        self.calibrated = False
        self.parameters = parameters
        self.data = targets
        self.len = batch_size
        self.resolution = resolution
        self.convention = convention

    def set_valid_keys(self, keys):
        self.valid_keys = keys

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        data = dict()

        for key in self.valid_keys:
            data[key] = self.data[key][index]

        data['frame_id'] = index

        return data
