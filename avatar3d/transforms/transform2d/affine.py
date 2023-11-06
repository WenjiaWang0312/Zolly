from .transform2d import Transform2d


class AffineTransform(Transform2d):

    def __init__(self) -> None:
        pass


class RandomAffineTranform(AffineTransform):

    def __init__(self) -> None:
        super().__init__()