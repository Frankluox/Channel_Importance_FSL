from torchvision.datasets import ImageFolder

class general_dataset(ImageFolder):
    r"""The standard dataset. ::
         
        root
        |
        |
        |---train
        |    |--n01532829
        |    |   |--n0153282900000005.jpg
        |    |   |--n0153282900000006.jpg
        |    |              .
        |    |              .
        |    |--n01558993
        |        .
        |        .
        |---val
        |---test  
    Args:
        root: Root directory path.
        transform: pytorch transforms for transforms and tensor conversion
    """
    def __init__(self, root: str, transform) -> None:
        IMAGE_PATH = root
        super().__init__(IMAGE_PATH, transform)
        self.label = self.targets
