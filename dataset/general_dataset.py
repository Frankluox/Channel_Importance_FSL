from torchvision.datasets import ImageFolder

class general_dataset(ImageFolder):
    r"""The standard dataset.
    Args:
        root: Root directory path.
        transform: pytorch transforms for transforms and tensor conversion
    """
    def __init__(self, root: str, transform) -> None:
        IMAGE_PATH = root
        super().__init__(IMAGE_PATH, transform)
        self.label = self.targets
