from torchvision.datasets import ImageFolder
import os

class miniImageNet(ImageFolder):
    r"""The miniImageNet dataset.
    Args:
        root: Root directory path.
        transform: pytorch transforms for transforms and tensor conversion
    """
    def __init__(self, root: str, transform) -> None:
        IMAGE_PATH = os.path.join(root, "test")
        self.transform = transform

        super().__init__(IMAGE_PATH, self.transform)

        self.label = self.targets  # label of all data

    def __getitem__(self, i):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

