from torchvision.datasets import Omniglot
from torch.utils.data import Dataset
from PIL import Image
import os

class omniglot(Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        self.dataset = Omniglot(root, "test", transform)
        self.label = []
        for pair in self.dataset._flat_character_images:
            self.label.append(pair[1])

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image_name, character_class = self.dataset._flat_character_images[index]
        image_path = os.path.join(self.dataset.target_folder, self.dataset._characters[character_class], image_name)
        image = Image.open(image_path, mode="r").convert('RGB')

        if self.dataset.transform:
            image = self.dataset.transform(image)

        if self.dataset.target_transform:
            character_class = self.dataset.target_transform(character_class)
        return image, character_class

    def __len__(self):
        return len(self.dataset)

