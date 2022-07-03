from torchvision.datasets import Omniglot
from torch.utils.data import Dataset

class omniglot(Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        self.dataset = Omniglot(root, "test", transform)
        self.label = []
        for pair in self.dataset._flat_character_images:
            self.label.append(pair[1])
    def __getitem__(self, index):
        return self.dataset[index]
    def __len__(self):
        return len(self.dataset)

