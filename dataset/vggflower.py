from PIL import Image
from torch.utils.data import Dataset
from scipy.io import loadmat
import torch

class OxfordFlowers102Dataset(Dataset):
    """
    Oxford 102 Category Flower Dataset
    https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html
    """

    def __init__(self, root, transform):

        self.transform = transform
        self.root = root


        labels_filename = self.root + "/imagelabels.mat"
        # shift labels from 1-index to 0-index
        self.label = loadmat(labels_filename)["labels"].flatten() - 1

    def __getitem__(self, index):
        filepath = self.root + "/jpg" + f"/image_{index+1:05}.jpg"
        img = Image.open(filepath).convert('RGB')
        img = self.transform(img)
        label = self.label[index]
        label = torch.tensor(label, dtype=torch.long)
        return img, label

    def __len__(self):
        return len(self.labels)
