from torch.utils.data import Dataset
from PIL import Image
import os.path as osp

class miniImageNet(Dataset):
    r"""The miniImageNet dataset.
    Args:
        root: Root directory path.
        transform: pytorch transforms for transforms and tensor conversion
    """
    def __init__(self, root: str, transform) -> None:
        
        IMAGE_PATH = osp.join(root, 'images')
        SPLIT_PATH = osp.join(root, 'split')

        csv_path = osp.join(SPLIT_PATH, 'test.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(IMAGE_PATH, name)
            if wnid not in wnids:
                wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data  # data path of all data
        self.label = label  # label of all data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label

