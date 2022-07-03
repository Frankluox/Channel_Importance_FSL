from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
import os

class ISIC(Dataset):
    def __init__(self, root, transform):
        """
        Args:
            root: Root directory path.
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.img_path = os.path.join(root, "ISIC2018_Task3_Training_Input")
        self.csv_path = os.path.join(root, "ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv")

        

        self.transform = transform


        # Read the csv file
        self.data_info = pd.read_csv(self.csv_path, skiprows=[0], header=None)

        # First column contains the image paths
        self.image_name = np.asarray(self.data_info.iloc[:, 0])

        self.label = np.asarray(self.data_info.iloc[:, 1:])

        # print(self.labels[:10])
        self.label = (self.label!=0).argmax(axis=1)

        # Calculate len
        self.data_len = len(self.label)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_name[index]
        # Open image
        image = Image.open(os.path.join(self.img_path, single_image_name + ".jpg")).convert('RGB')
        image = self.transform(image)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label[index]

        return (image, single_image_label)

    def __len__(self):
        return self.data_len

