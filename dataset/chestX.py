from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
import os
class Chest(Dataset):
    def __init__(self, root, transform):
        """
        Args:
            root: Root directory path.
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.img_path = os.path.join(root, 'images')
        self.csv_path = os.path.join(root, 'Data_Entry_2017.csv')
        self.used_labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax"]

        self.labels_maps = {"Atelectasis": 0, "Cardiomegaly": 1, "Effusion": 2, "Infiltration": 3, "Mass": 4, "Nodule": 5,  "Pneumothorax": 6}
        
        labels_set = []


        # Read the csv file
        self.data_info = pd.read_csv(self.csv_path, skiprows=[0], header=None)

        # First column contains the image paths
        self.image_name_all = np.asarray(self.data_info.iloc[:, 0])
        self.labels_all = np.asarray(self.data_info.iloc[:, 1])

        self.image_name  = []
        self.label = []

        self.transform = transform

        for name, label in zip(self.image_name_all,self.labels_all):
            label = label.split("|")

            if len(label) == 1 and label[0] != "No Finding" and label[0] != "Pneumonia" and label[0] in self.used_labels:
                self.label.append(self.labels_maps[label[0]])
                self.image_name.append(name)
    
        self.data_len = len(self.image_name)

        self.image_name = np.asarray(self.image_name)
        self.label = np.asarray(self.label)        

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_name[index]

        # Open image
        image = Image.open(os.path.join(self.img_path, single_image_name)).convert('RGB')
        image = self.transform(image)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label[index]

        return (image, single_image_label)

    def __len__(self):
        return self.data_len
