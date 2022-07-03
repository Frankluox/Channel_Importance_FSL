from PIL import Image
from torch.utils.data import Dataset
import torch
import json

class coco(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
    
        with open(f"{root}/annotations/instances_val2017.json", 'r') as f:   
            fra = json.load(f) 
        annotations = fra["annotations"]
        self.label = []
        self.image_ids = []
        category_id2label = {}
        label_id = -1
        for ann in annotations:
            category = ann["category_id"]
            
            if ann["category_id"] not in [80, 87, 89]:
                if category not in category_id2label:
                    label_id += 1
                    category_id2label[category] = label_id
                self.image_ids.append(ann["image_id"])
                self.label.append(category_id2label[category])

    def __getitem__(self, index):
        image_id = str(self.image_ids[index])

        image_id = '0'*(12-len(image_id))+image_id
        path = f"{self.root}/val2017/{image_id}.jpg"
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        label = self.label[index]
        label = torch.tensor(label, dtype=torch.long)
        return img, label
    
    def __len__(self):
        return len(self.label)
