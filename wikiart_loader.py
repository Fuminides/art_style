import os
import pandas as pd
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import torch

def read_image(path):
    img = Image.open(path)
    # transform = transforms.Compose([transforms.ToTensor()])
    return img 


class WikiArtLoader(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, samples=1.0):
        if samples < 1.0:
            self.img_labels = annotations_file.sample(frac=samples)
        else:
            self.img_labels = annotations_file
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])

        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        
        try:
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
        except IOError:
            print("Error in transform")
            dummy_torch = torch.rand(3, 224, 224)
            dummy_label = 0
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            dummy_torch = dummy_torch.to(device)
            
            return dummy_torch, dummy_label
        
        return image, label
