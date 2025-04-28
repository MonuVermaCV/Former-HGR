from __future__ import print_function, division
import os.path
from numpy.random import randint
from torch.utils import data
import glob
import os
#from dataloader.video_transform import *
import numpy as np

import sys
import argparse
import enum
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


namerafdb = {'1':'C1','2':'C2','3':'C3','4':'C4','5':'C5','6':'C6','7':'C7'}
DatasetsLocation = {
    'ouhands':{10: 'Test'}
}


DatasetsMeanAndStd = {
    'ouhands':{
        10: {'mean': [0.5754, 0.4499, 0.4017], 'std':  [0.2636, 0.2403, 0.2388]}
    }
    

}


class ClassesRAFDB(enum.Enum):
    C1 = 0
    C2 = 1
    C3 = 2
    C4 = 3
    C5 = 4
    C6 = 5
    C7 = 6
    



classes_map = {
    "ouhands": {10: ClassesRAFDB}
    
}



        
def image_loader(path, transform):
    img = Image.open(path)
   
    # Check if the image is grayscale (1 channel)
    if img.mode == '1' or 'L':  # 'L' means grayscale
        
        img = img.convert('RGB')  # Convert grayscale to RGB (3 channels)
    else:
        print(f"The image is not grayscale, it is in {img.mode} mode.")
    
    # Apply the transformation
    img = transform(img)
    return img


class ClassificationDataset(Dataset):

    def __init__(self, train, arch_split_file):
        super().__init__()
        dataset="ouhands"
        no_of_classes=10
        self.base_dir = DatasetsLocation.get(dataset).get(no_of_classes)
        # print(self.base_dir)
        
        self.image_dir = os.path.join("data", dataset.upper(), self.base_dir)
        # print(self.image_dir)
        self.image_paths = []
        self.labels = []
        self.loader = image_loader
        for line in open(arch_split_file):
                    index = line.find('path:')
                    index2 = line.find('label:')
                    self.image_paths.append(line[index:index2].split(":")[-1].strip())
                    self.labels.append(int(line[index2:].split(":")[-1].strip()))
        
        self.transform = transforms.Compose([])
        if dataset == 'ouhands': 
            self.transform.transforms.append(transforms.Resize((112,112)))
        if train==True:
            # self.transform.transforms.append(transforms.Grayscale(3))
            
            self.transform.transforms.append(transforms.RandomRotation(15))

            self.transform.transforms.append(transforms.RandomHorizontalFlip())
            self.transform.transforms.append(transforms.RandomVerticalFlip())
            self.transform.transforms.append(transforms.ColorJitter())

        
        self.transform.transforms.append(transforms.ToTensor())
        self.transform.transforms.append(
            transforms.Normalize(DatasetsMeanAndStd.get(dataset).get(no_of_classes).get('mean'),
                                  DatasetsMeanAndStd.get(dataset).get(no_of_classes).get('std')))


        # print(f'Number Of Videos in {args.dataset.upper()} dataset split: {arch_split_file}: {len(self.image_paths)}')

    def __getitem__(self, index):
        path = self.image_paths[index]
        # print(path)
        label = self.labels[index]
        # print(label)
        img = self.loader(path, self.transform)
        return {'image': img, 'label': label}

    def __len__(self):
        return len(self.image_paths)


def train_data_loader():
    
    path1 = "dataloader/OUHANDS_train.txt"
        
    train_data = ClassificationDataset( train=True, arch_split_file=path1)

    

    path2 = "dataloader/OUHANDS_val.txt"
        
    val_data = ClassificationDataset( train=False, arch_split_file=path2)
        
    return train_data, val_data


def test_data_loader():
    path2 = "dataloader/OUHANDS_test.txt"
        
    test_data = ClassificationDataset( train=False, arch_split_file=path2)
    return test_data
