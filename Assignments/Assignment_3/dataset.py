from typing import Tuple
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from glob import glob
from annotation import read_groundtruth_file
import os
import numpy as np
from tqdm import tqdm
from PIL import Image


class MMP_Dataset(torch.utils.data.Dataset):
    """Exercise 3.2"""

    def __init__(self, path_to_data: str, image_size: int):
        """
        @param path_to_data: Path to the folder that contains the images and annotation files, e.g. dataset_mmp/train
        @param image_size: Desired image size that this dataset should return
        """
        self.size = image_size
        self.imgs = sorted(glob(os.path.join(path_to_data, '*.jpg')))
        self.annotations = {}
        for img in self.imgs:
            gt_file = img.replace('jpg', 'gt_data.txt')
            self.annotations[img] = read_groundtruth_file(gt_file)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        @return: Tuple of image tensor and label. The label is 0 if there is one person and 1 if there a multiple people.
        """
        img_file = self.imgs[idx]
        img = Image.open(img_file)    

        img = TF.to_tensor(img)
        c, h, w = img.size()
        pad_to = max(h, w)
        img = TF.pad(img, [0, max(0, w - pad_to), 0, max(0, h - pad_to)])
        img = TF.resize(img, [self.size, self.size], antialias=True)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        annotation = self.annotations[img_file]

        return (img, 1 if len(annotation) > 1 else 0)

    def __len__(self) -> int:
        return len(self.imgs)


def get_dataloader(
    path_to_data: str, image_size: int, batch_size: int, num_workers: int
, is_train=False) -> DataLoader:
    """Exercise 3.2d"""
    dataset = MMP_Dataset(path_to_data, image_size)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, drop_last=is_train, shuffle=is_train)


def main():
    loader = get_dataloader('dataset_mmp/train', 256, 16, 1)
    loop = tqdm(enumerate(loader), total=len(loader), leave=False)

    for b_nr, (input, target) in loop:
        print(input.size())


if __name__ == "__main__":
    main()
    
