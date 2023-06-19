from typing import Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from anchor_grid import get_anchor_grid
from label_grid import draw_anchor, get_label_grid
from glob import glob
from annotation import read_groundtruth_file, AnnotationRect
import os
from pathlib import Path
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from PIL import Image
import cv2
from transformations import wrap_sample, unwrap_sample, get_train_transforms, to_cv2_img


class MMP_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path_to_data: str,
        image_size: int,
        anchor_grid: np.ndarray,
        min_iou: float,
        is_test: bool,
        apply_transforms_on_init: bool = False,
        transform=None,
    ):
        """
        @param anchor_grid: The anchor grid to be used for every image
        @param min_iou: The minimum IoU that is required for an overlap for the label grid.
        @param is_test: Whether this is the test set (True) or the validation/training set (False)
        """
        self.size = image_size
        self.imgs = sorted(glob(os.path.join(path_to_data, '*.jpg')))
        self.annotations = {}
        self.transformed_annotations = {}
        self.ids = {}
        self.anchor_grid = anchor_grid
        self.is_test = is_test
        self.min_iou = min_iou
        self.transform = transform

        for img_file in self.imgs:
            # Get image id
            img_path = Path(img_file)
            img_id = int(img_path.stem)

            self.ids[img_file] = img_id
            if not is_test:
                # Get annotations
                gt_file = img_file.replace('jpg', 'gt_data.txt')
                annotation = read_groundtruth_file(gt_file)
                self.annotations[img_id] = annotation
                # If apply_transforms_on_init is True, apply transforms to the annotations beforehand so they are present for multiple workers 
                if apply_transforms_on_init:
                    img = Image.open(img_file)
                    img = TF.to_tensor(img)
                    c, h, w = img.size()
                    pad_to = max(h, w)
                    for i in range(len(annotation)):
                        box = np.array(annotation[i])
                        box = (box * (self.size / pad_to)).astype(int)
                        annotation[i] = AnnotationRect.fromarray(box)
                    self.transformed_annotations[img_id] = annotation


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        @return: 3-tuple of image tensor, label grid, and image (file-)number
        """
        img_file = self.imgs[idx]
        img_id = self.ids[img_file]
        img = Image.open(img_file)
        img = TF.to_tensor(img)
        annotation = []

        if not self.is_test:
            annotation = self.annotations[img_id]

        if self.transform is not None:
            sample = wrap_sample(img, annotation)
            sample = self.transform(sample)
            img, annotation = unwrap_sample(sample)

        # Transform image
        c, h, w = img.size()
        pad_to = max(h, w)
        padding = [0, 0, max(0, pad_to - w), max(0, pad_to - h)]
        img = TF.pad(img, padding=padding)
        img = TF.resize(img, [self.size, self.size], antialias=True)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Transform annotations
        if not self.is_test:
            for i in range(len(annotation)):
                box = np.array(annotation[i])
                box = (box * (self.size / pad_to)).astype(int)
                annotation[i] = AnnotationRect.fromarray(box)
                
            label_grid = torch.tensor(get_label_grid(self.anchor_grid, annotation, self.min_iou), dtype=torch.long)
        else:
            label_grid = torch.tensor(0)

        return img, label_grid, img_id

    def __len__(self) -> int:
        return len(self.imgs)


def get_dataloader(
    path_to_data: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    anchor_grid: np.ndarray,
    is_test: bool,
    min_iou = 0.7,
    transforms = None,
    apply_transforms_on_init: bool = False,
) -> DataLoader:
    dataset = MMP_Dataset(path_to_data, image_size, anchor_grid, min_iou, is_test, apply_transforms_on_init, transform=transforms)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, drop_last=(not is_test), shuffle=(not is_test))


def main():
    # Anchor grid parameters
    IMSIZE = 224
    SCALE_FACTOR = 32
    WIDTHS = [IMSIZE * i for i in [0.8, 0.65, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2]]
    ASPECT_RATIOS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]

    anchor_grid = get_anchor_grid(int(IMSIZE / SCALE_FACTOR), int(IMSIZE / SCALE_FACTOR), scale_factor=SCALE_FACTOR, anchor_widths=WIDTHS, aspect_ratios=ASPECT_RATIOS)
    transforms = get_train_transforms(crop=True)
    print(transforms)
    dataset = MMP_Dataset('./dataset_mmp/train/', IMSIZE, anchor_grid, 0.7, False, False, transforms)

    image, target, id = dataset[0]
    image = to_cv2_img(image)
    idxs = torch.nonzero(target, as_tuple=True)        
    for box in anchor_grid[idxs]:
        image = draw_anchor(image, np.array(box))
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()  
