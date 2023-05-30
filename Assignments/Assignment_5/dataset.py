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


class MMP_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path_to_data: str,
        image_size: int,
        anchor_grid: np.ndarray,
        min_iou: float,
        is_test: bool,
    ):
        """
        @param anchor_grid: The anchor grid to be used for every image
        @param min_iou: The minimum IoU that is required for an overlap for the label grid.
        @param is_test: Whether this is the test set (True) or the validation/training set (False)
        """
        self.size = image_size
        self.imgs = sorted(glob(os.path.join(path_to_data, '*.jpg')))
        self.annotations = {}
        self.anchor_grid = anchor_grid
        self.is_test = is_test
        self.min_iou = min_iou

        if is_test:
            return

        for img in self.imgs:
            gt_file = img.replace('jpg', 'gt_data.txt')
            annotations = read_groundtruth_file(gt_file)
            self.annotations[img] = annotations

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        @return: 3-tuple of image tensor, label grid, and image (file-)number
        """
        img_file = self.imgs[idx]
        img = Image.open(img_file)    

        # Transform image
        img = TF.to_tensor(img)
        c, h, w = img.size()
        pad_to = max(h, w)
        padding = [0, 0, max(0, pad_to - w), max(0, pad_to - h)]
        img = TF.pad(img, padding=padding)
        img = TF.resize(img, [self.size, self.size], antialias=True)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Transform annotations
        if not self.is_test:
            annotation = self.annotations[img_file]
            for i in range(len(annotation)):
                box = np.array(annotation[i])
                box = (box * (self.size / pad_to)).astype(int)
                annotation[i] = AnnotationRect.fromarray(box)
            label_grid = torch.tensor(get_label_grid(self.anchor_grid, annotation, self.min_iou), dtype=torch.long)
        else:
            label_grid = torch.tensor(0)

        # Get image id
        img_path = Path(img_file)
        img_id = int(img_path.stem)

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
    min_iou = 0.7
) -> DataLoader:
    dataset = MMP_Dataset(path_to_data, image_size, anchor_grid, min_iou, is_test)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, drop_last=(not is_test), shuffle=(not is_test))


def to_cv2_img(img):
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    unnormalize = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    img = (unnormalize(img).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    return np.ascontiguousarray(img[:, :, (2, 1, 0)])


def main():
    imsize = 224
    scale_factor = 8
    widths = [imsize * i for i in [1.0, 0.75, 0.5, 0.375, 0.25]]

    anchor_grid = get_anchor_grid(int(imsize / scale_factor), int(imsize / scale_factor), scale_factor=scale_factor, anchor_widths=widths, aspect_ratios=[1.0, 1.5, 2.0, 3.0])

    train_dataloader = get_dataloader('./dataset_mmp/train/', imsize, 8, 1, anchor_grid, False)
    test_dataloader = get_dataloader('./dataset_mmp/test/', imsize, 8, 1, anchor_grid, True)

    for batch_nr, (imgs, label_grids, img_ids) in enumerate(train_dataloader):
        if batch_nr >= 12: 
            break
        img, grid, id = to_cv2_img(imgs[5]), label_grids[5], img_ids[5]
        print(img.shape)
        idxs = torch.nonzero(grid, as_tuple=True)        
        for box in anchor_grid[idxs]:
            img = draw_anchor(img, np.array(box))
        cv2.imwrite(f"4_3_output/batch_{batch_nr}.jpg", img)


if __name__ == '__main__':
    main()  
