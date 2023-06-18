from typing import List

from annotation import AnnotationRect, read_groundtruth_file, draw_bounding_boxes

import numpy as np
from PIL import Image
import cv2
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as F
from torchvision import datapoints 


def to_cv2_img(img):
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    unnormalize = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    img = (unnormalize(img).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    return np.ascontiguousarray(img[:, :, (2, 1, 0)])


def get_train_transforms(gaussian_blur=False, solarize=False, horizontal_flip=False, crop=False, normalize=False):
    transforms = []
    if gaussian_blur:
        transforms.append(T.GaussianBlur(3, sigma=(0.1, 2.0)))
    if solarize:
        transforms.append(T.RandomSolarize(0.7, 0.5))
    if crop:
        transforms.append(T.RandomIoUCrop(min_aspect_ratio=0.5, max_aspect_ratio=2.0, sampler_options=[0.1, 0.3, 0.5]))
    if horizontal_flip:
        transforms.append(T.RandomHorizontalFlip(0.5))
    if normalize:
        transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    transforms.append(T.ClampBoundingBox())
    transforms.append(T.SanitizeBoundingBox(labels_getter=None))
    return T.Compose(transforms)


def wrap_sample(img, boxes):
    size = img.size()
    return img, {'boxes': datapoints.BoundingBox(np.array([np.array(box) for box in boxes]), format=datapoints.BoundingBoxFormat.XYXY, spatial_size=size[-2:])}


def unwrap_sample(sample):
    return sample[0], [AnnotationRect.fromarray(box) for box in sample[1]['boxes']]


def main():
    size = 224
    """Read an image and try the augmentation pipeline. Then draw the bounding boxes on the image."""
    img_id = '00015219'
    img_file = f'dataset_mmp/train/{img_id}.jpg'
    gt_file = f'dataset_mmp/train/{img_id}.gt_data.txt'
    img = Image.open(img_file)
    img = F.to_tensor(img)
    transform = get_train_transforms(solarize=True, normalize=True)
    sample = wrap_sample(img, read_groundtruth_file(gt_file))
    sample = transform(sample)
    img, boxes = unwrap_sample(sample)
    img = to_cv2_img(img)
    draw_bounding_boxes(img, boxes)
    print(np.array(boxes[0]))
    cv2.imwrite(f'{img_id}.jpg', img)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
if __name__ == '__main__':
    main()
