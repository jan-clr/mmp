from os.path import isfile
from typing import Sequence
import torch
import torchvision.models as m
import torchvision.transforms as tf
import os
from PIL import Image
import json


def build_batch(paths: Sequence[str], transform=None) -> torch.Tensor:
    """Exercise 1.1

    @param paths: A sequence (e.g. list) of strings, each specifying the location of an image file.
    @return: Returns one single tensor that contains every image.
    """
    if transform is None:
        transform = tf.Compose([
                tf.Resize((256, 256), interpolation=tf.InterpolationMode.BILINEAR),
                tf.CenterCrop(224, 224)
                tf.ToTensor(),
                tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    batch = []
    for path in paths:
        if not os.path.isfile(path):
            continue
        img = Image.open(path)
        if img is not None:
            batch.append(transform(img))

    batch = torch.stack(batch)
    return batch
    

def get_model() -> torch.nn.Module:
    """Exercise 1.2

    @return: Returns a neural network, initialised with pretrained weights.
    """
    model = m.resnet18(weights=m.ResNet18_Weights.IMAGENET1K_V1)
    return model

def main():
    """Exercise 1.3

    Put all your code for exercise 1.3 here.
    """
    model = get_model()
    
    paths = [os.path.join('images', img) for img in os.listdir('images')] 

    with open('imagenet_classes.json') as json_file:
        classes = json.load(json_file)

    transform_b_128 = tf.Compose([
            tf.Resize((128, 128), interpolation=tf.InterpolationMode.BILINEAR),
            tf.CenterCrop(224, 224)
            tf.ToTensor(),
            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    transform_b_512 = tf.Compose([
            tf.Resize((512, 512), interpolation=tf.InterpolationMode.BILINEAR),
            tf.CenterCrop(224, 224)
            tf.ToTensor(),
            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    transform_c = tf.Compose([
            tf.Resize((256,256), interpolation=tf.InterpolationMode.BILINEAR),
            tf.CenterCrop(224, 224)
            tf.ToTensor(),
            tf.RandomVerticalFlip(p=1.0),
            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    infer(model, paths, classes, report_suffix='_a')
    infer(model, paths, classes, transform=transform_b_128, report_suffix='_b_128')
    infer(model, paths, classes, transform=transform_b_512, report_suffix='_b_512')
    infer(model, paths, classes, transform=transform_c, report_suffix='_c')
    


def infer(model, paths, classes, transform=None, report_suffix=
          ''):
    
    input = build_batch(paths, transform)
    with torch.no_grad():
        out = model(input)
        print(out.shape)

    preds = torch.max(out, dim=1)
    report = []
    for i, path in enumerate(paths):
        report.append({
                    'image': path,
                    'class': classes[preds.indices[i]],
                    'score': preds.values[i].item()
                })

    with open(f"report{report_suffix}.json", 'w') as json_file:
        json.dump(report, json_file, indent=4)


if __name__ == "__main__":
    main()
