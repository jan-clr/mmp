from typing import List, Tuple
import numpy as np
import os
from glob import glob
import cv2
from pathlib import Path


class AnnotationRect:
    """Exercise 3.1"""

    def __init__(self, x1, y1, x2, y2):
        self.x1 = int(min(x1, x2))
        self.y1 = int(min(y1, y2))
        self.x2 = int(max(x1, x2))
        self.y2 = int(max(y1, y2))

    def area(self):
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def __array__(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2])

    @staticmethod
    def fromarray(arr: np.ndarray):
        return AnnotationRect(arr[0], arr[1], arr[2], arr[3])


def read_groundtruth_file(path: str) -> List[AnnotationRect]:
    """Exercise 3.1b"""
    if not os.path.isfile(path):
        print(f"ERROR: Path '{path}' is not a file.")
        return []

    rects = []
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            data = line.split(' ')
            rects.append(AnnotationRect(int(data[0]), int(data[1]), int(data[2]), int(data[3])))
        
    return rects


# put your solution for exercise 3.1c wherever you deem it right
def get_maximum_annotation_img(train_dir: str) -> Tuple[str, List[AnnotationRect]]:
    annotation_files = glob(os.path.join(train_dir, '*.gt_data.txt'))
    max_annotations = []
    max_ann_file = ''
    for file in annotation_files:
        boxes = read_groundtruth_file(file)
        if len(boxes) > len(max_annotations):
            max_annotations = boxes
            max_ann_file = file
    return max_ann_file.replace('gt_data.txt', 'jpg'), max_annotations


def draw_bounding_boxes(img: np.ndarray, boxes: List[AnnotationRect]):
    for box in boxes:
        box = np.array(box)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)

    return img


def visualize_max_annotations(img_path: str, boxes: List[AnnotationRect], out_path: str):
    img = cv2.imread(img_path)
    img = draw_bounding_boxes(img, boxes)
    cv2.imwrite(out_path, img)


def main():
    max_file, boxes = get_maximum_annotation_img(os.path.join('dataset_mmp', 'train'))
    print(max_file, len(boxes))
    max_path = Path(max_file)
    visualize_max_annotations(max_file, boxes, max_path.stem + '_annotated' + max_path.suffix)


if __name__ == "__main__":
    main()

