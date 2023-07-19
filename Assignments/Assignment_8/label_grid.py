from typing import Sequence
import numpy as np
from annotation import AnnotationRect, read_groundtruth_file
from anchor_grid import get_anchor_grid
import cv2
import torch


def iou(rect1: AnnotationRect, rect2: AnnotationRect) -> float:
    width = min(rect1.x2, rect2.x2) - max(rect1.x1, rect2.x1)
    height = min(rect1.y2, rect2.y2) - max(rect1.y1, rect2.y1)

    if width <= 0 or height <= 0:
        return 0.0
    else:
        area_inter = width * height
        area_union = rect1.area() + rect2.area() - area_inter
        return area_inter / area_union


def get_label_grid(
    anchor_grid: np.ndarray, gts: Sequence[AnnotationRect], min_iou: float
) -> np.ndarray:
    num_sizes, num_ratios, num_rows, num_cols, _ = anchor_grid.shape
    label_grid = np.ones((num_sizes, num_ratios, num_rows, num_cols))
    for size_idx in range(num_sizes):
        for ratio_idx in range(num_ratios):
            for row_idx in range(num_rows):
                for col_idx in range(num_cols):
                    anchor = anchor_grid[size_idx, ratio_idx, row_idx, col_idx, :]
                    anchor_rect = AnnotationRect.fromarray(anchor)
                    max_iou = 0.0
                    for gt in gts:
                        max_iou = max(max_iou, iou(gt, anchor_rect))
                    if max_iou < min_iou:
                        label_grid[size_idx, ratio_idx, row_idx, col_idx] = 0

    return label_grid


def draw_anchor(img, anchor, color=(0, 0, 255)):
    cv2.rectangle(img, (anchor[0], anchor[1]), (anchor[2], anchor[3]), color, 1)
    return img


def main():
    # Anchor grid parameters
    IMSIZE = 224
    SCALE_FACTOR = 32
    WIDTHS = [IMSIZE * i for i in [0.8, 0.65, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15]]
    ASPECT_RATIOS = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]

    anchor_grid = get_anchor_grid(int(IMSIZE / SCALE_FACTOR), int(IMSIZE / SCALE_FACTOR), scale_factor=SCALE_FACTOR, anchor_widths=WIDTHS, aspect_ratios=ASPECT_RATIOS)

    gts = [AnnotationRect.fromarray(np.array([ 45,  43, 153, 217]))]
    # aspec ratio of gts
    print((217 - 43) / (153 - 45))

    label_grid = get_label_grid(anchor_grid, gts, 0.7)
    print(torch.sum(torch.tensor(label_grid)))


if __name__ == '__main__':
    main()
