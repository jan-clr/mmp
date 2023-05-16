from typing import Sequence
import numpy as np
from annotation import AnnotationRect, read_groundtruth_file
from anchor_grid import get_anchor_grid
import cv2


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
    img = cv2.imread('./dataset_mmp/train/00542033.jpg')
    annotations = read_groundtruth_file('./dataset_mmp/train/00542033.gt_data.txt')
    scale_factor = 8.0
    h, w, c = img.shape
    print(h, w, c)
    widths = [w * i for i in [0.5, 0.375, 0.25]]
    print(widths)
    anchor_grid = get_anchor_grid(int(h / scale_factor), int(w / scale_factor), scale_factor, widths, [1.5, 2.0, 3.0])
    num_sizes, num_ratios, num_rows, num_cols, _ = anchor_grid.shape
    for idx in np.ndindex(num_sizes, num_ratios, num_rows, num_cols):
        anchor = anchor_grid[idx]
        anchor_rect = AnnotationRect.fromarray(anchor)
        max_iou = 0.0
        #print(f"Max Iou for {anchor}:")
        for gt in annotations:
            max_iou = max(max_iou, iou(gt, anchor_rect))
            #print(f"\t{np.array(gt)}:\t{max_iou}")
        if max_iou >= 0.7:
            draw_anchor(img, anchor)

    for gt in annotations:
        draw_anchor(img, np.array(gt), color=(0, 255, 0))

    cv2.imwrite('output.jpg', img) 


if __name__ == '__main__':
    main()
