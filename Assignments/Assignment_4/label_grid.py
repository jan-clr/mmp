from pathlib import Path
from typing import Sequence
import numpy as np
from PIL import Image, ImageDraw
import math
import matplotlib.pyplot as plt

from solutions.a3.annotation import AnnotationRect, read_groundtruth_file
from solutions.a4.anchor_grid import get_anchor_grid


def iou(rect1: AnnotationRect, rect2: AnnotationRect) -> float:
    inter_width = min(rect1.x2, rect2.x2) - max(rect1.x1, rect2.x1)
    inter_height = min(rect1.y2, rect2.y2) - max(rect1.y1, rect2.y1)

    if inter_width <= 0 or inter_height <= 0:
        intersection = 0
    else:
        intersection = inter_width * inter_height

    union = rect1.area() + rect2.area() - intersection

    return intersection / union


def get_label_grid(
    anchor_grid: np.ndarray, gts: Sequence[AnnotationRect], min_iou: float
) -> np.ndarray:
    num_scales, num_ratios, num_rows, num_cols, c = anchor_grid.shape
    anchor_iou = np.zeros(shape=(num_scales, num_ratios, num_rows, num_cols))
    if len(gts) == 0:
        return anchor_iou >= min_iou

    for index in np.ndindex(num_scales, num_ratios, num_rows, num_cols):
        box = AnnotationRect.fromarray(anchor_grid[index])
        anchor_iou[index] = max((iou(box, g) for g in gts))

    return anchor_iou >= min_iou


def _draw_boxes():
    """Not part of the template"""
    folderpath = Path(__file__).parent.absolute()
    scales = [75.0, 150.0]
    ratios = [1.0, 2.0]
    scale_factor = 8.0

    img = Image.open(folderpath / "00542033.jpg")
    rectangles = read_groundtruth_file(folderpath / "00542033.gt_data.txt")

    draw = ImageDraw.Draw(img)

    w, h = img.size

    # ceil the rows and cols just in case there are some very small areas with detected humans on the right edge
    anchors = get_anchor_grid(
        num_rows=math.ceil(h / scale_factor),
        num_cols=math.ceil(w / scale_factor),
        scale_factor=scale_factor,
        scales=scales,
        aspect_ratios=ratios,
    )
    label_grid = get_label_grid(anchors, rectangles, min_iou=0.7)

    # TODO: plot anchor overlaps as an exercise? Would be nice to know how it looks like
    plt.imshow(label_grid[0, 0, :, :])
    plt.show()

    height, width, num_scales, num_ratios, c = anchors.shape

    # we could probably use .nonzero or .where instead
    for col in range(width):
        for row in range(height):
            for scale in range(num_scales):
                for ratio in range(num_ratios):
                    if label_grid[row, col, scale, ratio]:
                        draw_coords = anchors[row, col, scale, ratio]
                        draw.rectangle(
                            xy=draw_coords, fill=None, width=1, outline="red"
                        )

    img.save("output.jpg")


if __name__ == "__main__":
    _draw_boxes()
