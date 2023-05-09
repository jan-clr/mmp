from typing import Sequence
import numpy as np
from ..Assignment_3.annotation import AnnotationRect


def iou(rect1: AnnotationRect, rect2: AnnotationRect) -> float:
    raise NotImplementedError()


def get_label_grid(
    anchor_grid: np.ndarray, gts: Sequence[AnnotationRect], min_iou: float
) -> np.ndarray:
    raise NotImplementedError()
