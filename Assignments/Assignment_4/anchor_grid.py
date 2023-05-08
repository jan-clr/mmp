from typing import Sequence
import numpy as np


def get_anchor_grid(
    num_rows: int,
    num_cols: int,
    scale_factor: float,
    scales: Sequence[float],
    aspect_ratios: Sequence[float],
) -> np.ndarray:
    raise NotImplementedError()
