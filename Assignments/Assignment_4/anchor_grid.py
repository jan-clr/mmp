from typing import Sequence
import numpy as np


def get_anchor_grid(
    num_rows: int,
    num_cols: int,
    scale_factor: float,
    anchor_widths: Sequence[float],
    aspect_ratios: Sequence[float],
) -> np.ndarray:
    grid = np.zeros((len(anchor_widths), len(aspect_ratios), num_rows, num_cols, 4))
    #with np.nditer(grid, flags=['multi_index'], op_flags=['readwrite']) as it:
    #    for coord in it:
    #        width_idx, scale_idx, row, col, i = it.multi_index
    #        coord = (row + 0.5) * scale_factor - anchor_widths[width_idx] * 
    for width_idx, width in enumerate(anchor_widths):
        for aspect_idx, aspect in enumerate(aspect_ratios):
            for row in range(num_rows):
                for col in range(num_cols):
                    grid[width_idx, aspect_idx, row, col, :] = [
                                (col + 0.5) * scale_factor - 0.5 * width,
                                (row + 0.5) * scale_factor - 0.5 * width * aspect,
                                (col + 0.5) * scale_factor + 0.5 * width,
                                (row + 0.5) * scale_factor + 0.5 * width * aspect,
                            ]

    return grid


def main():
    grid = get_anchor_grid(4, 4, 8, [2, 4, 8], [1.0, 0.5])
    print(grid.shape)
    print(grid)


if __name__ == '__main__':
    main()
