import torch
import numpy as np
from annotation import AnnotationRect


def iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """
    Calculates the IoU of two boxes (x1, y1, x2, y2)
    """
    width = min(box1[2], box2[2]) - max(box1[0], box2[0])
    height = min(box1[3], box2[3]) - max(box1[1], box2[1])

    if width <= 0 or height <= 0:
        return 0.0
    else:
        area_inter = width * height
        area_union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - area_inter
        return area_inter / area_union


def match_boxes_for_bbr(model_output: torch.Tensor, labels: torch.Tensor, anchor_grid: np.ndarray, min_iou=0.5):
    """
    For each box in the model output, find the box with the highest iou with it in the labels
    and match them together for calculating the bbr loss.
    If the iou is lower than min_iou, the box is ignored.
    @param model_output: Output of the model
    @param labels: Labels for the model output
    @param anchor_grid: Anchor grid
    """


def get_bbr_loss(
    anchor_boxes: torch.Tensor,
    adjustments: torch.Tensor,
    groundtruths: torch.Tensor,
):
    """
    @param anchor_boxes: Batch of box coordinates from the anchor grid
    @param adjustments: Batch of adjustments of the prediction (#data, 4)
    @param groundtruths: Batch of ground truth data given as (x1, y1, x2, y2) (#data, 4)
    """
    widths = anchor_boxes[:, 2] - anchor_boxes[:, 0]
    heights = anchor_boxes[:, 3] - anchor_boxes[:, 1]
    divby = torch.stack([widths, heights, widths, heights], dim=1)
    subtract = torch.stack([anchor_boxes[:, 0], anchor_boxes[:, 1], anchor_boxes[:, 0], anchor_boxes[:, 1]], dim=1)
    ideal_adjustments = (groundtruths - subtract) / divby
    loss = torch.nn.MSELoss()
    return loss(adjustments, ideal_adjustments)


def apply_bbr(anchor_box: np.ndarray, adjustment: torch.Tensor) -> AnnotationRect:
    """Calculates an AnnotationRect based on a given anchor box and adjustments

    @param anchor_box: Single box coordinates from the anchor grid
    @param adjustment: Adjustments, generated by the model
    """
    x1, y1, x2, y2 = anchor_box.astype(float)
    width = x2 - x1
    height = y2 - y1
    offset_x, offset_y, scale_w, scale_h = adjustment

    x1_new = x1 + offset_x * width
    y1_new = y1 + offset_y * height

    return AnnotationRect(x1_new, y1_new, x1_new + width * scale_w, y1_new + height * scale_h)


def apply_bbr_batch(anchor_boxes: torch.Tensor, adjustments: torch.Tensor) -> torch.Tensor:
    """Calculates a batch of adjusted boxes based on a given batch of anchor boxes and adjustments

    @param anchor_boxes: Batch of box coordinates from the anchor grid
    @param adjustments: Batch of adjustments, generated by the model
    """
    widths = anchor_boxes[:, 2] - anchor_boxes[:, 0]
    heights = anchor_boxes[:, 3] - anchor_boxes[:, 1]
    multby = torch.stack([widths, heights, widths, heights], dim=1)
    subtract = torch.stack([anchor_boxes[:, 0], anchor_boxes[:, 1], anchor_boxes[:, 0], anchor_boxes[:, 1]], dim=1)
    return (anchor_boxes * multby) + subtract


def main():
    anchor_boxes = torch.tensor([[0, 0, 20, 10], [0, 0, 10, 20]])
    adjustments = torch.tensor([[0.05, 0.1, .15, .2], [0.1, 0.05, .2, .15]])
    groundtruths = torch.tensor([[1, 1, 3, 2], [1, 1, 2, 3]])
    for i in range(anchor_boxes.shape[0]):
        print(np.array(apply_bbr(np.array(anchor_boxes[i]), adjustments[i])))
    print(get_bbr_loss(anchor_boxes, adjustments, groundtruths))


if __name__ == '__main__':
    main()