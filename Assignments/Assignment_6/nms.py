from typing import List, Sequence, Tuple
from annotation import AnnotationRect, draw_bounding_boxes
import numpy as np
import pandas as pd
import cv2
from label_grid import iou
import time


def non_maximum_suppression(
    boxes_scores: Sequence[Tuple[AnnotationRect, float]], threshold: float
) -> List[Tuple[AnnotationRect, float]]:
    """Exercise 6.1
    @param boxes_scores: Sequence of tuples of annotations and scores
    @param threshold: Threshold for NMS

    @return: A list of tuples of the remaining boxes after NMS together with their scores
    """
    final_boxes_scores = []
    boxes_scores = sorted(boxes_scores, key=lambda x: x[1], reverse=True)
    while len(boxes_scores) > 1:
        box1, score1 = boxes_scores[0]
        final_boxes_scores.append((box1, score1))
        boxes_scores = boxes_scores[1:]
        for i in range(len(boxes_scores) - 1, -1, -1):
            box2, score2 = boxes_scores[i]
            box_iou = iou(box1, box2)
            if box_iou > threshold:
                del boxes_scores[i]

    return final_boxes_scores


def non_maximum_suppression_old(
    boxes_scores: Sequence[Tuple[AnnotationRect, float]], threshold: float
) -> List[Tuple[AnnotationRect, float]]:
    """Exercise 6.1
    @param boxes_scores: Sequence of tuples of annotations and scores
    @param threshold: Threshold for NMS

    @return: A list of tuples of the remaining boxes after NMS together with their scores
    """
    final_boxes_scores = []
    boxes_scores = sorted(boxes_scores, key=lambda x: x[1], reverse=True)
    while len(boxes_scores) > 1:
        box1, score1 = boxes_scores[0]
        final_boxes_scores.append((box1, score1))
        boxes_scores = boxes_scores[1:]
        for box2, score2 in boxes_scores:
            box_iou = iou(box1, box2)
            if box_iou > threshold:
                boxes_scores.remove((box2, score2))

    return final_boxes_scores


def main():
    score_list = pd.read_csv('model_output.txt', header=None, sep=' ', names=['id', 'x1', 'y1', 'x2', 'y2', 'score'], dtype={'id': str, 'x1': int, 'y1': int, 'x2': int, 'y2': int, 'score': float})
    score_list_grouped = score_list.groupby('id')
    threshold = 0.3
    for id, frame in score_list_grouped:
        print(id)
        boxes_scores = []
        for index, row in frame.iterrows():
            boxes_scores.append((AnnotationRect(row['x1'], row['y1'], row['x2'], row['y2']), row['score']))
        filtered_boxes_scores = non_maximum_suppression(boxes_scores, threshold)
        print(len(filtered_boxes_scores))
        img = cv2.imread('dataset_mmp/test/' + str(id) + '.jpg')
        img = draw_bounding_boxes(img, [box for box, score in filtered_boxes_scores])
        cv2.imwrite(str(id) + '_nms.jpg', img)


if __name__ == '__main__':
    main()
