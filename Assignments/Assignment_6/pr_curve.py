from typing import Sequence, Tuple
from numpy.typing import NDArray
import torch
import numpy as np
from model import MmpNet
from nms import non_maximum_suppression
from annotation import AnnotationRect
import torch.optim as optim
from model import MmpNet
from dataset import get_anchor_grid, get_dataloader
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
from evallib import calculate_ap_pr
from pprint import pprint
from main import batch_inference
from matplotlib import pyplot as plt


def calc_prcurve(model: MmpNet, loader: DataLoader, device: torch.device, threshold: float, anchor_grid):
    det_boxes_scores = {}
    loop = tqdm(enumerate(loader), total=len(loader), leave=False)
    for b_nr, (input, target, img_id) in loop:
        detected = batch_inference(model, input, device, anchor_grid, threshold)    
        # filter out boxes with score < 0.5
        det_boxes_scores.update({img_id[i].item(): detected[i] for i in range(len(img_id))})
    _, pr, rc = calculate_ap_pr(det_boxes_scores, loader.dataset.transformed_annotations)
    return pr, rc


def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 16
    NUM_WORKERS = 8
    
    IMSIZE = 224
    SCALE_FACTOR = 32
    WIDTHS = [IMSIZE * i for i in [0.8, 0.65, 0.5, 0.4, 0.3, 0.2]]
    ASPECT_RATIOS = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    NSM_THRESHOLD = 0.3

    RUN_ROOT_DIR = './runs'
    #run_dir = f'{RUN_ROOT_DIR}/correctannot_filter_0.5_sgd_gridv3_sf_{SCALE_FACTOR}_negr{NEGATIVE_RATIO}_nsm_{NSM_THRESHOLD}_lr_{LR}_bs_{BATCH_SIZE}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    run_dir = f'{RUN_ROOT_DIR}/best_until_now'

    anchor_grid = get_anchor_grid(int(IMSIZE / SCALE_FACTOR), int(IMSIZE / SCALE_FACTOR), scale_factor=SCALE_FACTOR, anchor_widths=WIDTHS, aspect_ratios=ASPECT_RATIOS)

    val_dataloader = get_dataloader('./dataset_mmp/val/', IMSIZE, BATCH_SIZE, NUM_WORKERS, anchor_grid, is_test=False, apply_transforms_on_init=True)
    model = MmpNet(len(WIDTHS), len(ASPECT_RATIOS), IMSIZE, SCALE_FACTOR).to(DEVICE)
    model.load_state_dict(torch.load(f'{run_dir}/best_model.pth'))
    pr, rc = calc_prcurve(model, val_dataloader, DEVICE, 0.5, anchor_grid)
    plt.plot(rc, pr)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(f'pr_curve.png')


if __name__ == '__main__':
    main()
