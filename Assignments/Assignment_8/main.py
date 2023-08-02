from typing import List, Tuple
import torch
import numpy as np
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
from transformations import get_train_transforms
from argparse import ArgumentParser
from bbr import get_bbr_loss, match_boxes_for_bbr, apply_bbr, apply_bbr_batch, get_adjustment_tensor_from_model_output
from time import time
from matplotlib import pyplot as plt
import torch.multiprocessing as mp

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR = 1e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
BATCH_SIZE = 32
NUM_WORKERS = 8
EPOCHS = 30

# Anchor grid parameters
IMSIZE = 320
SCALE_FACTOR = 32
WIDTHS = [IMSIZE * i for i in [0.8, 0.65, 0.5, 0.4, 0.3, 0.2]]
ASPECT_RATIOS = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
LG_MIN_IOU = 0.5

MINING_ENABLED = True
NEGATIVE_RATIO = 20.0
NMS_THRESHOLD = 0.3

RUN_ROOT_DIR = './runs'
DEBUG = False
PLOT_PR_ON_EVAL = True


def batch_inference(
    model: MmpNet, images: torch.Tensor, device: torch.device, anchor_grid: np.ndarray, nms_threshold: float = 0.3, filter_threshold=0.0, stretch_factor = 1.0
) -> List[List[Tuple[AnnotationRect, float]]]:
    images = images.to(device)
    model.eval()
    with torch.no_grad():
        output = model(images)
        anchor_output = torch.softmax(output[0], dim=1)
        bbr_output = output[1]
        anchor_output = anchor_output.cpu().numpy()
        batch_boxes_scores = []
        st = time()
        for i in range(len(anchor_output)):
            boxes_scores = []
            for idx in np.ndindex(anchor_output[i][1].shape):
                box = stretch_factor * anchor_grid[idx]
                adjustments = bbr_output[i][:, idx[0], idx[1], idx[2], idx[3]]
                box = AnnotationRect.fromarray(box)
                boxes_scores.append((box, anchor_output[i][1][idx], adjustments))
            load_time = time()
            if filter_threshold > 0.0:
                if DEBUG:
                    print(f"Filtering boxes with scores lower than {filter_threshold}.")
                boxes_scores = [(box, score, adjustment) for box, score, adjustment in boxes_scores if score > 0.5]
            boxes_scores = non_maximum_suppression(boxes_scores, nms_threshold)
            nms_time = time()
            for idx, (box, score, adjustments) in enumerate(boxes_scores):
                boxes_scores[idx] = (apply_bbr(np.array(box), adjustments), score)
            bbrt = time()
            if DEBUG:
                print(f"Load time: {load_time - st}, NMS time: {nms_time - load_time}, BBR time: {bbrt - nms_time}")
            batch_boxes_scores.append(boxes_scores)

    return batch_boxes_scores


def evaluate(model: MmpNet, loader: DataLoader, device: torch.device, anchor_grid: np.ndarray, nms_threshold:float = 0.3, filter_threshold=0.0, plot_pr=False, save_dir='.', pr_suffix='') -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluates a specified model on the whole validation dataset.

    @return: AP for the validation set as a float.

    You decide which arguments this function should receive
    """
    det_boxes_scores = {}
    loop = tqdm(enumerate(loader), total=len(loader), leave=False)
    for b_nr, (input, target, img_id, annotations) in loop:
        detected = batch_inference(model, input, device, anchor_grid, nms_threshold, filter_threshold)
        # filter out boxes with score < 0.5
        det_boxes_scores.update({img_id[i]: detected[i] for i in range(len(img_id))})
    ap, pr, rc = calculate_ap_pr(det_boxes_scores, loader.dataset.transformed_annotations)
    if plot_pr:
        fig = plt.figure(pr_suffix)
        plt.plot(rc, pr)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.savefig(f'{save_dir}/pr_curve{pr_suffix}.png')
    return ap, pr, rc
            

def evaluate_test(model: MmpNet, loader: DataLoader, device: torch.device, anchor_grid: np.ndarray, out_file: str, nms_threshold:float = 0.3, stretch_factor:float = 1.0, filter_threshold=0.5):
    """Generates predictions on the provided test dataset.
    This function saves the predictions to a text file.
    
    You decide which arguments this function should receive
    """
    det_boxes_scores = {}
    with torch.no_grad():   
        loop = tqdm(enumerate(loader), total=len(loader), leave=False)
        for b_nr, (input, target, img_id, annotations) in loop:
            detected = batch_inference(model, input, device, anchor_grid, nms_threshold, stretch_factor=stretch_factor, filter_threshold=filter_threshold)

            det_boxes_scores.update({f'{img_id[i]:08}': detected[i] for i in range(len(img_id))})

    with open(out_file, 'w') as f:
        for img_id, boxes_scores in det_boxes_scores.items():
            for box, score in boxes_scores:
                f.write(f"{img_id} {box.x1} {box.y1} {box.x2} {box.y2} {score}\n")


def step(
    model: MmpNet,
    criterion,
    optimizer: optim.Optimizer,
    img_batch: torch.Tensor,
    lbl_batch: torch.Tensor,
    annotations: List[List[AnnotationRect]] = [],
    anchor_grid: torch.Tensor = None,
    mining_enabled: bool = False,
    negative_ratio: float = 2.0,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Performs one update step for the model

    @return: The loss for the specified batch. Return a float and not a PyTorch tensor
    """ 
    preds = model(img_batch)
    
    if mining_enabled:
        unfiltered_loss = criterion(preds[0], lbl_batch)
        mask = get_random_sampling_mask(lbl_batch, negative_ratio)
        assert unfiltered_loss.shape == mask.shape
        anchor_loss = torch.mean(unfiltered_loss * mask)
    else:
        anchor_loss = criterion(preds[0], lbl_batch)

    # make bbr branch only learn from positive samples = near samples
    adjustments = get_adjustment_tensor_from_model_output(preds[1])
    boxes, adjustments, gts = match_boxes_for_bbr(adjustments, labels=annotations, label_grid=lbl_batch, anchor_grid=anchor_grid)
    bbr_loss = get_bbr_loss(boxes, adjustments, gts.to(device))

    loss = anchor_loss * bbr_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def get_random_sampling_mask(labels: torch.Tensor, neg_ratio: float) -> torch.Tensor:
    """
    @param labels: The label tensor that is returned by your data loader
    @return: A tensor with the same shape as labels
    """
    mask = torch.zeros_like(labels)
    # iterate over batch so samples are balanced for each image
    for i in range(labels.shape[0]):
        pos_indices = torch.where(labels[i] == 1)
        neg_indices = torch.where(labels[i] == 0)
        num_pos_indices = len(pos_indices[0])
        num_neg_samples = int(num_pos_indices * neg_ratio)
        if DEBUG:
            print(f"num_pos_indices: {num_pos_indices}, num_neg_samples: {num_neg_samples}")
        neg_samples = torch.randperm(len(neg_indices[0]))[:num_neg_samples]
        mask[i][pos_indices] = 1
        # index with tuples of permuted indices
        mask[i][tuple(idx[neg_samples] for idx in neg_indices)] = 1
        if DEBUG:
            print(f"num_used_samples: {torch.sum(mask[i])}")

    return mask


def train_epoch(
    model: MmpNet,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    anchor_grid: np.ndarray,
    mining_enabled: bool = False,
    device=torch.device('cpu'),
    negative_ratio: float = 2.0,
):
    model.train()
    loop = tqdm(enumerate(loader), total=len(loader), leave=False)
    total_loss = 0
    total_batches = 0
    anchor_grid = torch.from_numpy(anchor_grid).to(device)

    for b_nr, (input, label_grid, img_id, annotations) in loop:
        input, target = input.to(device), label_grid.to(device)

        loss = step(model, criterion, optimizer, input, target, annotations, anchor_grid, mining_enabled, negative_ratio, device)
        total_loss += loss
        total_batches += 1

        loop.set_description(f"Avg Loss: {total_loss / total_batches:.4f}")

    return total_loss / total_batches


def main():
    parser = ArgumentParser()
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--horizontal_flip', action='store_true')
    parser.add_argument('--solarize', action='store_true')
    parser.add_argument('--gauss_blur', action='store_true')

    args = parser.parse_args()

    run_dir = f'{RUN_ROOT_DIR}/bn_do/crop_{args.crop}_flip_{args.horizontal_flip}_solarize_{args.solarize}_gauss_{args.gauss_blur}_adam_gridv3_sf_{SCALE_FACTOR}_negr{NEGATIVE_RATIO}_nsm_{NMS_THRESHOLD}_lgminiou_{LG_MIN_IOU}_nodes_{int(len(WIDTHS) * len(ASPECT_RATIOS) * (IMSIZE / SCALE_FACTOR) ** 2)}_lr_{LR}_bs_{BATCH_SIZE}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    #run_dir = f'{RUN_ROOT_DIR}/best_until_now'
    print(run_dir)

    anchor_grid = get_anchor_grid(int(IMSIZE / SCALE_FACTOR), int(IMSIZE / SCALE_FACTOR), scale_factor=SCALE_FACTOR, anchor_widths=WIDTHS, aspect_ratios=ASPECT_RATIOS)

    transforms = get_train_transforms(horizontal_flip=args.horizontal_flip, crop=args.crop, solarize=args.solarize, gaussian_blur=args.gauss_blur)
    #transforms = None

    train_dataloader = get_dataloader('./dataset_mmp/train/', IMSIZE, BATCH_SIZE, NUM_WORKERS, anchor_grid, is_test=False, apply_transforms_on_init=True, transforms=transforms, min_iou=LG_MIN_IOU)
    val_dataloader = get_dataloader('./dataset_mmp/val/', IMSIZE, BATCH_SIZE, NUM_WORKERS, anchor_grid, is_test=False, apply_transforms_on_init=True)
    model = MmpNet(len(WIDTHS), len(ASPECT_RATIOS), IMSIZE, SCALE_FACTOR).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss() if not MINING_ENABLED else torch.nn.CrossEntropyLoss(reduction='none')

    #optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    writer = SummaryWriter(log_dir=run_dir)
    # Continue Training
    #model.load_state_dict(torch.load(f'{run_dir}/best_model.pth'))

    best_ap = 0
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer, anchor_grid, mining_enabled=MINING_ENABLED, device=DEVICE, negative_ratio=NEGATIVE_RATIO)
        #train_loss = 0
        writer.add_scalar('Training/Loss', train_loss, global_step=epoch)
        if epoch % 2 == 0:
            ap, _, _ = evaluate(model, val_dataloader, DEVICE, anchor_grid, nms_threshold=NMS_THRESHOLD, filter_threshold=0.0, plot_pr=PLOT_PR_ON_EVAL, save_dir=run_dir, pr_suffix=f'_epoch_{epoch}')
            writer.add_scalar('Validation/mAP', ap, global_step=epoch)
            print(f"Epoch {epoch} - Training Loss: {train_loss:.4f} - Validation mAP: {ap:.4f}")
            if ap > best_ap:
                best_ap = ap
                torch.save(model.state_dict(), f'{run_dir}/best_model.pth')
                print(f"New best model saved with mAP {best_ap:.4f}")
        else:
            print(f"Epoch {epoch} - Training Loss: {train_loss:.4f}")

    model.load_state_dict(torch.load(f'{run_dir}/best_model.pth'))
    ap, _, _ = evaluate(model, val_dataloader, DEVICE, anchor_grid, nms_threshold=NMS_THRESHOLD, filter_threshold=0.0, plot_pr=PLOT_PR_ON_EVAL, save_dir=run_dir, pr_suffix=f'_best')


if __name__ == '__main__':
    main()
