from typing import List, Tuple
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


def batch_inference(
    model: MmpNet, images: torch.Tensor, device: torch.device, anchor_grid: np.ndarray, threshold: float = 0.3
) -> List[List[Tuple[AnnotationRect, float]]]:
    images = images.to(device)
    model.eval()
    with torch.no_grad():
        output = model(images)
        output = output.cpu().numpy()
        batch_boxes_scores = []
        for i in range(len(output)):
            boxes_scores = []
            for idx in np.ndindex(output[i][1].shape):
                boxes_scores.append((AnnotationRect(*anchor_grid[idx]), output[i][1][idx]))
            batch_boxes_scores.append(boxes_scores)

    batch_boxes_scores = [non_maximum_suppression(boxes_scores, threshold) for boxes_scores in batch_boxes_scores]
    return batch_boxes_scores


def evaluate(model: MmpNet, loader: DataLoader, device: torch.device, anchor_grid: np.ndarray, threshold:float = 0.3) -> float:
    """Evaluates a specified model on the whole validation dataset.

    @return: AP for the validation set as a float.

    You decide which arguments this function should receive
    """
    det_boxes_scores = {}
    loop = tqdm(enumerate(loader), total=len(loader), leave=False)
    for b_nr, (input, target, img_id) in loop:
        detected = batch_inference(model, input, device, anchor_grid, threshold)    
        det_boxes_scores.update({img_id[i].item(): detected[i] for i in range(len(img_id))})

    ap, _, _ = calculate_ap_pr(det_boxes_scores, loader.dataset.annotations)
    return ap
            

def evaluate_test(model: MmpNet, loader: DataLoader, device: torch.device, anchor_grid: np.ndarray, out_file: str, threshold:float = 0.3): 
    """Generates predictions on the provided test dataset.
    This function saves the predictions to a text file.
    
    You decide which arguments this function should receive
    """
    det_boxes_scores = {}
    with torch.no_grad():   
        loop = tqdm(enumerate(loader), total=len(loader), leave=False)
        for b_nr, (input, target, img_id) in loop:
            detected = batch_inference(model, input, device, anchor_grid, threshold)    
            det_boxes_scores.update({f'{img_id[i]:08}': detected[i] for i in range(len(img_id))})

    with open(out_file, 'w') as f:
        for img_id, boxes_scores in det_boxes_scores.items():
            for box, score in boxes_scores:
                f.write(f"{img_id} {box.x} {box.y} {box.w} {box.h} {score}\n")


def step(
    model: MmpNet,
    criterion,
    optimizer: optim.Optimizer,
    img_batch: torch.Tensor,
    lbl_batch: torch.Tensor,
    mining_enabled: bool = False,
) -> float:
    """Performs one update step for the model

    @return: The loss for the specified batch. Return a float and not a PyTorch tensor
    """ 
    preds = model(img_batch)
    
    if mining_enabled:
        unfiltered_loss = criterion(preds, lbl_batch)
        mask = get_random_sampling_mask(lbl_batch, 2.0)
        assert unfiltered_loss.shape == mask.shape
        loss = torch.mean(unfiltered_loss * mask)
    else:
        loss = criterion(preds, lbl_batch)
        
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
        neg_samples = torch.randperm(len(neg_indices[0]))[:num_neg_samples]
        mask[i][pos_indices] = 1
        # index with tuples of permuted indices
        mask[i][tuple(idx[neg_samples] for idx in neg_indices)] = 1
        assert torch.sum(mask[i]) == num_pos_indices + num_neg_samples

    return mask


def train_epoch(
    model: MmpNet,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    mining_enabled: bool = False,
    device=torch.device('cpu'),
):
    model.train()
    loop = tqdm(enumerate(loader), total=len(loader), leave=False)
    total_loss = 0
    total_batches = 0

    for b_nr, (input, label_grid, img_id) in loop:
        input, target = input.to(device), label_grid.to(device)

        loss = step(model, criterion, optimizer, input, target, mining_enabled)
        total_loss += loss
        total_batches += 1

        loop.set_description(f"Avg Loss: {total_loss / total_batches:.4f}")

    return total_loss / total_batches


def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LR = 1e-2
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    BATCH_SIZE = 16
    NUM_WORKERS = 8
    EPOCHS = 100
    
    IMSIZE = 224
    SCALE_FACTOR = 8
    WIDTHS = [IMSIZE * i for i in [1.0, 0.75, 0.5, 0.375, 0.25]]
    ASPECT_RATIOS = [1.0, 1.5, 2.0, 3.0]
    MINING_ENABLED = False

    RUN_ROOT_DIR = './runs'
    run_dir = f'{RUN_ROOT_DIR}/upconv_sf_{SCALE_FACTOR}_lr_{LR}_bs_{BATCH_SIZE}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    anchor_grid = get_anchor_grid(int(IMSIZE / SCALE_FACTOR), int(IMSIZE / SCALE_FACTOR), scale_factor=SCALE_FACTOR, anchor_widths=WIDTHS, aspect_ratios=ASPECT_RATIOS)

    train_dataloader = get_dataloader('./dataset_mmp/train/', IMSIZE, BATCH_SIZE, NUM_WORKERS, anchor_grid, is_test=False)
    val_dataloader = get_dataloader('./dataset_mmp/val/', IMSIZE, BATCH_SIZE, NUM_WORKERS, anchor_grid, is_test=False)
    test_dataloader = get_dataloader('./dataset_mmp/test/', IMSIZE, BATCH_SIZE, NUM_WORKERS, anchor_grid, is_test=True)
    model = MmpNet(len(WIDTHS), len(ASPECT_RATIOS), IMSIZE, SCALE_FACTOR).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss() if not MINING_ENABLED else torch.nn.CrossEntropyLoss(reduction='none')

    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    writer = SummaryWriter(log_dir=run_dir)

    best_ap = 0
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer, mining_enabled=MINING_ENABLED, device=DEVICE)
        writer.add_scalar('Training/Loss', train_loss, global_step=epoch)
        if epoch % 5 == 0:
            ap = evaluate(model, val_dataloader, DEVICE, anchor_grid)
            writer.add_scalar('Validation/mAP', ap, global_step=epoch)
            print(f"Epoch {epoch} - Training Loss: {train_loss:.4f} - Validation mAP: {ap:.4f}")
            if ap > best_ap:
                best_ap = ap
                torch.save(model.state_dict(), f'{run_dir}/best_model.pth')
                print(f"New best model saved with mAP {best_ap:.4f}")
        else:
            print(f"Epoch {epoch} - Training Loss: {train_loss:.4f}")

    model.load_state_dict(torch.load(f'{run_dir}/best_model.pth'))
    evaluate_test(model, test_dataloader, DEVICE, anchor_grid, f'{run_dir}/test_results.txt')


if __name__ == '__main__':
    main()
