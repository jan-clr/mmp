from typing import Tuple
import torch
import torch.optim as optim
from model import MmpNet
from dataset import get_anchor_grid, get_dataloader
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime


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
    device='cpu',
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


def eval_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device='cpu') -> Tuple[float, float]:
    """    
    @param model: The model that should be evaluated
    @param loader: The DataLoader that contains the evaluation data

    @return: Returns the IoU of the target class and the loss over the full validation dataset as a float."""
    model.eval()
    intersect = 0
    union = 0
    total_loss = 0
    total_batches = 0
    with torch.no_grad():   
        loop = tqdm(enumerate(loader), total=len(loader), leave=False)
        for b_nr, (input, target, img_id) in loop:
            input, target = input.to(device), target.to(device)
            preds = model(input)
            loss = criterion(preds, target)
            pred_class = torch.argmax(preds, dim=1)

            total_loss += loss.item()
            total_batches += 1

            intersect += torch.sum(torch.logical_and(pred_class, target))
            union += torch.sum(torch.logical_or(pred_class, target))
            #print(intersect, union)

    return float(intersect / union), float(total_loss / total_batches)


def main():
    """Put your training code for exercises 5.2 and 5.3 here"""
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LR = 1e-2
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    BATCH_SIZE = 16
    NUM_WORKERS = 8
    EPOCHS = 50
    
    IMSIZE = 224
    SCALE_FACTOR = 8
    WIDTHS = [IMSIZE * i for i in [1.0, 0.75, 0.5, 0.375, 0.25]]
    ASPECT_RATIOS = [1.0, 1.5, 2.0, 3.0]
    MINING_ENABLED = False

    anchor_grid = get_anchor_grid(int(IMSIZE / SCALE_FACTOR), int(IMSIZE / SCALE_FACTOR), scale_factor=SCALE_FACTOR, anchor_widths=WIDTHS, aspect_ratios=ASPECT_RATIOS)

    train_dataloader = get_dataloader('./dataset_mmp/train/', IMSIZE, BATCH_SIZE, NUM_WORKERS, anchor_grid, is_test=False)
    val_dataloader = get_dataloader('./dataset_mmp/val/', IMSIZE, BATCH_SIZE, NUM_WORKERS, anchor_grid, is_test=False)
    model = MmpNet(len(WIDTHS), len(ASPECT_RATIOS), IMSIZE, SCALE_FACTOR).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss() if not MINING_ENABLED else torch.nn.CrossEntropyLoss(reduction='none')
    val_criterion = torch.nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    writer = SummaryWriter(log_dir=f'runs/negmining_{MINING_ENABLED}_lr_{LR}_bs_{BATCH_SIZE}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer, mining_enabled=MINING_ENABLED, device=DEVICE)
        writer.add_scalar('Training/Loss', train_loss, global_step=epoch)
        acc, val_loss = eval_epoch(model=model, loader=val_dataloader, criterion=val_criterion, device=DEVICE)
        print(f"Epoch {epoch} - Training Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f} - Validation IoU: {acc:.4f}")
        writer.add_scalar(tag='Validation/IoU', scalar_value=acc, global_step=epoch)
        writer.add_scalar(tag='Validation/Loss', scalar_value=val_loss, global_step=epoch)


if __name__ == "__main__":
    main()
