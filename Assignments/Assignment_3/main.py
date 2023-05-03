import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as m
from tqdm import tqdm
from dataset import get_dataloader, MMP_Dataset
from torch.utils.tensorboard.writer import SummaryWriter


class MmpNet(nn.Module):
    """Exercise 2.1"""

    def __init__(self, num_classes: int):
        super(MmpNet, self).__init__()
        self.backbone = m.mobilenet_v2(weights=m.MobileNet_V2_Weights.IMAGENET1K_V2)
        self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features=1280, out_features=num_classes) 
            )

    def forward(self, x: torch.Tensor):
        return self.backbone(x)


def get_criterion_optimizer(model: nn.Module, lr=1e-3) -> tuple[nn.Module, optim.Optimizer]:
    """Exercise 2.3a

    @param model: The model that is being trained.
    @return: Returns a tuple of the criterion and the optimizer.
    """
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(params=model.parameters(), lr=lr)

    return crit, opt


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device='cpu'
):
    """Exercise 2.3b

    @param model: The model that should be trained
    @param loader: The DataLoader that contains the training data
    @param criterion: The criterion that is used to calculate the loss for backpropagation
    @param optimizer: Executes the update step
    """
    model.train()
    loop = tqdm(enumerate(loader), total=len(loader), leave=False)

    for b_nr, (input, target) in loop:
        input, target = input.to(device), target.to(device)
        preds = model(input)

        loss = criterion(preds, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def eval_epoch(model: nn.Module, loader: DataLoader, device='cpu') -> float:
    """Exercise 2.3c

    @param model: The model that should be evaluated
    @param loader: The DataLoader that contains the evaluation data

    @return: Returns the accuracy over the full validation dataset as a float."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        loop = tqdm(enumerate(loader), total=len(loader), leave=False)
        for b_nr, (input, target) in loop:
            input, target = input.to(device), target.to(device)
            preds = model(input)
            pred_class = torch.argmax(preds, dim=1)

            total += pred_class.shape[0]
            correct += torch.sum(pred_class == target)

    return float(correct / total)


def main():
    """Put your code for Exercise 3.3 in here"""
    # Should be global, don't wanna risk tripping up testing
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LR = 1e-3
    BATCH_SIZE = 16
    NUM_WORKERS = 8
    EPOCHS = 5

    model = MmpNet(2).to(DEVICE)
    crit, opt = get_criterion_optimizer(model=model, lr=LR)
    train_loader = get_dataloader(is_train=True, path_to_data='./dataset_mmp/train', batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, image_size=256)
    val_loader = get_dataloader(is_train=False, path_to_data='./dataset_mmp/val', batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, image_size=256)
    writer = SummaryWriter(log_dir='')

    for epoch in tqdm(range(EPOCHS)):
        train_epoch(model=model, loader=train_loader, criterion=crit, optimizer=opt, device=DEVICE)
        acc = eval_epoch(model=model, loader=val_loader, device=DEVICE)
        print(f"Epoch: {epoch}, accuracy: {acc}")
 

if __name__ == "__main__":
    main()
