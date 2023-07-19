import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as m
import torchvision.datasets as ds
import torchvision.transforms as tf
from tqdm import tqdm
import json



# these are the labels from the Cifar10 dataset:
CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


class MmpNet(nn.Module):
    """Exercise 2.1"""

    def __init__(self, num_classes: int):
        super(MmpNet, self).__init__()
        self.backbone = m.mobilenet_v2(weights=m.MobileNet_V2_Weights.IMAGENET1K_V2)
        self.backbone.anchor_branch = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features=1280, out_features=num_classes) 
            )

    def forward(self, x: torch.Tensor):
        return self.backbone(x)


def get_dataloader(
    is_train: bool, data_root: str, batch_size: int, num_workers: int
) -> DataLoader:
    """Exercise 2.2

    @param is_train: Whether this is the training or validation split
    @param data_root: Where to download the dataset to
    @param batch_size: Batch size for the data loader
    @param num_workers: Number of workers for the data loader
    """
    transform = tf.Compose([
            tf.ToTensor(),
            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    dataset = ds.CIFAR10(root=data_root, train=is_train, transform=transform, download=True)
    loader = DataLoader(dataset=dataset, shuffle=is_train, batch_size=batch_size, num_workers=num_workers, drop_last=is_train)

    return loader


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


def generate_report(report, epoch, accuracy, filename):
    report.append({
                'epoch': epoch,
                'accuracy': accuracy,
            })
    with open(filename, 'w') as json_file:
        json.dump(report, json_file, indent=4)
    return report


def main():
    """Exercise 2.3d"""
    
    # Should be global, don't wanna risk tripping up testing
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LR = 1e-3
    BATCH_SIZE = 16
    NUM_WORKERS = 8
    EPOCHS = 5

    model = MmpNet(len(CLASSES)).to(DEVICE)
    crit, opt = get_criterion_optimizer(model=model, lr=LR)
    train_loader = get_dataloader(is_train=True, data_root='./data', batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    val_loader = get_dataloader(is_train=False, data_root='./data', batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    report = []

    for epoch in tqdm(range(EPOCHS)):
        train_epoch(model=model, loader=train_loader, criterion=crit, optimizer=opt, device=DEVICE)
        acc = eval_epoch(model=model, loader=val_loader, device=DEVICE)
        print(f"Epoch: {epoch}, accuracy: {acc}")
        generate_report(report, epoch + 1, acc, 'report.json')
        

if __name__ == "__main__":
    main()
