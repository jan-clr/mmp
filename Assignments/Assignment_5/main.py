import torch
import torch.optim as optim
from model import MmpNet


def step(
    model: MmpNet,
    criterion,
    optimizer: optim.Optimizer,
    img_batch: torch.Tensor,
    lbl_batch: torch.Tensor,
) -> float:
    """Performs one update step for the model

    @return: The loss for the specified batch. Return a float and not a PyTorch tensor
    """ 
    preds = model(img_batch)

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
    raise NotImplementedError()


def main():
    """Put your training code for exercises 5.2 and 5.3 here"""
    raise NotImplementedError()


if __name__ == "__main__":
    main()
