import torch 
from torch import nn


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




def main():
    """Put your code for Exercise 3.3 in here"""
    raise NotImplementedError()


if __name__ == "__main__":
    main()
