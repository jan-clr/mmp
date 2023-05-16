import torch
import torch.nn as nn
from torch.nn.functional import interpolate
import torchvision.models as m
import torch.nn.functional as F


class MmpNet(torch.nn.Module):
    def __init__(self, num_sizes: int, num_aspect_ratios: int):
        super(MmpNet, self).__init__()
        self.num_sizes = num_sizes
        self.num_aspect_ratios = num_aspect_ratios

        self.backbone = m.mobilenet_v2(weights=m.MobileNet_V2_Weights.IMAGENET1K_V2).features
        self.classifier = nn.Sequential(
                # Final output channels = num_classes * num_sizes * num_aspect_ratios * (imsize / scale_factor)
                nn.Conv2d(in_channels=1280, out_channels=(2 * self.num_sizes * self.num_aspect_ratios), kernel_size=3, padding=1),
                nn.ReLU6()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        # upsample to match label grid dimensions
        features = F.interpolate(features, size=28)
        output = self.classifier(features)
        bs, out_chan, h, w = output.shape
        out_shape = (bs, 2, self.num_sizes, self.num_aspect_ratios, h, w)
        return torch.reshape(output, out_shape)


def main():
    model = MmpNet(6, 6)
    input = torch.randn(1, 3, 224, 224)
    output = model(input)
    print(output.shape)


if __name__ == '__main__':
    main()
