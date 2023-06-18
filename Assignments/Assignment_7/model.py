import torch
import torch.nn as nn
from torch.nn.functional import interpolate
import torchvision.models as m
import torch.nn.functional as F


class MmpNet(torch.nn.Module):
    def __init__(self, num_sizes: int, num_aspect_ratios: int, imsize: int=224, scale_factor: int=8):
        super(MmpNet, self).__init__()
        self.num_sizes = num_sizes
        self.num_aspect_ratios = num_aspect_ratios
        self.imsize = imsize
        self.scale_factor = scale_factor

        self.backbone = m.mobilenet_v2(weights=m.MobileNet_V2_Weights.IMAGENET1K_V2).features
        self.classifier = nn.Sequential(
                # Deconvolution that upsamples the feature map to 28 from 7
                #nn.ConvTranspose2d(in_channels=1280, out_channels=256, kernel_size=4, stride=4),
                #nn.Conv2d(in_channels=1280, out_channels=256, kernel_size=1),
                # Final output channels = num_classes * num_sizes * num_aspect_ratios * (imsize / scale_factor)
                nn.Conv2d(in_channels=1280, out_channels=(2 * self.num_sizes * self.num_aspect_ratios), kernel_size=1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
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

