import torch
import torch.nn as nn
import torchvision
import math


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels=512):
        super(RegionProposalNetwork, self).__init__()
        self.scales = [128, 256, 512]
        self.aspect_ratios = [0.5, 1, 2]
        self.num_anchors = len(self.scales) * len(self.aspect_ratios)

        # 3x3 convolution - first layer or RPN
        # 9 anchor boxes at each location of the feature map being fed into
        # the RPN
        self.rpn_conv = nn.Conv2d(in_channels,
                                  in_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)

        # 1x1 classification layer for if object exists or not in region of interest (RoI)
        self.cls_layer = nn.Conv2d(in_channels,
                                   self.num_anchors,
                                   kernel_size=1,
                                   stride=1)

        # 1x1 regression layer for bounding box of RoI
        self.bbox_reg_layer = nn.Conv2d(in_channels,
                                        self.num_anchors * 4,
                                        kernel_size=1,
                                        stride=1)

