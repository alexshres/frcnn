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

        # 9 anchor boxes in this case
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

    def generate_anchors(self, image, feat):
        """
        params:
        image: dimension (NxCxHxW)
        feat : dimension (NxCxHxW)
        """
        grid_h, grid_w = feat.shape[-2:]
        img_h, img_w = image.shape[-2:]

        stride_h = torch.tensor(img_h // grid_h,
                                dtype=torch.int64,
                                device=feat.device)

        stride_w = torch.tensor(image_w // grid_w,
                                dtype=torch.int64,
                                device=feat.device
                                )

        scales = torch.as_tensor(self.scales,
                                 dtype=feat.dtype,
                                 device=feat.device
                                 )

        aspect_ratios = torch.as_tensor(self.aspect_ratios,
                                        dtype=feat.dtype,
                                        device=feat.device
                                        )
        pass

    def forward(self, image, feat, target):
        # Call RPN layers (use chaining since nn.ReLU() returns a function)
        rpn_feat = nn.ReLU()(self.rpn_conv(feat))
        cls_scores = self.cls_layer(rpn_feat)
        box_transform_pred = self.bbox_reg_layer(rpn_feat)

        # Generating the anchors
        anchors = self.generate_anchors(image, feat)
