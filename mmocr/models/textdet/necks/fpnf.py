import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.runner import auto_fp16
from torch import nn

from mmdet.models.builder import NECKS


@NECKS.register_module()
class FPNF(nn.Module):
    """FPN-like fusion module in Shape Robust Text Detection with Progressive
    Scale Expansion Network."""

    def __init__(
            self,
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            fusion_type='concat',  # 'concat' or 'add'
            upsample_ratio=1):
        super().__init__()
        conv_cfg = None
        norm_cfg = dict(type='BN')
        act_cfg = dict(type='ReLU')

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.backbone_end_level = len(in_channels)
        for i in range(self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)

            if i < self.backbone_end_level - 1:
                fpn_conv = ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(fpn_conv)

        self.fusion_type = fusion_type

        if self.fusion_type == 'concat':
            feature_channels = 1024
        elif self.fusion_type == 'add':
            feature_channels = 256
        else:
            raise NotImplementedError

        self.output_convs = ConvModule(
            feature_channels,
            out_channels,
            3,
            padding=1,
            conv_cfg=None,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False)
        self.upsample_ratio = upsample_ratio

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # step 1: upsample to level i-1 size and add level i-1
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, mode='nearest')
            # step 2: smooth level i-1
            laterals[i - 1] = self.fpn_convs[i - 1](laterals[i - 1])

        # upsample and cont
        bottom_shape = laterals[0].shape[2:]
        for i in range(1, used_backbone_levels):
            laterals[i] = F.interpolate(
                laterals[i], size=bottom_shape, mode='nearest')

        if self.fusion_type == 'concat':
            out = torch.cat(laterals, 1)
        elif self.fusion_type == 'add':
            out = laterals[0]
            for i in range(1, used_backbone_levels):
                out += laterals[i]
        else:
            raise NotImplementedError
        out = self.output_convs(out)

        return out
