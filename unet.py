# -*- coding: utf-8 -*-
#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD-3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
'''
This is a PyTorch implementation of the CVPR 2020 paper:
"Deep Local Parametric Filters for Image Enhancement": https://arxiv.org/abs/2003.13985

Please cite the paper if you use this code

Tested with Pytorch 1.7.1, Python 3.7.9

Authors: Sean Moran (sean.j.moran@gmail.com), 
         Pierre Marza (pierre.marza@gmail.com)

'''
import torch
import torch.nn as nn

import fixes


def _pad_to_match(x, skip):
    """Pad an upsampled decoder map so it can be concatenated with its skip connection.

    Odd input sizes lose a row/column at each max-pool, so after nearest
    upsampling ``x`` can be one pixel shorter than ``skip`` in height and/or
    width. Pads a single zero column on the left and/or a single zero row at
    the bottom (only the combinations the original code handled).
    """
    if x.shape[3] != skip.shape[3] and x.shape[2] != skip.shape[2]:
        x = torch.nn.functional.pad(x, (1, 0, 0, 1))
    elif x.shape[2] != skip.shape[2]:
        x = torch.nn.functional.pad(x, (0, 0, 0, 1))
    elif x.shape[3] != skip.shape[3]:
        x = torch.nn.functional.pad(x, (1, 0, 0, 0))
    return x


class UNet(nn.Module):
    """U-Net backbone used to extract per-pixel features from the input image.

    A standard encoder-decoder U-Net with skip connections built from
    :class:`LocalNet` double-convolution blocks. The decoder pads feature maps
    as needed so skip connections align for arbitrary input sizes, and a global
    residual connection adds the input image back to the output.
    """

    def __init__(self):
        """Build the U-Net encoder-decoder layers.

        :returns: N/A
        :rtype: N/A

        """
        super().__init__()

        self.conv1 = nn.Conv2d(16, 64, 1)
        self.conv2 = nn.Conv2d(32, 64, 1)
        self.conv3 = nn.Conv2d(64, 64, 1)

        self.local_net = LocalNet(16)

        self.dconv_down1 = LocalNet(3, 16)
        self.dconv_down2 = LocalNet(16, 32)
        self.dconv_down3 = LocalNet(32, 64)
        self.dconv_down4 = LocalNet(64, 128)
        self.dconv_down5 = LocalNet(128, 128)

        self.maxpool = nn.MaxPool2d(2, padding=0)

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.up_conv1x1_1 = nn.Conv2d(128, 128, 1)
        self.up_conv1x1_2 = nn.Conv2d(128, 128, 1)
        self.up_conv1x1_3 = nn.Conv2d(64, 64, 1)
        self.up_conv1x1_4 = nn.Conv2d(32, 32, 1)

        self.dconv_up4 = LocalNet(256, 128)
        self.dconv_up3 = LocalNet(192, 64)
        self.dconv_up2 = LocalNet(96, 32)
        self.dconv_up1 = LocalNet(48, 16)

        self.conv_last = LocalNet(16, 3)

    def forward(self, x):
        """UNet implementation

        :param x: image
        :returns: predicted image
        :rtype: Tensor

        """
        x_in_tile = x.clone()

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)

        x = self.dconv_down5(x)

        x = self.up_conv1x1_1(self.upsample(x))
        
        x = _pad_to_match(x, conv4)
        
        x = torch.cat([x, conv4], dim=1)

        x = self.dconv_up4(x)
        x = self.up_conv1x1_2(self.upsample(x))

        x = _pad_to_match(x, conv3)

        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.up_conv1x1_3(self.upsample(x))

        x = _pad_to_match(x, conv2)

        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.up_conv1x1_4(self.upsample(x))

        x = _pad_to_match(x, conv1)

        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)
        out = out + x_in_tile


        # v2: multi-scale features for the filter heads.
        #
        # conv1/conv2/conv3 are 1x1 projections of the three finest encoder
        # scales (16, 32 and 64 channels) onto the 64 channels the heads
        # consume. They have always been declared in __init__ but were never
        # called, so in v1 they sat at their initialisation in every released
        # checkpoint. Wiring them gives the heads access to encoder features
        # directly, instead of only to the 3-channel image the decoder
        # collapses to. See docs/RESEARCH_DIRECTIONS.md, observation 1.
        # Only compute this when it will be used: under --fixes=none the result
        # is discarded, and three convolutions plus two interpolations per
        # forward pass is not free.
        ms_feat = None
        if fixes.enabled('wiring'):
            ms_feat = self.conv1(conv1)
            for proj, skip in ((self.conv2, conv2), (self.conv3, conv3)):
                ms_feat = ms_feat + torch.nn.functional.interpolate(
                    proj(skip), size=ms_feat.shape[2:], mode='bilinear',
                    align_corners=False)

        del conv1, conv2, conv3

        return out, ms_feat


class LocalNet(nn.Module):
    """Double 3x3 convolution block with reflection padding and LeakyReLU.

    The basic building block of the U-Net encoder and decoder: two reflection-
    padded 3x3 convolutions each followed by a LeakyReLU activation.
    """

    def forward(self, x_in):
        """Double convolutional block

        :param x_in: image features
        :returns: image features
        :rtype: Tensor

        """
        x = self.lrelu(self.conv1(self.refpad(x_in)))
        x = self.lrelu(self.conv2(self.refpad(x)))

        return x

    def __init__(self, in_channels=16, out_channels=64):
        """Double convolutional block

        :param in_channels:  number of input channels
        :param out_channels: number of output channels
        :returns: N/A
        :rtype: N/A

        """
        super(LocalNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 0, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 0, 1)
        self.lrelu = nn.LeakyReLU()
        self.refpad = nn.ReflectionPad2d(1)


# Model definition
class UNetModel(nn.Module):
    """U-Net backbone wrapper that produces a 64-channel feature map.

    Runs the :class:`UNet` and projects its 3-channel output to a 64-channel
    feature map with a final reflection-padded 3x3 convolution. This feature map
    is what the parametric filter head consumes.
    """

    def __init__(self):
        """UNet model definition

        :returns: N/A
        :rtype: N/A

        """

        super(UNetModel, self).__init__()

        self.unet = UNet()
        self.final_conv = nn.Conv2d(3, 64, 3, 1, 0, 1)
        self.refpad = nn.ReflectionPad2d(1)

        # v2: gate on the multi-scale encoder features, initialised to zero so
        # an untrained v2 model is numerically identical to v1 and the feature
        # path has to earn its contribution during training.
        self.ms_gate = nn.Parameter(torch.zeros(1))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Let v1 checkpoints load cleanly under ``strict=True``.

        ``ms_gate`` is new in v2, so checkpoints released with v1 do not carry
        it. Supplying the zero initialisation here means such a checkpoint
        loads into a v2 model and reproduces v1 numerics exactly, rather than
        failing as a missing key.
        """
        key = prefix + 'ms_gate'
        if key not in state_dict:
            state_dict[key] = self.ms_gate.detach().clone()
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, img):
        """Extract features from the input image.

        :param img: input RGB image Tensor of shape BxCxHxW
        :returns: 64-channel feature map Tensor of shape Bx64xHxW
        :rtype: Tensor

        """

        output_img, ms_feat = self.unet(img)
        x = self.final_conv(self.refpad(output_img))

        # Channels 0:3 are the backbone-enhanced image Y1, which the filter
        # heads and the final skip both depend on, so only the feature channels
        # 3:64 take the multi-scale contribution.
        if not fixes.enabled('wiring'):
            return x

        return torch.cat(
            [x[:, 0:3], x[:, 3:64] + self.ms_gate * ms_feat[:, 3:64]], dim=1)
