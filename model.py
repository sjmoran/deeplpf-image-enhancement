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

Code-to-paper map (section / equation numbers refer to the arXiv version):

* ``unet.UNetModel``            backbone, Sec. 3.1 (produces the H x W x C feature map;
                                its first three channels are the backbone-enhanced image Y1)
* ``CubicFilter``               polynomial ("cubic-20") filter, Sec. 3.2.4, Eq. 6 -> Y2
* ``GraduatedFilter``           graduated filter, Sec. 3.2.2, Eqs. 1-3, fused per Eq. 7 -> s_g
* ``EllipticalFilter``          elliptical filter, Sec. 3.2.3, Eq. 4, fused per Eq. 7 -> s_e
* ``DeepLPFParameterPrediction`` fusion S = s_g + s_e, Y3 = S * Y2, Y = Y3 + Y1, Sec. 3.1
* ``DeepLPFLoss``               training loss, Sec. 3.4, Eq. 8
'''
import torch
import torch.nn as nn
import unet
import fixes
from math import exp
import math
import torch.nn.functional as F
from util import ImageProcessing


_GRID_CACHE = {}


def _coord_grids(H, W, device):
    """Normalised pixel-coordinate grids shared by the three filter branches.

    ``x_axis`` varies along dim 2 (H), ``y_axis`` along dim 3 (W), both in
    [0, 1). Built exactly as the branches used to build them inline, but
    cached per (H, W, device): FiveK has 16 distinct image sizes, and building
    them per call cost six host-to-device copies per forward pass.
    """
    key = (H, W, device)
    grids = _GRID_CACHE.get(key)
    if grids is None:
        x_axis = torch.arange(H).view(-1, 1).repeat(1, W).to(device) / H
        y_axis = torch.arange(W).repeat(H, 1).to(device) / W
        grids = _GRID_CACHE[key] = (x_axis, y_axis)
    return grids


def _coord_grid_powers(H, W, device):
    """``(x, x^2, x^3, y, y^2, y^3)`` of the grids from :func:`_coord_grids`.

    The cubic filter's polynomial needs the squares and cubes of the
    coordinate grids; they depend only on the image size, so compute them
    once per size rather than on every forward pass.
    """
    key = ('pow', H, W, device)
    powers = _GRID_CACHE.get(key)
    if powers is None:
        x_axis, y_axis = _coord_grids(H, W, device)
        powers = _GRID_CACHE[key] = (
            x_axis, x_axis ** 2, x_axis ** 3, y_axis, y_axis ** 2, y_axis ** 3)
    return powers


class DeepLPFLoss(nn.Module):
    """DeepLPF training loss (paper Sec. 3.4, Eq. 8).

    ``L = w_lab * ||Lab(Y_hat) - Lab(Y)||_1 + w_msssim * (1 - MS-SSIM(L(Y_hat), L(Y)))``

    with the L1 term taken over all three CIELab channels (rescaled to [0, 1]
    by :func:`util.ImageProcessing.rgb_to_lab`) and the MS-SSIM term on the L
    (lightness) channel only. The weights are hard-coded to the values the
    paper reports for MIT-Adobe-5K-DPE: ``w_lab = 1``, ``w_msssim = 1e-3``.
    Each image in the batch is scored separately and the terms are averaged
    over the batch.
    """

    def __init__(self, ssim_window_size=5, alpha=0.5):
        """Initialisation of the DeepLPF loss function

        :param ssim_window_size: size of averaging window for SSIM
        :param alpha: unused; retained for backward compatibility of the
            constructor signature (the L1/MS-SSIM weighting is fixed in forward)
        :returns: N/A
        :rtype: N/A

        """
        super(DeepLPFLoss, self).__init__()
        self.alpha = alpha
        self.ssim_window_size = ssim_window_size
        # Per-(device, dtype) caches of the constant tensors the loss uses, so
        # they are copied to the device once rather than on every call.
        self._window_cache = {}
        self._weights_cache = {}

    def create_window(self, window_size, num_channel):
        """Window creation function for SSIM metric. Gaussian weights are applied to the window.
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

        :param window_size: size of the window to compute statistics
        :param num_channel: number of channels
        :returns: Tensor of shape Cx1xWindow_sizexWindow_size
        :rtype: Tensor

        """
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(
            _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(
            num_channel, 1, window_size, window_size).contiguous()
        return window

    def gaussian(self, window_size, sigma):
        """
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
        :param window_size: size of the SSIM sampling window e.g. 11
        :param sigma: Gaussian variance
        :returns: 1xWindow_size Tensor of Gaussian weights
        :rtype: Tensor

        """
        gauss = torch.Tensor(
            [exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def compute_ssim(self, img1, img2):
        """Computes the structural similarity index between two images. This function is differentiable.
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

        :param img1: image Tensor BxCxHxW
        :param img2: image Tensor BxCxHxW
        :returns: mean SSIM
        :rtype: float

        """
        (_, num_channel, _, _) = img1.size()
        key = (num_channel, img1.device, img1.dtype)
        window = self._window_cache.get(key)
        if window is None:
            # Move the SSIM window to the same device and dtype as the input
            window = self.create_window(self.ssim_window_size, num_channel).type_as(img1)
            self._window_cache[key] = window

        mu1 = F.conv2d(
            img1, window, padding=self.ssim_window_size // 2, groups=num_channel)
        mu2 = F.conv2d(
            img2, window, padding=self.ssim_window_size // 2, groups=num_channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(
            img1 * img1, window, padding=self.ssim_window_size // 2, groups=num_channel) - mu1_sq
        sigma2_sq = F.conv2d(
            img2 * img2, window, padding=self.ssim_window_size // 2, groups=num_channel) - mu2_sq
        sigma12 = F.conv2d(
            img1 * img2, window, padding=self.ssim_window_size // 2, groups=num_channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map1 = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
        ssim_map2 = ((mu1_sq + mu2_sq + C1) *
                     (sigma1_sq + sigma2_sq + C2))
        ssim_map = ssim_map1 / ssim_map2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)

        return ssim_map.mean(), cs


    def compute_msssim(self, img1, img2):
        """Computes the multi scale structural similarity index between two images. This function is differentiable.
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

        :param img1: image Tensor BxCxHxW
        :param img2: image Tensor BxCxHxW
        :returns: mean SSIM
        :rtype: float

        """
        if img1.shape[2]!=img2.shape[2]:
                img1=img1.transpose(2,3)

        if img1.shape != img2.shape:
            raise RuntimeError('Input images must have the same shape (%s vs. %s).',
                       img1.shape, img2.shape)
        if img1.ndim != 4:
            raise RuntimeError('Input images must have four dimensions, not %d',
                       img1.ndim)

        device = img1.device
        weights = self._weights_cache.get(device)
        if weights is None:
            weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
            self._weights_cache[device] = weights
        levels = weights.size()[0]
        ssims = []
        mcs = []
        for _ in range(levels):
            ssim, cs = self.compute_ssim(img1, img2)
            ssims.append(ssim)
            mcs.append(cs)

            img1 = F.avg_pool2d(img1, (2, 2))
            img2 = F.avg_pool2d(img2, (2, 2))

        ssims = torch.stack(ssims)
        mcs = torch.stack(mcs)

        # Map SSIM / contrast-sensitivity from [-1, 1] to [0, 1] before raising
        # to the fractional per-scale weights (avoids NaN from a negative base).
        # This "simple normalisation" departs from the original Wang et al.
        # definition and is kept because the released models were trained with it.
        ssims = (ssims + 1) / 2
        mcs = (mcs + 1) / 2

        pow1 = mcs ** weights
        pow2 = ssims ** weights

        # Combine the per-scale contrast-sensitivity terms with the finest-scale SSIM
        # (cf. the Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/).
        #
        # v2 fix: this read `torch.prod(pow1[:-1] * pow2[-1])`, which broadcasts
        # pow2[-1] into all four factors so it entered the product four times
        # rather than once. Upstream jorge-pessoa/pytorch-msssim, which the rest
        # of this function follows, groups it as below.
        #
        # Written as explicit products rather than torch.prod: prod's CUDA
        # backward counts the zeros in its input with a host sync, which is
        # illegal inside a CUDA graph capture (--cuda_graphs). Bitwise equal
        # to prod on CPU; on CUDA and MPS within 1 ulp of the MS-SSIM factor.
        if fixes.enabled('msssim'):
            output = pow1[0] * pow1[1] * pow1[2] * pow1[3] * pow2[-1]
        else:
            p = pow1[:-1] * pow2[-1]
            output = p[0] * p[1] * p[2] * p[3]
        return output

    def forward(self, predicted_img_batch, target_img_batch):
        """Forward function for the DeepLPF loss

        :param predicted_img_batch: predicted image Tensor of shape BxCxHxW
        :param target_img_batch: ground-truth image Tensor of shape BxCxHxW
        :returns: value of loss function
        :rtype: Tensor

        """
        if predicted_img_batch.shape[2]!=target_img_batch.shape[2]:
                target_img_batch=target_img_batch.transpose(2,3)

        num_images = target_img_batch.shape[0]

        # Accumulate the loss on the same device/dtype as the predictions
        ssim_loss_value = predicted_img_batch.new_zeros((1, 1))
        l1_loss_value = predicted_img_batch.new_zeros((1, 1))

        for i in range(0, num_images):

            target_img = target_img_batch[i, :, :, :]
            predicted_img = predicted_img_batch[i, :, :, :]

            # Lab(.) of Eq. 8: CIELab with every channel rescaled to [0, 1]
            predicted_img_lab = ImageProcessing.rgb_to_lab(
                predicted_img.squeeze(0))
            target_img_lab = ImageProcessing.rgb_to_lab(target_img.squeeze(0))

            # L(.) of Eq. 8: the lightness channel, as a 1x1xHxW batch for MS-SSIM
            target_img_L_ssim = target_img_lab[0, :, :].unsqueeze(0)
            predicted_img_L_ssim = predicted_img_lab[0, :, :].unsqueeze(0)
            target_img_L_ssim = target_img_L_ssim.unsqueeze(0)
            predicted_img_L_ssim = predicted_img_L_ssim.unsqueeze(0)

            ssim_value = self.compute_msssim(
                predicted_img_L_ssim, target_img_L_ssim)
            ssim_loss_value += (1.0 - ssim_value)

            l1_loss_value += F.l1_loss(predicted_img_lab, target_img_lab)

        l1_loss_value = l1_loss_value/num_images
        ssim_loss_value = ssim_loss_value/num_images
        # Eq. 8 with w_lab = 1. The paper's MIT-Adobe-5K-DPE setting is
        # w_msssim = 1e-3, at which the structural term contributes about 0.07%
        # of the L1 term's gradient - measured in
        # tests/test_loss_terms_contribute.py - so what trains the model is L1
        # in Lab space. fixes.msssim_weight() returns that published value
        # unless --msssim_weight overrides it.
        deeplpf_loss = l1_loss_value + fixes.msssim_weight()*ssim_loss_value
        return deeplpf_loss


class _SignSTE(torch.autograd.Function):
    """``sign`` with a straight-through gradient.

    ``torch.sign`` has zero gradient everywhere, so a binarisation layer built
    on it alone blocks learning. This is the estimator of Courbariaux et al.:
    the forward pass binarises, and the backward pass passes the gradient
    through unchanged except where the input has saturated (``|input| > 1``),
    where it is zeroed.

    Implemented as an ``autograd.Function`` because that is the only thing
    PyTorch calls ``backward`` on -- see :class:`BinaryLayer`.
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * (input.abs() <= 1).to(grad_output.dtype)


class BinaryLayer(nn.Module):
    """Binarisation layer with a straight-through estimator.

    Used by the graduated filter for the binarised inversion indicator
    ``g_inv_hat = (sgn(g_inv) + 1) / 2`` of paper Sec. 3.2.2.

    v2 fix: this class previously defined ``backward`` as a plain method on an
    ``nn.Module``. Autograd only calls ``backward`` on
    ``torch.autograd.Function`` subclasses, so that method was never invoked
    and the zero gradient of ``torch.sign`` was what actually flowed back --
    leaving the three inversion indicators untrained at their initialisation in
    every v1 checkpoint. The estimator now lives in :class:`_SignSTE`, where
    autograd will use it, which is what Sec. 3.2.2 describes.
    """

    def forward(self, input):
        """Binarise the input with a straight-through gradient.

        :param input: data
        :returns: sign of data
        :rtype: Tensor

        """
        if fixes.enabled('ste'):
            return _SignSTE.apply(input)
        return torch.sign(input)


class CubicFilter(nn.Module):
    """Cubic-polynomial ("cubic-20") filter branch of DeepLPF (paper Sec. 3.2.4, Eq. 6).

    Regresses 60 coefficients (20 per RGB channel, ``{A..T}`` in Eq. 6) from the
    backbone features and evaluates the cubic polynomial ``f(x, y, i)`` in the
    normalised pixel coordinates ``(x, y)`` and the channel intensity ``i`` at
    every pixel. Note the code applies it as a residual, ``i' = clamp(i + f)``,
    whereas Eq. 6 writes ``i' = f``.

    The parameter-prediction sub-network follows Sec. 3.2.1: conv/max-pool
    stages, global average pooling (so it is resolution agnostic), dropout 0.5,
    and a fully-connected regressor. The input is first resized to 300x300.
    """

    def __init__(self, num_in_channels=64, num_out_channels=64, batch_size=1):
        """Build the cubic-filter branch.

        :param num_in_channels: number of input feature-map channels
        :param num_out_channels: number of channels used by the conv stack
        :param batch_size: image batch size (only 1 is supported)
        :returns: N/A
        :rtype: N/A

        """
        super(CubicFilter, self).__init__()

        self.cubic_layer1 = ConvBlock(num_in_channels, num_out_channels)
        self.cubic_layer2 = MaxPoolBlock()
        self.cubic_layer3 = ConvBlock(num_out_channels, num_out_channels)
        self.cubic_layer4 = MaxPoolBlock()
        self.cubic_layer5 = ConvBlock(num_out_channels, num_out_channels)
        self.cubic_layer6 = MaxPoolBlock()
        self.cubic_layer7 = ConvBlock(num_out_channels, num_out_channels)
        self.cubic_layer8 = GlobalPoolingBlock(2)
        self.fc_cubic = torch.nn.Linear(
            num_out_channels, 60)  # cubic
        self.upsample = torch.nn.Upsample(size=(300, 300), mode='bilinear',align_corners=False)
        self.dropout = nn.Dropout(0.5)

    def get_cubic_mask(self, feat, img):
        """Cubic filter definition

        :param feat: feature map
        :param img:  image
        :returns: cubic scaling map
        :rtype: Tensor

        """
        # Parameter prediction (Sec. 3.2.1): image + backbone features -> 60 coefficients
        feat_cubic = torch.cat((feat, img), 1)
        feat_cubic = self.upsample(feat_cubic)
        return self.mask_from_input(feat_cubic, img)

    def mask_from_input(self, feat_cubic, img):
        """As :meth:`get_cubic_mask`, given the already resized 300x300 input.

        :param feat_cubic: (B, 64, 300, 300) resized concatenation of features and image
        :param img: image the filter is applied to, (B, 3, H, W)
        :returns: cubic-filtered image, (B, 3, H, W)
        :rtype: Tensor

        """
        x = self.cubic_layer1(feat_cubic)
        x = self.cubic_layer2(x)
        x = self.cubic_layer3(x)
        x = self.cubic_layer4(x)
        x = self.cubic_layer5(x)
        x = self.cubic_layer6(x)
        x = self.cubic_layer7(x)
        x = self.cubic_layer8(x)
        x = x.view(x.size()[0], -1)
        x = self.dropout(x)

        R = self.fc_cubic(x)

        # Normalised pixel coordinates in [0, 1): x_axis varies along dim 2 (H),
        # y_axis along dim 3 (W). Shared by all three filter branches.
        x_axis, x_axis2, x_axis3, y_axis, y_axis2, y_axis3 = _coord_grid_powers(
            img.shape[2], img.shape[3], img.device)

        # Evaluate Eq. 6 for every image in the batch and every RGB channel.
        # R has shape (B, 60): 20 coefficients per channel, r[0..19] = A..T in
        # the paper's order (x^3, x^2 y, x^2 i, x^2, x y^2, x y i, x y, x i^2,
        # x i, x, y^3, y^2 i, y^2, y i^2, y i, y, i^3, i^2, i, 1).
        # The x/y coordinate grids are shared across the batch and channels;
        # the coefficients are reshaped to (B, 3, 1, 1) so one expression
        # evaluates all three channels (this used to loop over c, with the
        # same per-element arithmetic).
        R = R.view(-1, 3, 20)
        r = [R[:, :, k].reshape(-1, 3, 1, 1) for k in range(20)]
        img_c = img  # (B, 3, H, W)
        cubic_mask = r[0] * x_axis3 + r[1] * x_axis2 * y_axis + r[2] * (
            x_axis2) * img_c + r[3] * x_axis2 + r[4] * x_axis * y_axis2 + r[
            5] * x_axis * y_axis * img_c \
            + r[6] * x_axis * y_axis + r[7] * x_axis * (img_c ** 2) + r[
            8] * x_axis * img_c + r[9] * x_axis + r[10] * (
            y_axis3) + r[11] * y_axis2 * img_c \
            + r[12] * y_axis2 + r[13] * y_axis * (img_c ** 2) + r[
            14] * y_axis * img_c + r[15] * y_axis + r[16] * (
            img_c ** 3) + r[17] * (img_c ** 2) \
            + r[18] * \
            img_c + r[19]

        img_cubic = torch.clamp(img + cubic_mask, 0, 1)
        return img_cubic


class GraduatedFilter(nn.Module):
    """Graduated-filter branch of DeepLPF (paper Sec. 3.2.2, Eqs. 1-3; Sec. 3.3, Eq. 7).

    Emulates a photographic graduated filter: three parallel lines split the
    image into a 100% region, two linear-decay bands and a 0% region. The
    network regresses ``G = 24`` values: for each of three filter instances a
    binarised inversion indicator ``g_inv`` (via :class:`BinaryLayer`), the
    central-line slope ``m``, two offsets ``o1, o2`` (so ``d1 = o1 cos(alpha)``,
    ``d2 = o2 cos(alpha)``, ``alpha = atan(m)``), and three per-channel scaling
    factors ``s_g^R, s_g^G, s_g^B``. Each instance yields a (B, 3, H, W)
    scaling map; the three instances are fused by element-wise multiplication
    (Eq. 7) into ``s_g``. Scalings are bounded to [0, max_scale = 2].
    """

    def __init__(self, num_in_channels=64, num_out_channels=64):
        """Initialisation function for the graduated filter

        :param num_in_channels:  input channels
        :param num_out_channels: output channels
        :returns: N/A
        :rtype: N/A

        """
        super(GraduatedFilter, self).__init__()

        self.graduated_layer1 = ConvBlock(num_in_channels, num_out_channels)
        self.graduated_layer2 = MaxPoolBlock()
        self.graduated_layer3 = ConvBlock(num_out_channels, num_out_channels)
        self.graduated_layer4 = MaxPoolBlock()
        self.graduated_layer5 = ConvBlock(num_out_channels, num_out_channels)
        self.graduated_layer6 = MaxPoolBlock()
        self.graduated_layer7 = ConvBlock(num_out_channels, num_out_channels)
        self.graduated_layer8 = GlobalPoolingBlock(2)
        self.fc_graduated = torch.nn.Linear(
            num_out_channels, 24)
        self.upsample = torch.nn.Upsample(size=(300, 300), mode='bilinear',align_corners=False)
        self.dropout = nn.Dropout(0.5)
        self.bin_layer = BinaryLayer()

    def tanh01(self, x):
        """Adjust Tanh to return values between 0 and 1

        :param x: Tensor arbitrary range
        :returns: Tensor between 0 and 1
        :rtype: tensor

        """
        tanh = nn.Tanh()
        return 0.5 * (tanh(x) + 1)

    def get_inverted_mask(self, factor, invert, d1, d2, max_scale, top_line):
        """Builds one channel's graduated scaling map ``s(x, y)`` (paper Eqs. 1-3).

        ``top_line`` plays the role of the signed distance ``l(x, y)`` to the
        central line; the two ramps ``(... / d1) * top_line`` and
        ``(... / d2) * top_line`` are the linear decays of Eqs. 1 and 2, and
        ``invert`` is the binarised ``g_inv`` of Eq. 3 that swaps which side of
        the line receives the full scaling. The four cases (inverted or not,
        scale factor above or below 1) are the branches of Eq. 3 written out
        so that the scaling always ramps between 1 (no change) and ``factor``.

        All arguments broadcast against each other; the per-image scalars are
        passed with trailing singleton dims so they broadcast over the (H, W)
        grid. The caller evaluates all nine (instance, channel) masks in one
        call with ``factor`` of shape (B, 3, 3, 1, 1), the per-instance
        ``invert``, ``d1``, ``d2`` of shape (B, 3, 1, 1, 1) and ``top_line`` of
        shape (B, 3, 1, H, W); the arithmetic per element is exactly what a
        single (B, H, W) call performed.

        :param factor: scale factor s_g
        :param invert: binary indicator variable g_inv in {0, 1}
        :param d1: distance between top and mid line
        :param d2: distance between bottom and mid line
        :param max_scale: maximum scaling factor possible
        :param top_line: soft above/below-line map
        :returns: scaling mask, the broadcast shape of the inputs
        :rtype: Tensor

        """
        # The original code branched with `(invert == 1).all()` / `(factor >= 1).all()`,
        # which collapses the batch to a single decision. To support a batch whose
        # images take different branches, all four branch expressions are evaluated
        # and selected per image with torch.where; the clamp bounds (which depend
        # only on factor >= 1) are likewise applied element-wise.
        f = factor
        inv = (invert == 1)
        fac_ge1 = (factor >= 1)

        # Every branch has the form  base + (A / d1) * top_line + (B / d2) * top_line
        # with per-image scalars base, A, B; only those differ between the
        # branches. Select them per image first, then evaluate the ramp once
        # at full resolution instead of four times. The selected map is
        # arithmetically identical to the branch it came from.
        diff_hi = (f - 1) / 2 + 1
        diff_lo = (1 - f) / 2 + f
        # invert == 1
        A_inv = torch.where(fac_ge1, diff_hi - f, diff_lo - f)
        B_inv = torch.where(fac_ge1, 1 - diff_hi, 1 - diff_lo)
        # invert != 1
        A_non = torch.where(fac_ge1, diff_hi - f, diff_lo - 1)
        B_non = torch.where(fac_ge1, f - diff_hi, f - diff_lo)

        def ramp(base, A, B):
            return base + (A / d1) * top_line + (B / d2) * top_line

        if fixes.enabled('blend'):
            branch_inv = ramp(f, A_inv, B_inv)
            branch_non = ramp(1, A_non, B_non)
        else:
            mask_scale = ramp(torch.where(inv, f, torch.ones_like(f)),
                              torch.where(inv, A_inv, A_non),
                              torch.where(inv, B_inv, B_non))

        if fixes.enabled('blend'):
            # A torch.where CONDITION is not differentiable, so selecting on
            # `inv` severs g_inv from the loss - a second cut, one line after
            # the one the `ste` fix repairs. Even with a working straight-through
            # estimator the inversion indicators receive no gradient.
            #
            # Blend the two branches with the binarised indicator instead. The
            # forward value is unchanged, because `invert` is exactly 0 or 1, but
            # now the STE's gradient reaches g_inv. Requires `ste` to be on;
            # without it torch.sign contributes its zero gradient and this is
            # merely a slower way of writing the same select.
            w = invert.to(branch_inv.dtype)
            mask_scale = w * branch_inv + (1.0 - w) * branch_non

        # factor >= 1 branches clamp to [1, max_scale]; factor < 1 branches to [0, 1].
        lower = torch.where(fac_ge1, torch.ones_like(f), torch.zeros_like(f))
        upper = torch.where(fac_ge1, torch.full_like(f, float(max_scale)), torch.ones_like(f))
        mask_scale = torch.minimum(torch.maximum(mask_scale, lower), upper)

        mask_scale = torch.clamp(mask_scale, 0, max_scale)
        return mask_scale

    def get_graduated_mask(self, feat, img):
        """ Graduated filter definition

        :param feat: features
        :param img: image
        :returns: scaling map
        :rtype: Tensor

        """
        feat_graduated = torch.cat((feat, img), 1)
        feat_graduated = self.upsample(feat_graduated)
        return self.mask_from_input(feat_graduated, img)

    def mask_from_input(self, feat_graduated, img):
        """As :meth:`get_graduated_mask`, given the already resized 300x300 input.

        :param feat_graduated: (B, 64, 300, 300) resized concatenation of features and image
        :param img: image the filter is applied to, (B, 3, H, W)
        :returns: scaling map, (B, 3, H, W)
        :rtype: Tensor

        """
        eps = 1e-10

        # Normalised pixel coordinates (see CubicFilter.get_cubic_mask)
        x_axis, y_axis = _coord_grids(img.shape[2], img.shape[3], img.device)

        # Parameter prediction (Sec. 3.2.1): image + backbone features -> 24 values
        x = self.graduated_layer1(feat_graduated)
        x = self.graduated_layer2(x)
        x = self.graduated_layer3(x)
        x = self.graduated_layer4(x)
        x = self.graduated_layer5(x)
        x = self.graduated_layer6(x)
        x = self.graduated_layer7(x)
        x = self.graduated_layer8(x)
        x = x.view(x.size()[0], -1)
        x = self.dropout(x)
        G = self.fc_graduated(x)

        # G has shape (B, 24); every parameter below is a per-image vector of
        # shape (B,). Layout: [0:3] g_inv, [3:6] slope m, [6:9] and [9:12] the
        # central-line intercept c (which doubles as offset o1; [9:12] is
        # lower-bounded by [6:9]), [12:15] offset o2 (upper-bounded by o1),
        # [15:24] per-channel scale factors s_g (three instances x R,G,B).
        #
        # The three instances are evaluated together along a leading "instance"
        # dim: every per-instance quantity below has shape (B, 3), and the nine
        # (instance, channel) masks come out of one get_inverted_mask call as
        # (B, 3, 3, H, W). This replaced three copies of every line and nine
        # mask calls; the per-element arithmetic is unchanged.
        #
        # Inversion indicator g_inv_hat = (sgn(g_inv) + 1) / 2 in {0, 1} (Sec. 3.2.2)
        above_or_below_line = ((self.bin_layer(G[:, 0:3]))+1)/2

        slope = G[:, 3:6].clone()

        y_axis_dist = self.tanh01(G[:, 6:9]) + eps

        # clamp to [y_axis_distN, 1.0]; expressed with maximum/clamp because modern
        # torch.clamp does not accept a tensor min together with a scalar max.
        y_axis_dist = torch.clamp(torch.maximum(self.tanh01(G[:, 9:12]), y_axis_dist.data), max=1.0)

        # clamp to [0, y_axis_distN]
        y_axis_dist_lo = torch.clamp(torch.minimum(self.tanh01(G[:, 12:15]), y_axis_dist.data), min=0)

        # Scales: (B, 9) laid out instance-major, i.e. [inst0 R,G,B, inst1 R,G,B, inst2 R,G,B]
        max_scale = 2

        scale_factor = self.tanh01(G[:, 15:24]) * max_scale

        slope_angle = torch.atan(slope)

        # Distances between the central line and the two outer lines:
        # d1 = o1 cos(alpha), d2 = o2 cos(alpha), alpha = atan(m) (Sec. 3.2.2)
        d_top = self.tanh01(y_axis_dist*torch.cos(slope_angle))
        d_bot = self.tanh01(y_axis_dist_lo*torch.cos(slope_angle))

        # Soft side-of-line map: tanh01 of l(x, y) = y - (m x + c) shifted by d1,
        # i.e. ~1 above the top line, ~0 below, ramping in between.
        # Broadcast the per-image line parameters (B, 3) over the (H, W) grid -> (B, 3, H, W)
        top_line = self.tanh01(y_axis - (slope.view(-1, 3, 1, 1) * x_axis + y_axis_dist.view(-1, 3, 1, 1) + d_top.view(-1, 3, 1, 1)))

        # Three filter instances, each with a per-channel (R, G, B) scaling map:
        # (B, instance, channel, H, W).
        mask_scale = self.get_inverted_mask(
            scale_factor.view(-1, 3, 3, 1, 1), above_or_below_line.view(-1, 3, 1, 1, 1),
            d_top.view(-1, 3, 1, 1, 1), d_bot.view(-1, 3, 1, 1, 1), max_scale,
            top_line.unsqueeze(2))
        mask_scale = torch.clamp(mask_scale, 0, max_scale)

        # Fuse the three instances by element-wise multiplication: s_g = prod_i s_gi (Eq. 7)
        mask_scale = torch.clamp(
            mask_scale[:, 0]*mask_scale[:, 1]*mask_scale[:, 2], 0, max_scale)

        return mask_scale


class EllipticalFilter(nn.Module):
    """Elliptical-filter branch of DeepLPF (paper Sec. 3.2.3, Eq. 4; Sec. 3.3, Eq. 7).

    Regresses ``E = 24`` values: for each of three ellipse instances the centre
    ``(h, k)``, semi-axes ``(a, b)``, rotation ``theta`` and three per-channel
    scaling factors ``s_e^R, s_e^G, s_e^B``. Inside an ellipse the scaling is
    ``s_e`` at the centre and decays linearly to 1 at the boundary (the code's
    linear-in-radius form of Eq. 4); outside it is 1 (no change). Each instance
    yields a (B, 3, H, W) map; the three are fused by element-wise
    multiplication (Eq. 7) into ``s_e``. Scalings are bounded to [0, max_scale = 2].
    """

    def __init__(self, num_in_channels=64, num_out_channels=64):
        """Build the elliptical-filter branch.

        :param num_in_channels: number of input feature-map channels
        :param num_out_channels: number of channels used by the conv stack
        :returns: N/A
        :rtype: N/A

        """
        super(EllipticalFilter, self).__init__()

        self.elliptical_layer1 = ConvBlock(num_in_channels, num_out_channels)
        self.elliptical_layer2 = MaxPoolBlock()
        self.elliptical_layer3 = ConvBlock(num_out_channels, num_out_channels)
        self.elliptical_layer4 = MaxPoolBlock()
        self.elliptical_layer5 = ConvBlock(num_out_channels, num_out_channels)
        self.elliptical_layer6 = MaxPoolBlock()
        self.elliptical_layer7 = ConvBlock(num_out_channels, num_out_channels)
        self.elliptical_layer8 = GlobalPoolingBlock(2)
        self.fc_elliptical = torch.nn.Linear(
            num_out_channels, 24)  # elliptical
        self.upsample = torch.nn.Upsample(size=(300, 300), mode='bilinear',align_corners=False)
        self.dropout = nn.Dropout(0.5)
        self._sel_cache = {}  # per-device (instance, channel) selector, see mask_from_input

    def tanh01(self, x):
        """Adjust Tanh to return values between 0 and 1

        :param x: Tensor arbitrary range
        :returns: Tensor between 0 and 1
        :rtype: tensor

        """
        tanh = nn.Tanh()
        return 0.5 * (tanh(x) + 1)

    def where(self, cond, x_1, x_2):
        """Differentiable where function to compare two Tensors

        :param cond: condition e.g. <
        :param x_1: Tensor 1
        :param x_2: Tensor 2
        :returns: Boolean comparison result
        :rtype: Tensor

        """
        cond = cond.float()
        return (cond * x_1) + ((1 - cond) * x_2)

    def get_mask(self, x_axis, y_axis, shift_x=0, shift_y=0, semi_axis_x=1, semi_axis_y=1, alpha=0,
                 scale_factor=2, max_scale=2, eps=1e-7, radius=1):
        """Gets the elliptical scaling mask according to the equation of a
        rotated ellipse

        :param x_axis: normalised x-coordinate grid
        :param y_axis: normalised y-coordinate grid
        :param shift_x: ellipse centre x-coordinate
        :param shift_y: ellipse centre y-coordinate
        :param semi_axis_x: ellipse semi-axis along x
        :param semi_axis_y: ellipse semi-axis along y
        :param alpha: rotation angle of the ellipse
        :param scale_factor: peak scaling applied at the ellipse centre
        :param max_scale: maximum permitted scaling factor
        :param eps: small constant to avoid division by zero
        :param radius: ellipse radius at the current angle
        :returns: scaling mask
        :rtype: Tensor

        """
        # Rotated-ellipse membership test (the bracketed term of Eq. 4):
        # [(x-h)cos t + (y-k)sin t]^2 / a^2 + [(x-h)sin t - (y-k)cos t]^2 / b^2 < 1
        ellipse_equation_part1 = (((x_axis - shift_x)*torch.cos(alpha) + (y_axis - shift_y)*torch.sin(alpha)) ** 2) / ((semi_axis_x)**2)
        ellipse_equation_part2 = (((x_axis - shift_x)*torch.sin(alpha) - (y_axis - shift_y)*torch.cos(alpha)) ** 2) / ((semi_axis_y)**2)

        # Inside: scale_factor at the centre, decaying linearly with the distance
        # from the centre to reach 1 at the ellipse boundary (`radius` is the
        # boundary distance along the pixel's direction). Outside: 1 (no change).
        mask_scale = self.where(ellipse_equation_part1+ellipse_equation_part2 < 1,
                                (torch.sqrt((x_axis - shift_x) ** 2 + (y_axis - shift_y) ** 2 + eps) * (1 - scale_factor)) / radius + scale_factor, 1)

        # Returns a (B, H, W) per-image, per-ellipse scaling map.
        mask_scale = torch.clamp(mask_scale, 0, max_scale)

        return mask_scale

    def get_elliptical_mask(self, feat, img):
        """Gets the elliptical scaling mask according to the equation of a
        rotated ellipse

        :param feat: features from the backbone
        :param img: image
        :returns: elliptical adjustment maps for each channel
        :rtype: Tensor

        """

        # The two eps parameters are used to avoid numerical issues in the learning
        eps2 = 1e-7
        eps1 = 1e-10

        # max_scale is the maximum an ellipse can scale the image R,G,B values by
        max_scale = 2

        feat_elliptical = torch.cat((feat, img), 1)
        feat_elliptical = self.upsample(feat_elliptical)
        return self.mask_from_input(feat_elliptical, img)

    def mask_from_input(self, feat_elliptical, img):
        """As :meth:`get_elliptical_mask`, given the already resized 300x300 input.

        :param feat_elliptical: (B, 64, 300, 300) resized concatenation of features and image
        :param img: image the filter is applied to, (B, 3, H, W)
        :returns: elliptical adjustment maps for each channel, (B, 3, H, W)
        :rtype: Tensor

        """
        # The two eps parameters are used to avoid numerical issues in the learning
        eps2 = 1e-7
        eps1 = 1e-10

        # max_scale is the maximum an ellipse can scale the image R,G,B values by
        max_scale = 2

        # Parameter prediction (Sec. 3.2.1): image + backbone features -> 24 values
        x = self.elliptical_layer1(feat_elliptical)
        x = self.elliptical_layer2(x)
        x = self.elliptical_layer3(x)
        x = self.elliptical_layer4(x)
        x = self.elliptical_layer5(x)
        x = self.elliptical_layer6(x)
        x = self.elliptical_layer7(x)
        x = self.elliptical_layer8(x)
        x = x.view(x.size()[0], -1)
        x = self.dropout(x)
        G = self.fc_elliptical(x)

        # The next code implements a rotated ellipse according to:
        # https://math.stackexchange.com/questions/426150/what-is-the-general-equation-of-the-ellipse-that-is-not-in-the-origin-and-rotate
        
        # Normalised coordinates for x and y-axes, we instantiate the ellipses in these coordinates
        x_axis, y_axis = _coord_grids(img.shape[2], img.shape[3], img.device)

        # G has shape (B, 24). The three ellipse instances are evaluated
        # together along a leading "instance" dim: each per-instance parameter
        # is a (B, 3) vector reshaped to (B, 3, 1, 1, 1) so it broadcasts over
        # the (H, W) coordinate grids, and the nine (instance, channel) masks
        # come out of one get_mask call as (B, 3, 3, H, W). This replaced three
        # copies of every line and nine get_mask calls; the per-element
        # arithmetic is unchanged.
        # Layout: [0:3] h, [3:6] k, [6:9] a, [9:12] b, [12:15] theta,
        # [15:24] per-channel scale factors s_e (three instances x R,G,B).
        # Centre of ellipse, x-coordinate (h in Eq. 4)
        x_coord = (self.tanh01(G[:, 0:3]) + eps1).view(-1, 3, 1, 1, 1)

        # Centre of ellipse, y-coordinate (k in Eq. 4)
        y_coord = (self.tanh01(G[:, 3:6]) + eps1).view(-1, 3, 1, 1, 1)

        # Semi-major axis a
        a = (self.tanh01(G[:, 6:9]) + eps1).view(-1, 3, 1, 1, 1)

        # Semi-minor axis b
        b = (self.tanh01(G[:, 9:12]) + eps1).view(-1, 3, 1, 1, 1)

        # Rotation angle theta in [0, pi]
        A = (self.tanh01(G[:, 12:15]) * math.pi + eps1).view(-1, 3, 1, 1, 1)

        # Per-channel scale factors s_e in [0, max_scale] for the three
        # instances: (B, instance, channel, 1, 1)
        scale = (self.tanh01(G[:, 15:24]) * max_scale + eps1).view(-1, 3, 3, 1, 1)

        # Polar angle of every pixel about the ellipse centre, measured from the
        # y semi-axis and offset by the ellipse rotation. The clamp keeps acos
        # inside its domain so its gradient stays finite. (B, 3, 1, H, W)
        angle = torch.acos(torch.clamp((y_axis-y_coord) /
            (torch.sqrt((x_axis-x_coord)**2 + (y_axis-y_coord)**2 + eps1)), -1+eps2, 1-eps2))-A

        # Distance from the centre to the ellipse boundary along each pixel's angle,
        # r(phi) = a b / sqrt(a^2 sin^2 phi + b^2 cos^2 phi); this is the normaliser
        # for the linear decay in get_mask.
        # https://math.stackexchange.com/questions/432902/how-to-get-the-radius-of-an-ellipse-at-a-specific-angle-by-knowing-its-semi-majo
        radius = (a*b)/torch.sqrt((a**2)*(torch.sin(angle)**2)+(b**2)*(torch.cos(angle)**2) + eps1)

        # v2 fix: instance 2's blue channel used semi_axis_y=b3, a different
        # ellipse from its red and green siblings. b2 makes the three channels
        # share one shape, as instances 1 and 3 already do.
        if fixes.enabled('ellipse'):
            semi_axis_y = b
        else:
            sel = self._sel_cache.get(b.device)
            if sel is None:
                sel = torch.zeros(1, 3, 3, 1, 1, dtype=torch.bool, device=b.device)
                sel[0, 1, 2] = True
                self._sel_cache[b.device] = sel
            semi_axis_y = torch.where(sel, b[:, 2:3], b)

        # Every instance: one ellipse geometry, three per-channel (R, G, B)
        # scalings -> (B, instance, channel, H, W)
        mask_scale = self.get_mask(x_axis, y_axis,
                                   shift_x=x_coord, shift_y=y_coord, semi_axis_x=a, semi_axis_y=semi_axis_y,
                                   alpha=angle, scale_factor=scale, radius=radius)
        mask_scale_rad = torch.clamp(mask_scale, 0, max_scale)

        # Fuse the three instances by element-wise multiplication: s_e = prod_i s_ei (Eq. 7)
        mask_scale_elliptical = torch.clamp(
            mask_scale_rad[:, 0] * mask_scale_rad[:, 1] * mask_scale_rad[:, 2], 0, max_scale)

        return mask_scale_elliptical


class Block(nn.Module):
    """Base class providing shared convolution helpers for the conv blocks."""

    def __init__(self):
        """Initialisation for a lower-level DeepLPF conv block

        :returns: N/A
        :rtype: N/A

        """
        super(Block, self).__init__()

    def conv3x3(self, in_channels, out_channels, stride=1):
        """Represents a convolution of shape 3x3

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param stride: the convolution stride
        :returns: convolution function with the specified parameterisation
        :rtype: function

        """
        return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                         stride=stride, padding=1, bias=True)


class ConvBlock(Block, nn.Module):
    """3x3 strided convolution followed by a LeakyReLU non-linearity."""

    def __init__(self, num_in_channels, num_out_channels, stride=1):
        """Initialise function for the higher level convolution block

        :param num_in_channels: number of input channels
        :param num_out_channels: number of output channels
        :param stride: unused; the internal convolution uses a fixed stride of 2
        :returns: N/A
        :rtype: N/A

        """
        super(Block, self).__init__()
        self.conv = self.conv3x3(num_in_channels, num_out_channels, stride=2)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        """ Forward function for the higher level convolution block

        :param x: Tensor representing the input BxCxWxH, where B is the batch size, C is the number of channels, W and H are the width and image height
        :returns: Tensor representing the output of the block
        :rtype: Tensor

        """
        img_out = self.lrelu(self.conv(x))
        return img_out


class MaxPoolBlock(Block, nn.Module):
    """2x2 max-pooling block with stride 2 (halves the spatial resolution)."""

    def __init__(self):
        """Initialise function for the max pooling block

        :returns: N/A
        :rtype: N/A

        """
        super(Block, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """ Forward function for the max pooling block

        :param x: Tensor representing the input BxCxWxH, where B is the batch size, C is the number of channels, W and H are the width and image height
        :returns: Tensor representing the output of the block
        :rtype: Tensor

        """
        img_out = self.max_pool(x)
        return img_out


class GlobalPoolingBlock(Block, nn.Module):
    """Global average-pooling block collapsing each feature map to a scalar."""

    def __init__(self, receptive_field):
        """Implementation of the global pooling block. Takes the average over a 2D receptive field.
        :param receptive_field: nominal receptive field size (unused; an
            adaptive average pool to a 1x1 output is used instead)
        :returns: N/A
        :rtype: N/A

        """
        super(Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """Forward function for the high-level global pooling block

        :param x: Tensor of shape BxCxAxA
        :returns: Tensor of shape BxCx1x1, where B is the batch size
        :rtype: Tensor

        """
        out = self.avg_pool(x)
        return out


class DeepLPFParameterPrediction(nn.Module):
    """Applies the three parametric filters and fuses them into an enhanced image.

    This is the two-path head of paper Fig. 3 / Sec. 3.1. Given the backbone
    feature map (whose first three channels are the backbone-enhanced image
    ``Y1``), it runs the single-stream cubic (polynomial) filter to get ``Y2``,
    then the two-stream graduated and elliptical filters on ``Y2`` to get the
    scaling maps ``s_g`` and ``s_e``, and fuses them as

        ``S = s_g + s_e``,  ``Y3 = S * Y2``,  ``Y = Y3 + Y1``  (global skip).

    Every intermediate is clamped to a valid range ([0, 1] for images,
    [0, 2] for scaling maps).
    """

    def __init__(self, num_in_channels=64, num_out_channels=64, batch_size=1):
        """Initialisation function

        :param num_in_channels:  Number of input feature maps
        :param num_out_channels: Number of output feature maps
        :param batch_size: Size of image batch
        :returns: N/A
        :rtype: N/A

        """
        super(DeepLPFParameterPrediction, self).__init__()
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.cubic_filter = CubicFilter()
        self.graduated_filter = GraduatedFilter()
        self.elliptical_filter = EllipticalFilter()
      

    def forward(self, x):
        """DeepLPF combined architecture fusing cubic, graduated and elliptical filters

        :param x: forward the data Tensor x through the network
        :returns: Tensor representing the predicted image batch of shape BxCxWxH
        :rtype: Tensor

        """
        feat = x[:, 3:64, :, :]  # C' = C - 3 backbone features
        img = x[:, 0:3, :, :]    # Y1: backbone-enhanced image

        # Each branch's parameter predictor consumes cat(feat, image) resized
        # to 300x300. The resize is bilinear and per channel, so
        # upsample(cat(a, b)) == cat(upsample(a), upsample(b)) exactly, and
        # the feature part is shared by all three branches while the graduated
        # and elliptical branches share the whole input. Resize each part once.
        upsample = self.cubic_filter.upsample
        feat_up = upsample(feat)

        # Single-stream path: Y2 = polynomial filter applied to Y1
        img_cubic = self.cubic_filter.mask_from_input(
            torch.cat((feat_up, upsample(img)), 1), img)

        # Two-stream path: graduated and elliptical scaling maps estimated from Y2
        feat_img_cubic_up = torch.cat((feat_up, upsample(img_cubic)), 1)
        mask_scale_graduated = self.graduated_filter.mask_from_input(
            feat_img_cubic_up, img_cubic)
        mask_scale_elliptical = self.elliptical_filter.mask_from_input(
            feat_img_cubic_up, img_cubic)

        # S = s_g + s_e (simple addition, Sec. 3.1), Y3 = S * Y2
        #
        # Each map is neutral at 1, so summing them puts S at 2 when both
        # filters are neutral. The `fusion` fix sums the DEVIATIONS from neutral
        # instead, so that two neutral filters compose to neutral.
        if fixes.enabled('fusion'):
            mask_scale_fuse = torch.clamp(
                1.0 + (mask_scale_graduated - 1.0) + (mask_scale_elliptical - 1.0), 0, 2)
        else:
            mask_scale_fuse = torch.clamp(
                mask_scale_graduated+mask_scale_elliptical, 0, 2)

        img_fuse = torch.clamp(img_cubic*mask_scale_fuse, 0, 1)

        # Global skip connection: Y = Y3 + Y1
        img = torch.clamp(img_fuse+img, 0, 1)

        return img


class DeepLPFNet(nn.Module):
    """End-to-end DeepLPF network.

    Composes the U-Net backbone (:class:`unet.UNetModel`) with the parametric
    filter head (:class:`DeepLPFParameterPrediction`) to map an input RGB image
    to an enhanced RGB image.
    """

    def __init__(self):
        """Initialisation function

        :returns: initialises the parameters of the neural network
        :rtype: N/A

        """
        super(DeepLPFNet, self).__init__()
        self.backbonenet = unet.UNetModel()
        self.deeplpfnet = DeepLPFParameterPrediction()
        
    def forward(self, img):
        """Neural network forward function

        :param img: input RGB image Tensor of shape BxCxHxW
        :returns: enhanced RGB image Tensor of shape BxCxHxW
        :rtype: Tensor

        """
        feat = self.backbonenet(img)
        img = self.deeplpfnet(feat)
        
        return img
