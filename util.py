# -*- coding: utf-8 -*-
#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#Copyright (C) 2026. Sean Moran. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.
'''
This is a PyTorch implementation of the CVPR 2020 paper:
"Deep Local Parametric Filters for Image Enhancement": https://arxiv.org/abs/2003.13985

Please cite the paper if you use this code

Tested with Pytorch 1.7.1, Python 3.7.9

Authors: Sean Moran (sean.j.moran@gmail.com), 
         Pierre Marza (pierre.marza@gmail.com)

'''
import matplotlib
matplotlib.use('agg')
import inspect
import numpy as np
import torch
import sys
from PIL import Image
from skimage.metrics import structural_similarity as ssim

np.set_printoptions(threshold=sys.maxsize)

# scikit-image >= 0.19 renamed the ``multichannel=True`` argument of
# ``structural_similarity`` to ``channel_axis``. Detect which one this
# installation supports so the code works across skimage versions. The images
# passed to SSIM are HxWx3, so the channel axis is the last one.
if "channel_axis" in inspect.signature(ssim).parameters:
    _SSIM_MULTICHANNEL_KWARGS = {"channel_axis": -1}
else:  # older skimage (e.g. the pinned 0.18.1)
    _SSIM_MULTICHANNEL_KWARGS = {"multichannel": True}


_LAB_CONSTANTS = {}


def _lab_constants(device):
    """The constant matrices/vectors of :meth:`ImageProcessing.rgb_to_lab`,
    built once per device rather than copied host-to-device on every call."""
    consts = _LAB_CONSTANTS.get(device)
    if consts is None:
        # Linear RGB -> XYZ (D65), then normalise by the D65 white point
        rgb_to_xyz = torch.FloatTensor([  # X        Y          Z
                                                [0.412453, 0.212671,
                                                    0.019334],  # R
                                                [0.357580, 0.715160,
                                                    0.119193],  # G
                                                [0.180423, 0.072169,
                                                    0.950227],  # B
                                                ]).to(device)
        white = torch.FloatTensor([1/0.950456, 1.0, 1/1.088754]).to(device)
        # (fx, fy, fz) -> (L, a, b): L = 116 fy - 16, a = 500 (fx - fy), b = 200 (fy - fz)
        fxfyfz_to_lab = torch.FloatTensor([[0.0,  500.0,    0.0],  # fx
                                                    [116.0, -500.0,  200.0],  # fy
                                                    [0.0,    0.0, -200.0],  # fz
                                                    ]).to(device)
        lab_offset = torch.FloatTensor([-16.0, 0.0, 0.0]).to(device)
        # Per-channel rescaling to [0, 1]: L / 100, (a / 110 + 1) / 2, (b / 110 + 1) / 2.
        # Written as x / d + o with d = (100, 220, 220), o = (0, 0.5, 0.5):
        # (x / 110 + 1) / 2 and x / 220 + 0.5 round identically in IEEE
        # arithmetic since halving is exact, so this is the same map in one
        # kernel instead of three in-place slice writes.
        rescale_div = torch.FloatTensor([100.0, 220.0, 220.0]).view(3, 1, 1).to(device)
        rescale_off = torch.FloatTensor([0.0, 0.5, 0.5]).view(3, 1, 1).to(device)
        consts = _LAB_CONSTANTS[device] = (
            rgb_to_xyz, white, fxfyfz_to_lab, lab_offset, rescale_div, rescale_off)
    return consts


class ImageProcessing(object):
    """Stateless image-processing helpers (colour conversion and metrics).

    All methods are static; the class is used purely as a namespace. Tensor
    methods operate on CHW image tensors, while the numpy metric helpers operate
    on batched arrays of shape BxCxHxW.
    """

    @staticmethod
    def rgb_to_lab(img):
        """ PyTorch implementation of RGB to LAB conversion: https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
        Based roughly on a similar implementation here: https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py

        :param img: RGB image Tensor of shape CxHxW with values in [0, 1]
        :returns: CIELAB image Tensor of shape CxHxW with each channel rescaled to [0, 1]
        :rtype: Tensor

        """
        # CHW -> WHC -> (W*H, 3) so each pixel is a row for the matrix products below
        img = img.permute(2, 1, 0)
        shape = img.shape
        img = img.contiguous()
        img = img.view(-1, 3)

        # sRGB gamma expansion to linear RGB (piecewise: linear below 0.04045)
        img = (img / 12.92) * img.le(0.04045).float() + (((torch.clamp(img,
                                                                       min=0.000001) + 0.055) / 1.055) ** 2.4) * img.gt(0.04045).float()

        (rgb_to_xyz, white, fxfyfz_to_lab, lab_offset,
         rescale_div, rescale_off) = _lab_constants(img.device)

        # Linear RGB -> XYZ (D65), then normalise by the D65 white point
        img = torch.matmul(img, rgb_to_xyz)
        img = torch.mul(img, white)

        # CIE f(t): cube root above (6/29)^3, linear below (the clamps keep the
        # gradient of the cube root finite at zero)
        epsilon = 6/29

        img = ((img / (3.0 * epsilon**2) + 4.0/29.0) * img.le(epsilon**3).float()) + \
            (torch.clamp(img, min=0.0001)**(1.0/3.0) * img.gt(epsilon**3).float())

        # (fx, fy, fz) -> (L, a, b): L = 116 fy - 16, a = 500 (fx - fy), b = 200 (fy - fz)
        img = torch.matmul(img, fxfyfz_to_lab) + lab_offset

        img = img.view(shape)
        img = img.permute(2, 1, 0)

        '''
        L_chan: black and white with input range [0, 100]
        a_chan/b_chan: color channels with input range ~[-110, 110], not exact
        [0, 100] => [0, 1],  ~[-110, 110] => [0, 1]
        '''
        img = img / rescale_div + rescale_off

        # Zero any NaNs. A boolean-mask assignment here would run nonzero() and
        # so synchronise the GPU with the host on every call; where() does not.
        img = torch.where(img != img, torch.zeros((), dtype=img.dtype, device=img.device), img)

        img = img.contiguous()

        return img

    @staticmethod
    def swapimdims_3HW_HW3(img):
        """Move the image channels to the first dimension of the numpy
        multi-dimensional array

        :param img: numpy nd array representing the image
        :returns: numpy nd array with permuted axes
        :rtype: numpy nd array

        """
        if img.ndim == 3:
            return np.swapaxes(np.swapaxes(img, 1, 2), 0, 2)
        elif img.ndim == 4:
            return np.swapaxes(np.swapaxes(img, 2, 3), 1, 3)

    @staticmethod
    def load_image(img_filepath, normaliser):
        """Loads an image from file as a numpy multi-dimensional array

        :param img_filepath: filepath to the image
        :param normaliser: value to divide the pixel intensities by (e.g. 255)
        :returns: image as a multi-dimensional numpy array
        :rtype: multi-dimensional numpy array

        """
        img = np.array(Image.open(img_filepath))
        if img.ndim == 2:
            # Greyscale: repeat the single channel, since every convolution in
            # the network expects three. Without this a greyscale photograph
            # reached the first conv and failed with a shape mismatch.
            img = np.stack([img] * 3, axis=2)
        if img.ndim == 3 and img.shape[2] == 4:
            # Drop an alpha channel if present (RGBA -> RGB); the network is
            # trained on 3-channel RGB input.
            img = img[:, :, :3]
        img = ImageProcessing.normalise_image(img, normaliser)  # NB: imread normalises to 0-1

        return img

    @staticmethod
    def normalise_image(img, normaliser):
        """Normalises image data to be a float between 0 and 1

        :param img: Image as a numpy multi-dimensional image array
        :returns: Normalised image as a numpy multi-dimensional image array
        :rtype: Numpy array

        """
        img = img.astype('float32') / normaliser
        return img

    @staticmethod
    def compute_mse(original, result):
        """Computes the mean squared error between two RGB images represented as multi-dimensional numpy arrays.

        :param original: input RGB image as a numpy array
        :param result: target RGB image as a numpy array
        :returns: the mean squared error between the input and target images
        :rtype: float

        """
        return ((original - result) ** 2).mean()

    @staticmethod
    def compute_psnr(image_batchA, image_batchB, max_intensity):
        """Computes the PSNR for a batch of input and output images

        :param image_batchA: numpy nd-array representing image batch A of shape Bx3xHxW
        :param image_batchB: numpy nd-array representing image batch B of shape Bx3xHxW
        :param max_intensity: maximum intensity possible in the image (e.g. 255)
        :returns: average PSNR for the batch of images
        :rtype: float

        """
        num_images = image_batchA.shape[0]
        psnr_val = 0.0

        for i in range(0, num_images):
            imageA = image_batchA[i, 0:3, :, :]
            imageB = image_batchB[i, 0:3, :, :]
            imageB = np.maximum(0, np.minimum(imageB, max_intensity))
            psnr_val += 10 * \
                np.log10(max_intensity ** 2 /
                         ImageProcessing.compute_mse(imageA, imageB))

        return psnr_val / num_images

    @staticmethod
    def compute_ssim(image_batchA, image_batchB):
        """Computes the SSIM for a batch of input and output images

        :param image_batchA: numpy nd-array representing image batch A of shape Bx3xHxW
        :param image_batchB: numpy nd-array representing image batch B of shape Bx3xHxW
        :returns: average SSIM for the batch of images
        :rtype: float

        """
        num_images = image_batchA.shape[0]
        ssim_val = 0.0

        for i in range(0, num_images):
            imageA = ImageProcessing.swapimdims_3HW_HW3(
                image_batchA[i, 0:3, :, :])
            imageB = ImageProcessing.swapimdims_3HW_HW3(
                image_batchB[i, 0:3, :, :])
            ssim_val += ssim(imageA, imageB, data_range=imageA.max() - imageA.min(),
                             gaussian_weights=True, win_size=11, **_SSIM_MULTICHANNEL_KWARGS)

        return ssim_val / num_images
