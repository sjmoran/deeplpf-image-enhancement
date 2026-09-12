# -*- coding: utf-8 -*-
#Copyright (C) 2026. Sean Moran. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.
"""The colour conversion and the metrics, against independent implementations.

Eq. 8 is an L1 loss in CIELab, so ``rgb_to_lab`` decides what the network is
trained to match, and every reported number comes from ``compute_psnr``. Both
are hand-written here, and nothing compared either against a reference: a slow
drift in the conversion would move every result in the repository without
failing a single test.

scikit-image is the reference. It is already a dependency, used by
``compute_ssim``.
"""

import numpy as np
import torch
from skimage import color
from skimage.metrics import peak_signal_noise_ratio as reference_psnr

from util import ImageProcessing

#: The repository rescales Lab into [0, 1] as L/100 and (a, b)/220 + 0.5.
L_DIVISOR = 100.0
AB_DIVISOR = 220.0


def as_scaled_lab(rgb_chw):
    """Convert with scikit-image, then apply this repository's [0, 1] scaling."""
    lab = color.rgb2lab(np.transpose(rgb_chw, (1, 2, 0)))
    return np.stack([lab[:, :, 0] / L_DIVISOR,
                     lab[:, :, 1] / AB_DIVISOR + 0.5,
                     lab[:, :, 2] / AB_DIVISOR + 0.5], axis=0)


def test_rgb_to_lab_matches_scikit_image():
    """The conversion the loss is computed in must be the real one."""
    rng = np.random.default_rng(0)
    for _ in range(3):
        rgb = rng.random((3, 48, 48)).astype(np.float32)

        ours = ImageProcessing.rgb_to_lab(torch.from_numpy(rgb).clone()).numpy()

        assert np.abs(ours - as_scaled_lab(rgb)).max() < 1e-3


def test_rgb_to_lab_handles_the_gamma_boundary():
    """The sRGB expansion is piecewise; both sides and the join must be right."""
    values = np.array([0.0, 0.02, 0.04045, 0.05, 0.5, 1.0], dtype=np.float32)
    rgb = np.repeat(values[None, None, :], 3, axis=0)  # 3 x 1 x 6

    ours = ImageProcessing.rgb_to_lab(torch.from_numpy(rgb).clone()).numpy()

    assert np.abs(ours - as_scaled_lab(rgb)).max() < 1e-3


def test_compute_psnr_matches_scikit_image():
    """PSNR is what every result in the repository is quoted in."""
    rng = np.random.default_rng(1)
    a = rng.random((1, 3, 32, 32)).astype(np.float32)
    b = np.clip(a + 0.05 * rng.standard_normal(a.shape), 0, 1).astype(np.float32)

    ours = ImageProcessing.compute_psnr(a, b, 1.0)

    assert abs(ours - reference_psnr(a, b, data_range=1.0)) < 1e-4


def test_identical_images_score_infinite_psnr():
    """A perfect prediction has zero error, and PSNR must not divide by it."""
    img = np.full((1, 3, 8, 8), 0.5, dtype=np.float32)

    assert np.isinf(ImageProcessing.compute_psnr(img, img.copy(), 1.0))
