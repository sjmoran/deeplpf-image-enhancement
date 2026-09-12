# -*- coding: utf-8 -*-
#Copyright (C) 2026. Sean Moran. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.
"""Training must stop when the network dies, and the loss will not tell it.

A 1000-epoch run went NaN at epoch 716 and trained on for another 284 epochs,
logging ``psnr_valid: nan`` and saving scheduled checkpoints of a dead model.
The reason nothing noticed is the subject of the first test here: the loss
stayed finite and constant throughout, because the CIELab conversion replaces
NaN with zero. Checking the parameters is what detects it.
"""

import math
import os

import torch

import train
from losses import DeepLPFLoss
from util import ImageProcessing


def test_a_dead_network_still_produces_a_finite_loss():
    """The loss cannot be used on its own to detect divergence.

    ``rgb_to_lab`` zeroes NaN, so an all-NaN prediction becomes a black image
    in Lab space and the L1 term against it is perfectly finite. Any guard that
    watches only the loss will train through a network that has died.
    """
    prediction = torch.full((1, 3, 32, 32), float('nan'))
    target = torch.rand(1, 3, 32, 32)

    lab = ImageProcessing.rgb_to_lab(prediction.squeeze(0).clone())
    assert int(torch.isnan(lab).sum()) == 0, 'the conversion no longer zeroes NaN'

    loss = DeepLPFLoss()(torch.clamp(prediction, 0, 1), target)
    assert math.isfinite(float(loss)), (
        'if this now fails, the loss has become a usable divergence signal and '
        'the parameter check could be relaxed')


def test_non_finite_parameters_stop_training(tmp_path):
    """A NaN in any parameter must end the run and save the failed weights."""
    net = torch.nn.Linear(2, 2)
    with torch.no_grad():
        net.weight[0, 0] = float('nan')

    try:
        for name, parameter in net.named_parameters():
            if not torch.isfinite(parameter).all():
                train._abort(net, str(tmp_path), 715, 'parameter %s is no longer finite' % name)
    except SystemExit as exit_reason:
        assert 'weight' in str(exit_reason)
        assert 'epoch 716' in str(exit_reason)
        assert os.path.isfile(os.path.join(str(tmp_path), 'diverged_epoch_716.pt'))
    else:
        raise AssertionError('a NaN parameter must stop training')
