# -*- coding: utf-8 -*-
#Copyright (C) 2026. Sean Moran. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.
"""Every term in the loss must actually influence training.

A loss term can appear in the paper's equation, appear in the code, and still be
decorative - if its weight is small enough, or its implementation degenerate
enough, that it never moves a parameter. Nothing errors; the term is simply
along for the ride, and the paper describes an objective the model was not
trained on.

DeepLPF's Eq. 8 is ``L = w_lab * L1(Lab) + w_msssim * (1 - MS-SSIM(L))`` with
``w_lab = 1`` and ``w_msssim = 1e-3``. This file measures what that weighting
actually buys, by comparing the gradient each term contributes rather than the
loss values, since gradients are what training consumes.

The numbers are asserted as ranges rather than pinned exactly: the point is to
catch a term becoming irrelevant, not to freeze a measurement.
"""

import os

import torch

import model
from util import ImageProcessing


def _term_gradient_norms(seed=0):
    """Gradient norm over all parameters from each loss term separately.

    :returns: ``(l1_norm, ssim_norm)``, each the L2 norm of the whole network's
              flattened gradient under that term alone
    :rtype: tuple
    """

    def grads_from(term):
        torch.manual_seed(seed)
        net = model.DeepLPFNet()
        net.train()
        torch.manual_seed(seed + 1)
        predicted = net(torch.rand(1, 3, 64, 64))
        target = torch.rand(1, 3, 64, 64)

        criterion = model.DeepLPFLoss()
        # Mirror DeepLPFLoss.forward: Lab with channels rescaled to [0, 1],
        # and MS-SSIM on the lightness channel alone (Eq. 8's Lab(.) and L(.)).
        predicted_lab = ImageProcessing.rgb_to_lab(predicted.squeeze(0))
        target_lab = ImageProcessing.rgb_to_lab(target.squeeze(0))

        if term == 'l1':
            value = torch.nn.functional.l1_loss(predicted_lab, target_lab)
        else:
            value = 1.0 - criterion.compute_msssim(
                predicted_lab[0, :, :].unsqueeze(0).unsqueeze(0),
                target_lab[0, :, :].unsqueeze(0).unsqueeze(0))

        net.zero_grad()
        value.backward()
        return torch.cat([p.grad.flatten() for p in net.parameters()
                          if p.grad is not None]).norm()

    return float(grads_from('l1')), float(grads_from('ssim'))


def test_both_loss_terms_produce_a_gradient():
    """Neither term is disconnected from the parameters.

    The weaker assertion, but the one that catches a term wired up so that it
    cannot influence anything at all.
    """
    l1_norm, ssim_norm = _term_gradient_norms()

    assert l1_norm > 0, 'the Lab L1 term produces no gradient'
    assert ssim_norm > 0, 'the MS-SSIM term produces no gradient'


def test_the_msssim_term_is_negligible_at_its_published_weight():
    """Record the structural term's share at the published weight.

    At w_msssim = 1e-3 the term contributes a fraction of a percent of the L1
    term's gradient, so what trains the model is effectively L1 in Lab space.
    Anyone reasoning about the structural term should know that scale. Asserted
    so a change to the weight, or to compute_msssim, surfaces here rather than
    being discovered later by measurement.
    """
    l1_norm, ssim_norm = _term_gradient_norms()

    weighted_ratio = (1e-3 * ssim_norm) / l1_norm

    assert weighted_ratio < 0.05, (
        'the MS-SSIM term now contributes %.1f%% of the L1 term\'s gradient; '
        'if the weight or the implementation changed, update this test and the '
        'claim in docs that the loss is effectively Lab-L1' % (100 * weighted_ratio))


def test_msssim_multiplies_the_finest_scale_term_once():
    """The finest-scale SSIM enters the MS-SSIM product exactly once.

    Grouping it as ``pow1[:-1] * pow2[-1]`` broadcasts that term into all four
    factors, so it is raised to the fourth power - a silent scaling of the
    structural term rather than an error.
    """
    import torch

    criterion = model.DeepLPFLoss()
    mcs = torch.tensor([0.9, 0.8, 0.85, 0.95, 0.7])
    ssims = torch.tensor([0.6, 0.7, 0.75, 0.8, 0.9])
    weights = torch.tensor([0.4, 0.3, 0.2, 0.06, 0.04])

    pow1, pow2 = mcs ** weights, ssims ** weights
    expected = pow1[0] * pow1[1] * pow1[2] * pow1[3] * pow2[-1]

    got = criterion.combine_scales(pow1, pow2) if hasattr(criterion, 'combine_scales') else None
    if got is None:
        # The grouping lives inline in compute_msssim; assert on the identity it
        # must satisfy rather than on a private helper.
        wrong = pow1[:-1] * pow2[-1]
        wrong = wrong[0] * wrong[1] * wrong[2] * wrong[3]
        assert not torch.isclose(wrong, expected), 'the two groupings must differ'
        source = open(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'losses.py')).read()
        assert 'pow1[0] * pow1[1] * pow1[2] * pow1[3] * pow2[-1]' in source
        assert 'p = pow1[:-1] * pow2[-1]' not in source
