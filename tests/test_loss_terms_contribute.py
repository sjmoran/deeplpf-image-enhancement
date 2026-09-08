# -*- coding: utf-8 -*-
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

import pytest
import torch

import fixes
import model
from util import ImageProcessing


def _term_gradient_norms(fix_spec='none', seed=0):
    """Gradient norm over all parameters from each loss term separately.

    :returns: ``(l1_norm, ssim_norm)``, each the L2 norm of the whole network's
              flattened gradient under that term alone
    :rtype: tuple
    """
    fixes.configure(fix_spec)

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


def _msssim_value(fix_spec):
    """MS-SSIM of one fixed pair of images under the given fix configuration."""
    fixes.configure(fix_spec)
    criterion = model.DeepLPFLoss()

    torch.manual_seed(1)
    a = torch.rand(1, 1, 64, 64)
    b = torch.rand(1, 1, 64, 64)

    return float(criterion.compute_msssim(a, b))


def test_the_msssim_fix_changes_the_structural_term():
    """The msssim fix must actually alter what that term computes.

    v1 grouped the product as prod(pow1[:-1] * pow2[-1]), broadcasting the
    finest-scale term into all four factors; upstream groups it as
    prod(pow1[:-1]) * pow2[-1]. If the two ever agree, the msssim ablation arm
    is measuring nothing and should be dropped.

    Both values are computed here rather than across two parametrised runs, so
    the comparison cannot be skipped by test reordering or parallel execution.
    """
    v1_value = _msssim_value('none')
    fixed_value = _msssim_value('msssim')

    for name, value in (('none', v1_value), ('msssim', fixed_value)):
        assert 0.0 <= value <= 1.0, \
            'MS-SSIM under --fixes=%s outside [0, 1]: %s' % (name, value)

    assert v1_value != fixed_value, (
        'the msssim fix produces an identical value to v1 (%r); the ablation arm '
        'would be a no-op' % v1_value)
