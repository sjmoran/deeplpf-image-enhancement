# -*- coding: utf-8 -*-
#Copyright (C) 2026. Sean Moran. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.
"""The `colour` feature must add capability without changing the starting model.

The head exists because every filter in the published model is diagonal: no
head can express a cross-channel operation, although white balance, saturation
and hue shifts are most of what the expert retouch does. Two properties make an
arm using it interpretable.

It must be the *identity* at initialisation, so a colour arm and its control
begin from the same predictions rather than from a perturbed model. And it must
draw no RNG before the existing modules, so at a fixed seed the two arms share
every other initial weight and the difference between them is the feature
alone, not the initialisation.
"""

import torch

import model


def _built(colour_knots=None):
    torch.manual_seed(42)
    return model.DeepLPFNet(colour_knots=colour_knots)


def test_identity_at_initialisation():
    """A zero-initialised head must reproduce the ungated model exactly."""
    x = torch.rand(1, 3, 64, 64)

    plain = _built()
    plain.eval()
    with torch.no_grad():
        want = plain(x)

    coloured = _built(colour_knots=16)
    coloured.eval()
    with torch.no_grad():
        got = coloured(x)

    assert torch.allclose(got, want, atol=1e-6), \
        'the colour head is not the identity at init; max diff %.2e' % \
        float((got - want).abs().max())


def test_existing_weights_are_untouched_at_the_same_seed():
    """Constructing the head last must leave every other parameter identical."""
    plain = _built()
    coloured = _built(colour_knots=16)

    plain_state = dict(plain.named_parameters())
    for name, param in coloured.named_parameters():
        if 'colour_head' in name:
            continue
        assert name in plain_state, '%s appeared unexpectedly' % name
        assert torch.equal(param, plain_state[name]), \
            '%s differs between the arms; the RNG stream shifted' % name


def test_the_head_receives_gradient():
    """Zero-initialised is not the same as dead: it must still train."""
    net = _built(colour_knots=16)
    net(torch.rand(1, 3, 64, 64)).mean().backward()

    grad = net.deeplpfnet.colour_head.fc.weight.grad
    assert grad is not None and grad.abs().sum() > 0, \
        'the colour head receives no gradient and can never leave the identity'


def test_the_mixer_is_genuinely_cross_channel():
    """The point of the head: an output channel must depend on other inputs."""
    head = model.ColourHead(64, knots=0)
    with torch.no_grad():
        # Pure channel swap: R_out = G_in.
        head.fc.bias[0:9] = torch.tensor(
            [-1., 1., 0., 0., 0., 0., 0., 0., 0.])

    img = torch.rand(1, 3, 8, 8)
    out = head(torch.zeros(1, 64, 8, 8), img)

    assert torch.allclose(out[:, 0], img[:, 1], atol=1e-6), \
        'the mixer cannot express a cross-channel map, which is its only job'


def test_curve_knots_zero_leaves_the_mixer_alone():
    """--colour_knots 0 must drop the curve parameters entirely."""
    head = model.ColourHead(64, knots=0)
    assert head.fc.out_features == 12

    head = model.ColourHead(64, knots=16)
    assert head.fc.out_features == 12 + 3 * 16
