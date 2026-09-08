# -*- coding: utf-8 -*-
"""Every trainable parameter must receive a gradient, or be listed as not doing so.

A parameter that is silently severed from the loss is invisible: the code
imports, trains, and produces a working model, and the only symptom is a
parameter that never moves. Nothing throws, so no test catches it by accident.

The expected set is written down rather than asserted empty. This model has
legitimate exceptions, and a bare "every parameter trains" assertion would fail
for known reasons and then be skipped or deleted by the next person to meet one.
"""

import pytest
import torch

import fixes
import model


# Parameters that legitimately receive no gradient on the first backward pass.
#
# unet.conv1/2/3 are the three 1x1 encoder projections, declared in v1 and never
# called; the `wiring` fix connects them. unet.local_net is a LocalNet built at
# unet.py:64 that nothing references in any configuration - dead in v1 and dead
# now, kept only so released checkpoints keep loading under strict=True.
# backbonenet.ms_gate gates the wiring contribution and exists only when that
# fix is on.
DEAD_IN_V1 = {
    'backbonenet.unet.conv1.weight', 'backbonenet.unet.conv1.bias',
    'backbonenet.unet.conv2.weight', 'backbonenet.unet.conv2.bias',
    'backbonenet.unet.conv3.weight', 'backbonenet.unet.conv3.bias',
}
NEVER_REFERENCED = {
    'backbonenet.unet.local_net.conv1.weight', 'backbonenet.unet.local_net.conv1.bias',
    'backbonenet.unet.local_net.conv2.weight', 'backbonenet.unet.local_net.conv2.bias',
}

EXPECTED_STARVED = {
    # v1: the projections and local_net are unreachable, and ms_gate is unused.
    'none': DEAD_IN_V1 | NEVER_REFERENCED | {'backbonenet.ms_gate'},
    'ste': DEAD_IN_V1 | NEVER_REFERENCED | {'backbonenet.ms_gate'},
    # wiring: ms_gate starts at zero, so on the FIRST step the projections it
    # gates still see no gradient. The gate itself does, moves off zero, and
    # from step 1 they train - see test_the_gated_projections_start_training.
    'wiring': DEAD_IN_V1 | NEVER_REFERENCED,
    'all': DEAD_IN_V1 | NEVER_REFERENCED,
}


def _starved_after_one_backward(fix_spec):
    """Names of trainable parameters with no gradient after one backward pass.

    A gradient of None means the parameter never entered the graph; an all-zero
    gradient means it entered but is disconnected from the loss - the
    ``torch.sign`` signature. Both mean it cannot train.
    """
    fixes.configure(fix_spec)
    torch.manual_seed(0)

    net = model.DeepLPFNet()
    net.train()

    predicted = net(torch.rand(1, 3, 64, 64))
    net.zero_grad()
    model.DeepLPFLoss()(predicted, torch.rand(1, 3, 64, 64)).backward()

    return {name for name, param in net.named_parameters()
            if param.requires_grad and (param.grad is None or not param.grad.any())}


@pytest.mark.parametrize('fix_spec', sorted(EXPECTED_STARVED))
def test_starved_parameters_are_exactly_the_known_set(fix_spec):
    """No parameter is silently starved of gradient in any configuration.

    Equality, not a subset check: a new starved parameter is a regression, and a
    parameter that stops being starved means the model changed in a way this
    file should be updated to describe.
    """
    starved = _starved_after_one_backward(fix_spec)
    expected = EXPECTED_STARVED[fix_spec]

    assert starved == expected, (
        'unexpected: %s | no longer starved: %s'
        % (sorted(starved - expected), sorted(expected - starved)))


def test_the_ste_fix_alone_does_not_free_the_inversion_indicators():
    """`ste` and `blend` are only useful together, and this pins that.

    The binarisation reaches the loss through two places: the estimator itself,
    and a ``torch.where`` whose condition is not differentiable. Enabling either
    fix alone leaves the other path severed, so the indicators receive gradient
    only under ``ste,blend``. Asserted so that a change to either one shows up
    here rather than silently making an ablation arm a no-op.
    """
    for fix_spec, should_train in (('none', False), ('ste', False),
                                   ('blend', False), ('ste,blend', True)):
        fixes.configure(fix_spec)
        torch.manual_seed(0)
        net = model.DeepLPFNet()
        net.train()
        predicted = net(torch.rand(1, 3, 64, 64))
        net.zero_grad()
        model.DeepLPFLoss()(predicted, torch.rand(1, 3, 64, 64)).backward()

        # Rows 0:3 of fc_graduated drive g_inv, which reaches the loss only
        # through the binarisation.
        g_inv_rows = net.deeplpfnet.graduated_filter.fc_graduated.weight.grad[0:3]
        assert bool(g_inv_rows.any()) is should_train, (
            'g_inv gradient under --fixes=%s should be %s; if this now trains, the '
            'where-condition at model.py:494 was made differentiable and both this '
            'test and the ste arm need revisiting' % (fix_spec, should_train))


def test_the_gated_projections_start_training_after_the_first_step():
    """ms_gate opens on step 0, so the wiring path is cold for one step only.

    Worth asserting because a gate initialised to zero looks, on a single
    backward pass, exactly like a disconnected subnetwork.
    """
    fixes.configure('wiring')
    torch.manual_seed(0)
    net = model.DeepLPFNet()
    net.train()
    optimiser = torch.optim.Adam(net.parameters(), lr=1e-3)

    saw_gradient = []
    for _ in range(2):
        predicted = net(torch.rand(1, 3, 64, 64))
        optimiser.zero_grad()
        model.DeepLPFLoss()(predicted, torch.rand(1, 3, 64, 64)).backward()
        saw_gradient.append(bool(net.backbonenet.unet.conv1.weight.grad.any()))
        optimiser.step()

    assert saw_gradient == [False, True], (
        'expected the projections to be cold on step 0 and training on step 1, got %s'
        % saw_gradient)
    assert float(net.backbonenet.ms_gate) != 0.0, 'ms_gate never left zero'
