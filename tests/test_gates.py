# -*- coding: utf-8 -*-
"""The `gates` feature must make filter count learnable without changing the model.

Two properties carry the whole idea. A gate at zero has to remove its instance
*exactly* - if it only nearly removes it, the L1 penalty is trading reconstruction
quality for sparsity and the learned count means nothing. And the gates have to
receive gradient, or the count is fixed at whatever initialisation produced, which
is the failure mode this repository already has elsewhere.
"""

import torch

import fixes
import model


def _net(spec):
    fixes.configure(spec)
    torch.manual_seed(0)
    return model.DeepLPFNet()


def test_closed_gate_is_exactly_the_identity():
    """g=0 must make an instance exactly neutral, so switching it off is free."""
    fixes.configure('ellipse,fusion,gates')
    mask = torch.rand(2, 3, 3, 8, 8) * 2
    gates = torch.zeros(2, 3)

    gated = model._apply_gates(mask, gates)

    assert torch.equal(gated, torch.ones_like(gated)), \
        'a closed gate left a residue; sparsity would then cost reconstruction'


def test_open_gate_leaves_the_instance_untouched():
    """g=1 must reproduce the ungated model exactly."""
    fixes.configure('ellipse,fusion,gates')
    mask = torch.rand(2, 3, 3, 8, 8) * 2

    gated = model._apply_gates(mask, torch.ones(2, 3))

    assert torch.allclose(gated, mask, atol=0, rtol=0), \
        'a fully open gate changed the filter'


def test_gates_receive_gradient():
    """The gate rows of both FC layers must be trained, or the count is fixed."""
    net = _net('ellipse,fusion,gates')
    out = net(torch.rand(1, 3, 64, 64))
    (out.mean() + net.gate_penalty).backward()

    for name in ('graduated_filter.fc_graduated', 'elliptical_filter.fc_elliptical'):
        module = net.deeplpfnet.get_submodule(name)
        grad = module.weight.grad[24:27]
        assert grad is not None and grad.abs().sum() > 0, \
            '%s gate rows receive no gradient; the filter count cannot be learned' % name


def test_penalty_is_the_mean_gate():
    """The penalised quantity must be the expected active-filter fraction."""
    net = _net('ellipse,fusion,gates')
    net(torch.rand(1, 3, 64, 64))

    both = torch.cat((net.deeplpfnet.graduated_filter.gates,
                      net.deeplpfnet.elliptical_filter.gates), 1)
    assert torch.allclose(net.gate_penalty, both.mean())
    assert 0.0 <= float(net.gate_penalty) <= 1.0


def test_ungated_model_is_untouched():
    """Without the feature the layer widths and outputs must be as published."""
    net = _net('none')
    assert net.deeplpfnet.graduated_filter.fc_graduated.out_features == 24
    assert net.deeplpfnet.elliptical_filter.fc_elliptical.out_features == 24
    assert net(torch.rand(1, 3, 64, 64)).shape == (1, 3, 64, 64)
