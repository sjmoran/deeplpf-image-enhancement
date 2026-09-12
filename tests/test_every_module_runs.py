# -*- coding: utf-8 -*-
#Copyright (C) 2026. Sean Moran. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.
"""Every module the network declares must run in a forward pass.

A module that is built and never called costs parameters, appears in every
checkpoint, and looks like part of the architecture to anyone reading the
model. Forward hooks on every leaf module are the cheapest way to keep the
declared architecture and the executed one in step.
"""

import torch

import model

#: Leaf modules that legitimately do not fire on a plain forward pass: the two
#: upsamplers are only reached for inputs larger than their fixed size.
KNOWN_UNCALLED = {
    'deeplpfnet.graduated_filter.upsample',
    'deeplpfnet.elliptical_filter.upsample',
}


def _modules_that_never_ran(net):
    """Names of the leaf modules that no forward hook saw."""
    ran = set()
    handles = []
    for name, module in net.named_modules():
        if list(module.children()):
            continue
        handles.append(module.register_forward_hook(
            lambda mod, inputs, output, name=name: ran.add(name)))

    net.eval()
    with torch.no_grad():
        net(torch.rand(1, 3, 64, 64))

    for handle in handles:
        handle.remove()

    declared = {name for name, module in net.named_modules()
                if not list(module.children())}
    return declared - ran


def test_uncalled_modules_are_exactly_the_known_set():
    """Nothing may be declared and left unused beyond the documented set."""
    never_ran = _modules_that_never_ran(model.DeepLPFNet())

    assert never_ran == KNOWN_UNCALLED, (
        'modules built but never called: %s' % sorted(never_ran - KNOWN_UNCALLED))


def test_the_colour_head_runs_when_it_is_asked_for():
    """The optional colour head must fire once enabled."""
    never_ran = _modules_that_never_ran(model.DeepLPFNet(colour_knots=16))

    assert not any(name.startswith('deeplpfnet.colour_head')
                   for name in never_ran)
