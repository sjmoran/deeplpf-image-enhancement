# -*- coding: utf-8 -*-
"""Toggles for the v2 changes, so each can be measured against the v1 model.

Every change v2 makes to the published model is gated here. With no fix enabled
the network reproduces v1 exactly, which means the control arm of an ablation
runs the same code path as every treatment - no separate branch, and no risk of
an incidental difference between control and treatment landing in the measured
effect.

``wiring``
    Feed the three 1x1 encoder projections (``UNet.conv1/2/3``) into the filter
    head input, gated by ``ms_gate`` which starts at zero.
``ellipse``
    Use ``semi_axis_y=b2`` for instance 2's third channel, so its three channels
    share one ellipse, as instances 1 and 3 do.
``ste``
    Route the graduated filter's binarisation through an ``autograd.Function``
    so the straight-through estimator of paper Sec. 3.2.2 is applied.
``blend``
    Combine the graduated filter's two branches as
    ``g_inv_hat * m_inv + (1 - g_inv_hat) * m_non`` rather than selecting with a
    ``torch.where``, whose condition carries no gradient. Needed alongside
    ``ste`` for the inversion indicators to receive any gradient. Forward values
    are unchanged, since ``invert`` is exactly 0 or 1.
``fusion``
    Sum the filter maps' deviations from neutral,
    ``S = 1 + (s_g - 1) + (s_e - 1)``, rather than the maps themselves, so that
    two neutral filters compose to neutral.
``msssim``
    Group the MS-SSIM product as ``prod(pow1[:-1]) * pow2[-1]``, matching
    upstream jorge-pessoa/pytorch-msssim.

Set the active fixes once at startup with :func:`configure`; read them with
:func:`enabled`.
"""

ALL_FIXES = ('wiring', 'ellipse', 'ste', 'msssim', 'fusion', 'blend')

PUBLISHED_MSSSIM_WEIGHT = 1e-3

_active = frozenset()
_msssim_weight = PUBLISHED_MSSSIM_WEIGHT


def msssim_weight():
    """Weight on the MS-SSIM term of Eq. 8.

    Defaults to the published 1e-3. At that value the term contributes roughly
    0.07% of the L1 term's gradient, so it cannot influence training - not
    because the term is degenerate (unweighted it is within a factor of 1.5 of
    L1) but because of the weight alone. Raising it is the only way to find out
    whether the structural term does anything when it is not decorative.

    :returns: the active weight
    :rtype: float

    """
    return _msssim_weight


def configure(spec, msssim_weight=None):
    """Set the active fixes from a command-line spec.

    :param spec: ``'none'``, ``'all'``, or a comma-separated subset of
                 :data:`ALL_FIXES` (e.g. ``'wiring,ste'``)
    :param msssim_weight: override for the Eq. 8 MS-SSIM weight; ``None`` keeps
                          the published :data:`PUBLISHED_MSSSIM_WEIGHT`
    :returns: the active fixes, sorted
    :rtype: tuple
    :raises ValueError: if a name is not one of :data:`ALL_FIXES`

    """
    global _active, _msssim_weight

    _msssim_weight = (PUBLISHED_MSSSIM_WEIGHT if msssim_weight is None
                      else float(msssim_weight))

    spec = (spec or 'none').strip().lower()
    if spec == 'all':
        names = set(ALL_FIXES)
    elif spec == 'none':
        names = set()
    else:
        names = {part.strip() for part in spec.split(',') if part.strip()}
        unknown = names - set(ALL_FIXES)
        if unknown:
            raise ValueError(
                'unknown fix(es) %s; choose from %s, or "all"/"none"'
                % (', '.join(sorted(unknown)), ', '.join(ALL_FIXES)))

    _active = frozenset(names)
    return tuple(sorted(_active))


def enabled(name):
    """Is this fix active?

    :param name: one of :data:`ALL_FIXES`
    :returns: whether the fix is enabled
    :rtype: bool

    """
    return name in _active


def active():
    """The active fixes, sorted.

    :returns: names of the enabled fixes
    :rtype: tuple

    """
    return tuple(sorted(_active))
