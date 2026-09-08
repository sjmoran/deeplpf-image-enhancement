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

Separately from the fixes, :data:`FEATURES` holds additions that change what
the model can express rather than correcting it. They are opt-in by name and
are deliberately *not* included in ``'all'``, so that ``--fixes=all`` keeps
meaning "every correction, and nothing else".

``colour``
    Apply a global colour mixer and per-channel tone curve to Y1 before the
    cubic filter. Every existing head is diagonal - the cubic filter's
    coefficients are per channel and every term multiplies that same channel -
    so no filter in the published model can express white balance, saturation
    or a hue shift, which is most of what the expert retouch does. The mixer
    ``Y1' = M Y1 + b`` supplies the cross-channel term and the piecewise-linear
    curve supplies a tone shape the cubic's fixed polynomial cannot. Both are
    zero-initialised, so the feature is the identity at initialisation and the
    model it starts from is exactly the published one.

``gates``
    Predict one gate per filter instance and scale that instance's deviation
    from neutral by it, so an instance can switch itself off. With an L1
    penalty on the gates the model learns how many filter instances the image
    needs, instead of always using the hardcoded three per branch. Requires
    ``fusion``: without it two neutral branches compose to ``S = 2`` rather
    than the identity, so switching filters off is not free and the penalty
    fights the reconstruction loss.

Set the active fixes once at startup with :func:`configure`; read them with
:func:`enabled`.
"""

ALL_FIXES = ('wiring', 'ellipse', 'ste', 'msssim', 'fusion', 'blend')

#: Opt-in capabilities, excluded from ``'all'`` (see the module docstring).
FEATURES = ('gates', 'colour')

#: Instances per filter branch that a gate can switch off.
GATES_PER_BRANCH = 3

PUBLISHED_GATE_WEIGHT = 3e-3

#: Knots in the per-channel tone curve of the ``colour`` feature. 0 means the
#: colour mixer alone, which is the ablation that separates the two halves.
COLOUR_KNOTS = 16

PUBLISHED_MSSSIM_WEIGHT = 1e-3

_active = frozenset()
_msssim_weight = PUBLISHED_MSSSIM_WEIGHT
_gate_weight = PUBLISHED_GATE_WEIGHT
_colour_knots = COLOUR_KNOTS


def colour_knots():
    """Knots in the ``colour`` feature's per-channel tone curve.

    Zero disables the curve and leaves the colour mixer alone, which is the
    arm that says which of the two halves carries any gain.

    :returns: the active knot count
    :rtype: int

    """
    return _colour_knots


def gate_weight():
    """Weight on the L1 gate penalty that drives the learnable filter count.

    The penalty is the mean gate value over all six instances, so it lies in
    [0, 1] and the weighted term is directly comparable with the Lab L1 loss,
    which sits around 0.035 once trained. Too small and every gate pins at 1
    (the published fixed-three model); too large and the branches collapse to
    the identity. Only meaningful with the ``gates`` feature enabled.

    :returns: the active weight
    :rtype: float

    """
    return _gate_weight


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


def configure(spec, msssim_weight=None, gate_weight=None, colour_knots=None):
    """Set the active fixes from a command-line spec.

    :param spec: ``'none'``, ``'all'``, or a comma-separated subset of
                 :data:`ALL_FIXES` and :data:`FEATURES` (e.g. ``'wiring,ste'``).
                 ``'all'`` means every fix in :data:`ALL_FIXES`; features are
                 never implied and must be named.
    :param msssim_weight: override for the Eq. 8 MS-SSIM weight; ``None`` keeps
                          the published :data:`PUBLISHED_MSSSIM_WEIGHT`
    :param gate_weight: override for the ``gates`` L1 penalty weight; ``None``
                        keeps :data:`PUBLISHED_GATE_WEIGHT`
    :returns: the active fixes, sorted
    :rtype: tuple
    :raises ValueError: if a name is not one of :data:`ALL_FIXES`

    """
    global _active, _msssim_weight, _gate_weight, _colour_knots

    _msssim_weight = (PUBLISHED_MSSSIM_WEIGHT if msssim_weight is None
                      else float(msssim_weight))
    _gate_weight = (PUBLISHED_GATE_WEIGHT if gate_weight is None
                    else float(gate_weight))
    _colour_knots = (COLOUR_KNOTS if colour_knots is None else int(colour_knots))

    spec = (spec or 'none').strip().lower()
    if spec == 'all':
        names = set(ALL_FIXES)
    elif spec == 'none':
        names = set()
    else:
        names = {part.strip() for part in spec.split(',') if part.strip()}
        unknown = names - set(ALL_FIXES) - set(FEATURES)
        if unknown:
            raise ValueError(
                'unknown fix(es) %s; choose from %s, the features %s, '
                'or "all"/"none"'
                % (', '.join(sorted(unknown)), ', '.join(ALL_FIXES),
                   ', '.join(FEATURES)))
        if 'gates' in names and 'fusion' not in names:
            raise ValueError(
                'the "gates" feature requires "fusion": without it two neutral '
                'branches compose to S = 2 rather than the identity, so '
                'switching a filter off is not free')

    _active = frozenset(names)
    return tuple(sorted(_active))


def enabled(name):
    """Is this fix active?

    :param name: one of :data:`ALL_FIXES` or :data:`FEATURES`
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
