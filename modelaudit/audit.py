"""modelaudit -- find the model defects that do not raise an exception.

Four classes of defect, each of which leaves a model that imports, trains and
produces plausible numbers, and none of which a normal test suite catches:

  DEAD_PARAM    a trainable parameter receives no gradient
  DEAD_MODULE   a submodule is built in __init__ and never called by forward
  INERT         a component does nothing in a trained checkpoint
  LOSS_SHARE    a loss term's weighted gradient contribution is negligible

Usage:

    from modelaudit import audit

    report = audit(
        build=lambda: MyNet(),
        forward=lambda m: m(torch.randn(1, 3, 64, 64)),
        ckpt="pretrained/model.pt",            # optional but by far the best check
        loss_terms=lambda m: {                  # optional
            "recon": l1(m(x), y),
            "perceptual": lpips(m(x), y),
        },
        weights={"recon": 1.0, "perceptual": 1e-3},
    )
    report.print()

Everything runs on CPU: one forward pass, and one backward per loss term. No
training, no dataset, no GPU. Only dependency is torch.

Measured base rates, so a hit can be read in proportion: a mechanical scan of 17
randomly sampled CVPR 2023 repositories found a verified starved parameter in 2
and a verified dead module in 3; a hand audit of 6 CVPR 2021 repositories found
dead modules in 4. A parameterless dead module is ordinary. A component inert in
a trained checkpoint is not.

WHAT IT CANNOT SEE, stated so a clean result is not over-read: code that runs
and computes the wrong thing (wrong sign, wrong normalisation, off-by-one);
anything requiring real data, since inputs here are random; defects that appear
only during training, such as a component that saturates after convergence;
config-conditional deadness, since it reports on the configuration you give it;
and every evaluation-side problem - metric implementation, test-set leakage,
checkpoint selection. Those need a person.
"""
import math
import torch
import torch.nn as nn

ZERO = 1e-12          # |grad| below this counts as no gradient
REL_CONST = 1e-5      # out-std/in-std below this counts as a constant activation
INIT_TOL = 1.02       # slack when testing "still inside the init interval"
SHARE = 0.05          # loss term below this share of the largest term = decorative


# ---------------------------------------------------------------- report ----
class Report:
    def __init__(self):
        self.dead_params = []      # (name, reason)
        self.dead_modules = []     # (name, type)
        self.inert = []            # (name, reason)
        self.loss_shares = {}      # name -> (grad_norm, share)
        self.errors = []           # (stage, message)
        self.init_only = []        # dead at init but alive after perturbation
        self.used_functionally = []  # hook silent but params got gradient -> NOT dead
        self.n_params = 0
        self.n_modules = 0

    def unexplained_dead(self):
        """Dead params NOT accounted for by a never-fired module.

        A module that never ran trivially has no gradient, and class 2 already
        reports it. What is left over is the more interesting case: the module
        DID run, but the parameter still got nothing -- a torch.where condition,
        a comparison, a stray .detach(), argmax/sign. Reported separately so the
        two classes are not double-counted.
        """
        dead_mods = {n for n, _ in self.dead_modules}
        out = []
        for n, r in self.dead_params:
            owner = n.rsplit(".", 1)[0]
            if owner in dead_mods or any(owner.startswith(d + ".") for d in dead_mods):
                continue
            out.append((n, r))
        return out

    def flags(self):
        return {
            "DEAD_PARAM": bool(self.dead_params),
            "DEAD_MODULE": bool(self.dead_modules),
            "INERT": bool(self.inert),
            "LOSS_SHARE": any(s < SHARE for _, s in self.loss_shares.values()),
            "DEAD_PARAM_UNEXPLAINED": bool(self.unexplained_dead()),
        }

    def print(self):
        print(f"params={self.n_params} leaf_modules={self.n_modules}")
        for stage, msg in self.errors:
            print(f"  ERROR [{stage}] {msg}")
        for k, v in self.flags().items():
            print(f"  {k}: {'HIT' if v else 'clean'}")
        for n, r in self.dead_params:
            tag = "dead param  " if (n, r) in self.unexplained_dead() else "dead(module)"
            print(f"    {tag} {n}: {r}")
        if self.used_functionally:
            print(f"    [{len(self.used_functionally)} modules called functionally "
                  f"(hook silent, params got gradient), not counted dead]")
        for n, t in self.dead_modules:
            print(f"    never fired  {n} ({t})")
        if self.init_only:
            print(f"    [{len(self.init_only)} params dead at init only (zero-init artefact), "
                  f"not counted: {self.init_only[:4]}...]")
        for n, r in self.inert:
            print(f"    inert        {n}: {r}")
        for n, (g, s) in sorted(self.loss_shares.items(), key=lambda x: -x[1][1]):
            print(f"    loss term    {n}: |grad|={g:.4g} share={s:.3%}")


# Modules whose parameters PyTorch uses FUNCTIONALLY, so __call__ -- and hence any
# forward hook -- never fires even though the weights are fully live. Not defects.
FUNCTIONAL_ARTIFACTS = ("NonDynamicallyQuantizableLinear",)  # MultiheadAttention.out_proj


# ------------------------------------------------------- 2: dead modules ----
def _leaf_modules(model):
    """Leaf modules only. A container that never fires is implied by its leaves."""
    return [(n, m) for n, m in model.named_modules()
            if n and not list(m.children())]


def _instrument(model, fired, acts):
    handles = []
    for name, mod in _leaf_modules(model):
        def hook(m, inp, out, name=name):
            fired.add(name)
            t = out if torch.is_tensor(out) else (
                out[0] if isinstance(out, (tuple, list)) and out and torch.is_tensor(out[0]) else None)
            if t is not None and t.is_floating_point() and t.numel() > 1:
                i = inp[0] if inp and torch.is_tensor(inp[0]) else None
                iscale = i.detach().float().std().item() if (i is not None and i.numel() > 1) else 1.0
                acts[name] = (t.detach().float().std().item(),
                              t.detach().float().mean().item(), iscale)
        handles.append(mod.register_forward_hook(hook))
    return handles


# ---------------------------------------------------------- 3: inertness ----
def _init_bound(mod):
    """Default init interval half-width for Linear/Conv: 1/sqrt(fan_in).

    nn.Linear and nn.ConvNd both use kaiming_uniform_(a=sqrt(5)), which reduces
    to U(-1/sqrt(fan_in), +1/sqrt(fan_in)). A weight still entirely inside that
    interval, with the variance uniform predicts, most likely never moved.
    """
    if isinstance(mod, nn.Linear):
        fan_in = mod.in_features
    elif isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d,
                          nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        fan_in = mod.in_channels * int(torch.tensor(mod.kernel_size).prod())
    else:
        return None
    return 1.0 / math.sqrt(fan_in) if fan_in > 0 else None


def _check_inert(model, acts, rep, have_ckpt, acts2=None):
    # (a) and (b) compare against INITIALISATION, so they only mean anything once
    # real weights are loaded. On a freshly built model every layer is trivially
    # inside its init interval and every scale is trivially at its init value.
    if not have_ckpt:
        rep.errors.append(("inert", "no checkpoint supplied: checks 3a/3b (init-interval, "
                                    "pinned scales) skipped; only constant-activation 3c ran"))
    if have_ckpt:
      # (a) weights that never left their initialisation interval
      for name, mod in model.named_modules():
          b = _init_bound(mod)
          w = getattr(mod, "weight", None)
          if b is None or w is None or w.numel() < 16:
              continue
          mx = w.detach().abs().max().item()
          if mx > b * INIT_TOL:
              continue
          # uniform on (-b,b) has std b/sqrt(3); trained weights rarely match
          expect = b / math.sqrt(3)
          got = w.detach().std().item()
          if abs(got - expect) / expect < 0.05:
              rep.inert.append((f"{name}.weight",
                                f"inside init interval (max|w|={mx:.4g} <= 1/sqrt(fan_in)={b:.4g}), "
                                f"std {got:.4g} matches uniform init {expect:.4g} -- suspected untrained"))

      # (b) scale-like parameters pinned at exactly 0 or 1
      for name, p in model.named_parameters():
          if p.numel() > 64:
              continue
          v = p.detach().flatten()
          for target in (0.0, 1.0):
              if torch.allclose(v, torch.full_like(v, target), atol=1e-7):
                  rep.inert.append((name, f"all elements == {target:g} (scale factor is a no-op)"))
                  break

    # (c) activations that came out constant.
    # Relative to the module's own input scale: a small-but-varying activation is
    # not inert, and an absolute threshold flags every low-magnitude branch.
    for name, (std, mean, iscale) in acts.items():
        # Require the same verdict on the SECOND, independently-sampled forward.
        # Stochastic depth / dropout can zero a whole branch at small batch size,
        # which looks exactly like a dead component on any single pass.
        if acts2:
            a2 = acts2.get(name)
            if a2 is None:
                continue
            std2, _, iscale2 = a2
            steady = (std2 == 0.0) if std == 0.0 else (iscale2 > 0 and std2 / iscale2 < REL_CONST)
            if not steady:
                continue
        if std == 0.0:
            rep.inert.append((name, f"output EXACTLY constant at {mean:.4g} (std=0) -- no-op"))
        elif iscale > 0 and std / iscale < REL_CONST:
            rep.inert.append((name, f"output ~constant at {mean:.4g} "
                                    f"(std={std:.3g}, {std/iscale:.2g}x its input scale) -- suspected saturated"))


# ----------------------------------------------------------------- audit ----
def audit(build, forward, ckpt=None, loss_terms=None, weights=None, train_mode=True):
    """Run checks 1-4.

    build()            -> nn.Module
    forward(model)     -> a tensor (loss, or any output; sum() is used as loss)
    ckpt               -> optional state_dict (already loaded, or path/dict to load)
    loss_terms(model)  -> optional dict name -> scalar tensor, for check 4
    weights            -> optional dict name -> float multiplier for those terms
    train_mode         -> run forward in train() (default) or eval()
    """
    rep = Report()
    model = build()
    # train mode by default: training-only branches are part of what forward does,
    # and calling a module dead because eval() skipped it would be a false positive.
    model.train(train_mode)

    if ckpt is not None:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False) if isinstance(ckpt, str) else ckpt
        for k in ("state_dict", "model", "net", "params"):
            if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
                sd = sd[k]
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            rep.errors.append(("ckpt", f"{len(missing)} missing keys, e.g. {missing[:3]}"))

    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    rep.n_params = len(params)
    leaves = _leaf_modules(model)
    rep.n_modules = len(leaves)

    fired, acts = set(), {}
    handles = _instrument(model, fired, acts)
    try:
        out = forward(model)
    finally:
        for h in handles:
            h.remove()

    # 1 -- parameters that receive no gradient
    loss = out if (torch.is_tensor(out) and out.ndim == 0) else _to_scalar(out)
    model.zero_grad(set_to_none=True)
    loss.backward(retain_graph=loss_terms is not None)
    dead0 = _dead_now(params)

    # 2 -- declared but never called.
    # Run AFTER the backward, because a fired hook is not the ground truth: PyTorch
    # invokes plenty of modules functionally (MultiheadAttention.out_proj, and any
    # `nn.Embedding` read as `.weight`), so the hook stays silent while the weights
    # are fully live. A module whose parameters received gradient is NOT dead.
    for name, mod in leaves:
        if name in fired:
            continue
        own = [p for p in mod.parameters(recurse=False) if p.requires_grad]
        if own and any(p.grad is not None and p.grad.abs().max().item() > ZERO for p in own):
            rep.used_functionally.append((name, type(mod).__name__))
            continue
        if not own and not list(mod.buffers(recurse=False)):
            rep.dead_modules.append((name, type(mod).__name__ + " [no params: cosmetic]"))
        elif not own:
            rep.dead_modules.append((name, type(mod).__name__ + " [frozen/no trainable params]"))
        else:
            rep.dead_modules.append((name, type(mod).__name__))

    # A model is audited at INITIALISATION, and some inits make a parameter dead
    # only momentarily: anything behind a zero-initialised layer (zero-init gamma,
    # zero-init residual gate) gets no gradient on step 0 but is live from step 1.
    # Re-test after jittering every parameter. Structural deadness -- an unreachable
    # module, a torch.where condition, a .detach() -- survives the jitter; deadness
    # that is only an artefact of the init does not. Report the intersection.
    acts2 = {}
    dead1 = _dead_after_perturb(build, forward, ckpt, train_mode, rep, acts2)
    if dead1 is None:
        rep.errors.append(("perturb", "perturbation re-test failed; class-1 hits are "
                                      "AT-INIT only and may include zero-init artefacts"))
        confirmed = dead0
    else:
        rep.init_only = sorted(set(dead0) - dead1)
        confirmed = {n: r for n, r in dead0.items() if n in dead1}
    rep.dead_params = sorted(confirmed.items())

    # 3 -- inert under the loaded weights
    _check_inert(model, acts, rep, ckpt is not None, acts2)

    # 4 -- per-loss-term weighted gradient share
    if loss_terms is not None:
        terms = loss_terms(model)
        norms = {}
        for name, t in terms.items():
            model.zero_grad(set_to_none=True)
            w = (weights or {}).get(name, 1.0)
            if not (torch.is_tensor(t) and t.requires_grad and t.grad_fn is not None):
                # a term detached from the graph trains nothing at all: 0 gradient,
                # which is the strongest possible version of this defect class
                rep.errors.append(("loss", f"term '{name}' has no grad_fn -- "
                                           "detached from the graph, contributes no gradient"))
                norms[name] = 0.0
                continue
            try:
                (w * t).backward(retain_graph=True)
            except RuntimeError as e:
                rep.errors.append(("loss", f"term '{name}' backward failed: {str(e)[:100]}"))
                norms[name] = 0.0
                continue
            norms[name] = math.sqrt(sum(float(p.grad.pow(2).sum())
                                        for _, p in params if p.grad is not None))
        top = max(norms.values())
        if top <= 0:
            # every term contributed nothing: the terms were not wired to the graph.
            # That is a harness failure, not a finding about the repo.
            rep.errors.append(("loss", "ALL loss terms had zero gradient -- treating "
                                       "class 4 as NOT MEASURED, not as a hit"))
            rep.loss_shares = {}
        else:
            rep.loss_shares = {n: (v, v / top) for n, v in norms.items()}

    return rep


def _dead_now(params):
    out = {}
    for n, p in params:
        if p.grad is None:
            out[n] = "grad is None (not reached by autograd)"
        elif p.grad.abs().max().item() <= ZERO:
            out[n] = "grad is all-zero"
    return out


def _dead_after_perturb(build, forward, ckpt, train_mode, rep, acts_out, scale=0.05):
    """Rebuild, jitter every parameter, and see which params are STILL dead."""
    try:
        m2 = build()
        m2.train(train_mode)
        if ckpt is not None and not isinstance(ckpt, str):
            m2.load_state_dict(ckpt, strict=False)
        with torch.no_grad():
            for p in m2.parameters():
                # std() of a 1-element tensor is undefined and warns; those are
                # exactly the scalar gates this check most wants to perturb.
                sd = p.detach().float().std().item() if p.numel() > 1 else 0.0
                p.add_(torch.randn_like(p) * (scale * (sd if sd > 0 else 1.0)))
        ps = [(n, p) for n, p in m2.named_parameters() if p.requires_grad]
        fired2, acts2 = set(), {}
        hs = _instrument(m2, fired2, acts2)
        try:
            out2 = forward(m2)
        finally:
            for h in hs:
                h.remove()
        m2.zero_grad(set_to_none=True)
        _to_scalar_or_self(out2).backward()
        acts_out.update(acts2)
        return set(_dead_now(ps))
    except BaseException as e:
        rep.errors.append(("perturb", f"{type(e).__name__}: {str(e)[:120]}"))
        return None


def _to_scalar_or_self(out):
    return out if (torch.is_tensor(out) and out.ndim == 0) else _to_scalar(out)


def _to_scalar(out):
    """Reduce whatever forward returned to one scalar to backprop from.

    Recurses through nested lists/tuples/dicts and keeps ONLY graph-connected
    tensors: models routinely return detached logging scalars, index tensors and
    masks alongside the real outputs, and summing those kills the backward.
    """
    parts = []

    def walk(o, depth=0):
        if depth > 6:
            return
        if torch.is_tensor(o):
            if o.is_floating_point() and o.requires_grad:
                parts.append(o.float().sum())
        elif isinstance(o, dict):
            for v in o.values():
                walk(v, depth + 1)
        elif isinstance(o, (list, tuple, set)):
            for v in o:
                walk(v, depth + 1)

    walk(out)
    if not parts:
        raise TypeError(f"no graph-connected float tensor found in {type(out).__name__} output")
    return sum(parts)


# ------------------------------------------------------------ self-check ----
def _demo():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.used = nn.Linear(8, 8)
            self.orphan = nn.Linear(8, 8)          # class 2: never called
            self.gate = nn.Parameter(torch.zeros(4))  # class 1+3: only in a condition
            self.scale = nn.Parameter(torch.ones(4))  # class 3: identity scale
            self.frozen = nn.Linear(64, 8)         # class 3: left at init
            # zero-init pair: lin1 is dead ONLY at init, alive from step 1.
            # Must land in init_only, NOT in dead_params.
            self.lin1 = nn.Linear(8, 8)
            self.lin2 = nn.Linear(8, 8)
            nn.init.zeros_(self.lin2.weight); nn.init.zeros_(self.lin2.bias)

        def forward(self, x):
            h = self.used(x)
            # gate is used only as a torch.where CONDITION -> no gradient
            h = h + self.lin2(self.lin1(h))
            return torch.where(self.gate.repeat(2) > 0, h, h * 0.5) * self.scale.repeat(2)

    # a stand-in "released checkpoint": trained weights everywhere except the
    # parts the defects left untouched, so checks 3a/3b have something to find.
    ck = Net().state_dict()
    ck["used.weight"] = ck["used.weight"] * 50

    rep = audit(Net, lambda m: m(torch.randn(2, 8)), ckpt=ck,
                loss_terms=lambda m: {"main": m(torch.randn(2, 8)).pow(2).mean(),
                                      "decor": m(torch.randn(2, 8)).abs().mean()},
                weights={"main": 1.0, "decor": 1e-4})
    rep.print()
    names = [n for n, _ in rep.dead_params]
    assert "gate" in names, names                       # 1: where-condition kills grad
    assert "orphan" in [n for n, _ in rep.dead_modules]  # 2
    inert = [n for n, _ in rep.inert]
    assert "scale" in inert and "frozen.weight" in inert, inert   # 3
    assert "used.weight" not in inert, "trained weight must NOT be flagged"
    # the zero-init pair must be classified as init-only, not as a real defect
    assert "lin1.weight" in rep.init_only, rep.init_only
    assert "lin1.weight" not in names, "zero-init artefact must not count as dead"
    assert "frozen.weight" in names, names   # structurally dead survives the jitter
    assert rep.loss_shares["decor"][1] < SHARE           # 4
    print("\nself-check OK")


if __name__ == "__main__":
    _demo()
