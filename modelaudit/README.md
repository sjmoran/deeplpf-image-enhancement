# modelaudit

Find the model defects that do not raise an exception.

One forward pass, one backward per loss term, CPU only, `torch` the sole
dependency. Point it at any PyTorch model.

```python
from modelaudit import audit

report = audit(
    build=lambda: MyNet(),
    forward=lambda m: m(torch.randn(1, 3, 64, 64)),
    ckpt="pretrained/model.pt",           # optional, and the strongest check
    loss_terms=lambda m: {"recon": ..., "perceptual": ...},   # optional
    weights={"recon": 1.0, "perceptual": 1e-3},
)
report.print()
```

## What it checks

| Check | Finds |
|---|---|
| `DEAD_PARAM` | a trainable parameter that receives no gradient |
| `DEAD_MODULE` | a submodule built in `__init__` and never called by `forward` |
| `INERT` | a component that does nothing in a *released checkpoint* |
| `LOSS_SHARE` | a loss term whose weighted gradient contribution is negligible |

`DEAD_PARAM` catches the whole family of silently-severed gradients: custom
autograd written as a method on an `nn.Module` (PyTorch only calls `backward` on
`autograd.Function` subclasses), a value used as a `torch.where` **condition**
(booleans carry no gradient), a stray `.detach()`, an `argmax` or `sign` on a
path the paper says is learned.

`INERT` is the one that needs trained weights and finds what reading cannot:
activations that came out constant, scale factors pinned at 0 or 1, and weight
blocks still inside their initialisation interval — `nn.Linear(n, m)` initialises
uniform on ±1/√n, so a weight that never left that interval most likely never
trained. This is the check worth running first on any released checkpoint.

## Why these four

Each check targets a failure that is silent by construction — the code imports,
trains, and produces a working model, and the only symptom is something that
never changes. They are the failures a test suite does not catch by accident,
because nothing throws.

`DEAD_PARAM` and `DEAD_MODULE` are visible from a careful reading, given enough
time. `INERT` and `LOSS_SHARE` generally are not: a component can run on every
forward pass and still contribute nothing if training drove it into a saturated
region, and a loss term can appear in the paper's equation, appear in the code,
and still be too lightly weighted to move a parameter. Both need measurement.

## Calibration — why the naive version of each check is wrong

These are the false positives that had to be engineered out before the tool was
usable on real code. Each one produces confident, wrong output in the naive
version of the check:

- **Zero-init designs.** A layer behind a zero-initialised layer gets no gradient
  on step 0 and trains fine from step 1. One repository produced 160 false hits
  this way. Fixed by re-testing after jittering every parameter and reporting
  only the intersection.
- **Functionally-called modules.** `MultiheadAttention.out_proj`, and any
  `nn.Embedding` read as `.weight`, never fire a forward hook although the
  weights are fully live. Fixed with a better rule: *a module whose parameters
  received gradient is not dead.*
- **Stochastic depth.** `drop_path` at batch size 1 zeroes a whole branch, which
  looks exactly like a dead component. Fixed by requiring the same verdict on
  two independently sampled forward passes.
- **Init-interval checks without a checkpoint** flag every layer in an untrained
  model. Gated on a checkpoint being supplied.
- **Absolute activation thresholds** flag every low-magnitude branch. The check
  is relative to each module's own input scale.

## Reading a result in proportion

Measured base rates, so a hit is neither ignored nor over-read:

| Sample | Starved parameter | Dead module |
|---|---|---|
| 17 random CVPR 2023 repos (mechanical) | 2 | 3 |
| 6 random CVPR 2021 repos (by hand) | — | 4 |

A parameterless dead module — an unused `Dropout`, an inherited `avgpool` — is
ordinary and worth a line in a changelog at most. **A component inert in a
released checkpoint is not ordinary**, and is worth investigating properly,
because it means the shipped model does not contain something the paper
describes.

## What it cannot see

Stated plainly, so a clean result is not over-read:

- **Code that runs and computes the wrong thing** — wrong sign, wrong
  normalisation constant, off-by-one indexing. Every check here asks "does
  gradient flow", never "is the answer right".
- **Data-dependent defects.** Inputs are random tensors; branches gated on real
  data statistics are never taken.
- **Defects that appear only at convergence** — a component that saturates
  during training is invisible in an untrained or freshly-loaded model.
- **Config-conditional deadness.** It reports on the configuration you hand it;
  a module dead in one config may be live in another.
- **Everything on the evaluation side** — metric implementation, test-set
  leakage, checkpoint selection, results copied across protocols. Those need a
  person.

## Self-check

```bash
python3 modelaudit/audit.py
```

Builds a synthetic model containing one of each defect plus three negative
controls, and asserts every check fires exactly where it should.
`tests/test_modelaudit.py` then runs it against this repository's model and
asserts the results are stable, including that modules which genuinely run are
never reported.
