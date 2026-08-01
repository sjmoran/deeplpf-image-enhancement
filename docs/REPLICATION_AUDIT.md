# Replication audit: do the post-CVPR2020 changes preserve the paper results?

**Date:** 2026-07-31
**Audited commit:** `fd05d12` (`origin/master`, 0 ahead / 0 behind — this clone mirrors
the GitHub repo exactly)
**Question:** the code in `sjmoran/deeplpf-image-enhancement` has been modernised
since the CVPR 2020 release. Do any of those changes alter the model's numerical
output or the reported PSNR/SSIM — i.e. are they non-impacting for paper
reproduction?

## Scope and definitions

"Paper reproduction" here means the **batch=1 inference / eval path** — the path
that runs the pretrained `adobe_dpe` checkpoint and reports per-image PSNR/SSIM.
That is what `tests/test_replication.py` guards and what produces the paper-era
numbers.

"Non-impacting" means: for every input the code previously handled correctly, the
change is behaviour-preserving on the batch=1 path — output values and reported
metrics are numerically identical (modulo library-version floating-point drift,
which is bounded empirically below).

Two headline results are distinct:

- **Per-image checkpoint faithfulness** — the pretrained model reproduces its 2020
  per-image PSNR to within 0.5 dB. This is verified (see Evidence).
- **The aggregate 23.90 dB / 0.911 SSIM** — a test-set average over the 500-image
  Adobe-DPE test split. This is *not* re-confirmed here; it requires running the
  prepared test set. Nothing in the audited changes threatens it, but the tests
  verify per-image fidelity, not the aggregate.

## Method

Two independent lines of evidence:

1. **Static, per-hunk audit.** Every merged PR that touches a code file was read
   diff-by-diff and each computational hunk classified as behaviour-preserving or
   not, on the batch=1 path. Diffs pulled via `gh pr diff <n>`.
2. **Empirical guard.** The full test suite — including the checkpoint-faithfulness
   test — was run on the audited commit, which is *after* all of these PRs.

## Verdict summary

Every code change since the original CVPR-2020 code is **non-impacting for paper
replication at batch=1**.

| PR | Change | Files | Verdict (batch=1 / eval / inference) |
|----|--------|-------|--------------------------------------|
| #31 | Vectorise parametric filter heads | model, main, conftest | Non-impacting — mathematically equivalent at B=1, hunk-by-hunk |
| #26 | Device-agnostic (CPU/MPS/CUDA) | model, metric, util, main | Non-impacting — pure device placement, no dtype/math change |
| #27 | SSIM API fix (`multichannel`→`channel_axis`) | util | SSIM preserved — same `data_range`, window, gaussian defaults |
| #25 | Docstrings + dead-code removal | all code | Non-computational — removals were provably-unused locals |
| #30 | CI + faithfulness guard | util, tests, ci | Non-computational — only adds RGBA→RGB for formerly-broken inputs |
| #37 | LICENSE consistency | all code | Non-computational — license header comments only |
| #38 | `--crop_size` for batch>1 | data, main | Non-impacting — opt-in, doubly-gated off the eval path |
| #28 | Dependency bump (torch 1.7→2.13) | requirements | Library-level — bounded to <0.5 dB by the passing replication test |

PRs touching only README / docs / prep-scripts / test files (#29, #32, #33, #34,
#35) are non-computational by construction and not detailed further.

## Per-PR detail

### PR #38 — `--crop_size` for batch>1 training

Purely additive and opt-in. The new random-crop block in `data.py::__getitem__`
is doubly-gated: it runs only when `not self.is_valid` (training, not validation)
**and** `self.crop_size` is truthy. `crop_size` defaults to `0` in `main.py`, which
is converted to `None` before reaching the training `Dataset`; validation and
inference datasets never receive it, and inference has its own separate branch that
never reaches the crop code. With `crop_size` unset, `__getitem__` is behaviourally
byte-identical to the pre-PR code — the only new state is `self.crop_size = None`.
When the crop *does* run (batch>1 training only), it applies the same box to input
and target and clamps to `min(crop_size, width, height)`.

Paths that reach the new crop code:

| Path | Reaches crop code? | Why |
|------|--------------------|-----|
| Inference (`is_inference=True`) | No | Separate branch |
| Validation (`is_valid=True`) | No | Fails the `if not is_valid` gate |
| Test-set eval | No | Batch-1 / valid-style load; fails the same gate |
| Batch=1 training (paper repro) | No | `crop_size` defaults to `0` → `None` |
| Batch>1 training (new opt-in) | Yes | The only new behaviour |

### PR #31 — Vectorise the parametric filter heads (highest risk)

The only PR that rewrote the forward math. Each hunk in `model.py` was checked for
mathematical equivalence at batch=1:

- **`get_cubic_mask`** — unrolled per-channel assignments replaced by a
  `for c in range(3)` loop with `R[:, 20*c+k].view(-1,1,1)` broadcasting.
  Coefficient-to-channel mapping (0-19→R, 20-39→G, 40-59→B) preserved. Equivalent.
- **`get_inverted_mask`** — scalar `.all()` branching replaced by evaluating all
  four branches and selecting with `torch.where`. Each of the four pre-clamp
  formulas and their per-branch clamps (`[1, max_scale]` / `[0, 1]`) reproduced
  exactly; divisors `d1..d6 = tanh01(...) ∈ (0,1)` are strictly positive, so no new
  division-by-zero, and `torch.where` does not propagate unselected NaN/inf.
  Equivalent.
- **`get_graduated_mask`** — clamp reorders (`clamp(maximum(x, yd), max=1)` etc.),
  `G[0,k]`→`G[:,k]` indexing with `.view(-1,1,1)`, and `cat(dim=0).unsqueeze(0)`→
  `stack(dim=1)` combine. All element-wise / broadcast; same channel order and
  values. Equivalent.
- **`get_mask` / `get_elliptical_mask`** — dropped `.unsqueeze(0)` and
  `cat(dim=0).unsqueeze(0)`→`stack(dim=1)`; ellipse equation, angle/radius terms
  unchanged. Equivalent.
- **Downstream fuse** — unchanged; element-wise on `(1,3,H,W)`, no reduction over
  batch anywhere.

`main.py` / `conftest.py` are plumbing: `--batch_size` defaults to 1 (preserving
prior behaviour) and wires only the *training* loader; eval/test/validation and
inference loaders are hard-coded to batch_size=1. `conftest.py` only adds the repo
root to `sys.path` for tests.

**Batch>1-only footnote:** the vectorised `get_inverted_mask` evaluates all four
branches, so *gradients* flow through unselected branches during batch>1 training —
a training-dynamics difference vs the old scalar branching. It cannot change forward
values and is irrelevant at batch=1. The forward-only batch-invariance test would
not catch it. Worth knowing only if you train at batch>1 and compare convergence.

### PR #26 — Device-agnostic (CPU / MPS / CUDA)

Every hunk changes only device placement or memory-cache calls. `.cuda()` →
`.to(device)` / `type_as` / `new_zeros`; `torch.load(..., map_location=device)`;
`empty_cache()` guarded behind `torch.cuda.is_available()`. No dtype, reduction-order,
or constant changes — the RGB→LAB matrices, coordinate `arange`s, and SSIM window are
identical values on whichever device the model sits on. Numerically neutral on a given
device, including CPU.

### PR #27 — SSIM API fix

`structural_similarity(..., multichannel=True)` → version-detected
`channel_axis=-1` (skimage ≥0.19), falling back to `multichannel=True` on older
skimage. Images are HxWx3; in skimage `multichannel=True` is defined as
`channel_axis=-1`, so channel handling is identical. `data_range`, `win_size=11`,
and `gaussian_weights=True` (sigma default 1.5) are unchanged. Reported SSIM is
preserved across both branches — a pure API-compatibility shim.

### PR #25 — Docstrings + dead-code removal

Every hunk is a docstring, comment, argparse help string, or removal of a
provably-unused local. Removed: `min_scale = 0` (two sites), the
`right_x/left_x/right_y/left_y` block, a redundant class-body
`import torch.nn.functional as F` (shadowing the module-level import at line 25),
and a write-only `shape = x.shape`. All confirmed to have zero references. argparse
`default=` values unchanged. No executable expression altered.

### PR #30 — CI + faithfulness guard

CI workflow, `conftest.py` sys.path shim, and `tests/*` are non-runtime. The one
runtime change is `util.py::load_image`, which now drops the alpha channel on
4-channel inputs (`if img.ndim == 3 and img.shape[2] == 4: img = img[:, :, :3]`).
For grayscale and 3-channel RGB inputs — i.e. every valid FiveK input — the returned
array is bit-identical to before. The change only rescues formerly-broken RGBA
inputs; it does not alter any previously-valid computation.

### PR #37 — LICENSE consistency

New `LICENSE` file (data, not imported) plus two `#`-prefixed header comment lines
per `.py` file ("BSD 0-Clause" → "BSD-3-Clause"). No executable line changed.

### PR #28 — Dependency bump

`torch 1.7.1 → 2.13.0`, `scikit-image 0.18.1 → 0.26.0`, `numpy 1.22 → 2.4.6`, plus
removal of a bogus `skimage==0.0` stub. This is a library-level change, not code,
and is the one drift source that cannot be proven non-impacting by reading a diff.
It is the "PyTorch version differences (paper used 1.7.1)" caveat noted in
`docs/ADOBE_DPE_DATASET.md`. The passing replication test empirically bounds the
entire torch 1.7→2.x jump to <0.5 dB per-image.

## Evidence

Full suite on the audited commit (`fd05d12`):

```
$ python3 -m pytest -q
........                                                                 [100%]
8 passed
```

Faithfulness detail:

- `test_replication.py::test_pretrained_checkpoint_reproduces_reference` — PASSED.
  Runs the shipped `adobe_dpe` epoch-424 checkpoint through the inference pipeline
  on the bundled examples; 4 of 5 reproduce their 2020 PSNR to within 0.5 dB (a4774
  is run but not value-checked — its stored input file is mismatched; see the test's
  module docstring).
- `test_batch_invariance.py` (both cases) — PASSED. Per-image forward results are
  batch-invariant (max diff < 1e-5) and the batch>1 training step works.

## Conclusion

The cleanup and feature changes since CVPR 2020 are non-impacting for paper
replication on the batch=1 inference/eval path. Static per-hunk analysis and the
empirical faithfulness guard agree.

Not yet re-confirmed: the **aggregate 23.90 dB / 0.911 SSIM** over the full
500-image Adobe-DPE test split. That requires a prepared test set (see
`docs/ADOBE_DPE_DATASET.md`) and is the outstanding Tier-2 check.

## Limitations

- The audit covers the batch=1 path. It does not certify batch>1 *training
  dynamics* — see the PR #31 gradient-path footnote.
- Library-version floating-point drift (PR #28) is bounded empirically (<0.5 dB per
  image on the bundled examples), not proven identical.
- The clone is shallow (depth 1); the audit reads current file state and PR diffs
  via `gh`, not full local history.
