# DeepLPF: research directions

A survey of where DeepLPF sits against post-2020 work on MIT-Adobe FiveK, and
what is worth building on top of it. Written against the code in this repo
(`model.py`, `unet.py`, `data.py`, `main.py`, `metric.py`), not against the
paper alone.

Conjecture is labelled as such throughout. Published numbers carry a citation;
anything without one is an expectation, not a result.

## 1. What the code actually does

**Pipeline.** `unet.UNetModel` runs a 5-level U-Net (16-32-64-128-128 channels,
reflection-padded double 3x3 convs) that outputs a **3-channel image**
(`out + x_in`). `final_conv` (3 -> 64, one 3x3 conv) produces the "64-channel
feature map". `DeepLPFParameterPrediction` splits that into `img = x[:, 0:3]`
and `feat = x[:, 3:64]`. Each of the three heads concatenates `feat` with the
current image, bilinearly resizes to 300x300, runs 4 stride-2 convs + 3
max-pools + global average pool + dropout(0.5) + FC, and regresses a per-image
parameter vector:

| Head | Parameters | Application |
|---|---|---|
| `CubicFilter` | 60 coefficients (20 per channel) | cubic in normalised (x, y, intensity), applied as residual `clamp(img + f)` |
| `GraduatedFilter` | 24 = 3 lines x (invert bit, slope, two band offsets) + 9 per-channel scales in [0, 2] | `tanh01(y - (slope*x + offset + d))` — a sigmoid over normalised coordinates |
| `EllipticalFilter` | 24 = 3 ellipses x (centre, semi-axes, angle) + 9 per-channel scales in [0, 2] | inside the ellipse, scale decays linearly from centre to 1 at the boundary via a hard `where` |

Fusion: `S = clamp(s_g + s_e, 0, 2)`, `Y3 = clamp(Y2 * S)`, `Y = clamp(Y3 + Y1)`.

**Loss** (`DeepLPFLoss`): L1 in rescaled CIELab + `1e-3 * (1 - MS-SSIM on L)`.
**Training** (`main.py`): Adam 1e-4, no LR schedule, batch 1 by default
(batch > 1 requires `--crop_size`), flips only, best-validation-PSNR checkpoint.
**Data** (`data.py`): FiveK DPE rendering, long edge ~512, paired.
**Eval** (`metric.py`): per-image PSNR (numpy, clipped) and skimage SSIM at
batch 1.

## 2. Implementation notes

Points about the current implementation that bear on the directions below,
recorded so the proposals can be read against what the code actually does.

1. **The heads receive a 3-channel image, not decoder features.** `feat` is a
   single 3x3 convolution of the U-Net's 3-channel output, so the backbone's
   representation is compressed to RGB before parameter prediction. This is the
   main architectural constraint on *localised* filters, which need content cues
   to place themselves, and it is what Rank 1 addresses. Note that passing an
   image between stages was normal practice for the period - DPE, HDRNet and
   CURL all do a version of it.
2. **Neutral fusion is not identity.** Both scaling maps are neutral at 1 and
   are summed, so `S = 2` when both filters are neutral. The `fusion` flag sums
   deviations instead.
3. **The graduated inversion indicators reach the loss twice through
   non-differentiable operations** - `torch.sign` and a `torch.where` condition -
   so both the `ste` and `blend` flags are needed for them to train.
4. **Filter supports are differentiable at opposite extremes.** The elliptical
   hard `where` gives gradient only through the inside branch; the graduated
   `tanh01` line is very soft. Neither support has a learnable sharpness.
5. **The head input is resized to a fixed 300x300** and the cubic polynomial is
   in normalised (x, y), tying the model to aspect ratio. Fine on FiveK, weaker
   as a general tool.
6. **The training recipe is 2020-era**: batch 1, no schedule, flips only,
   dropout 0.5 on the regressor. LUT papers that report higher numbers train at
   480p with crops, colour jitter and schedules, so part of the reported gap is
   recipe rather than model. Testable.
7. **Benchmark comparability.** DeepLPF is reported at 23.63, 23.90, 24.48 and
   24.73 under different protocols - see `docs/BENCHMARK_TABLE.md`. Any new
   comparison must state which protocol it uses.

## 3. Where the field is

Established results, with sources.

- **Global 3D-LUT family** (FiveK 480p, Expert C, 4500/500, AdaInt table):
  HDRNet 24.66, DeepLPF 24.73/0.916, CSRNet 25.17, 3D-LUT 25.29, AdaInt
  25.49/0.926 ([AdaInt](https://arxiv.org/abs/2204.13983)). Successors: SepLUT
  25.02–25.32 ([SepLUT](https://arxiv.org/abs/2207.08351)), CLUT 25.53 and
  ICELUT 25.27 ([ICELUT](https://arxiv.org/html/2403.19238v2)), LoR-LUT 25.53
  with 118k parameters ([LoR-LUT](https://arxiv.org/html/2602.22607)). The
  family has plateaued at roughly 25.3–25.5 dB for four years.
- **Interpretable local/region methods.** RSFNet (ICCV 2023) uses 10 named
  Lightroom-like filters times soft region maps with a ResNet18 backbone:
  **25.49/0.924 at 480p, matching AdaInt**, and 24.39 vs AdaInt's 24.24 at full
  resolution ([RSFNet](https://arxiv.org/abs/2303.08682)). LTMNet learns a grid
  of local tone curves ([LTMNet](https://github.com/SamsungLabs/ltmnet)).
  NamedCurves+ (TPAMI 2026) uses named-colour tone curves plus a transformer
  and explicitly claims interpretability and user adjustability
  ([arXiv 2607.08185](https://arxiv.org/abs/2607.08185)). LLF-LUT
  ([arXiv 2310.17190](https://arxiv.org/abs/2310.17190)) and CoTF
  ([CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Real-Time_Exposure_Correction_via_Collaborative_Transformations_and_Adaptive_Sampling_CVPR_2024_paper.html))
  both show global LUT + local component beats global alone.
- **Sequential operators.** NeurOp (ECCV 2022), 3 learned colour operators with
  scalar strengths and 28k parameters: 25.09 on FiveK-Lite vs DeepLPF's 23.63
  in the same table, 24.32 on FiveK-Dark
  ([NeurOp](https://arxiv.org/abs/2207.08080), Table 1).
- **Implicit neural representations.** NILUT
  ([arXiv 2306.11920](https://arxiv.org/abs/2306.11920)) and INRetouch
  ([arXiv 2412.03848](https://arxiv.org/html/2412.03848v4)) are style emulators
  and one-shot transfer, not fidelity improvers on the expert benchmark.
- **VLM + differentiable renderer.** VeraRetouch (2026) reports 26.85/0.939 on
  its own "FiveK-Bench" with RSFNet at 25.07 there — not comparable to the 480p
  protocol, but it signals where instruction-following retouching is heading
  ([VeraRetouch](https://arxiv.org/html/2604.27375)). RetouchLLM is in the same
  vein ([arXiv 2510.08054](https://arxiv.org/pdf/2510.08054)).
- **Diffusion, Retinex, flow, zero-reference.** Diff-Retinex, Reti-Diff,
  CLE-Diffusion, LLFlow, Zero-DCE(++) all target low-light (LOL, SICE) rather
  than expert retouching; none reports a competitive FiveK Expert-C PSNR
  ([Diff-Retinex](https://arxiv.org/pdf/2308.13164),
  [Zero-DCE++](https://li-chongyi.github.io/Proj_Zero-DCE++.html)).

**Net.** The ~0.8 dB gap between DeepLPF (24.73) and the front (25.5) has been
closed both by global LUTs *and* by an interpretable region-filter method.
Interpretability is not what costs the dB. Content-adaptivity of the local
support and the training recipe are.

## 4. Ranked directions

### Rank 1 — content-adaptive supports for parametric filters

Keep the three named filter families, but make each filter's *support* a
product of a geometric prior and a learned, bounded refinement:
`M_k = clamp(G_k(x, y; theta_k) * (1 + eps * R_k(feat)), 0, 1)`, where `G_k` is
the existing ellipse or graduated shape (user-draggable handles) and `R_k` is a
low-resolution map predicted from **real U-Net decoder features**
(`dconv_up1`, 16 ch — requires `UNetModel` to expose them, a few lines). Add a
small library of named per-region operators beyond RGB scaling (exposure,
contrast, saturation, temperature, shadows/highlights, as differentiable closed
forms in the RSFNet style), each with per-filter scalar strengths from the
existing FC heads, and compose additively:
`I' = I + sum_k M_k (F_k(I) - I)`. Replace the fusion with an
identity-preserving one (observation 2), and give each filter a learnable mask
sharpness (observation 6). Roughly 200 lines in `model.py` plus a backbone tap
in `unet.py`; loss unchanged initially.

**Why it fits.** This is DeepLPF's thesis — few, human-meaningful local
filters — pushed to where the evidence says the headroom is. RSFNet showed that
named filters times soft masks reach 25.49, but its masks are free-form CNN
outputs with no editable geometry. The hybrid keeps Lightroom-style semantics
("radial filter plus range-mask refinement") with a bound (`eps`) on how far
the mask can depart from the drawn shape, so interpretability degrades
gracefully and *measurably*.

**Expected gain.** Established: RSFNet is +0.76 dB over DeepLPF at 480p on the
same protocol. Conjecture: the hybrid recovers most of that (+0.5–0.8 dB, to
~25.3–25.5) while retaining geometric handles. Beating RSFNet outright is not
evidenced. The contribution is the controllable interpretability/fidelity
trade-off curve over `eps`, plus editability results — not a headline record.

**Cost and risk.** One to two weeks of implementation; training cost unchanged
(hours per run on one GPU at 480p). Medium risk: the free-form residual may
dominate, masks collapse to RSFNet-like blobs, and the geometry becomes
vestigial. That outcome is itself a publishable negative result about
geometric priors.

**Falsification.** 3D-LUT 480p protocol, PSNR/SSIM/dE, sweeping `eps` over
{0, 0.25, 0.5, 1, free}. Falsified if `eps = 0` lands within 0.2 dB of
`eps = free`, or if `eps = free` fails to exceed DeepLPF by 0.4 dB.
Interpretability check: a mask-edit consistency test (drag an ellipse centre,
confirm the pixel change is confined to the new support) plus a small user
study on editability against RSFNet.

### Rank 2 — fix the bottleneck and the recipe, then re-baseline

Feed the heads real decoder features instead of `final_conv(3->64)` of the RGB
output; use identity-preserving fusion; correct the ellipse `b3` typo; drop or
straight-through-estimate the invert bit; correct MS-SSIM and give it a weight
that matters; adopt a modern recipe (batch 8–16 with `--crop_size`, cosine LR,
colour jitter, EMA, train and evaluate under the 480p protocol). All within
`model.py`, `unet.py` and `main.py`.

**Why it fits.** No change to the interpretability claim. It establishes what
the *existing* filter set can do when not handicapped, and any Rank 1
comparison is unfair without it.

**Expected gain.** Conjecture only: +0.3–0.8 dB. The indirect evidence is the
recipe ablations in the LUT papers plus the size of observations 1 and 2 as
information bottlenecks. Days of work, low risk. **Falsified** if the retrained
model lands within 0.2 dB of 24.73 under the 480p protocol — which would say
the filter parametrisation itself is the ceiling, and would be a strong
motivation for Rank 1.

### Rank 3 — local monotone tone curves / low-rank LUTs blended by the masks

The cubic in (x, y, i) is the workhorse but is non-monotone, aspect-ratio tied
and hard to read. Replace it with K per-channel monotone piecewise-linear
curves (CURL-style) or a small low-rank 3D LUT (LoR-LUT reaches 25.53 with 118k
parameters), spatially blended by the graduated and elliptical masks: local
LUTs with geometric supports. Touches `CubicFilter` only.

Curves and LUTs are the most familiar retouching primitives, and making them
local with visible supports is the natural DeepLPF/CURL synthesis. Evidence
that local beats global: LLF-LUT and CoTF (established). Gain over Rank 1 is
conjecture. About a week; medium risk that it lands at SepLUT/LoR-LUT numbers
with locality contributing little, since expert edits on FiveK are mostly
global. Falsified if a K=1 (global curve) ablation equals the local version.

### Rank 4 — sequential application with a strength predictor

Wrap `DeepLPFParameterPrediction` in T = 2–3 shared-weight iterations, each
conditioned on the intermediate image, with a scalar strength per filter
exposed as a slider. Around 30 lines in `forward`. NeurOp reaches 25.09 vs
DeepLPF's 23.63 on FiveK-Lite (established, but confounded by different
operators and recipe), so the gain here is conjecture. Low cost, runtime times
T — DeepLPF is already 32 ms at 480p against roughly 1–2 ms for LUTs. Risk of
training instability with clamps at every stage. Falsified if T=1 and T=3
differ by under 0.15 dB.

### Rank 5 — few-shot expert or user adaptation via the parameter heads

FiveK has five experts. Train on C, then adapt the FC heads only (a few
thousand parameters) on N in {5, 20, 100} pairs from expert A/B/D/E, or refine
the parameter vector per image at test time against a no-reference prior. Needs
a loader change in `data.py` to select expert folders and a fine-tune mode in
`main.py`. The per-image low-dimensional parameter vector is the asset here —
dense-pixel methods cannot adapt this cheaply or transparently. Gain is
conjecture; PieNet is the baseline. Low cost and low risk, but modest novelty
unless paired with Rank 1.

## 5. Not worth it

- **Diffusion-based enhancement.** FiveK is a deterministic fidelity benchmark.
  No diffusion method reports a competitive Expert-C PSNR under the standard
  protocol, inference is 100–1000x slower, and it discards the
  interpretable-parameter claim with nothing to show for it on this task.
- **Retinex-based deep methods and normalizing flows.** Built for low-light
  noise and illumination (LOL, SICE). The decomposition prior does not describe
  expert retouching, and they are dense-pixel.
- **Zero-reference / unpaired methods.** FiveK is paired; unpaired training
  costs several dB, and the curve estimators are per-pixel — less interpretable
  than DeepLPF, not more. Relevant only if the goal changes to in-the-wild data.
- **Swapping the U-Net for a transformer.** The bottleneck is what the heads
  receive (observation 1), not backbone capacity — 28k–600k-parameter LUT
  models already beat a 1.7M U-Net. Do Rank 2 first and keep a backbone swap as
  an ablation at most.
- **Replacing DeepLPF with an image-adaptive 3D LUT.** That reproduces
  AdaInt/SepLUT. The family is saturated at 25.5 and has no locality or spatial
  handles.
- **Adding more ellipses or lines of the same kind.** Conjecture, but strongly
  suggested by the design: with per-image global placement and 2x maximum
  scale, extra fixed shapes add parameters without content-awareness, which is
  the missing ingredient.
- **INR methods as enhancers.** NILUT and INRetouch emulate given styles and
  presets; they are not a fidelity mechanism for FiveK.
- **Loss engineering alone.** PSNR on FiveK is dominated by colour and tone
  accuracy, and dE-type losses are close to what Lab-L1 already does. Fix the
  MS-SSIM bug, but do not expect a paper from losses.

## 6. Suggested sequencing

Rank 2 first — days of work, retrained under the 480p protocol, and a credible
repo release on its own as a "DeepLPF v2 baseline". Then Rank 1 as the paper,
with Rank 3 as a second contribution if the cubic-to-local-curve swap adds
measurably. Ranks 4 and 5 are cheap add-ons, not headline claims.
