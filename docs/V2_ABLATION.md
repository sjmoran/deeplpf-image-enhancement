# The v2 ablation

Four changes to the published model are gated behind `--fixes`. This is how
their effect is measured.

## The changes

| `--fixes` name | What it changes | Effect on the forward pass |
|---|---|---|
| `wiring` | Feeds `UNet.conv1/2/3`, three 1x1 encoder projections, into the filter-head input. | None at initialisation: the contribution is gated by `ms_gate`, which starts at zero. Diverges once trained. |
| `ellipse` | `EllipticalFilter` instance 2 uses `semi_axis_y=b2` for its third channel, so its three channels share one ellipse, as instances 1 and 3 do. | Changes it. 0.0037 max abs difference on the released checkpoint at 128x128. |
| `ste` | Routes the graduated filter's binarisation through an `autograd.Function`, applying the straight-through estimator of paper Sec. 3.2.2. | None. Gradient-only. |
| `blend` | Combines the graduated branches by interpolation rather than a `torch.where`, whose condition carries no gradient. Needed alongside `ste`. | None. Forward values unchanged. |
| `fusion` | Sums the filter maps' deviations from neutral rather than the maps themselves, so two neutral filters compose to neutral. | Changes it. |
| `msssim` | Groups the MS-SSIM product as upstream `jorge-pessoa/pytorch-msssim` does. | None. Loss-only. |

Two of these are invisible in a forward pass but change what training converges to, which is why they need a training run rather than an inspection.

## Design

Every arm runs identical code and differs only in the flag. `--fixes=none` is
the default and is numerically the published v1 model, so the control is the
same code path as the treatments rather than a different branch. That removes
the usual risk of an incidental difference between control and treatment
landing in the measured effect.

One instance per arm, five in parallel:

```bash
export BUCKET=s3://your-bucket \
       KEY_NAME=your-keypair \
       SECURITY_GROUP=sg-... \
       SUBNETS="subnet-a subnet-b subnet-c" \
       AMI=ami-...            # a Deep Learning AMI for your region
experiments/launch_ablation.sh
experiments/check_ablation.sh   # progress, by tag; no ids hardcoded
```

`BUCKET` must already hold `code.tgz` (a `git archive` of the revision under
test) and `data.tgz` (`adobe5k_dpe_data/` plus `adobe5k_dpe/`).

## Reading the results

Compare **validation PSNR**, not training loss. The `msssim` arm changes the
loss function itself, so its loss is on a different scale and is not comparable
with the others; PSNR is measured identically for all five.

Results sync to `$BUCKET/results/<arm>/` every ten minutes.

## Results

### The fixes and features, 500 epochs

Every arm below ran 500 epochs, seed 42, batch 1, on the recovered DPE split
(2250 train / 2250 valid / 498 test), except `guessed_repl`, which ran on the
superseded best-guess split and whose test column is therefore a different set
of images.

| Arm | `--fixes` | valid PSNR | test PSNR | test SSIM |
|---|---|---|---|---|
| `baseline` | none (v1) | 23.049 | 23.381 | 0.897 |
| `wiring` | wiring | 23.030 | 23.355 | 0.890 |
| `msssimw` | msssim | 22.901 | 23.333 | 0.897 |
| `steblend` | blend, ste | 23.034 | 23.394 | 0.896 |
| `ellipse` | ellipse | 23.213 | 23.565 | 0.902 |
| `fusion` | fusion | 23.225 | 23.598 | 0.901 |
| `comb_ef` | ellipse, fusion | 23.322 | 23.556 | 0.901 |
| `comb_efsb` | blend, ellipse, fusion, ste | 23.256 | 23.763 | 0.901 |
| `colour` | colour, fusion, 16 curve knots | 23.293 | 23.477 | 0.905 |
| `colour_mix` | colour, fusion, 0 knots (mixer only) | 23.236 | 23.501 | 0.902 |
| `gates_hi` | + gates, `--gate_weight` 1e-2 | 23.143 | 23.698 | 0.901 |
| `gates_lo` | + gates, `--gate_weight` 3e-3 | 23.222 | 23.768 | 0.901 |
| `gates_vlo` | + gates, `--gate_weight` 3e-4 | 23.298 | 23.592 | 0.900 |
| `guessed_repl` | none (v1), best-guess split | 22.810 | 23.742 | 0.910 |

`wiring` and `msssim` do not help on their own. `ellipse` and `fusion` carry
what gain there is, and combining them with `ste` and `blend` is the best of
the fix-only arms.

**None of this is resolved.** Scoring each arm's best-validation checkpoint on
the 498 test images and bootstrapping the paired per-image differences against
`baseline` gives, for the four leading arms, +0.28, +0.21, +0.18 and +0.04 dB
with 95% intervals of roughly +/-0.28 dB. Every interval contains zero, and
that is the optimistic bound: it covers test-set sampling only, not the seed
and checkpoint-selection variability that this document elsewhere identifies as
the larger term. The three gate weights land within 0.05 dB of each other, so
the gate weight is doing nothing measurable at n=498 either. Separating these
arms needs several seeds each, not a larger test set.

### The MS-SSIM weight stays at the paper's value

Eq. 8's `w_msssim = 1e-3` makes the structural term contribute about 0.07% of
the L1 gradient, so the model is trained by L1 in Lab space alone. Raising it
was tested and abandoned: five arms at 1e-3 to 1.0 for 150 epochs, then the two
most promising for 1000 epochs. The arm that led the short sweep by 0.18 dB
finished 0.13 dB behind on test, with validation and test disagreeing about the
ordering throughout. The weight is now fixed at the published value and there is
no flag to change it.

The 1000-epoch run at that published weight is the checkpoint shipped as
`pretrained_models/adobe_dpe_v2_reconstructed/`: 24.18 dB test PSNR on the
reconstructed split, against 23.90 for the checkpoint released in 2020 on that
same split, and not comparable with any DPE-protocol number.

## Cost and shutdown

Measured on a g5.xlarge (A10G), 2250 training images per epoch:

| Recipe | per epoch | per arm, 500 epochs |
|---|---|---|
| batch 1, full image (the paper's) | 4m36s | ~38 hrs |
| batch 8, crop 256 | 1m20s | ~11 hrs |

The model is latency-bound at batch size 1: the GPU sits at 33-45% while a
4-vCPU host stays at load 1.3, waiting on kernel launches for many small
operations. TF32 buys 2.5%.

`torch.compile` initially appeared unusable — a first attempt produced no epoch
in 13 minutes — and this document previously recorded the cause as FiveK's
variable image sizes forcing endless recompilation. **That was wrong.** FiveK
has only 16 distinct image sizes, which Dynamo reduces to 9 unique graphs with
no graph breaks and no recompilation after warm-up. The real cause was
PyTorch's default `cache_size_limit` of 8: the ninth shape exceeded it and
silently fell back to eager, permanently. `main.py` now raises the limit when
`--compile` is passed.

The launch overhead itself was addressed directly instead. Profiling one
training step found the U-Net backbone — which does all the arithmetic — issues
82 dispatched operations, while the three mask heads issue 1,818, and the
backward pass 3,281. Building nine full-resolution masks one instance at a time
in Python, not the convolutions, is what the model spends its time on. Batching
those into single broadcast expressions cut the step from ~6,500 dispatched
operations to ~2,700, with the forward pass bitwise identical and parameter
gradients differing by at most 1.5e-8 from the changed reduction order.

Batch 8 with a crop is three times cheaper, but its baseline no longer
reproduces the published number, so it answers "do the fixes help" and not
"does v1 replicate".

Instances terminate on three independent paths, none of which needs an
operator session: normal completion, an `EXIT` trap covering crashes and
staging failures, and a hard-cap `shutdown` armed at boot.

## Training speed

Everything below the first table was measured on CPU and Apple MPS, with no
GPU available; CUDA numbers are estimates and are labelled as such. The one
GPU measurement is the 4m36s epoch above, taken before the launch-count work
of `da571ab`: 2250 images in 276 s is 123 ms per step on an A10G.

### The profile

One step, batch 1, 341x512, `--fixes=none`. "Launches" counts dispatched aten
ops that run a kernel (views, `detach`, `select` and friends excluded); Adam is
the CUDA `foreach` path, which is ~15 launches there and 722 per-parameter ops
on CPU/MPS.

| Phase | Launches | Note |
|---|---|---|
| backbone forward | 82 | 25 convolutions |
| filter heads forward | 293 | mask construction, already batched over the nine instances |
| loss | 262 | two `rgb_to_lab`, five-scale MS-SSIM |
| backward | 837 | |
| Adam | ~15 (CUDA) | |
| **total** | **~1,490** | 2,552 dispatched ops including the view-like ones |

What those launches do: 72.6 GFLOP (forward + backward, `torch.utils.flop_counter`)
in 129 convolution kernels, and 1,343 elementwise kernels that read and write
5.2 GB between them. At A10G rates - fp32 convolutions at 10-20 TFLOP/s for
these small-channel shapes, 600 GB/s for the rest - the GPU work in the step is
roughly 4-7 ms of convolution and ~9 ms of unfused elementwise traffic. The
measured 123 ms is neither: it is the CPU issuing ~2,600 (now ~1,490) ops at
40-60 us each while the GPU waits. That is the number to remove.

Under `torch.compile` (CPU inductor backend as a proxy, since it is the same
fusion pass) the forward + backward of network and loss compiles to 316
generated kernels plus the ~130 convolutions, so ~450 launches per step, and
the elementwise traffic drops with them because fused chains keep their
intermediates in registers.

### What this branch adds

`--cuda_graphs` (`trainstep.GraphedStep`): captures forward, loss, backward
and Adam in one CUDA graph per image shape and replays it, so the step costs
one launch plus the GPU time. FiveK has 16 shapes; each is captured on first
sight after three eager warm-up steps whose parameter and optimiser-state
changes are rolled back. The kernels and arithmetic are the eager ones; the
two departures are Adam `capturable=True`, whose bias corrections are fp32
tensors instead of Python doubles (~1e-7 relative in the step size), and the
dropout RNG stream, which the warm-ups advance, so the masks differ from an
eager run with the same seed. Memory: each shape's graph keeps its own
activation pool, on the order of the eager step's working set per shape.

On the A10G the first capture failed with `cudaErrorStreamCaptureInvalidated`.
Capturing one forward + backward per component located it: backbone, heads
and `rgb_to_lab` capture; `compute_msssim` does not, because `torch.prod`'s
CUDA backward counts the zeros in its input with a host synchronisation. Both
`prod` calls are now explicit four-factor products: bitwise equal on CPU,
within 1 ulp on CUDA and MPS on a term weighted 1e-3.

Numerics on the A10G, eval mode so dropout does not confound, two shapes, six
steps: the graphed step's loss equals the eager step's bitwise on the first
step and stays within 2e-7 after; parameters drift to 2e-4 by step six.
Running the eager step with `capturable=True` and no graph produces the same
drift (1.5e-8 after one step, 2.2e-4 after six), so it is Adam's fp32 bias
correction amplified by its normalised update on near-zero gradients, not the
graph. `tests/test_trainstep.py` covers the bookkeeping (per-shape buffers,
warm-up rollback) on CPU with an eager stand-in for the capture. The eager
path reproduces the previous loop exactly on CPU and MPS.

`--amp bf16`: autocasts the network's forward pass; the loss stays fp32
because its SSIM variances are `E[x^2] - mu^2`, which cancels badly in bf16
(with the loss autocast too, gradients moved by up to 30% of their norm).
Against fp32 on an untrained model: max abs prediction difference 0.010-0.013,
loss 1e-5 to 6e-5, gradient difference 0.6-1.3% of the gradient norm (worst
single parameter 6-28%, always one with a tiny gradient). Measured on CPU and
MPS; MPS bf16 is slower than fp32 (165 vs 142 ms), which says nothing about
A10G tensor cores.

Not done, with the reason:

- *Caching decoded PNGs.* Decode + tensor conversion is ~14 ms per pair on one
  core; six workers deliver a pair every ~2.5 ms, and at four vCPUs every
  ~3.5 ms. Off the critical path at 123 ms and still off it at 12 ms.
- *Masks at reduced resolution.* Nine full-resolution masks cost ~5 ms of
  memory traffic when every elementwise op is its own kernel, and under a
  quarter of that once fused. A low-resolution build buys at most a few ms in
  the eager-plus-graphs configuration and almost nothing under compile, and it
  changes the gradients. Not worth a flag.
- *Fusing the loss by hand.* 262 launches, but they are exactly what inductor
  fuses; hand-fusing them buys nothing `--compile` does not.

### Measured on the A10G

Three epochs each, 2250 images, batch 1, `--fixes=none --seed=42`, same box
and data:

| Configuration | s/epoch (steady) | ms/step | vs old code | vs current eager |
|---|---|---|---|---|
| old code (before `da571ab`) | 266 | 118 | 1x | |
| current eager (`da571ab` + this branch, default path) | 94 | 42 | 2.8x | 1x |
| `--cuda_graphs` | 57 | 25.3 | **4.7x** | **1.65x** |
| `--cuda_graphs --compile` | 47.5 | 21.1 | **5.6x** | **2.0x** |

The graphed run's first epoch took 65 s including the 16 captures; the
graphs hold 13.3 GB of the 24 GB. During replay the GPU reports 97%
utilisation, so at 25 ms the step is GPU-bound on eager kernels: the launch
overhead is gone and what remains is the kernels' own time, ~1,500 of them,
most of them elementwise passes over full-resolution tensors. That is
higher than the 16-22 ms estimated below from bytes and FLOPs, i.e. the
unfused kernels are slower than a bandwidth model says. Test PSNR after three
epochs was 19.39 (graphs) against 20.06 (eager) and 19.08 (old); with
different dropout draws and three epochs, that spread is noise.

`--cuda_graphs --compile` composes (inductor's kernels capture) and is the
fastest configuration measured, but fusion bought only 4 ms of the 25: the
step is now dominated by the convolutions, not by elementwise traffic. Its
first epoch took 20 minutes, almost all of it compiling forward and backward
for the 16 shapes, and the evaluation pass recompiles again for the test
images' shapes in eval mode. Test PSNR after three epochs: 20.53. The next
unmeasured lever on this configuration is `--tf32`, which the existing flag
already exposes and which inductor itself suggests at compile time; it
changes the convolution arithmetic.

### Ceiling by constraint (A10G estimates)

Baseline: 123 ms per step, batch 1, full resolution, fp32, eager. The GPU
floor is 4-7 ms of convolutions plus the elementwise traffic, which is ~9 ms
unfused and ~2-3 ms fused. Launch gaps inside a graph are ~1-2 us per kernel.

| Kept | Given up | Configuration | Est. ms/step | Est. speedup |
|---|---|---|---|---|
| batch 1, fp32, eager numerics, eager kernels | nothing | as of `da571ab` (1,490 launches) | 55-75 | 1.6-2.2x |
| batch 1, fp32, eager numerics | nothing but dropout draws | `--cuda_graphs` | **25 measured** | **4.7x measured** |
| batch 1, fp32, same recipe | reduction order inside fused kernels | `--compile --cuda_graphs` | **21 measured** | **5.6x measured** |
| batch 1, same recipe | fp32 convolutions | `--tf32` or `--amp bf16` on top | not measured | convolution-bound now, so this is where the rest is |
| full images, fp32 | the paper's batch size | batch 8 with `--crop_size 256` (measured 1m20s/epoch above) | 3.4x now; multiplies with the rows above | |

Where the uncertainty is: the two things that cannot be measured here are the
actual conv efficiency at these shapes (the 4-7 ms) and how well cudnn behaves
inside a captured graph across 16 shapes. The eager-plus-graphs row is the
most solid, because it is arithmetic on measured kernel counts and bytes; the
estimates for the fused step were 12-18 ms and it measured 21: the
convolutions take more of the step than the FLOP model gave them. A plain
10x from the old code is not reachable at batch 1 in fp32 with any launch or
fusion work; what is left is convolution time, which only reduced precision
(TF32, bf16) or a larger batch can cut. None of the rows below the second one keeps
bitwise numerics, and none changes the recipe (batch size, resolution,
optimiser, loss).
