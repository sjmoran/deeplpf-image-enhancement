# FiveK enhancement benchmarks: which number belongs to which protocol

DeepLPF is reported in the literature as **23.63**, **23.90**, **23.93**, **24.48**
and **24.73** dB. All five are correct. They are five different protocols, and
they are not comparable with one another.

This is the single most common source of confusion when entering this field, and
it is not specific to DeepLPF - every method on FiveK has the same problem. This
page states which number belongs to which protocol, what each protocol actually
is, and - where we have them - which split files reproduce it.

Numbers not measured in this repository are attributed to the paper that
reported them. We have not re-run other authors' methods, and say so rather than
implying a verification we did not perform.

## The protocols

| Protocol | Train / test | Input rendering | Target | Resolution |
|---|---|---|---|---|
| **DPE** | 2250 / 498 | `InputAsShotZeroed` | Expert C | long edge 512 |
| **UPE** (DeepUPE) | 4500 / 500 | as-shot | Expert C | varies |
| **3D-LUT 480p** | 4500 / 500 | as-shot, 8-bit sRGB | Expert C | 480p |
| **FiveK-Lite** (NeurOp) | 4500 / 500 | as-shot | Expert C | 480p |
| **DSN** | per Zhao et al. 2021 | unstated | Expert C | unstated |

The two axes that matter most are **the test set** (498 scattered ids for DPE
versus 500 for the LUT line) and **the input rendering**, which changes the
inputs by tens of levels - see
[`adobe5k_dpe/SPLIT_PROVENANCE.md`](../adobe5k_dpe/SPLIT_PROVENANCE.md) and the
[dataset guide](./ADOBE_DPE_DATASET.md).

## DeepLPF, by protocol

| Protocol | PSNR | SSIM | Reported by | Reproducible here? |
|---|---|---|---|---|
| DPE | 23.93 | 0.903 | NamedCurves (ECCV 2024), Tab. 1 | yes, with the shipped splits |
| UPE | 24.48 | 0.887 | NamedCurves Tab. 1; StarEnhancer (ICCV 2021) agree exactly | no UPE split shipped |
| 3D-LUT 480p | 24.73 | 0.916 | AdaInt (CVPR 2022) Tab. 1; RSFNet (ICCV 2023) Tab. 1 agree | no 480p split shipped |
| FiveK-Lite | 23.63 | — | NeurOp (ECCV 2022) Tab. 1 | not attempted |
| DSN | 23.63 | 0.875 | Zhao et al. ICCV 2021, **as corrected** | not attempted |
| PPR10K 480p | 23.47 | 0.892 | RSFNet Tab. 2 | different dataset |
| LOL | 16.58 | 0.678 | Zhao et al. ICCV 2021 | different dataset |
| HDR+ | 25.73 | 0.902 | PQDynamicISP (2024) | different dataset |

### A caution about evaluation determinism

Re-evaluating one retrained checkpoint gave 23.333 dB where the training run had
logged 23.514 for the same weights on the same images, the difference being the
device (MPS versus CUDA). That is the same order as the differences between
model variants, so figures compared with each other should be produced by one
evaluation pass on one device rather than taken from training logs.

### The DSN errata

Zhao et al., *Deep Symmetric Network for Underexposed Image Enhancement*, ICCV
2021, originally reported DeepLPF at **21.82 / 0.845**. That was wrong: the
authors had trained it at batch size 30 on 250x250 crops, where the released
configuration is batch size 1. After being told, they retrained and issued an
[errata](https://www.shaopinglu.net/proj-iccv21/ImageEnhancement.html) giving
**23.63 / 0.875**.

The corrected values are on the authors' own copy of the paper. **The CVF
open-access version still carries the uncorrected 21.82 / 0.845 and the
corrupted qualitative figure**, with no errata pointer, and that is the copy
most people download. If you are citing DeepLPF numbers from that paper, use the
errata.

## The wider field, 3D-LUT 480p protocol

Reported by the papers cited; not re-run here. Included because the shape of the
table matters more than any single row: the global-LUT family has sat at
25.3-25.5 dB for four years, and RSFNet reached the same place with named,
interpretable filters.

| Method | Year | PSNR | SSIM | Source |
|---|---|---|---|---|
| HDRNet | 2017 | 24.66 | — | AdaInt Tab. 1 |
| DeepLPF | 2020 | 24.73 | 0.916 | AdaInt Tab. 1 |
| CSRNet | 2020 | 25.17 | — | AdaInt Tab. 1 |
| 3D-LUT | 2020 | 25.29 | — | AdaInt Tab. 1 |
| SepLUT | 2022 | 25.02-25.32 | — | SepLUT (table-dependent) |
| ICELUT | 2024 | 25.27 | — | ICELUT |
| AdaInt | 2022 | 25.49 | 0.926 | AdaInt Tab. 1 |
| RSFNet | 2023 | 25.49 | 0.924 | RSFNet Tab. 1 |
| CLUT / LoR-LUT | 2023 / 2026 | 25.53 | — | ICELUT; LoR-LUT |
| NamedCurves+ | 2026 | 25.75 | 0.940 | TPAMI 2026 |

## Reproducing a number

Only the DPE protocol is fully reproducible from this repository, because it is
the only one whose split files we hold:

```bash
# 1. verify your Lightroom export matches the reference rendering
python3 data_prep/verify_export.py ./adobe5k_dpe_data adobe5k_dpe/MANIFEST.json.gz

# 2. verify the splits pair up
python3 data_prep/verify_dataset.py ./adobe5k_dpe_data

# 3. train, or evaluate a checkpoint, against the recovered DPE lists
python3 main.py --training_img_dirpath=./adobe5k_dpe_data/ \
  --train_img_list_path=./adobe5k_dpe/images_train.txt \
  --valid_img_list_path=./adobe5k_dpe/images_valid.txt \
  --test_img_list_path=./adobe5k_dpe/images_test.txt --batch_size=1
```

For the UPE, 3D-LUT 480p and FiveK-Lite protocols we do not ship split files and
have not reproduced the numbers. Getting those lists from their respective
authors, and adding them here, is the obvious next contribution to this page.

## What would make this table better

- The 3D-LUT 480p split lists, which most current work uses.
- A DeepLPF number trained and tested wholly on the recovered DPE lists, which
  the ablation in `docs/V2_ABLATION.md` is producing.
- Verified SSIM figures. DPE's own released code defines SSIM and never calls
  it, and its `test.py` is a zero-byte file, so DPE-protocol SSIM comparisons
  rest on the papers alone.
- Confirmation of which protocol each cross-paper row was actually produced
  under. Several rows above agree exactly across independent papers, which is
  good evidence they share a protocol; others may not.

Corrections welcome, particularly from authors whose numbers appear here.
