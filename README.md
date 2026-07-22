# DeepLPF: Deep Local Parametric Filters for Image Enhancement (CVPR 2020)

[![CI](https://github.com/sjmoran/deeplpf-image-enhancement/actions/workflows/ci.yml/badge.svg)](https://github.com/sjmoran/deeplpf-image-enhancement/actions/workflows/ci.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2003.13985-b31b1b.svg)](https://arxiv.org/abs/2003.13985)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
[![Stars](https://img.shields.io/github/stars/sjmoran/deeplpf-image-enhancement?style=social)](https://github.com/sjmoran/deeplpf-image-enhancement/stargazers)

[Sean Moran](https://sjmoran.github.io/), [Pierre Marza](https://pierremarza.github.io/), [Steven McDonagh](https://smcdonagh.github.io/), [Sarah Parisot](https://parisots.github.io/), [Greg Slabaugh](https://eecs.qmul.ac.uk/~gslabaugh/) — **Huawei Noah's Ark Lab**

[[Paper]](https://arxiv.org/abs/2003.13985) [[Poster]](https://github.com/sjmoran/sjmoran.github.io/blob/main/pdfs/DeepLPF_CVPR20_poster.pdf) [[Video]](https://www.youtube.com/watch?v=Sxach3FM6FY) [[Supplementary]](https://github.com/sjmoran/sjmoran.github.io/blob/7775d1fc39d14baeb6935f6c750f923e1251f491/pdfs/DeepLPF_supplementary.pdf)

Official PyTorch implementation of the CVPR 2020 paper **DeepLPF: Deep Local Parametric Filters for Image Enhancement**. Instead of predicting output pixels directly, DeepLPF regresses the parameters of a small set of spatially localised image filters (cubic, graduated and elliptical) and applies them, giving an interpretable retouching model. On the Adobe-DPE benchmark the bundled pre-trained model reaches **23.90 dB PSNR / 0.911 SSIM**.

<p align="center">
<img src="./images/teaser.png" width="80%"/>
</p>

## Contents

- [Quick start](#quick-start)
- [Results](#results)
- [How it works](#how-it-works)
- [Pre-trained models](#pre-trained-models)
- [Training](#training)
- [Datasets](#datasets)
- [Original (CVPR 2020) code](#original-cvpr-2020-code)
- [Citation](#citation)
- [License](#license)
- [Contributing](#contributing)
- [Errata](#errata)

## Quick start

```bash
git clone https://github.com/sjmoran/deeplpf-image-enhancement.git
cd deeplpf-image-enhancement
pip install -r requirements.txt

# Enhance the bundled example images with the pre-trained Adobe-DPE model.
# Results (with PSNR/SSIM in the filenames) appear in a timestamped log_* directory.
python3 main.py \
  --inference_img_list_path=./adobe5k_dpe/images_inference.txt \
  --inference_img_dirpath=./adobe5k_dpe/ \
  --checkpoint_filepath=./pretrained_models/adobe_dpe/deeplpf_validpsnr_23.378_validloss_0.033_testpsnr_23.904_testloss_0.031_epoch_424_model.pt
```

The code picks the best available device automatically: a CUDA GPU, Apple Silicon (MPS), or CPU. No configuration is needed.

## Results

The bundled `adobe_dpe` checkpoint (epoch 424) on the Adobe-DPE test set:

| Dataset | PSNR | SSIM |
|---|---|---|
| Adobe-DPE | 23.90 dB | 0.911 |

Input → expert-retouched label → DeepLPF output:

<table>
  <tr><th>Input</th><th>Label</th><th>Ours (DeepLPF)</th></tr>
  <tr>
    <td><img src="./adobe5k_dpe/deeplpf_example_test_input/a4576-DSC_0217_input.png"/></td>
    <td><img src="./adobe5k_dpe/deeplpf_example_test_output/a4576-DSC_0217_gt.png"/></td>
    <td><img src="./adobe5k_dpe/deeplpf_example_test_inference/a4576-DSC_0217_TEST_425_9_PSNR_34.596_SSIM_0.980.jpg"/></td>
  </tr>
  <tr>
    <td><img src="./adobe5k_dpe/deeplpf_example_test_input/a4742-Duggan_090331_6517_input.png"/></td>
    <td><img src="./adobe5k_dpe/deeplpf_example_test_output/a4742-Duggan_090331_6517_output.png"/></td>
    <td><img src="./adobe5k_dpe/deeplpf_example_test_inference/a4742-Duggan_090331_6517_TEST_425_1_PSNR_29.825_SSIM_0.984.jpg"/></td>
  </tr>
  <tr>
    <td><img src="./adobe5k_dpe/deeplpf_example_test_input/a4869-09-05-19-at-19h05m38s-_MG_9563.png"/></td>
    <td><img src="./adobe5k_dpe/deeplpf_example_test_output/a4869-09-05-19-at-19h05m38s-_MG_9563.png"/></td>
    <td><img src="./adobe5k_dpe/deeplpf_example_test_inference/a4869-09-05-19-at-19h05m38s-_MG_9563_TEST_500_29_PSNR_30.090_SSIM_0.978.jpg"/></td>
  </tr>
</table>

<details>
<summary>More examples</summary>

<table>
  <tr><th>Input</th><th>Label</th><th>Ours (DeepLPF)</th></tr>
  <tr>
    <td><img src="./adobe5k_dpe/deeplpf_example_test_input/a4582-DSC_0343_input.png"/></td>
    <td><img src="./adobe5k_dpe/deeplpf_example_test_output/a4582-DSC_0343_gt.png"/></td>
    <td><img src="./adobe5k_dpe/deeplpf_example_test_inference/a4582-DSC_0343_TEST_425_10_PSNR_18.942_SSIM_0.921.jpg"/></td>
  </tr>
  <tr>
    <td><img src="./adobe5k_dpe/deeplpf_example_test_input/a4591-Duggan_080411_5940.png"/></td>
    <td><img src="./adobe5k_dpe/deeplpf_example_test_output/a4591-Duggan_080411_5940.tif_GT.png"/></td>
    <td><img src="./adobe5k_dpe/deeplpf_example_test_inference/a4591-Duggan_080411_5940_TEST_425_20_PSNR_28.000_SSIM_0.951.jpg"/></td>
  </tr>
  <tr>
    <td><img src="./adobe5k_dpe/deeplpf_example_test_input/a4521-kme_0310.png"/></td>
    <td><img src="./adobe5k_dpe/deeplpf_example_test_output/a4521-kme_0310.png"/></td>
    <td><img src="./adobe5k_dpe/deeplpf_example_test_inference/a4521-kme_0310_TEST_800_5_PSNR_28.774_SSIM_0.897.jpg"/></td>
  </tr>
  <tr>
    <td><img src="./adobe5k_dpe/deeplpf_example_test_input/a4933-Duggan_090428_8040.png"/></td>
    <td><img src="./adobe5k_dpe/deeplpf_example_test_output/a4933-Duggan_090428_8040.png"/></td>
    <td><img src="./adobe5k_dpe/deeplpf_example_test_inference/a4933-Duggan_090428_8040_TEST_500_23_PSNR_19.011_SSIM_0.882.jpg"/></td>
  </tr>
  <tr>
    <td><img src="./adobe5k_dpe/deeplpf_example_test_input/a4783-20090321_at_19h16m32__MG_0220.png"/></td>
    <td><img src="./adobe5k_dpe/deeplpf_example_test_output/a4783-20090321_at_19h16m32__MG_0220.png"/></td>
    <td><img src="./adobe5k_dpe/deeplpf_example_test_inference/a4783-20090321_at_19h16m32__MG_0220_TEST_500_2_PSNR_26.362_SSIM_0.905.jpg"/></td>
  </tr>
  <tr>
    <td><img src="./adobe5k_dpe/deeplpf_example_test_input/a4514-kme_0258.png"/></td>
    <td><img src="./adobe5k_dpe/deeplpf_example_test_output/a4514-kme_0258.png"/></td>
    <td><img src="./adobe5k_dpe/deeplpf_example_test_inference/a4514-kme_0258_TEST_500_27_PSNR_27.426_SSIM_0.879.jpg"/></td>
  </tr>
</table>

</details>

## How it works

A U-Net backbone extracts per-pixel features from the input image. Three lightweight heads then regress the parameters of three interpretable filter families, which are applied in sequence to produce the enhanced image:

- **Cubic**: a global cubic polynomial in the pixel intensity and image coordinates (a learned tone/colour curve).
- **Graduated**: graduated neutral-density-style filters that scale the image along learned lines.
- **Elliptical**: radial/vignette-style filters that scale within learned ellipses.

Because the network predicts a small set of human-meaningful filter parameters rather than dense pixel values, its adjustments are interpretable. See the [paper](https://arxiv.org/abs/2003.13985) for the full formulation.

## Pre-trained models

Checkpoints are in `pretrained_models/`:

- **Adobe-DPE** (`pretrained_models/adobe_dpe/`): trained on Adobe5K using the splits and pre-processing from the DeepPhotoEnhancer paper. The best-validation checkpoint (epoch 424) reaches 23.90 dB PSNR / 0.911 SSIM on the Adobe-DPE test set and is the one used in the [Quick start](#quick-start).
- **Adobe-UPE** (`pretrained_models/adobe_upe/`): trained on Adobe5K using the splits and pre-processing from the DeepUPE paper. Contributed by Yucheng Lu (yucheng.l@outlook.com) and applied in [this paper](https://arxiv.org/abs/2106.14844).

## Training

Prepare the dataset first (see [Datasets](#datasets)), so you have a directory with `input/` and `output/` sub-folders. Then:

```bash
python3 main.py \
  --training_img_dirpath=./adobe5k_dpe_data/ \
  --train_img_list_path=./adobe5k_dpe/images_train.txt \
  --valid_img_list_path=./adobe5k_dpe/images_valid.txt \
  --test_img_list_path=./adobe5k_dpe/images_test.txt \
  --batch_size=1
```

Training supports a batch size greater than one via `--batch_size` for throughput; use `--batch_size=1` to reproduce the paper's setup. Evaluation and inference run at a batch size of 1 so per-image PSNR/SSIM are reported and saved. Checkpoints are written whenever validation PSNR improves.

<p align="center">
<img src="./images/deeplpf_training_loss.png" width="70%"/>
</p>

## Datasets

DeepLPF is trained on the [MIT-Adobe FiveK](https://data.csail.mit.edu/graphics/fivek/) photographs, processed through Lightroom with Expert C retouching as the target. For a step-by-step walkthrough (Lightroom export settings, the expected folder layout, and helper/verification scripts), see **[docs/ADOBE_DPE_DATASET.md](./docs/ADOBE_DPE_DATASET.md)**.

- **Adobe-DPE** (5000 RGB→RGB pairs): download [here](https://data.csail.mit.edu/graphics/fivek/), then pre-process per the DeepPhotoEnhancer (DPE) [paper](https://github.com/nothinglo/Deep-Photo-Enhancer) (Expert C as target, exported in sRGB); see the [DPE instructions](https://github.com/nothinglo/Deep-Photo-Enhancer/issues/38#issuecomment-449786636). The train/valid/test splits are in [`adobe5k_dpe/`](./adobe5k_dpe/) (note: a best guess at the original DPE splits, which were unavailable).
- **Adobe-UPE** (5000 RGB→RGB pairs): download [here](https://data.csail.mit.edu/graphics/fivek/), then pre-process per the DeepUPE [paper](https://github.com/wangruixing/DeepUPE) as detailed [here](https://github.com/wangruixing/DeepUPE/issues/26). Test images are [available here](https://drive.google.com/file/d/1HZnNgptNxjKJAhekz2K5yh0mW0yKIws2/view?usp=sharing).

## Original (CVPR 2020) code

This repository has been updated since publication: it runs device-agnostically on a CUDA GPU, Apple Silicon (MPS), or CPU, supports a training batch size greater than one, works with current dependencies, and has a test suite run in CI. The code exactly as published for the paper is preserved at the [`legacy`](https://github.com/sjmoran/deeplpf-image-enhancement/tree/legacy) branch and the [`original-cvpr2020`](https://github.com/sjmoran/deeplpf-image-enhancement/releases/tag/original-cvpr2020) tag.

## Citation

If you use DeepLPF, its pre-trained models, or this code in your research, please cite:

```
@InProceedings{Moran_2020_CVPR,
author = {Moran, Sean and Marza, Pierre and McDonagh, Steven and Parisot, Sarah and Slabaugh, Gregory},
title = {DeepLPF: Deep Local Parametric Filters for Image Enhancement},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

## License

Released under the BSD-3-Clause License — see [LICENSE](./LICENSE).

## Contributing

We appreciate all contributions. Bug fixes can be sent as pull requests directly. For new features, utility functions, or extensions to the core, please open an issue to discuss first — the core may be heading in a direction a large unsolicited PR doesn't fit. If you are training your own DeepLPF and run into difficulties, open a GitHub issue; the authors are happy to help.

## Errata

**[Deep Symmetric Network for Underexposed Image Enhancement with Recurrent Attentional Learning](https://www.shaopinglu.net/publications_files/ICCV21_Image_Enhancement.pdf)** — the DeepLPF results in Fig. 1 of this paper are incorrect. An example inference for one of those images is [here](./images/004668_TEST_25_354_PSNR_21.848_SSIM_0.858.jpg), and our pre-trained model for their dataset is [here](./pretrained_models/adobe_distort_and_recover/deeplpf_validpsnr_23.629675866286313_validloss_0.030986817553639412_testpsnr_23.629675866286313_testloss_0.030986817553639412_epoch_49_model.pt). The correct DeepLPF results in their Table 1 should be **23.63 dB / 0.875 SSIM**. On 29 September 2021 the authors kindly published an [errata](https://www.shaopinglu.net/proj-iccv21/ImageEnhancement.html) to their ICCV paper; we thank them for re-checking the result.
