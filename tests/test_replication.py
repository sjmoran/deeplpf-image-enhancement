"""Faithfulness check: the shipped adobe_dpe checkpoint still reproduces the
paper-era results on the bundled example images.

The example inference filenames encode the PSNR/SSIM the epoch-424 model
produced in 2020 (the ``TEST_425`` tag == epoch 424). This test runs that same
checkpoint through the repo's inference pipeline on those five images and
asserts the PSNR still matches.

Tolerance is deliberately generous. On the reference platform four of the five
images reproduce PSNR to three decimals; a4774 sits near a BinaryLayer
above/below-line decision and drifts ~0.8 dB across torch versions. The test
guards against real regressions (a broken filter shifts PSNR by many dB), not
against last-digit numerical noise, and stays robust across platforms.
"""
import glob
import os
import re

import torch

import model
import metric
from data import Adobe5kDataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Reference PSNR/SSIM parsed from the shipped TEST_425 (epoch-424) filenames.
REFERENCE = {
    "a4576": 34.596,
    "a4582": 18.942,
    "a4591": 28.000,
    "a4742": 29.825,
    "a4774": 24.233,
}
PSNR_TOL_DB = 2.0


def test_pretrained_checkpoint_reproduces_reference(tmp_path):
    ckpts = glob.glob(os.path.join(ROOT, "pretrained_models", "adobe_dpe", "*.pt"))
    assert ckpts, "adobe_dpe checkpoint not found"

    id_list = tmp_path / "ids.txt"
    id_list.write_text("\n".join(REFERENCE) + "\n")

    net = model.DeepLPFNet()
    net.load_state_dict(torch.load(ckpts[0], map_location="cpu"))
    net.to("cpu").eval()

    loader = Adobe5kDataLoader(
        data_dirpath=os.path.join(ROOT, "adobe5k_dpe") + "/",
        img_ids_filepath=str(id_list),
    )
    data_dict = loader.load_data()
    dataset = Dataset(data_dict=data_dict, normaliser=1, is_inference=True)
    dl = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    metric.Evaluator(model.DeepLPFLoss(), dl, "test", str(out_dir)).evaluate(net, epoch=0)

    produced = {}
    for fn in glob.glob(str(out_dir / "test" / "*.jpg")):
        base = os.path.basename(fn)
        iid = base.split("-")[0]
        m = re.search(r"PSNR_([0-9]+\.[0-9]+)", base)
        if m:
            produced[iid] = float(m.group(1))

    missing = set(REFERENCE) - set(produced)
    assert not missing, "no output produced for %s" % sorted(missing)

    for iid, ref_psnr in REFERENCE.items():
        delta = abs(produced[iid] - ref_psnr)
        assert delta <= PSNR_TOL_DB, (
            "%s: PSNR %.3f dB drifted %.3f dB from reference %.3f dB (tol %.1f dB)"
            % (iid, produced[iid], delta, ref_psnr, PSNR_TOL_DB)
        )
