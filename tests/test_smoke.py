"""Fast smoke tests: the model builds and runs, and the metric path works.

These need no data or checkpoint. The metric test exercises PSNR (numpy) and
SSIM (scikit-image) end-to-end, which is the path a scikit-image API change
would break.
"""
import torch

import model
import metric


def test_forward_pass_cpu():
    torch.manual_seed(0)
    net = model.DeepLPFNet().eval()
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        y = net(x)
    assert tuple(y.shape) == (1, 3, 64, 64)
    assert torch.isfinite(y).all()
    assert (y >= 0).all() and (y <= 1).all()  # output is clamped to [0, 1]


def test_loss_cpu():
    torch.manual_seed(0)
    loss_fn = model.DeepLPFLoss()
    a = torch.rand(1, 3, 64, 64)
    b = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        loss = loss_fn(a, b)
    assert torch.isfinite(loss).all()
    assert float(loss) >= 0.0


def test_metric_evaluator_cpu(tmp_path):
    torch.manual_seed(0)
    net = model.DeepLPFNet()
    crit = model.DeepLPFLoss()
    # Mimic a torch DataLoader (batch dim already present) with two tiny samples.
    loader = [
        {
            "input_img": torch.rand(1, 3, 48, 48),
            "output_img": torch.rand(1, 3, 48, 48),
            "name": ["sample_%d.png" % k],
        }
        for k in range(2)
    ]
    loss, psnr, ssim = metric.Evaluator(crit, loader, "test", str(tmp_path)).evaluate(net, epoch=0)
    assert psnr > 0.0
    assert 0.0 <= ssim <= 1.0
