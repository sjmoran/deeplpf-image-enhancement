"""Batch-size correctness for DeepLPF.

The network is trained and applied per image, so the enhancement of a given
image must not depend on which other images share its batch. These tests assert
that property (batch-invariance) and that a batch>1 training step runs.
"""
import torch

import model


def test_forward_is_batch_invariant():
    torch.manual_seed(0)
    net = model.DeepLPFNet().eval()
    torch.manual_seed(1)
    imgs = [torch.rand(1, 3, 64, 64) for _ in range(3)]
    with torch.no_grad():
        single = [net(im) for im in imgs]
        batched = net(torch.cat(imgs, dim=0))
    assert tuple(batched.shape) == (3, 3, 64, 64)
    for i in range(3):
        max_diff = (batched[i] - single[i][0]).abs().max().item()
        assert max_diff < 1e-5, "image %d differs by %.2e between batched and single" % (i, max_diff)


def test_training_step_batch_gt_one():
    torch.manual_seed(0)
    net = model.DeepLPFNet().train()
    crit = model.DeepLPFLoss()
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    x = torch.rand(4, 3, 64, 64)
    gt = torch.rand(4, 3, 64, 64)
    out = torch.clamp(net(x), 0, 1)
    loss = crit(out, gt)
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert tuple(out.shape) == (4, 3, 64, 64)
    assert torch.isfinite(loss).all()
    # gradients actually flowed
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())
