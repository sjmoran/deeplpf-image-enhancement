"""GraphedStep bookkeeping, checked without a GPU.

The CUDA capture itself cannot run here; a stub capture performs the warm-up
steps and returns an eager replay. That exercises everything else: the
per-shape static buffers, and the parameter/optimiser-state snapshot that
keeps the warm-up steps from advancing the training trajectory.
"""
import torch

import fixes
import model
import trainstep


class _EagerCapture(trainstep.GraphedStep):
    def _capture(self, static_x, static_y):
        for _ in range(self.warmup):
            static_loss = trainstep.EagerStep.__call__(self, static_x, static_y).clone()

        def replay():
            # The warm-up consumed dropout draws; re-seed so the replayed
            # step draws the same mask as the eager step it is compared with.
            torch.manual_seed(self.seed)
            static_loss.copy_(trainstep.EagerStep.__call__(self, static_x, static_y))
        return static_loss, replay


def _build(seed):
    fixes.configure('none')
    torch.manual_seed(seed)
    net = model.DeepLPFNet().train()
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    return net, model.DeepLPFLoss(ssim_window_size=5), opt


def test_graphed_step_matches_eager_step_and_warmup_does_not_advance_training():
    torch.manual_seed(0)
    batches = [(torch.rand(1, 3, h, w), torch.rand(1, 3, h, w))
               for h, w in ((40, 48), (48, 40), (40, 48), (48, 40), (40, 48))]

    net_e, crit_e, opt_e = _build(0)
    eager = trainstep.EagerStep(net_e, crit_e, opt_e)
    net_g, crit_g, opt_g = _build(0)
    graphed = _EagerCapture(net_g, crit_g, opt_g, warmup=3)

    for i, (x, y) in enumerate(batches):
        torch.manual_seed(100 + i)  # dropout draw
        loss_e = eager(x, y)
        graphed.seed = 100 + i
        loss_g = graphed(x, y)
        assert torch.equal(loss_e, loss_g), (i, loss_e, loss_g)

    assert len(graphed.graphs) == 2
    for pe, pg in zip(net_e.parameters(), net_g.parameters()):
        assert torch.equal(pe, pg)
    for pe, pg in zip(net_e.parameters(), net_g.parameters()):
        if pe in opt_e.state:
            assert torch.equal(opt_e.state[pe]['step'], opt_g.state[pg]['step'])
            assert torch.equal(opt_e.state[pe]['exp_avg_sq'], opt_g.state[pg]['exp_avg_sq'])
