import pytest
import torch

from deepgs.networks import HeterogeneousCoefficient


def cross_difference(net):
    theta = torch.tensor([[0.1], [0.1], [0.4], [0.4]], dtype=torch.float64)
    z = torch.tensor([[0.05], [0.25], [0.05], [0.25]], dtype=torch.float64)
    v = net(theta, z).detach().ravel()
    return float(v[0] - v[1] - v[2] + v[3])


def test_legacy_heterogeneous_branch_is_additive():
    torch.manual_seed(0)
    net = HeterogeneousCoefficient([4, 4], [3]).double()
    assert cross_difference(net) == pytest.approx(0, abs=1e-14)


def test_nonlinear_fusion_can_express_interaction():
    net = HeterogeneousCoefficient([1], [1], [1]).double()
    with torch.no_grad():
        for layer in [*net.theta_layers, *net.depth_layers, *net.fusion_layers, net.output]:
            layer.weight.fill_(1)
            layer.bias.zero_()
    assert abs(cross_difference(net)) > 1e-3
