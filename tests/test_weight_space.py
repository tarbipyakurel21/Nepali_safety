"""Numerical tests for bounded weight-space updates (no model downloads)."""
import unittest
import torch
from torch import nn
from torch.nn.utils import parametrize
from src.weight_space import WeightDelta, project


class WeightSpaceTests(unittest.TestCase):
    def test_projection_bounds_nominal_and_bfloat16_effective_delta(self):
        torch.manual_seed(5)
        original=torch.randn(32,32).to(torch.bfloat16)
        delta=torch.randn(32,32)*10
        stats=project(delta,original,.01)
        effective=((original.float()+delta).to(original.dtype).float()-original.float()).norm()
        self.assertLessEqual(float(delta.norm()),stats['radius']+1e-6)
        self.assertLessEqual(float(effective),stats['radius']+1e-6)
        self.assertAlmostEqual(float(effective),stats['effective_norm'])

    def test_zero_budget_and_zero_original(self):
        for original,epsilon in [(torch.ones(3,3),0),(torch.zeros(3,3),.1)]:
            delta=torch.ones(3,3)
            project(delta,original,epsilon)
            self.assertTrue(torch.equal(delta,torch.zeros_like(delta)))

    def test_gradient_ascent_suppresses_target_and_leaves_base_unchanged(self):
        torch.manual_seed(4)
        layer=nn.Linear(3,2,bias=False)
        layer.weight.requires_grad_(False)
        original=layer.weight.detach().clone()
        parametrize.register_parametrization(layer,'weight',WeightDelta(layer.weight))
        delta=layer.parametrizations.weight[0].delta
        inputs=torch.tensor([[1.,.5,-.2]])
        target=torch.tensor([0])
        before=nn.functional.cross_entropy(layer(inputs),target)
        before.backward()
        self.assertIsNone(layer.parametrizations.weight.original.grad)
        with torch.no_grad():delta.add_(delta.grad*.01)
        project(delta,layer.parametrizations.weight.original,.1)
        after=nn.functional.cross_entropy(layer(inputs),target)
        self.assertGreater(float(after.detach()),float(before.detach()))
        self.assertTrue(torch.equal(original,layer.parametrizations.weight.original))

if __name__=='__main__':unittest.main()
