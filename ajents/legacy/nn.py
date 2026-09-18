"""Parameterized models"""
from dataclasses import field
import flax.linen as nn


class MLP(nn.Module):
    """Multi-layer perceptron"""
    features: int
    layers: list = field(default_factory=lambda: [16, 16])

    @nn.compact
    def __call__(self, x):
        return nn.Sequential(sum(([nn.Dense(k), nn.gelu] for k in self.layers), []) + [nn.Dense(self.features)])(x)
