"""Basic forward-pass test for registered models."""

import os
import sys

import torch

# Ensure the package under ``src`` is importable when running tests directly
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from rhea_train.models import auto_model  # noqa: F401  (registers model)
from rhea_train.models.registry import create_model


def test_forward_smoke():
    """Models from the registry should produce finite outputs."""
    m = create_model(
        name="auto_model",
        input_size=8,
        num_layers=1,
        hidden_size=16,
        dropout=0.0,
    )
    x = torch.randn(4, 8)
    y = m(x)
    assert y.shape == (4, 1) and torch.isfinite(y).all()
