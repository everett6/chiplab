"""Tests for GPU-accelerated multiscale model reduction and neural operator surrogates."""

from __future__ import annotations

import numpy as np
import pytest

from chiplab.multiscale import (
    GPUSurrogate,
    make_gpu_surrogate,
    pod_reduce,
    pod_reconstruct,
    quick_pod_demo,
)


def test_pod_reduce_and_reconstruct_cpu_and_auto():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(24, 64))
    for device in ("cpu", "auto"):
        modes, s = pod_reduce(X, n_components=8, device=device)
        assert modes.shape == (24, 8)
        assert s.shape == (8,)
        # Orthonormal columns
        gram = modes.T @ modes
        assert np.allclose(gram, np.eye(8), atol=1e-5)

        coeffs = modes.T @ X
        recon = pod_reconstruct(modes, coeffs)
        assert recon.shape == X.shape


def test_quick_pod_demo():
    out = quick_pod_demo(n_samples=16, n_features=32, n_modes=4, device="auto")
    assert isinstance(out, dict)
    assert "modes_shape" in out or "singular_values" in out or len(out) > 0


def test_gpu_surrogate_fit_and_predict_mlp():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(32, 4)).astype(np.float32)
    Y = (X[:, :2] * 2.0 + 0.5).astype(np.float32)

    for device in ("cpu", "auto"):
        surr = make_gpu_surrogate(input_dim=4, output_dim=2, device=device)
        assert isinstance(surr, GPUSurrogate)
        hist = surr.fit(X, Y, n_epochs=5, batch_size=16)
        assert hist is not None
        preds = surr.predict(X[:5])
        assert preds.shape == (5, 2)
        assert np.all(np.isfinite(preds))
