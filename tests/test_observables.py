"""Tests of observables.py: one test per function, in the order of the file."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lattice import Cubic
from observables import (
    STATE_KEYS,
    autocorrelation,
    binomial,
    correlator_at,
    layer_plaquettes,
    layer_winding_nonzero,
    mode_between_operators,
    photon_correlator,
    photon_weights,
    state_observables,
)
from sse import DIAG, EMPTY, FLIP, SSE
from starts import layered_start, start_state

GOLDEN = json.loads((Path(__file__).parent / "golden.json").read_text())
FRACTIONS = np.arange(9) / 16


def new_chain():
    """A chain without charges and flux on 2x2x2 before its first sweep."""
    lat = Cubic(2)
    return SSE(lat, 0.5, start_state(lat, "g", 2), seed=2)


def swept_chain():
    """The chain of new_chain after 20 sweeps at beta = 2."""
    sse = new_chain()
    for _ in range(20):
        sse.sweep(2.0)
    return sse


def put(sse, *ops):
    """Make the string these operators followed by empty slots."""
    sse.ops = list(ops) + [EMPTY] * (sse.M - len(ops))


# ---------------------------------------------------------------- a configuration


def test_state_observables():
    lat = Cubic(4)
    observables = state_observables(lat, layered_start(lat))
    assert tuple(observables) == STATE_KEYS
    # Only the xy plaquettes are flippable: f = (1, 0, 0), so Q = 1.
    assert (observables["f0"], observables["f1"], observables["f2"]) == (1.0, 0.0, 0.0)
    assert observables["nflip"] == pytest.approx(1 / 3)
    assert (observables["Q2"], observables["Q4"], observables["ReQ3"]) == pytest.approx((1, 1, 1))


# ---------------------------------------------------------------- the photon correlator


def test_photon_weights():
    lat = Cubic(4)
    w = photon_weights(lat)
    assert np.allclose(np.abs(w.reshape(lat.n_sites, 3)), [0.0, 0.5, 0.0])
    assert w[lat.link((1, 0, 0), 1)] == pytest.approx(0.5j)
    assert abs(w.sum()) == pytest.approx(0.0)  # a mode with momentum has no uniform part


def test_photon_correlator():
    sse = new_chain()
    w = photon_weights(sse.lat)
    empty = photon_correlator(sse, w, FRACTIONS)
    assert np.allclose(empty, abs(np.sum(w * sse.state)) ** 2)

    sse = swept_chain()
    mode = mode_between_operators(sse, w)
    correlator = photon_correlator(sse, w, FRACTIONS)
    assert correlator.shape == (9,)
    assert correlator[0] == pytest.approx(np.mean(abs(mode) ** 2))
    assert np.all(correlator <= correlator[0] + 1e-12)


def test_mode_between_operators():
    sse = new_chain()
    w = photon_weights(sse.lat)
    assert len(mode_between_operators(sse, w)) == 0
    p = int(np.nonzero(sse.lat.flippable(sse.state))[0][0])
    put(sse, (DIAG, p), (FLIP, p), (FLIP, p))
    before, after = np.sum(w * sse.state), np.sum(w * sse.lat.flip(sse.state, p))
    assert np.allclose(mode_between_operators(sse, w), [before, before, after])


def test_autocorrelation():
    assert np.allclose(autocorrelation(np.array([2j, 2j, 2j])), [4.0, 4.0, 4.0])
    assert np.allclose(autocorrelation(np.array([1, -1, 1, -1])), [1.0, -1.0, 1.0, -1.0])
    # against the sum over the string, one distance at a time
    mode = np.random.RandomState(1).randn(37) + 1j * np.random.RandomState(2).randn(37)
    direct = [np.mean(np.roll(mode, -m) * np.conj(mode)).real for m in range(len(mode))]
    assert np.allclose(autocorrelation(mode), direct)


def test_correlator_at():
    by_distance = np.array([1.0, -1.0, 1.0, -1.0])
    assert correlator_at(by_distance, 0.0) == 1.0
    # The alternating sum of the binomial probabilities for n = 4, f = 1/2 is (1/2 - 1/2)^4 = 0.
    assert correlator_at(by_distance, 0.5) == pytest.approx(0.0)
    assert correlator_at(np.full(7, 3.0), 0.3) == pytest.approx(3.0)


def test_binomial():
    assert binomial(4, 0.5) == pytest.approx(np.array([1, 4, 6, 4, 1]) / 16)
    assert binomial(10, 0.3).sum() == pytest.approx(1.0)
    assert binomial(100000, 0.25).sum() == pytest.approx(1.0)  # no overflow for a long string


# ---------------------------------------------------------------- the layer winding


def test_layer_plaquettes():
    lat = Cubic(4)
    layers = layer_plaquettes(lat)
    assert layers == GOLDEN["layers"]
    assert layers["z"] == [3 * lat.site((0, 0, c)) + 0 for c in range(4)]  # xy plaquettes along z


def test_layer_winding_nonzero():
    lat = Cubic(2)
    layers = layer_plaquettes(lat)
    sse = new_chain()
    assert layer_winding_nonzero(sse, layers) == 0.0
    p = int(np.nonzero(lat.flippable(sse.state))[0][0])
    put(sse, (FLIP, p), (FLIP, p))  # there and back: no winding
    assert layer_winding_nonzero(sse, layers) == 0.0

    k = np.zeros(lat.n_plaq, dtype=int)
    k[layers["y"][0]] = 1
    assert layer_winding_nonzero(SimpleNamespace(k_vector=lambda: k), layers) == 1.0
