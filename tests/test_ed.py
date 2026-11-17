"""Tests of ed.py: one test per function, in the order of the file."""

import json
from pathlib import Path

import numpy as np
import pytest

from ed import component, exact_thermal, hamiltonian
from lattice import Cubic
from starts import start_state

GOLDEN = json.loads((Path(__file__).parent / "golden.json").read_text())


@pytest.fixture(scope="module")
def vacuum():
    """The 2x2x2 lattice and the flip component of its vacuum start state."""
    lat = Cubic(2)
    return lat, component(lat, start_state(lat, "g", 1))


def test_component(vacuum):
    lat, states = vacuum
    assert np.array_equal(states[0], start_state(lat, "g", 1))
    known = {s.tobytes() for s in states}
    assert len(known) == len(states)  # no state twice
    for s in states[:50]:
        assert not lat.gauss(s).any() and lat.winding(s) == (0, 0, 0)
        for p in np.nonzero(lat.flippable(s))[0]:
            assert lat.flip(s, p).tobytes() in known  # closed under flips


def test_hamiltonian(vacuum):
    lat, states = vacuum
    H = hamiltonian(lat, states, lam=0.5, J=0.5)
    assert np.allclose(H, H.T)
    assert np.allclose(np.diag(H), [0.25 * lat.n_flip(s) for s in states])
    assert set(np.unique(H - np.diag(np.diag(H)))) <= {-0.5, 0.0}
    assert ((H == -0.5).sum(axis=0) == [lat.n_flip(s) for s in states]).all()
    # At the RK point lam = 1 the equal superposition has energy zero.
    assert np.allclose(hamiltonian(lat, states, lam=1.0) @ np.ones(len(states)), 0.0)


def test_exact_thermal(vacuum):
    lat, states = vacuum
    ground = np.linalg.eigvalsh(hamiltonian(lat, states, 0.5, 0.5))[0]
    assert exact_thermal(lat, states, 0.5, 200.0, 0.5)[0] == pytest.approx(ground)
    # At infinite temperature every state is equally likely.
    energy, n_flip = exact_thermal(lat, states, 0.5, 0.0, 0.5)
    assert n_flip == pytest.approx(np.mean([lat.n_flip(s) for s in states]))
    assert energy == pytest.approx(0.25 * n_flip)


@pytest.mark.parametrize("start", sorted(GOLDEN["ed"]))
def test_exact_thermal_reference_values(start):
    lat = Cubic(2)
    states = component(lat, start_state(lat, start, 1))
    n_states, energy, n_flip = GOLDEN["ed"][start]
    assert len(states) == n_states
    assert exact_thermal(lat, states, 0.5, 4.0, 0.5) == pytest.approx((energy, n_flip), rel=1e-12)
