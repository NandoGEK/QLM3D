"""Tests of starts.py: one test per function, in the order of the file."""

import json
from pathlib import Path

import numpy as np
import pytest

from lattice import Cubic
from starts import (
    charged_start,
    field_path,
    flux_loop,
    layered_start,
    outgoing_links,
    random_flips,
    start_state,
    walk_until_closed,
    wound_start,
)

GOLDEN = json.loads((Path(__file__).parent / "golden.json").read_text())


@pytest.fixture
def lat():
    return Cubic(4)


def charges(lat, at):
    """The array `gauss` gives for the charges at = {site: charge}."""
    expected = np.zeros(lat.n_sites, dtype=int)
    for x, charge in at.items():
        expected[lat.site(x)] = 2 * charge
    return expected


def test_start_state(lat):
    assert np.array_equal(start_state(lat, "g", 1), wound_start(lat, 1, (0, 0, 0)))
    assert np.array_equal(start_state(lat, "w0_1_0", 1), wound_start(lat, 1, (0, 1, 0)))
    assert np.array_equal(start_state(lat, "q2", 1), charged_start(lat, 1, 2))


@pytest.mark.parametrize("name", sorted(GOLDEN["starts"]))
def test_start_state_reference_values(lat, name):
    start, seed = name.rsplit("_", 1)
    assert start_state(lat, start, int(seed)).tolist() == GOLDEN["starts"][name]


def test_layered_start(lat):
    s = layered_start(lat)
    assert set(s.tolist()) == {-1, 1}
    assert not lat.gauss(s).any() and lat.winding(s) == (0, 0, 0)


@pytest.mark.parametrize("w", [(0, 0, 0), (1, 0, 0), (0, -1, 0), (0, 0, 2)])
def test_wound_start(lat, w):
    s = wound_start(lat, seed=3, w=w)
    assert not lat.gauss(s).any() and lat.winding(s) == w
    assert not np.array_equal(s, wound_start(lat, seed=4, w=w))


@pytest.mark.parametrize("R", [1, 2])
def test_charged_start(lat, R):
    s = charged_start(lat, seed=3, R=R)
    assert np.array_equal(lat.gauss(s), charges(lat, {(0, 0, 0): -1, (R, 0, 0): 1}))


def test_random_flips(lat):
    start = layered_start(lat)
    s = random_flips(lat, start, np.random.RandomState(1))
    assert not np.array_equal(s, start)
    assert not lat.gauss(s).any() and lat.winding(s) == (0, 0, 0)


def test_outgoing_links(lat):
    s = start_state(lat, "q1", seed=1)
    out = outgoing_links(lat, s, (2, 2, 2))
    assert len(out) == 3  # without charge, three of the six links point away
    for link, mu, step in out:
        assert s[link] == step
        assert link == lat.link((2, 2, 2) if step > 0 else lat.shift((2, 2, 2), mu, -1), mu)
    assert len(outgoing_links(lat, s, (0, 0, 0))) == 2  # charge -1
    assert len(outgoing_links(lat, s, (1, 0, 0))) == 4  # charge +1


def test_field_path(lat):
    s = layered_start(lat)
    path = field_path(lat, s, (0, 0, 0), (2, 0, 0))
    assert len(set(path)) == len(path) >= 2
    s[path] *= -1
    assert np.array_equal(lat.gauss(s), charges(lat, {(0, 0, 0): -1, (2, 0, 0): 1}))


@pytest.mark.parametrize(("mu", "sign"), [(0, 1), (0, -1), (2, -1)])
def test_flux_loop(lat, mu, sign):
    s = layered_start(lat)
    loop = flux_loop(lat, s, mu, sign, np.random.RandomState(1))
    s[loop] *= -1
    expected = [0.0, 0.0, 0.0]
    expected[mu] = -sign
    assert not lat.gauss(s).any() and lat.winding(s) == tuple(expected)


def test_walk_until_closed(lat):
    s = layered_start(lat)
    loop, displacement = walk_until_closed(lat, s, (0, 0, 0), 0, 1, np.random.RandomState(1))
    assert len(set(loop)) == len(loop) >= 4
    assert all(d % 4 == 0 for d in displacement)  # a closed loop ends where it began
    s[loop] *= -1
    assert not lat.gauss(s).any()
