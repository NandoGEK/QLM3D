"""Tests of lattice.py: one test per function of Cubic, in the order of the file."""

import json
from pathlib import Path

import numpy as np
import pytest

from lattice import Cubic
from starts import layered_start, start_state

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


# ---------------------------------------------------------------- indices


def test_init():
    lat = Cubic((2, 4, 6))
    assert lat.dims == (2, 4, 6) and Cubic(4).dims == (4, 4, 4)
    assert (lat.n_sites, lat.n_links, lat.n_plaq) == (48, 144, 144)
    assert lat.plaq_links.shape == (144, 4)
    assert (np.bincount(lat.plaq_links.ravel()) == 4).all()  # every link is in four plaquettes


def test_site(lat):
    assert [lat.site(x) for x in lat.coords] == list(range(lat.n_sites))
    assert lat.site((1, 2, 3)) == 1 + 4 * 2 + 16 * 3
    assert lat.site((4, -1, 0)) == lat.site((0, 3, 0))


def test_link(lat):
    assert lat.link((1, 2, 3), 2) == 3 * lat.site((1, 2, 3)) + 2


def test_shift(lat):
    assert lat.shift((1, 2, 3), 0) == (2, 2, 3)
    assert lat.shift((3, 0, 0), 0) == (0, 0, 0)
    assert lat.shift((0, 0, 0), 2, -1) == (0, 0, 3)


def test_links_around(lat):
    bottom, right, top, left = lat.links_around((0, 0, 0), 0, 1)
    assert bottom == lat.link((0, 0, 0), 0) and right == lat.link((1, 0, 0), 1)
    assert top == lat.link((0, 1, 0), 0) and left == lat.link((0, 0, 0), 1)
    assert lat.plaq_links[0].tolist() == [bottom, right, top, left]


# ---------------------------------------------------------------- conserved quantities


def test_gauss(lat):
    s = layered_start(lat)
    assert not lat.gauss(s).any()
    s[lat.link((0, 0, 0), 0)] = -1  # this link pointed from (0, 0, 0) to (1, 0, 0)
    assert np.array_equal(lat.gauss(s), charges(lat, {(0, 0, 0): -1, (1, 0, 0): 1}))


def test_winding(lat):
    assert lat.winding(np.ones(lat.n_links, dtype=np.int8)) == (8, 8, 8)
    assert lat.winding(layered_start(lat)) == (0, 0, 0)


# ---------------------------------------------------------------- flippable plaquettes


def test_orient(lat):
    s = np.ones(lat.n_links, dtype=np.int8)
    assert lat.orient(s, 0) == 0
    s[lat.plaq_links[0]] = [1, 1, -1, -1]
    assert lat.orient(s, 0) == -1
    s[lat.plaq_links[0]] = [-1, -1, 1, 1]
    assert lat.orient(s, 0) == 1


def test_is_flippable(lat):
    s = layered_start(lat)
    assert lat.is_flippable(s, 0) and not lat.is_flippable(s, 1)


def test_flippable(lat):
    s = start_state(lat, "g", seed=1)
    assert lat.flippable(s).tolist() == [lat.is_flippable(s, p) for p in range(lat.n_plaq)]


def test_n_flip(lat):
    assert lat.n_flip(layered_start(lat)) == lat.n_sites  # every xy plaquette


def test_flip_fractions(lat):
    assert lat.flip_fractions(layered_start(lat)).tolist() == [1.0, 0.0, 0.0]


def test_flip(lat):
    s = start_state(lat, "w1_0_0", seed=1)
    p = int(np.nonzero(lat.flippable(s))[0][0])
    flipped = lat.flip(s, p)
    assert lat.orient(flipped, p) == -lat.orient(s, p)
    assert (flipped != s).sum() == 4 and np.array_equal(lat.flip(flipped, p), s)
    assert np.array_equal(lat.gauss(flipped), lat.gauss(s))
    assert lat.winding(flipped) == lat.winding(s)
