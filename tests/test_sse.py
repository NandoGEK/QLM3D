"""Tests of sse.py: one test per function of SSE, in the order of the file."""

import json
from pathlib import Path

import numpy as np
import pytest

from lattice import Cubic
from sse import DIAG, EMPTY, FLIP, IDENTITY, SSE
from starts import start_state

GOLDEN = json.loads((Path(__file__).parent / "golden.json").read_text())


def new_chain(start="g", lam=0.5, seed=2):
    lat = Cubic(2)
    return SSE(lat, lam, start_state(lat, start, seed), seed=seed)


def swept_chain(sweeps=5):
    sse = new_chain()
    for _ in range(sweeps):
        sse.sweep(2.0)
    return sse


def a_flippable_plaquette(sse):
    return int(np.nonzero(sse.lat.flippable(sse.state))[0][0])


def a_blocked_plaquette(sse):
    return int(np.nonzero(~sse.lat.flippable(sse.state))[0][0])


def put(sse, *ops):
    """Make the string these operators followed by empty slots."""
    sse.ops = [EMPTY] * sse.M
    for x, op in enumerate(ops):
        sse.ops[x] = op
    sse.n = len(ops)


# ---------------------------------------------------------------- operators


def test_init():
    sse = new_chain(lam=0.5)
    assert sse.ops == [EMPTY] * 50 and (sse.M, sse.n) == (50, 0)
    assert (sse.offset, sse.w_flippable, sse.w_blocked) == (0.375, 0.125, 0.375)
    negative = new_chain(lam=-3.0)
    assert negative.w_flippable > 0 and negative.w_blocked > 0


def test_weight():
    sse = new_chain(lam=0.5)
    s, p, q = sse.state, a_flippable_plaquette(sse), a_blocked_plaquette(sse)
    assert sse.weight(s, IDENTITY, -1) == 1.0
    assert (sse.weight(s, DIAG, p), sse.weight(s, DIAG, q)) == (0.125, 0.375)
    assert (sse.weight(s, FLIP, p), sse.weight(s, FLIP, q)) == (0.5, 0.0)


def test_apply():
    sse = new_chain()
    s, p = sse.state, a_flippable_plaquette(sse)
    assert sse.apply(s, DIAG, p) is s and sse.apply(s, IDENTITY, -1) is s
    assert np.array_equal(sse.apply(s, FLIP, p), sse.lat.flip(s, p))


# ---------------------------------------------------------------- the string


def test_entering_states():
    sse = new_chain()
    p = a_flippable_plaquette(sse)
    put(sse, (FLIP, p), (FLIP, p))
    entering = sse.entering_states()
    assert len(entering) == sse.M
    assert np.array_equal(entering[0], sse.state)
    assert np.array_equal(entering[1], sse.lat.flip(sse.state, p))
    assert np.array_equal(entering[2], sse.state)


def test_is_valid():
    sse = new_chain()
    p, q = a_flippable_plaquette(sse), a_blocked_plaquette(sse)
    assert sse.is_valid()
    put(sse, (FLIP, p), (FLIP, p))
    assert sse.is_valid()
    put(sse, (FLIP, p))  # the state does not come back
    assert not sse.is_valid()
    put(sse, (FLIP, q))  # weight zero
    assert not sse.is_valid()


def test_k_vector():
    sse = new_chain()
    p = a_flippable_plaquette(sse)
    put(sse, (FLIP, p), (DIAG, p))
    assert sse.k_vector()[p] == sse.lat.orient(sse.state, p) and abs(sse.k_vector()).sum() == 1
    put(sse, (FLIP, p), (FLIP, p))  # U_p and then U_p^dag
    assert not sse.k_vector().any()


def test_energy():
    sse = new_chain(lam=0.5)
    sse.n = 3
    assert sse.energy(beta=2.0) == -1.5 + 24 * 0.375


# ---------------------------------------------------------------- sweep


@pytest.mark.parametrize("start", ["g", "w1_0_0", "q1"])
def test_sweep(start):
    sse = new_chain(start)
    lat = sse.lat
    charges, flux = lat.gauss(sse.state), lat.winding(sse.state)
    for _ in range(30):
        sse.sweep(2.0)
        assert sse.is_valid()
        assert sse.n == sum(t != IDENTITY for t, _ in sse.ops) and sse.M == len(sse.ops)
    assert sse.n > 0 and any(t == FLIP for t, _ in sse.ops)
    assert np.array_equal(lat.gauss(sse.state), charges) and lat.winding(sse.state) == flux


def test_sweep_is_reproducible():
    a, b = new_chain(seed=7), new_chain(seed=7)
    for _ in range(5):
        a.sweep(2.0)
        b.sweep(2.0)
    assert a.ops == b.ops and np.array_equal(a.state, b.state)


def test_sweep_reference_values():
    lat = Cubic((2, 2, 4))
    sse = SSE(lat, 0.3, start_state(lat, "g", 4), seed=4)
    for _ in range(100):
        sse.sweep(3.0)
    golden = GOLDEN["string"]
    assert sse.state.tolist() == golden["state"]
    assert [[int(t), int(p)] for t, p in sse.ops] == golden["ops"]
    assert (sse.M, sse.n) == (golden["M"], golden["n"])
    assert sse.k_vector().tolist() == golden["k"]
    assert float(sse.rng.rand()) == golden["rand"]


# ---------------------------------------------------------------- diagonal update


def test_diagonal_update():
    sse = new_chain()
    state = sse.state.copy()
    sse.diagonal_update(beta=2.0)
    assert 0 < sse.n == sum(t == DIAG for t, _ in sse.ops)
    assert sse.is_valid() and np.array_equal(sse.state, state)
    sse.diagonal_update(beta=1e-9)  # at infinite temperature every operator is removed
    assert sse.n == 0 and sse.ops == [EMPTY] * sse.M


def test_try_insert():
    sse = new_chain()
    sse.try_insert(0, sse.state, beta=0.0)
    assert sse.ops[0] == EMPTY and sse.n == 0
    sse.try_insert(0, sse.state, beta=1e9)
    assert sse.ops[0][0] == DIAG and sse.n == 1


def test_try_remove():
    sse = new_chain()
    put(sse, (DIAG, 0))
    sse.try_remove(0, sse.state, beta=1e9)
    assert sse.ops[0] == (DIAG, 0) and sse.n == 1
    sse.try_remove(0, sse.state, beta=1e-9)
    assert sse.ops[0] == EMPTY and sse.n == 0


# ---------------------------------------------------------------- cluster update


def test_flippable_vertices():
    sse = new_chain()
    p, q = a_flippable_plaquette(sse), a_blocked_plaquette(sse)
    assert sse.flippable_vertices() == []
    put(sse, (DIAG, p), (DIAG, q), (FLIP, p), (FLIP, p))
    assert sse.flippable_vertices() == [0, 2, 3]


def test_cluster_update():
    sse = new_chain()
    assert not sse.cluster_update()  # nothing to start from in an empty string
    sse = swept_chain()
    accepted = 0
    for _ in range(200):
        ops, state = list(sse.ops), sse.state.copy()
        if sse.cluster_update():
            accepted += 1
            assert sse.ops != ops
        else:
            assert sse.ops == ops and np.array_equal(sse.state, state)
        assert sse.is_valid()
    assert 0 < accepted < 200


def test_grow_cluster():
    sse = new_chain(lam=0.5)
    p = a_flippable_plaquette(sse)
    put(sse, (DIAG, p))  # the lines meet no other operator and come back to the start
    assert sse.grow_cluster(0, sse.state) is None

    put(sse, (DIAG, p), (DIAG, p))  # the second operator closes the lines: two FLIPs
    proposal = sse.grow_cluster(0, sse.state)
    assert proposal is not None
    ops, state, log_weight_ratio, choices = proposal
    assert ops[:3] == [(FLIP, p), (FLIP, p), EMPTY]
    assert state is sse.state and choices == 0
    assert log_weight_ratio == pytest.approx(2 * np.log(0.5 / 0.125))
    assert sse.ops[:2] == [(DIAG, p), (DIAG, p)]  # the chain itself is untouched


def test_continue_cluster():
    sse = new_chain()
    s, p, q = sse.state, a_flippable_plaquette(sse), a_blocked_plaquette(sse)
    flipped = sse.lat.flip(s, p)
    # Flippable below, not above: a random choice.
    new_type, choice, ends = sse.continue_cluster(DIAG, p, s, s, np.zeros_like(s), 4)
    assert new_type in (DIAG, FLIP) and (choice, ends) == (1, False)
    # Not flippable below: DIAG, and a choice on the way back only if it was flippable before.
    assert sse.continue_cluster(FLIP, q, s, s, s, 4) == (DIAG, 0, False)
    assert sse.continue_cluster(FLIP, p, s, np.zeros_like(s), s, 4) == (DIAG, -1, False)
    # Flippable on both sides with other lines left: unchanged.
    assert sse.continue_cluster(FLIP, p, s, s, flipped, 8) == (FLIP, 0, False)
    # Flippable on both sides and the last lines: the type that closes them.
    assert sse.continue_cluster(FLIP, p, s, s, s, 4) == (DIAG, 0, True)
    assert sse.continue_cluster(DIAG, p, s, s, flipped, 4) == (FLIP, 0, True)


# ---------------------------------------------------------------- time shift and cutoff


def test_time_shift_update():
    sse = swept_chain()
    ops = list(sse.ops)
    sse.time_shift_update()
    shifts = [k for k in range(sse.M) if ops[k:] + ops[:k] == sse.ops]
    assert shifts and sse.is_valid()
    assert any(np.array_equal(sse.state, swept_chain().entering_states()[k]) for k in shifts)


def test_adjust_cutoff():
    sse = new_chain()
    sse.n = 35
    sse.adjust_cutoff()
    assert sse.M == 50
    sse.n = 40
    sse.adjust_cutoff()
    assert sse.M == len(sse.ops) == 53 and sse.ops[50:] == [EMPTY] * 3
