"""Tests of sse.py.

The first part compares the chain with the plain implementation in reference_sse.py, which does
the same steps slot by slot in Python. The second part has one test per function of SSE.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import reference_sse  # pyright: ignore[reportImplicitRelativeImport]
from lattice import Cubic
from sse import DIAG, EMPTY, FLIP, IDENTITY, NO_NUMBERS, SSE, uniform, use_generator_of
from starts import start_state

GOLDEN = json.loads((Path(__file__).parent / "golden.json").read_text())


class Injected:
    """Random numbers from a list for reference_sse.SSE, with the integer rule of sse.random_below."""

    def __init__(self, numbers):
        """Start at the first of the numbers."""
        self.numbers = numbers
        self.position = 0

    def rand(self):
        """The next number of the list."""
        u = self.numbers[self.position]
        self.position += 1
        return u

    def randint(self, n):
        """An integer below n from the next number of the list."""
        return min(int(self.rand() * n), n - 1)


def new_chain(start="g", lam=0.5, seed=2, **options):
    """A chain on 2x2x2 before its first sweep."""
    lat = Cubic(2)
    return SSE(lat, lam, start_state(lat, start, seed), seed=seed, **options)


def swept_chain(sweeps=5, **options):
    """The chain of new_chain after some sweeps at beta = 2."""
    sse = new_chain(**options)
    for _ in range(sweeps):
        sse.sweep(2.0)
    return sse


def a_flippable_plaquette(sse):
    """The first plaquette that is flippable in the state of the chain."""
    return int(np.nonzero(sse.lat.flippable(sse.state))[0][0])


def a_blocked_plaquette(sse):
    """The first plaquette that is not flippable in the state of the chain."""
    return int(np.nonzero(~sse.lat.flippable(sse.state))[0][0])


def put(sse, *ops):
    """Make the string these operators followed by empty slots."""
    sse.ops = list(ops) + [EMPTY] * (sse.M - len(ops))


def entering_states(sse):
    """The state entering each slot of the string of `sse`, by the plain implementation."""
    old = reference_sse.SSE(sse.lat, sse.lam, sse.state)
    old.ops, old.M, old.n = sse.ops, sse.M, sse.n
    return old.entering_states()


def index_of(sse):
    """What the index says about the string, in a form that can be compared."""
    n = sse.n
    return {
        "rec": sse.rec[:n].copy(),
        "slot_of": sse.slot_of[:n].copy(),
        "first_op": sse.first_op.copy(),
        "parity": sse.parity.copy(),
        "fp_mask": sse.fp_mask.copy(),
        "fenwick": sse.fenwick.copy(),
        "n_fp": int(sse.n_fp[0]),
    }


# ---------------------------------------------------------------- against the plain implementation

IDENTITY_CASES = [
    # dims, lam, beta, start, sweeps
    (2, 0.5, 2.0, "g", 150),
    (2, -0.5, 4.0, "w1_0_0", 150),
    (2, 0.0, 4.0, "q1", 150),
    (2, 0.25, 16.0, "g", 40),
    ((2, 2, 4), 0.3, 3.0, "g", 60),
]
SLOW_IDENTITY_CASES = [
    (4, 0.25, 2.0, "g", 40),
    (4, 0.5, 2.0, "q2", 40),
    (4, 0.25, 4.0, "w1_0_0", 25),
    (4, -1.0, 2.0, "g", 25),
]


def same_chain_as_reference(dims, lam, beta, start, sweeps):
    """Sweep both implementations on the same random numbers and compare them after every sweep."""
    lat = Cubic(dims)
    seed = 11
    numbers = np.random.RandomState(seed).rand(40_000_000 if lat.n_plaq > 100 else 4_000_000)
    new = SSE(lat, lam, start_state(lat, start, seed), seed=seed, abort_lines=0, abort_steps=0)
    new.inject(numbers)
    old = reference_sse.SSE(lat, lam, start_state(lat, start, seed), seed=seed)
    old.rng = Injected(numbers)  # pyright: ignore[reportAttributeAccessIssue]
    flips = 0
    for _ in range(sweeps):
        new.sweep(beta)
        old.sweep(beta)
        assert np.array_equal(new.state, old.state)
        assert new.ops == old.ops and (new.M, new.n) == (old.M, old.n)
        used = old.rng.position  # pyright: ignore[reportAttributeAccessIssue]
        assert new.position[0] == used  # the same number of random numbers was used
        assert new.is_valid() == old.is_valid() is True
        assert np.array_equal(new.k_vector(), old.k_vector())
        flips += sum(t == FLIP for t, _ in new.ops)
    assert flips > 0  # the cluster update did something
    entering = zip(old.entering_states(), old.ops, strict=True)
    assert np.array_equal(new.operator_states(), [s for s, (t, _) in entering if t != IDENTITY])


@pytest.mark.parametrize("case", IDENTITY_CASES, ids=str)
def test_same_chain_as_reference(case):
    same_chain_as_reference(*case)


@pytest.mark.slow
@pytest.mark.parametrize("case", SLOW_IDENTITY_CASES, ids=str)
def test_same_chain_as_reference_4x4x4(case):
    same_chain_as_reference(*case)


@pytest.mark.parametrize("start", ["g", "w1_0_0", "q1"])
def test_index_after_cluster_updates_equals_a_new_index(start):
    lat = Cubic((2, 2, 4))
    sse = SSE(lat, 0.3, start_state(lat, start, 5), seed=5)
    for _ in range(30):
        sse.diagonal_update(3.0)
        for _ in range(20):
            accepted = sse.cluster_updates(lat.n_plaq // 4)
            kept = index_of(sse)
            sse.build_index()
            new = index_of(sse)
            assert kept.keys() == new.keys()
            for key in kept:
                assert np.array_equal(kept[key], new[key]), (key, accepted)
            assert not sse.lines.any()
        sse.time_shift_update()
        sse.adjust_cutoff()
    assert sse.n_fp[0] == len(reference_flippable_vertices(sse))


def reference_flippable_vertices(sse):
    """The flippable vertices of the string of `sse`, counted by the plain implementation."""
    old = reference_sse.SSE(sse.lat, sse.lam, sse.state)
    old.ops, old.M, old.n = sse.ops, sse.M, sse.n
    return old.flippable_vertices()


def test_parity_array_grows():
    sse = new_chain(lam=0.25)  # at beta = 100 a link of the 2x2x2 lattice has over 127 operators
    for _ in range(40):
        sse.sweep(100.0)
    assert sse.parity.shape[1] > 2 and sse.is_valid()
    assert sse.n_fp[0] == len(reference_flippable_vertices(sse))


# ---------------------------------------------------------------- random numbers


def test_use_generator_of():
    a, b = new_chain(seed=7), new_chain(seed=7)
    alone = new_chain(seed=7)
    for _ in range(5):
        alone.sweep(2.0)
    for _ in range(5):  # two chains that take turns do not disturb each other
        a.sweep(2.0)
        b.sweep(2.0)
    assert a.ops == b.ops == alone.ops and np.array_equal(a.state, alone.state)
    assert new_chain(seed=8).ops == [EMPTY] * 50  # another seed ...
    c = swept_chain(seed=8)
    assert c.ops != swept_chain(seed=7).ops  # ... gives another chain


def test_inject():
    sse = new_chain()
    sse.inject([0.25, 0.75])
    assert uniform(sse.numbers, sse.position) == 0.25 and sse.position[0] == 1
    sse.inject([])
    use_generator_of(sse)
    assert 0.0 <= uniform(sse.numbers, sse.position) < 1.0


# ---------------------------------------------------------------- operators


def test_init():
    sse = new_chain(lam=0.5)
    assert sse.ops == [EMPTY] * 50 and (sse.M, sse.n) == (50, 0)
    assert (sse.offset, sse.w_flippable, sse.w_blocked) == (0.375, 0.125, 0.375)
    assert (sse.abort_lines, sse.abort_prob, sse.abort_steps) == (20, 0.1, 32)
    negative = new_chain(lam=-3.0)
    assert negative.w_flippable > 0 and negative.w_blocked > 0


def test_ops():
    sse = new_chain()
    p = a_flippable_plaquette(sse)
    sse.ops = [(DIAG, p), EMPTY, (FLIP, p), (FLIP, p)]
    assert (sse.M, sse.n) == (4, 3)
    assert sse.ops_t.tolist() == [DIAG, IDENTITY, FLIP, FLIP] and sse.ops_p.tolist() == [
        p,
        -1,
        p,
        p,
    ]
    assert sse.ops == [(DIAG, p), EMPTY, (FLIP, p), (FLIP, p)]


def test_weight():
    sse = new_chain(lam=0.5)
    s, p, q = sse.state, a_flippable_plaquette(sse), a_blocked_plaquette(sse)
    assert sse.weight(s, IDENTITY, -1) == 1.0
    assert (sse.weight(s, DIAG, p), sse.weight(s, DIAG, q)) == (0.125, 0.375)
    assert (sse.weight(s, FLIP, p), sse.weight(s, FLIP, q)) == (0.5, 0.0)


# ---------------------------------------------------------------- the string


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


def test_operator_states():
    sse = new_chain()
    assert sse.operator_states().shape == (0, sse.lat.n_links)
    p = a_flippable_plaquette(sse)
    put(sse, (DIAG, p), EMPTY, (FLIP, p), (FLIP, p))
    flipped = sse.lat.flip(sse.state, p)
    assert np.array_equal(sse.operator_states(), [sse.state, sse.state, flipped])


def test_mode():
    sse = swept_chain(sweeps=20)
    w = np.random.RandomState(1).randn(sse.lat.n_links) + 0.5j
    expected = [np.sum(w * s) for s in sse.operator_states()]
    assert len(expected) == sse.n > 0 and np.allclose(sse.mode(w), expected)


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
    flips = 0
    for _ in range(30):
        sse.sweep(2.0)
        assert sse.is_valid()
        assert sse.n == sum(t != IDENTITY for t, _ in sse.ops) and sse.M == len(sse.ops)
        flips += sum(t == FLIP for t, _ in sse.ops)
    assert sse.n > 0 and flips > 0
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
    assert [[t, p] for t, p in sse.ops] == golden["ops"]
    assert (sse.M, sse.n) == (golden["M"], golden["n"])
    assert sse.k_vector().tolist() == golden["k"]
    use_generator_of(sse)
    assert float(uniform(NO_NUMBERS, sse.position)) == golden["rand"]


# ---------------------------------------------------------------- diagonal update


def test_diagonal_update():
    sse = new_chain()
    state = sse.state.copy()
    sse.diagonal_update(beta=2.0)
    assert 0 < sse.n == sum(t == DIAG for t, _ in sse.ops)
    assert sse.is_valid() and np.array_equal(sse.state, state)
    sse.diagonal_update(beta=1e-9)  # at infinite temperature every operator is removed
    assert sse.n == 0 and sse.ops == [EMPTY] * sse.M


def test_diagonal_update_inserts_and_removes():
    sse = new_chain()
    sse.ops = [EMPTY]
    sse.inject([0.0, 0.999])  # plaquette 0, not accepted at beta = 0
    sse.diagonal_update(beta=0.0)
    assert sse.ops == [EMPTY] and sse.n == 0
    sse.inject([0.0, 0.999])  # accepted at a low temperature
    sse.diagonal_update(beta=1e9)
    assert sse.ops == [(DIAG, 0)] and sse.n == 1
    sse.inject([0.999])  # the DIAG stays at a low temperature ...
    sse.diagonal_update(beta=1e9)
    assert sse.ops == [(DIAG, 0)]
    sse.inject([0.999])  # ... and is removed at a high one
    sse.diagonal_update(beta=1e-9)
    assert sse.ops == [EMPTY] and sse.n == 0


def test_build_index():
    sse = new_chain()
    p, q = a_flippable_plaquette(sse), a_blocked_plaquette(sse)
    put(sse, (DIAG, p), (DIAG, q), EMPTY, (FLIP, p), (FLIP, p))
    sse.build_index()
    assert sse.n_fp[0] == 3 and sse.slot_of[:4].tolist() == [0, 1, 3, 4]
    flags = [bool(sse.rec[j, 0] >> 30 & 1) for j in range(4)]
    flips = [bool(sse.rec[j, 0] >> 31 & 1) for j in range(4)]
    assert flags == [True, False, True, True] and flips == [False, False, True, True]
    link = int(sse.lat.plaq_links[p][0])
    assert sse.first_op[link] == 0
    assert int(sse.rec[0, 0]) & (2**30 - 1) in (1, 2)  # the next operator on that link


# ---------------------------------------------------------------- cluster update


def test_cluster_updates():
    sse = new_chain()
    assert sse.cluster_updates(10) == 0  # nothing to start from in an empty string
    sse = new_chain(lam=0.25)
    for _ in range(20):
        sse.sweep(8.0)
    accepted = 0
    for _ in range(200):
        ops, state = sse.ops, sse.state.copy()
        if sse.cluster_updates(1):
            accepted += 1
            assert sse.ops != ops
        else:
            assert sse.ops == ops and np.array_equal(sse.state, state)
        assert sse.is_valid()
    assert 0 < accepted < 200


def test_cluster_update_of_two_operators():
    sse = new_chain(lam=0.5)
    p = a_flippable_plaquette(sse)
    put(sse, (DIAG, p))  # the lines meet no other operator and come back to the start
    sse.inject([0.0])
    assert sse.cluster_updates(1) == 0 and sse.ops[0] == (DIAG, p)
    assert sse.position[0] == 1  # only the start vertex was drawn: there was no proposal

    put(sse, (DIAG, p), (DIAG, p))  # the second operator closes the lines: two FLIPs
    sse.inject([0.0, 0.999])  # weight ratio (0.5 / 0.125)^2 = 16: accepted whatever the number
    assert sse.cluster_updates(1) == 1
    assert sse.ops[:3] == [(FLIP, p), (FLIP, p), EMPTY] and sse.is_valid()
    sse.inject([0.0, 1 / 16 + 1e-9])  # the way back has the ratio 1 / 16
    assert sse.cluster_updates(1) == 0 and sse.ops[:2] == [(FLIP, p), (FLIP, p)]
    sse.inject([0.0, 1 / 16 - 1e-9])
    assert sse.cluster_updates(1) == 1 and sse.ops[:2] == [(DIAG, p), (DIAG, p)]


def test_cluster_update_gives_up_long_walks():
    sse = new_chain(lam=0.5, abort_steps=1, abort_prob=1.0)  # every walk stops at its second vertex
    for _ in range(20):
        sse.sweep(2.0)
        assert sse.is_valid() and not sse.lines.any()
    lat = sse.lat
    assert sse.n_fp[0] == len(reference_flippable_vertices(sse))
    never = new_chain(lam=0.5, abort_lines=1, abort_prob=1.0)  # every walk stops at once
    for _ in range(20):
        never.sweep(2.0)
    assert not any(t == FLIP for t, _ in never.ops)
    assert np.array_equal(never.state, start_state(lat, "g", 2))


# ---------------------------------------------------------------- time shift and cutoff


def test_time_shift_update():
    sse = swept_chain()
    ops = sse.ops
    states = entering_states(sse)
    sse.time_shift_update()
    shifts = [k for k in range(sse.M) if ops[k:] + ops[:k] == sse.ops]
    assert shifts and sse.is_valid()
    assert any(np.array_equal(sse.state, states[k]) for k in shifts)


def test_adjust_cutoff():
    sse = new_chain()
    sse.n = 35
    sse.adjust_cutoff()
    assert sse.M == 50
    sse.n = 40
    sse.adjust_cutoff()
    assert sse.M == len(sse.ops) == 53 and sse.ops[50:] == [EMPTY] * 3
