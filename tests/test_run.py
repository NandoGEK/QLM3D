"""Tests of run.py: one test per function, in the order of the file."""

import json
from pathlib import Path

import numpy as np
import pytest

from lattice import Cubic
from run import (
    Job,
    check,
    job_of,
    jobs,
    main,
    mean_and_error,
    measure_bin,
    read_rows,
    run_chain,
    run_chains,
    sector,
)
from sse import SSE
from starts import start_state

GOLDEN = json.loads((Path(__file__).parent / "golden.json").read_text())
SHORT = Job(L=2, lam=0.5, beta=2.0, n_burn=5, n_sweeps=20, n_bins=2, seed=1, start="g")
FULL_KEYS = {"energy", "nflip", "f0", "f1", "f2", "Q2", "Q4", "ReQ3", "Ipos"}


def without_time(row):
    """A row without its run time, which differs from run to run."""
    return {key: value for key, value in row.items() if key != "seconds"}


@pytest.fixture
def results(tmp_path):
    """A results file with three rows: one chain twice (and an empty line), and a second chain."""
    row = {"L": 2, "lam": 0.5, "beta": 2.0, "n_burn": 5, "n_sweeps": 20, "seed": 1,
           "start": "g", "bins": [{"energy": 0.0}] * 2}  # fmt: skip
    full = {**row, "seed": 2, "bins": [{"energy": 0.0, "Q2": 0.0}] * 2}
    path = tmp_path / "chains.jsonl"
    path.write_text("\n\n".join(json.dumps(r) for r in (row, full, row)) + "\n")
    return path


# ---------------------------------------------------------------- one chain


@pytest.mark.parametrize(
    ("start", "observables", "bin_keys", "has_field"),
    [
        ("g", "full", FULL_KEYS | {"ct"}, False),
        ("g", "energy", {"energy", "ct"}, False),
        ("w1_0_0", "energy", {"energy"}, False),
        ("q1", "energy", {"energy"}, True),
    ],
)
def test_run_chain(start, observables, bin_keys, has_field):
    job = SHORT._replace(start=start, observables=observables)
    row = run_chain(*job)
    assert len(row["bins"]) == 2 and set(row["bins"][0]) == bin_keys
    assert ("field" in row) == has_field
    assert row["bad"] == 0 and job_of(row) == job
    json.dumps(row)  # a row can be stored as JSON


def assert_same_row(row, expected):
    """The same chain: integers and texts equal, measured numbers equal up to rounding."""
    # Sums of floating-point numbers come out differently in the last digits on another processor.
    assert row.keys() == expected.keys()
    for key in expected:
        if key == "bins":
            assert len(row["bins"]) == len(expected["bins"])
            for measured, stored in zip(row["bins"], expected["bins"], strict=True):
                assert measured.keys() == stored.keys()
                for name in stored:
                    assert measured[name] == pytest.approx(stored[name], rel=1e-9, abs=1e-12), name
        elif key == "field":
            assert row[key] == pytest.approx(expected[key], abs=1e-9)
        else:
            assert row[key] == expected[key], key


@pytest.mark.parametrize("row", [r for r in GOLDEN["rows"] if r["L"] == 2], ids=job_of)
def test_run_chain_reference_values(row):
    assert_same_row(without_time(run_chain(*job_of(row))), row)


@pytest.mark.parametrize("row", [r for r in GOLDEN["rows"] if r["L"] > 2], ids=job_of)
def test_run_chain_4x4x4_reference_values(row):
    assert_same_row(without_time(run_chain(*job_of(row))), row)


def test_sector():
    lat = Cubic(2)
    s = start_state(lat, "q1", seed=1)
    assert sector(lat, s) == (lat.gauss(s).tolist(), lat.winding(s))
    assert sector(lat, lat.flip(s, int(np.nonzero(lat.flippable(s))[0][0]))) == sector(lat, s)
    assert sector(lat, s) != sector(lat, start_state(lat, "g", seed=1))


def test_measure_bin():
    lat = Cubic(2)
    sse = SSE(lat, 0.5, start_state(lat, "g", 1), seed=1)
    means, state_sum = measure_bin(sse, beta=2.0, n_sweeps=12, observables="full", photon=True)
    assert set(means) == FULL_KEYS | {"ct"} and len(means["ct"]) == 9
    assert 0 <= means["nflip"] <= 1 and means["Ipos"] in (0.0, 1.0)
    assert state_sum.shape == (lat.n_links,) and abs(state_sum).max() <= 12
    means, _ = measure_bin(sse, beta=2.0, n_sweeps=1, observables="energy", photon=False)
    assert means == {"energy": sse.energy(2.0)}


def test_run_chains():
    job_list = [SHORT, SHORT._replace(seed=2)]
    serial = [without_time(row) for row in run_chains(job_list)]
    parallel = [without_time(row) for row in run_chains(job_list, workers=2)]
    assert serial == parallel and serial[0] != serial[1]
    assert [job_of(row) for row in serial] == job_list


# ---------------------------------------------------------------- results files


def test_read_rows(results):
    assert [row["seed"] for row in read_rows(results)] == [1, 2, 1]


def test_job_of(results):
    energy_only, full, _ = read_rows(results)
    assert job_of(energy_only) == SHORT._replace(observables="energy")
    assert job_of(full) == SHORT._replace(seed=2, observables="full")
    assert job_of({**full, "observables": "energy"}).observables == "energy"


def test_jobs(results):
    assert jobs(results) == [SHORT._replace(observables="energy"), SHORT._replace(seed=2)]


# ---------------------------------------------------------------- check


def test_check(capsys):
    worst = check(n_chains=2, n_sweeps=40, n_burn=10, workers=1, lams=(0.5,))
    printed = capsys.readouterr().out
    assert worst > 0 and printed.count("ED") == 6  # energy and nflip for three starts
    assert f"largest deviation {worst:.2f} sigma" in printed


def test_mean_and_error():
    rows = [{"bins": [{"x": 1.0}, {"x": 3.0}]}, {"bins": [{"x": 4.0}]}]  # chain means 2 and 4
    mean, error = mean_and_error(rows, "x")
    assert (mean, error) == (3.0, pytest.approx(np.sqrt(2) / np.sqrt(2)))


# ---------------------------------------------------------------- command line


def test_main(results, capsys):
    main(["jobs", str(results)])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("L=2 lam=0.5 beta=2.0 n_burn=5 n_sweeps=20 n_bins=2 seed=1 start=g")
    assert lines[-1] == "2 chains"

    options = "--L 2 --lam 0.5 --beta 2 --start g --n_burn 5 --n_sweeps 20 --n_bins 2"
    main(["chain", *options.split(), "--observables", "energy"])
    row = json.loads(capsys.readouterr().out)
    assert job_of(row) == SHORT._replace(observables="energy")

    main([])  # no command: the help text
    assert "usage: run.py" in capsys.readouterr().out
