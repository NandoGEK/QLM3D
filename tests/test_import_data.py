"""Tests of plots/import_data.py: one test per function, in the order of the file."""

import numpy as np

from import_data import chains, dropped_bins, fields, groups


def test_dropped_bins():
    assert [dropped_bins(L) for L in (4, 8, 10, 12, 18)] == [2, 2, 4, 8, 8]


def test_groups():
    found = groups("chains.h5")
    assert found == sorted(found) and "L16/lam0.25/q8" in found and "L4/lam0.5/g" in found
    assert "L16/lam0.5/q8" in groups("fields.h5")


def test_chains():
    frame = chains()
    assert list(frame.columns) == ["L", "lam", "start", "seed", "n_burn", "energy", "correlator"]
    assert sorted({int(n) for n in frame["L"]}) == [4, 6, 8, 10, 12, 14, 16, 18]
    assert {float(lam) for lam in frame["lam"]} == {0.25, 0.5}
    vacuum = frame[(frame["L"] == 8) & (frame["lam"] == 0.25) & (frame["start"] == "g")]
    correlators = np.array(vacuum["correlator"].tolist())
    # the correlator falls from tau = 0 to tau = beta / 2, and only the sector "g" has one
    assert correlators.shape == (len(vacuum), 9) and np.all(np.diff(correlators.mean(axis=0)) < 0)
    assert all(c is None for c in frame[frame["start"] == "w1_0_0"]["correlator"])
    assert np.all(np.asarray(frame["energy"]) < 0)


def test_fields():
    frame = fields()
    assert list(frame.columns) == ["L", "lam", "R", "seed", "field"]
    sector = frame[(frame["L"] == 8) & (frame["R"] == 2) & (frame["lam"] == 0.25)]
    field = np.stack(sector["field"].tolist())
    assert field.shape == (len(sector), 8, 8, 8, 3) and np.all(np.abs(field) <= 0.5)
    # Gauss's law holds in every chain: the divergence is the charge
    parts = [field[..., mu] - np.roll(field[..., mu], 1, axis=mu + 1) for mu in range(3)]
    divergence = np.sum(parts, axis=0)
    assert np.allclose(divergence[:, 0, 0, 0], -1, atol=1e-4)
    assert np.allclose(divergence[:, 2, 0, 0], 1, atol=1e-4)
    assert np.abs(divergence).sum(axis=(1, 2, 3)).max() < 2.01
