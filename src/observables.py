"""Measurements on a configuration (order parameter) and on a string (photon correlator)."""

import math

import numpy as np

from sse import IDENTITY

W3 = np.exp(2j * np.pi / 3)  # third root of unity in the order parameter Q = f0 + W3 f1 + W3^2 f2
STATE_KEYS = ("nflip", "f0", "f1", "f2", "Q2", "Q4", "ReQ3")  # what state_observables returns
# The layer winding I_a sums k over plaquettes of one plane on a line: a -> (plane, line direction).
LAYER_LINES = {"z": (0, 2), "y": (1, 1), "x": (2, 0)}

# ---------------------------------------------------------------- a configuration


def state_observables(lat, s):
    """Flippable fraction, its parts f0, f1, f2 per plane, and |Q|^2, |Q|^4, Re Q^3 of s."""
    f0, f1, f2 = lat.flip_fractions(s)
    Q = f0 + W3 * f1 + W3**2 * f2
    Q2 = abs(Q) ** 2
    return {
        "nflip": lat.n_flip(s) / lat.n_plaq,
        "f0": f0,
        "f1": f1,
        "f2": f2,
        "Q2": Q2,
        "Q4": Q2 * Q2,
        "ReQ3": (Q**3).real,
    }


# ---------------------------------------------------------------- the photon correlator


def photon_weights(lat):
    """Weights w of the photon mode O = sum_l w_l s_l: E_y with the smallest momentum along x."""
    w = np.zeros(lat.n_links, dtype=complex)
    for x in lat.coords:
        w[lat.link(x, 1)] = 0.5 * np.exp(2j * np.pi * x[0] / lat.dims[0])
    return w


def photon_correlator(sse, w, fractions):
    """The correlator <O(tau) O^dag(0)> of the current string at tau = fractions * beta."""
    mode = mode_between_operators(sse, w)
    if len(mode) == 0:
        return np.full(len(fractions), abs(np.sum(w * sse.state)) ** 2)
    by_distance = autocorrelation(mode)
    return np.array([correlator_at(by_distance, fraction) for fraction in fractions])


def mode_between_operators(sse, w):
    """The mode O on the state entering each operator of the string."""
    mode = []
    s = sse.state
    for t, p in sse.ops:
        if t != IDENTITY:
            mode.append(np.sum(w * s))
            s = sse.apply(s, t, p)
    return np.array(mode)


def autocorrelation(mode):
    """The mean of O[i + m] O[i]^* over the string, for every operator distance m."""
    return np.array([np.mean(np.roll(mode, -m) * np.conj(mode)).real for m in range(len(mode))])


def correlator_at(by_distance, fraction):
    """The correlator at tau = fraction * beta: operator distances weighted binomially."""
    n = len(by_distance)
    if fraction == 0.0:
        return by_distance[0]
    total = 0.0
    for m in range(n + 1):
        total += binomial(n, m, fraction) * by_distance[m % n]
    return total


def binomial(n, m, f):
    """The probability that m of n operators lie in a fraction f of imaginary time."""
    log_probability = (
        math.lgamma(n + 1.0)
        - math.lgamma(m + 1.0)
        - math.lgamma(n - m + 1.0)
        + m * math.log(f)
        + (n - m) * math.log(1.0 - f)
    )
    return math.exp(log_probability)


# ---------------------------------------------------------------- the layer winding


def layer_plaquettes(lat):
    """For each layer winding I_x, I_y, I_z the plaquettes whose k it sums."""
    layers = {}
    for name, (plane, line) in LAYER_LINES.items():
        sites = [[c if mu == line else 0 for mu in range(3)] for c in range(lat.dims[line])]
        layers[name] = [3 * lat.site(x) + plane for x in sites]
    return layers


def layer_winding_nonzero(sse, layers):
    """1.0 if the string winds around the lattice in a whole layer of flips, else 0.0."""
    k = sse.k_vector()
    return float(any(k[plaquettes].sum() != 0 for plaquettes in layers.values()))
