"""Exact diagonalisation on tiny lattices, among the configurations connected by flips."""

from collections import deque

import numpy as np


def component(lat, s0):
    """All configurations that plaquette flips reach from s0, as rows of an array, s0 first."""
    s0 = np.asarray(s0, dtype=np.int8)
    seen = {s0.tobytes()}
    states = [s0]
    queue = deque([s0])
    while queue:
        s = queue.popleft()
        for p in np.nonzero(lat.flippable(s))[0]:
            flipped = lat.flip(s, p)
            if flipped.tobytes() not in seen:
                seen.add(flipped.tobytes())
                states.append(flipped)
                queue.append(flipped)
    return np.array(states)


def hamiltonian(lat, states, lam, J=1.0):
    """The matrix of H = -J sum_p (U_p + U_p^dag) + lam J N_flip in the basis `states`."""
    index = {s.tobytes(): i for i, s in enumerate(states)}
    H = np.zeros((len(states), len(states)))
    for i, s in enumerate(states):
        H[i, i] = lam * J * lat.n_flip(s)
        for p in np.nonzero(lat.flippable(s))[0]:
            j = index[lat.flip(s, p).tobytes()]
            H[j, i] = -J
    return H


def exact_thermal(lat, states, lam, beta, J=1.0):
    """The exact thermal averages (<H>, <N_flip>) at inverse temperature beta."""
    energies, vectors = np.linalg.eigh(hamiltonian(lat, states, lam, J))
    boltzmann = np.exp(-beta * (energies - energies.min()))
    probability = boltzmann / boltzmann.sum()  # of each eigenstate
    n_flip = np.array([lat.n_flip(s) for s in states])
    n_flip_of_eigenstate = (vectors**2).T @ n_flip
    return float(probability @ energies), float(probability @ n_flip_of_eigenstate)
