"""Start states of a chain: a random configuration with given charges and flux."""

from collections import deque

import numpy as np

UNIT = np.eye(3, dtype=int)  # UNIT[mu] is one step in direction mu
RANDOM_FLIPS = 50  # flip attempts per plaquette that randomise a start state


def start_state(lat, start, seed):
    """The start state "g" (vacuum), "w<x>_<y>_<z>" (flux sector) or "q<R>" (two charges)."""
    if start == "g":
        return wound_start(lat, seed, (0, 0, 0))
    if start.startswith("w"):
        return wound_start(lat, seed, tuple(int(w) for w in start[1:].split("_")))
    assert start.startswith("q"), start
    return charged_start(lat, seed, int(start[1:]))


def layered_start(lat):
    """A fixed state without charges and without flux, for even extents."""
    assert all(extent % 2 == 0 for extent in lat.dims)
    s = np.zeros(lat.n_links, dtype=np.int8)
    for x in lat.coords:
        checkerboard = 1 if (x[0] + x[1]) % 2 == 0 else -1
        s[lat.link(x, 0)] = checkerboard
        s[lat.link(x, 1)] = -checkerboard
        s[lat.link(x, 2)] = 1 if x[0] % 2 == 0 else -1
    return s


def wound_start(lat, seed, w):
    """A random state without charges in the flux sector W = w."""
    rng = np.random.RandomState(seed)
    s = layered_start(lat)
    for mu in range(3):
        while (missing := w[mu] - lat.winding(s)[mu]) != 0:
            # Reversing a loop that runs along -mu raises W_mu by one.
            loop = flux_loop(lat, s, mu, -1 if missing > 0 else 1, rng)
            s[loop] *= -1
    s = random_flips(lat, s, rng)
    assert not lat.gauss(s).any() and lat.winding(s) == tuple(w)
    return s


def charged_start(lat, seed, R):
    """A random state with charge -1 at the origin and +1 at (R, 0, 0)."""
    rng = np.random.RandomState(seed)
    s = layered_start(lat)
    origin, target = (0, 0, 0), (R % lat.dims[0], 0, 0)
    # Reversing a line of field from the origin to the target moves one unit of charge.
    s[field_path(lat, s, origin, target)] *= -1
    s = random_flips(lat, s, rng)

    charge = np.zeros(lat.n_sites, dtype=int)
    charge[lat.site(origin)] = -2
    charge[lat.site(target)] = 2
    assert np.array_equal(lat.gauss(s), charge)
    return s


def random_flips(lat, s, rng):
    """Randomise s by flip attempts, which keep the charges and the flux."""
    for _ in range(RANDOM_FLIPS * lat.n_plaq):
        p = rng.randint(lat.n_plaq)
        if lat.is_flippable(s, p):
            s = lat.flip(s, p)
    return s


def outgoing_links(lat, s, x):
    """The links whose field points away from site x, as (link, direction, step +-1)."""
    forward = [(lat.link(x, mu), mu, +1) for mu in range(3)]
    backward = [(lat.link(lat.shift(x, mu, -1), mu), mu, -1) for mu in range(3)]
    return [f for f in forward if s[f[0]] > 0] + [b for b in backward if s[b[0]] < 0]


def field_path(lat, s, origin, target):
    """The links of a shortest path from origin to target that follows the field."""
    # site -> (previous site, link)
    reached_from: dict[tuple[int, ...], tuple[tuple[int, ...], int] | None] = {origin: None}
    queue = deque([origin])
    while target not in reached_from:
        x = queue.popleft()
        # Neighbours in the order +x, -x, +y, -y, +z, -z.
        for link, mu, step in sorted(outgoing_links(lat, s, x), key=lambda o: (o[1], -o[2])):
            y = lat.shift(x, mu, step)
            if y not in reached_from:
                reached_from[y] = (x, link)
                queue.append(y)

    path = []
    x = target
    while (previous := reached_from[x]) is not None:
        x, link = previous
        path.append(link)
    return path


def flux_loop(lat, s, mu, sign, rng):
    """The links of a closed loop of flux that winds once around direction sign * mu."""
    for _ in range(10000):
        start = lat.coords[rng.randint(lat.n_sites)]
        loop, displacement = walk_until_closed(lat, s, start, mu, sign, rng)
        if np.array_equal(displacement, sign * lat.dims[mu] * UNIT[mu]):
            return loop
    raise RuntimeError("no winding loop found")


def walk_until_closed(lat, s, x, mu, sign, rng):
    """Follow the field from x, preferring sign * mu, until a site repeats: (loop, shift)."""
    steps_to = {x: 0}  # site -> number of steps after which the walk was there
    links = []
    positions = [np.zeros(3, dtype=int)]  # not wrapped around the lattice
    while True:
        out = outgoing_links(lat, s, x)
        preferred = [o for o in out if o[1:] == (mu, sign)]
        if preferred and rng.rand() < 0.7:
            link, nu, step = preferred[0]
        else:
            link, nu, step = out[rng.randint(len(out))]
        links.append(link)
        positions.append(positions[-1] + step * UNIT[nu])
        x = lat.shift(x, nu, step)
        if x in steps_to:
            first = steps_to[x]
            return links[first:], positions[-1] - positions[first]
        steps_to[x] = len(links)
