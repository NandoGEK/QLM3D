"""The periodic cubic lattice of the quantum link model."""

import numpy as np

PLANES = ((0, 1), (0, 2), (1, 2))  # the directions (mu, nu) of the plaquettes xy, xz, yz
SIGNS = np.array([1, 1, -1, -1])  # U_p raises the first two links of a plaquette, lowers the rest


class Cubic:
    """A periodic cubic lattice with configurations s of one entry s[link] = 2 E = +-1 per link."""

    def __init__(self, dims):
        """Build the lattice with dims = (Lx, Ly, Lz) sites, or L for a cube."""
        self.dims = (dims, dims, dims) if isinstance(dims, int) else tuple(dims)
        Lx, Ly, Lz = self.dims
        self.n_sites = Lx * Ly * Lz
        self.n_links = 3 * self.n_sites  # link = 3 * site + direction
        self.n_plaq = 3 * self.n_sites  # plaquette = 3 * site + plane
        self.coords = [(a, b, c) for c in range(Lz) for b in range(Ly) for a in range(Lx)]
        self.plaq_links = np.array(
            [self.links_around(x, mu, nu) for x in self.coords for mu, nu in PLANES]
        )

    # ---------------------------------------------------------------- indices

    def site(self, x):
        """Index of the site with coordinates x = (a, b, c)."""
        Lx, Ly, Lz = self.dims
        return x[0] % Lx + Lx * (x[1] % Ly + Ly * (x[2] % Lz))

    def link(self, x, mu):
        """Index of the link from site x in direction mu."""
        return 3 * self.site(x) + mu

    def shift(self, x, mu, step=1):
        """Coordinates of site x moved by `step` in direction mu."""
        y = list(x)
        y[mu] = (y[mu] + step) % self.dims[mu]
        return tuple(y)

    def links_around(self, x, mu, nu):
        """The links of the plaquette at site x in the plane (mu, nu): bottom, right, top, left."""
        return [
            self.link(x, mu),
            self.link(self.shift(x, mu), nu),
            self.link(self.shift(x, nu), mu),
            self.link(x, nu),
        ]

    # ---------------------------------------------------------------- conserved quantities

    def gauss(self, s):
        """Twice the charge at every site: the field leaving it minus the field entering it."""
        charge = np.zeros(self.n_sites, dtype=int)
        for i, x in enumerate(self.coords):
            for mu in range(3):
                leaving = s[self.link(x, mu)]
                entering = s[self.link(self.shift(x, mu, -1), mu)]
                charge[i] += leaving - entering
        return charge

    def winding(self, s):
        """The flux (W_x, W_y, W_z) through the three planes at coordinate 0."""
        flux = [0, 0, 0]
        for x in self.coords:
            for mu in range(3):
                if x[mu] == 0:
                    flux[mu] += int(s[self.link(x, mu)])
        return tuple(f / 2 for f in flux)

    # ---------------------------------------------------------------- flippable plaquettes

    def orient(self, s, p):
        """+1 if U_p can act on plaquette p, -1 if U_p^dag can, 0 if p is not flippable."""
        links = s[self.plaq_links[p]]
        if np.array_equal(links, -SIGNS):
            return 1
        if np.array_equal(links, SIGNS):
            return -1
        return 0

    def is_flippable(self, s, p):
        """Whether plaquette p is flippable."""
        return self.orient(s, p) != 0

    def flippable(self, s):
        """Boolean array: which plaquettes are flippable."""
        links = s[self.plaq_links]
        return np.all(links == -SIGNS, axis=1) | np.all(links == SIGNS, axis=1)

    def n_flip(self, s):
        """Number of flippable plaquettes."""
        return int(self.flippable(s).sum())

    def flip_fractions(self, s):
        """Flippable plaquettes per site in the planes xy, xz, yz."""
        return self.flippable(s).reshape(self.n_sites, 3).sum(axis=0) / self.n_sites

    def flip(self, s, p):
        """A copy of s with the flippable plaquette p flipped."""
        assert self.is_flippable(s, p)
        flipped = s.copy()
        flipped[self.plaq_links[p]] *= -1
        return flipped
