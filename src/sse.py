"""Stochastic series expansion: a Markov chain of operator strings (Yan et al., PRB 99, 165135)."""

import numpy as np

IDENTITY, DIAG, FLIP = 0, 1, 2  # the types of operator in a slot
EMPTY: tuple[int, int] = (IDENTITY, -1)  # a slot without operator


class SSE:
    """One chain: `state` enters slot 0 of the string `ops`, which has n operators in M slots."""

    def __init__(self, lat, lam, state, seed=None, J=0.5, eps=0.25, M0=50):
        """Start from `state` with an empty string of M0 slots."""
        self.lat = lat
        self.lam = lam
        self.J = J
        self.offset = (max(0.0, lam) + eps) * J  # added to H so that every weight is positive
        self.w_flippable = self.offset - lam * J  # weight of DIAG on a flippable plaquette
        self.w_blocked = self.offset  # weight of DIAG on any other plaquette
        self.rng = np.random.RandomState(seed)
        self.state = np.array(state, dtype=np.int8)
        self.ops: list[tuple[int, int]] = [EMPTY] * M0  # one (type, plaquette) per slot
        self.M = M0
        self.n = 0

    # ---------------------------------------------------------------- operators

    def weight(self, s, t, p):
        """Matrix element of the operator (t, p) on the state s."""
        if t == IDENTITY:
            return 1.0
        flippable = self.lat.is_flippable(s, p)
        if t == DIAG:
            return self.w_flippable if flippable else self.w_blocked
        return self.J if flippable else 0.0

    def apply(self, s, t, p):
        """The state after the operator (t, p) has acted on s."""
        return self.lat.flip(s, p) if t == FLIP else s

    # ---------------------------------------------------------------- the string

    def entering_states(self):
        """The state entering each slot."""
        entering = []
        s = self.state
        for t, p in self.ops:
            entering.append(s)
            s = self.apply(s, t, p)
        return entering

    def is_valid(self):
        """Whether every operator has a positive weight and the last state equals the first."""
        s = self.state
        for t, p in self.ops:
            if self.weight(s, t, p) <= 0:
                return False
            s = self.apply(s, t, p)
        return bool(np.array_equal(s, self.state))

    def k_vector(self):
        """Per plaquette, the number of U_p minus the number of U_p^dag in the string."""
        k = np.zeros(self.lat.n_plaq, dtype=int)
        s = self.state
        for t, p in self.ops:
            if t == FLIP:
                k[p] += self.lat.orient(s, p)
            s = self.apply(s, t, p)
        return k

    def energy(self, beta):
        """The energy estimator -n / beta + N_p offset."""
        return -self.n / beta + self.lat.n_plaq * self.offset

    # ---------------------------------------------------------------- sweep

    def sweep(self, beta):
        """One Monte Carlo sweep: diagonal update, N_p cluster attempts, time shift, cutoff."""
        self.diagonal_update(beta)
        for _ in range(self.lat.n_plaq):
            self.cluster_update()
        self.time_shift_update()
        self.adjust_cutoff()

    # ---------------------------------------------------------------- diagonal update

    def diagonal_update(self, beta):
        """Go through the string once, inserting a DIAG in empty slots and removing DIAGs."""
        s = self.state
        for x, (t, p) in enumerate(self.ops):
            if t == IDENTITY:
                self.try_insert(x, s, beta)
            elif t == DIAG:
                self.try_remove(x, s, beta)
            else:
                s = self.apply(s, t, p)

    def try_insert(self, x, s, beta):
        """Put a DIAG on a random plaquette into the empty slot x, where the state is s."""
        p = self.rng.randint(self.lat.n_plaq)
        probability = beta * self.lat.n_plaq * self.weight(s, DIAG, p) / (self.M - self.n)
        if self.rng.rand() < probability:
            self.ops[x] = (DIAG, p)
            self.n += 1

    def try_remove(self, x, s, beta):
        """Remove the DIAG in slot x, where the state is s."""
        p = self.ops[x][1]
        probability = (self.M - self.n + 1) / (beta * self.lat.n_plaq * self.weight(s, DIAG, p))
        if self.rng.rand() < probability:
            self.ops[x] = EMPTY
            self.n -= 1

    # ---------------------------------------------------------------- cluster update

    def flippable_vertices(self, entering=None):
        """The slots whose operator sits on a plaquette that is flippable before and after it."""
        if entering is None:
            entering = self.entering_states()
        return [
            x
            for x, (t, p) in enumerate(self.ops)
            if t == FLIP or (t == DIAG and self.lat.is_flippable(entering[x], p))
        ]

    def cluster_update(self):
        """Propose a sweeping cluster from a random flippable vertex and accept or reject it."""
        entering = self.entering_states()
        vertices = self.flippable_vertices(entering)
        if not vertices:
            return False
        start = vertices[self.rng.randint(len(vertices))]
        proposal = self.grow_cluster(start, entering[start])
        if proposal is None:
            return False

        old = self.ops, self.state
        self.ops, self.state, log_weight_ratio, choices = proposal
        assert self.is_valid()
        # Weight ratio, times the ratio of the ways to pick the start vertex and the 1/2 choices.
        n_before, n_after = len(vertices), len(self.flippable_vertices())
        ratio = np.exp(log_weight_ratio) * n_before / n_after * 2.0**choices
        if self.rng.rand() < ratio:
            return True
        self.ops, self.state = old
        return False

    def grow_cluster(self, start, entering_start):
        """The proposal (ops, state, log weight ratio, choices) of the cluster from slot `start`."""
        links = self.lat.plaq_links
        old_type, p = self.ops[start]
        new_type = FLIP if old_type == DIAG else DIAG
        new_ops = list(self.ops)
        new_ops[start] = (new_type, p)
        log_weight_ratio = np.log(self.weight(entering_start, new_type, p)) - np.log(
            self.weight(entering_start, old_type, p)
        )
        new_state = self.state
        choices = 0  # random 1/2 choices on the way there minus on the way back

        # The lines mark the links where the new state differs from the old one.
        lines = np.zeros(self.lat.n_links, dtype=bool)
        lines[links[p]] = True
        old_below = self.apply(entering_start, old_type, p)
        x = start
        while True:
            x += 1
            if x == self.M:  # the lines cross the end of the string and change `state`
                x = 0
                new_state = self.state.copy()
                new_state[lines] *= -1
            if x == start:  # the cluster came back to where it started: no proposal
                return None

            t, q = self.ops[x]
            old_above = self.apply(old_below, t, q)
            if t != IDENTITY and lines[links[q]].any():
                new_below = old_below.copy()
                new_below[lines] *= -1
                new_type, choice, ends = self.continue_cluster(
                    t, q, old_below, new_below, old_above, int(lines.sum())
                )
                new_above = self.apply(new_below, new_type, q)
                log_weight_ratio += np.log(self.weight(new_below, new_type, q)) - np.log(
                    self.weight(old_below, t, q)
                )
                new_ops[x] = (new_type, q)
                lines[links[q]] = new_above[links[q]] != old_above[links[q]]
                choices += choice
                if ends:
                    return new_ops, new_state, log_weight_ratio, choices
            old_below = old_above

    def continue_cluster(self, t, q, old_below, new_below, old_above, n_lines):
        """For the operator (t, q) hit by the lines: (new type, change of choices, cluster ends)."""
        flippable_below = self.lat.is_flippable(new_below, q)
        flippable_above = self.lat.is_flippable(old_above, q)
        if flippable_below and not flippable_above:  # both types are possible: choose
            return (DIAG if self.rng.rand() < 0.5 else FLIP), 1, False
        if not flippable_below:  # only DIAG is possible, and the way back has a choice here
            return DIAG, (-1 if self.lat.is_flippable(old_below, q) else 0), False
        if n_lines > 4:  # other lines remain: keep the type and go on
            return t, 0, False
        # The four lines of q are the last ones: close them.
        closes_as_diag = np.array_equal(
            new_below[self.lat.plaq_links[q]], old_above[self.lat.plaq_links[q]]
        )
        return (DIAG if closes_as_diag else FLIP), 0, True

    # ---------------------------------------------------------------- time shift and cutoff

    def time_shift_update(self):
        """Rotate the string by a random number of slots."""
        k = self.rng.randint(self.M)
        s = self.state
        for t, p in self.ops[:k]:
            s = self.apply(s, t, p)
        self.ops = self.ops[k:] + self.ops[:k]
        self.state = s

    def adjust_cutoff(self):
        """Append empty slots when more than 70 % of the string is filled."""
        if self.n > 0.7 * self.M:
            longer = max(self.M + 1, self.n + self.n // 3)
            self.ops += [EMPTY] * (longer - self.M)
            self.M = longer  # pyright: ignore[reportConstantRedefinition]
