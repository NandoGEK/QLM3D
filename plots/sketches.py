"""Sketches of the model. They use no data.

    uv run python sketches.py

writes thesis/lattice_sketch.pdf and thesis/cube_sketch.pdf.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = Path(__file__).resolve().parent
BLUE = "#1f5fa8"
ORANGE = "#c2571a"
GREEN = "#3a8f5c"
GREY = "0.45"

plt.style.use(HERE / "thesis.mplstyle")

# ---------------------------------------------------------------- thesis/lattice_sketch.pdf
# A configuration of the two-dimensional model and the flip of one plaquette. Every link of a
# periodic 4 x 4 lattice carries E = +1/2 or -1/2, drawn as an arrow along the positive or
# negative direction of the link. The configuration is made from the pattern
# E_(x,1) = (-1)^(x_1 + x_2) / 2, E_(x,2) = -E_(x,1), which obeys Gauss's law, by flips of
# flippable plaquettes, which keep Gauss's law. Left: the configuration, flippable plaquettes
# shaded. Right: the same after the plaquette operator has acted on the marked plaquette.
L = 4
SEED = 3
FLIPS = 12
MARKED = (1, 1)  # the plaquette whose flip is shown
SIGNS = np.array([1, 1, -1, -1])

# E[a, b, mu] = 2 E on the link from site (a, b) in direction mu.
a, b = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
E = np.zeros((L, L, 2), dtype=int)
E[:, :, 0] = np.where((a + b) % 2 == 0, 1, -1)
E[:, :, 1] = -E[:, :, 0]

# The links of the plaquette at (a, b): bottom, right, top, left.
links = {
    (i, j): [(i, j, 0), ((i + 1) % L, j, 1), (i, (j + 1) % L, 0), (i, j, 1)]
    for i in range(L)
    for j in range(L)
}
rng = np.random.RandomState(SEED)
done = 0
while done < FLIPS:
    p = (rng.randint(L), rng.randint(L))
    values = np.array([E[link] for link in links[p]])
    if abs(values @ SIGNS) == 4 and p != MARKED:
        for link in links[p]:
            E[link] *= -1
        done += 1
values = np.array([E[link] for link in links[MARKED]])
assert abs(values @ SIGNS) == 4, "choose another seed: the marked plaquette is not flippable"
gauss = E[:, :, 0] - np.roll(E[:, :, 0], 1, axis=0) + E[:, :, 1] - np.roll(E[:, :, 1], 1, axis=1)
assert not gauss.any()

flipped = E.copy()
for link in links[MARKED]:
    flipped[link] *= -1

fig, axes = plt.subplots(1, 2, figsize=(6.1, 3.1), layout="constrained")
for ax, field, title in zip(
    axes,
    (E, flipped),
    (r"$|\psi\rangle$", "$U_{\u25a1}\\,|\\psi\\rangle$"),
    strict=True,
):
    for i in range(L):
        for j in range(L):
            values = np.array([field[link] for link in links[i, j]])
            if abs(values @ SIGNS) == 4:
                ax.fill(
                    [i, i + 1, i + 1, i],
                    [j, j, j + 1, j + 1],
                    color="#dbe7f3",
                    zorder=0,
                )
            for mu, (dx, dy) in enumerate(((1, 0), (0, 1))):
                marked = (i, j, mu) in links[MARKED]
                color = BLUE if marked else "0.35"
                ax.plot(
                    [i, i + dx],
                    [j, j + dy],
                    color=color,
                    linewidth=1.6 if marked else 0.9,
                )
                s = field[i, j, mu]
                ax.annotate(
                    "",
                    xy=(i + dx * (0.5 + 0.14 * s), j + dy * (0.5 + 0.14 * s)),
                    xytext=(i + dx * (0.5 - 0.14 * s), j + dy * (0.5 - 0.14 * s)),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": color,
                        "lw": 0,
                        "mutation_scale": 13,
                    },
                )
    # The links that leave the picture continue periodically.
    for k in range(L):
        ax.plot([L, L], [k, k + 1], color="0.75", linewidth=0.9, linestyle=":")
        ax.plot([k, k + 1], [L, L], color="0.75", linewidth=0.9, linestyle=":")
    sites = np.arange(L + 1)
    xs, ys = np.meshgrid(sites, sites)
    ax.scatter(xs, ys, s=14, color="0.15", zorder=3)
    ax.text(MARKED[0] + 0.5, MARKED[1] + 0.5, "\u25a1", ha="center", va="center", color=BLUE)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(-0.3, L + 0.3)
    ax.set_ylim(-0.3, L + 0.3)
    ax.axis("off")
fig.savefig(HERE / "thesis" / "lattice_sketch.pdf")
plt.close(fig)

# ---------------------------------------------------------------- thesis/cube_sketch.pdf
# Left: a site with its six links. Three arrows point away from the site and three towards it,
# which is Gauss's law without charge. Middle: the unit cube with one plaquette of each of the
# three planes, and the circulating arrows of a flippable plaquette. Right: the layer of links
# that leave one plane of a 4 x 4 x 4 lattice. The flux through the plane is the sum of the field
# on them.
UNIT = np.eye(3)
fig = plt.figure(figsize=(7.2, 2.7), layout="constrained")

# A site with its six links: the field leaves along three of them and enters along three.
site = fig.add_subplot(1, 3, 1, projection="3d")
away = {
    (0, 1): True,
    (0, -1): False,
    (1, 1): False,
    (1, -1): True,
    (2, 1): True,
    (2, -1): False,
}
for (mu, side), leaves in away.items():
    end = side * UNIT[mu]
    site.plot(*zip((0, 0, 0), end, strict=True), color=GREY, linewidth=0.9)
    start = 0.25 * end if leaves else 0.8 * end
    step = (0.5 if leaves else -0.5) * end
    site.quiver(*start, *step, color=BLUE if leaves else ORANGE, linewidth=1.6)
    site.scatter(*end, s=10, color="0.15")
site.scatter(0, 0, 0, s=26, color="0.15")
site.text(0.12, 0.0, -0.32, r"$x$")
site.text(1.12, 0, 0, r"$x + \hat{1}$", fontsize=8)
site.text(0, 1.1, 0, r"$x + \hat{2}$", fontsize=8)
site.text(0, 0, 1.12, r"$x + \hat{3}$", fontsize=8)
site.set_title(r"$\rho(x) = 0$: three out, three in")

# The unit cube with one plaquette per plane. The plaquette in the 1-2 plane is flippable.
cube = fig.add_subplot(1, 3, 2, projection="3d")
for a in (0, 1):
    for b in (0, 1):
        cube.plot([0, 1], [a, a], [b, b], color=GREY, linewidth=0.8)
        cube.plot([a, a], [0, 1], [b, b], color=GREY, linewidth=0.8)
        cube.plot([a, a], [b, b], [0, 1], color=GREY, linewidth=0.8)
faces = {
    BLUE: [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
    ORANGE: [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
    GREEN: [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)],
}
for color, corners in faces.items():
    cube.add_collection3d(Poly3DCollection([corners], facecolor=color, alpha=0.22))
corners = np.array(faces[BLUE], dtype=float)
for start, end in zip(corners, np.roll(corners, -1, axis=0), strict=True):
    cube.quiver(
        *(start + 0.2 * (end - start)),
        *(0.6 * (end - start)),
        color=BLUE,
        linewidth=1.6,
    )
cube.text(0.5, 0.5, 0.02, "1-2", color=BLUE, ha="center", fontsize=8)
cube.text(0.5, 0.0, 0.5, "1-3", color=ORANGE, ha="center", fontsize=8)
cube.text(0.0, 0.5, 0.5, "2-3", color=GREEN, ha="center", fontsize=8)
cube.set_title("plaquettes of the three planes")

# The links that leave the plane x_1 = 1 of a 4 x 4 x 4 lattice.
SIZE = 4
PLANE = 1  # the plane whose flux is shown
flux = fig.add_subplot(1, 3, 3, projection="3d")
grid = np.arange(SIZE)
for a in grid:
    for b in grid:
        flux.plot([0, SIZE - 1], [a, a], [b, b], color="0.8", linewidth=0.5)
        flux.plot([a, a], [0, SIZE - 1], [b, b], color="0.8", linewidth=0.5)
        flux.plot([a, a], [b, b], [0, SIZE - 1], color="0.8", linewidth=0.5)
top = SIZE - 1
corners = [(PLANE, 0, 0), (PLANE, top, 0), (PLANE, top, top), (PLANE, 0, top)]
flux.add_collection3d(Poly3DCollection([corners], facecolor=BLUE, alpha=0.15))
for b in grid:
    for c in grid:
        flux.quiver(PLANE + 0.15, b, c, 0.7, 0, 0, color=BLUE, linewidth=1.3)
flux.text(PLANE, top, top + 0.5, r"$x_1 = n$", color=BLUE, ha="center", fontsize=8)
flux.set_title(r"links that carry the flux $\Phi_1(n)$")

for ax, limit in ((site, (-1.0, 1.0)), (cube, (0.0, 1.0)), (flux, (0.0, top))):
    ax.set_xlim(*limit)
    ax.set_ylim(*limit)
    ax.set_zlim(*limit)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-58)
    ax.set_axis_off()
fig.savefig(HERE / "thesis" / "cube_sketch.pdf")
plt.close(fig)
