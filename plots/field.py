"""All figures of the electric field of two static charges, from the chains with charges.

The charges are -1 at the origin and +1 at (R, 0, 0). Every chain stores the mean field on every
link. The reference is the lattice Coulomb field: the solution of the lattice Poisson equation
plus the uniform background -R / L^3 on the links along the axis. Nothing is fitted.

    uv run python field.py

writes thesis/field_cube.pdf, field_map.pdf, field_all.pdf, field_transverse.pdf, flux_planes.pdf
and presentation/field_line.png, potential_line.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, LogNorm

import import_data as d

HERE = Path(__file__).resolve().parent
LAMS = [0.25, 0.5]
BLUE = "#1f5fa8"
COLORS = {0.25: BLUE, 0.5: "#c2571a"}
MARKERS = {0.25: "o", 0.5: "s"}
L, R = 16, 8  # the sector of the figures that show one sector


def coulomb_field(n, distance, background=True):
    """The lattice Coulomb field E[a, b, c, mu] on the link from site (a, b, c) in direction mu, and its potential."""
    rho = np.zeros((n, n, n))
    rho[0, 0, 0] = -1.0
    rho[distance, 0, 0] = 1.0
    k = 2 * np.pi * np.fft.fftfreq(n)
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    k2 = 4 * (np.sin(kx / 2) ** 2 + np.sin(ky / 2) ** 2 + np.sin(kz / 2) ** 2)
    k2[0, 0, 0] = 1.0  # rho(k = 0) = 0: the charges are neutral
    phi = np.fft.ifftn(np.fft.fftn(rho) / k2).real
    field = np.stack([phi - np.roll(phi, -1, axis=mu) for mu in range(3)], axis=-1)
    if background:
        field[..., 0] -= distance / n**3
    return field, phi


# The field of every chain of every sector: fields[L, R, lam][chain, a, b, c, mu].
frame = d.fields()
fields = {}
for n, distance, lam in sorted(set(zip(frame["L"], frame["R"], frame["lam"], strict=True))):
    sector = frame[(frame["L"] == n) & (frame["R"] == distance) & (frame["lam"] == lam)]
    fields[int(n), int(distance), float(lam)] = np.stack(sector["field"].tolist())
mean = {sector: field.mean(axis=0) for sector, field in fields.items()}
error = {s: field.std(axis=0, ddof=1) / np.sqrt(len(field)) for s, field in fields.items()}
coulomb, potential = coulomb_field(L, R)
x0 = L // 2 - (R + 1) // 2  # the shift that puts the two charges in the middle of the line

plt.style.use(HERE / "thesis.mplstyle")

# ---------------------------------------------------------------- thesis/field_cube.pdf
# The field in the whole lattice, measured and predicted: one arrow per link above the threshold.
LAM = 0.25
THRESHOLD = 0.004  # three times the statistical error of a link
shift = (L // 2 - R // 2, L // 2, L // 2)  # the charges are moved to the middle of the box
blues = LinearSegmentedColormap.from_list("blues", plt.get_cmap("Blues")(np.linspace(0.3, 1, 256)))
norm = LogNorm(vmin=THRESHOLD, vmax=0.17)
fig = plt.figure(figsize=(7.2, 3.6), layout="constrained")
axes = []
panels = ((mean[L, R, LAM], rf"measured, $\lambda / J = {LAM:g}$"), (coulomb, "lattice Coulomb"))
for column, (values, title) in enumerate(panels):
    ax = fig.add_subplot(1, 2, column + 1, projection="3d")
    axes.append(ax)
    values = np.roll(values, shift, axis=(0, 1, 2))
    a, b, c, mu = np.nonzero(np.abs(values) >= THRESHOLD)
    value = values[a, b, c, mu]
    # An arrow covers 80 % of its link and points along the field.
    step = np.zeros((len(value), 3))
    step[np.arange(len(value)), mu] = 0.8 * np.sign(value)
    middle = np.stack([a, b, c], axis=1).astype(float)
    middle[np.arange(len(value)), mu] += 0.5
    colors = blues(norm(np.abs(value)))
    ax.quiver(
        *(middle - step / 2).T,
        *step.T,
        colors=np.concatenate([colors, np.repeat(colors, 2, axis=0)]),
        linewidth=0.7,
        arrow_length_ratio=0.45,
    )
    ax.scatter(*shift, s=38, color="white", edgecolor="0.1", depthshade=False)
    ax.scatter(shift[0] + R, shift[1], shift[2], s=38, color="0.1", depthshade=False)
    for p in (0, L):  # the edges of the lattice
        for q in (0, L):
            ax.plot([0, L], [p, p], [q, q], color="0.7", linewidth=0.5)
            ax.plot([p, p], [0, L], [q, q], color="0.7", linewidth=0.5)
            ax.plot([p, p], [q, q], [0, L], color="0.7", linewidth=0.5)
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_zlim(0, L)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=16, azim=-66)
    ax.set_axis_off()
    ax.set_title(title)
fig.colorbar(ScalarMappable(norm=norm, cmap=blues), ax=axes, label=r"$|E_l|$", shrink=0.7, pad=0.02)
fig.savefig(HERE / "thesis" / "field_cube.pdf")
plt.close(fig)

# ---------------------------------------------------------------- thesis/field_map.pdf
# The field in the plane of the charges at both couplings and predicted. The uniform background
# is subtracted in all three panels.
panels = []
for lam in LAMS:
    title = rf"measured, $\lambda / J = {lam:g}$ ({len(fields[L, R, lam])} chains)"
    panels.append((title, mean[L, R, lam] - mean[L, R, lam].mean(axis=(0, 1, 2))))
panels.append(("lattice Coulomb", coulomb_field(L, R, background=False)[0]))
y0 = L // 2
a, b = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
ticks = range(0, L, 2)
norm = LogNorm(vmin=1e-3, vmax=0.2, clip=True)
fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2), layout="constrained")
for ax, (title, values) in zip(axes, panels, strict=True):
    plane = np.roll(values[:, :, 0, :], (x0, y0), axis=(0, 1))
    ex = (plane[:, :, 0] + np.roll(plane[:, :, 0], -1, axis=1)) / 2
    ey = (plane[:, :, 1] + np.roll(plane[:, :, 1], -1, axis=0)) / 2
    size = np.hypot(ex, ey)
    # One cell per plaquette, from site (a, b) to (a + 1, b + 1). Its borders are the links.
    cells = ax.pcolormesh(
        np.arange(L + 1),
        np.arange(L + 1),
        size.T,
        cmap="Blues",
        norm=norm,
        edgecolors="white",
        linewidth=0.6,
    )
    # The arrows have unit length. They are white on the dark cells next to the charges.
    ax.quiver(
        a + 0.5,
        b + 0.5,
        ex / np.maximum(size, 1e-12),
        ey / np.maximum(size, 1e-12),
        color=["white" if v > 0.03 else "0.25" for v in size.ravel()],
        pivot="middle",
        scale_units="xy",
        scale=1.7,
        width=0.035 / L,
        headwidth=5,
        headlength=6,
        headaxislength=5.5,
        zorder=2,
    )
    ax.scatter([x0, x0 + R], [y0, y0], s=150, c="white", edgecolors="0.15", linewidths=1, zorder=3)
    ax.text(x0, y0, "−", ha="center", va="center_baseline", fontsize=10, zorder=4)
    ax.text(x0 + R, y0, "+", ha="center", va="center_baseline", fontsize=10, zorder=4)
    ax.set_aspect("equal")
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_xticks(ticks, [str((i - x0) % L) for i in ticks])
    ax.set_yticks(ticks, [str((i - y0) % L) for i in ticks])
    ax.set_xlabel(r"$x_1$")
    ax.set_title(title, fontsize=10)
axes[0].set_ylabel(r"$x_2$")
fig.colorbar(
    ScalarMappable(norm=norm, cmap="Blues"),
    ax=axes,
    label=r"$\sqrt{E_1^2 + E_2^2}$, uniform background subtracted",
    shrink=0.85,
)
fig.suptitle(rf"plane $x_3 = 0$, $L = {L}$, $R = {R}$")
fig.savefig(HERE / "thesis" / "field_map.pdf")
plt.close(fig)

# ---------------------------------------------------------------- thesis/field_all.pdf
# The measured against the predicted field for all sectors.
# Top: the links of all sectors with L >= MIN_L are grouped by the magnitude of the predicted
# field. In each group the measured field, projected on the direction of the predicted one, is
# averaged over the links of a sector, with the error from the spread between chains, and the
# sectors are combined with their inverse variances.
# Bottom left: (measured - predicted) / error on all links of the same sectors.
# Bottom right: chi^2 per link of every sector.
MIN_L = 8  # lattice sizes of the two panels that pool the links
MIN_CHAINS = 20  # sectors with fewer chains have too coarse an error per link
EDGES = np.geomspace(2e-4, 0.4, 16)  # groups of the predicted magnitude
pulls = {lam: [] for lam in LAMS}
ratios = {lam: [[] for _ in EDGES[1:]] for lam in LAMS}
chi2_per_link = []
for (n, distance, lam), field in sorted(fields.items()):
    if len(field) < MIN_CHAINS:
        continue
    expected = coulomb_field(n, distance)[0]
    pull = (mean[n, distance, lam] - expected) / error[n, distance, lam]
    chi2_per_link.append((n, distance, lam, float((pull**2).mean())))
    if n < MIN_L:
        continue
    pulls[lam].append(pull.ravel())
    group = np.digitize(np.abs(expected), EDGES) - 1
    projected = field * np.sign(expected)
    for g in range(len(EDGES) - 1):
        inside = group == g
        if inside.any():
            per_chain = projected[:, inside].mean(axis=1, dtype=float)
            size = np.abs(expected[inside]).mean()
            spread = per_chain.std(ddof=1) / np.sqrt(len(per_chain))
            ratios[lam][g].append((size, per_chain.mean() / size, spread / size))

fig = plt.figure(figsize=(7.2, 5.4), layout="constrained")
grid = fig.add_gridspec(2, 2)
top = fig.add_subplot(grid[0, :])
left = fig.add_subplot(grid[1, 0])
right = fig.add_subplot(grid[1, 1])
top.axhline(1.0, color="0.3", linewidth=1.0)
for lam in LAMS:
    xs, ys, es = [], [], []
    for entries in ratios[lam]:
        if not entries:
            continue
        size, ratio, ratio_error = (np.array(v) for v in zip(*entries, strict=True))
        if np.all(ratio_error < 1e-5):
            # The group holds only the six links at each charge. Their sum is fixed by Gauss's
            # law, so their mean agrees exactly and tests nothing.
            continue
        weight = 1 / ratio_error**2
        xs.append(np.sum(weight * size) / np.sum(weight))
        ys.append(np.sum(weight * ratio) / np.sum(weight))
        es.append(1 / np.sqrt(np.sum(weight)))
    top.errorbar(
        np.array(xs) * (1.04 if lam == LAMS[0] else 0.96),
        ys,
        yerr=es,
        color=COLORS[lam],
        linestyle="none",
        marker=MARKERS[lam],
        markersize=4.5,
        elinewidth=0.9,
        capsize=2,
        label=rf"$\lambda / J = {lam:g}$",
    )
top.set_xscale("log")
top.set_ylim(0.8, 1.2)
top.set_xlabel(r"predicted field $|E_l^{\mathrm{Coulomb}}|$")
top.set_ylabel("measured / predicted")
top.set_title(rf"all links of all sectors with $L \geq {MIN_L}$", loc="left")
top.legend(frameon=False, loc="upper right")

z = np.linspace(-5, 5, 201)
for lam in LAMS:
    values = np.concatenate(pulls[lam])
    print(f"lam = {lam}: {len(values)} links, chi^2 per link {np.mean(values**2):.3f}")
    left.hist(
        values,
        bins=np.linspace(-5, 5, 51).tolist(),
        density=True,
        histtype="step",
        color=COLORS[lam],
        linewidth=1.1,
        label=rf"$\lambda / J = {lam:g}$",
    )
left.plot(z, np.exp(-(z**2) / 2) / np.sqrt(2 * np.pi), color="0.3", linewidth=1.0, label="normal")
left.set_xlim(-5, 5)
left.set_xlabel(r"$(E_l - E_l^{\mathrm{Coulomb}})\,/\,\mathrm{error}$")
left.set_ylabel("density")
left.legend(frameon=False, loc="upper left", fontsize=7)

# L is ordered, so one hue from light to dark.
sizes = sorted({n for n, _, _, _ in chi2_per_link})
shades = plt.get_cmap("Blues")(np.linspace(0.3, 1.0, len(sizes)))
right.axhline(1.0, color="0.3", linewidth=1.0)
for shade, n in zip(shades, sizes, strict=True):
    for lam in LAMS:
        points = [
            (d, chi2) for size, d, coupling, chi2 in chi2_per_link if (size, coupling) == (n, lam)
        ]
        right.plot(
            [d / n + (0.006 if lam == LAMS[1] else -0.006) for d, _ in points],
            [chi2 for _, chi2 in points],
            color=shade,
            linestyle="none",
            marker=MARKERS[lam],
            markersize=4.5,
            markerfacecolor=shade if lam == LAMS[0] else "white",
        )
    right.plot([], [], color=shade, marker="o", linestyle="none", label=f"$L = {n}$")
right.set_ylim(0.8, 2.9)
right.set_xlabel(r"$R / L$")
right.set_ylabel(r"$\chi^2$ per link")
right.legend(frameon=False, loc="upper right", ncols=2, fontsize=7, columnspacing=0.8)
for ax in (top, left, right):
    ax.grid(color="0.92", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
fig.savefig(HERE / "thesis" / "field_all.pdf")
plt.close(fig)

# ---------------------------------------------------------------- thesis/field_transverse.pdf
# -E_1 on the links of the two layers in the middle between the charges, against the distance r
# from the line through the charges. Links at the same r are averaged within every chain.
layers = [R // 2 - 1, R // 2]
side = np.minimum(np.arange(L), L - np.arange(L))
r2 = (side[:, None] ** 2 + side[None, :] ** 2).ravel()
shells = np.unique(r2)
expected = np.array([-coulomb[layers, :, :, 0].reshape(2, -1)[:, r2 == s].mean() for s in shells])
fig, ax = plt.subplots(figsize=(6.4, 4.2), layout="constrained")
ax.plot(np.sqrt(shells), expected, color="0.35", linewidth=1.2, label="lattice Coulomb", zorder=1)
for lam in LAMS:
    middle = -fields[L, R, lam][:, layers, :, :, 0].reshape(len(fields[L, R, lam]), 2, -1)
    per_chain = np.array([middle[:, :, r2 == s].mean(axis=(1, 2), dtype=float) for s in shells]).T
    value = per_chain.mean(axis=0)
    spread = per_chain.std(axis=0, ddof=1) / np.sqrt(len(per_chain))
    chi2 = float((((value - expected) / spread) ** 2).mean())
    ax.errorbar(
        np.sqrt(shells),
        value,
        yerr=spread,
        color=COLORS[lam],
        linestyle="none",
        marker=MARKERS[lam],
        markersize=4,
        elinewidth=0.9,
        capsize=2,
        label=rf"SSE, $\lambda / J = {lam:g}$ ($\chi^2 / \mathrm{{shell}} = {chi2:.2f}$)",
    )
ax.set_yscale("log")
ax.set_xlabel(r"distance $r$ from the line through the charges")
ax.set_ylabel(r"$-E_1$ in the middle between the charges")
ax.set_title(rf"$L = {L}$, $R = {R}$")
ax.legend(frameon=False, loc="upper right", fontsize=9)
ax.grid(color="0.92", linewidth=0.6)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(HERE / "thesis" / "field_transverse.pdf")
plt.close(fig)

# ---------------------------------------------------------------- thesis/flux_planes.pdf
# The flux through the planes perpendicular to the axis, for charges at the distance 5: measured,
# the solution of the Poisson equation alone, and that solution plus the uniform background.
R5 = 5
shift5 = L // 2 - (R5 + 1) // 2
flux = fields[L, R5, LAM][..., 0].sum(axis=(2, 3), dtype=float)
print(f"flux per plane over {len(flux)} chains: smallest {flux.min(axis=0).round(4)}")
print(f"                                 largest {flux.max(axis=0).round(4)}")
poisson = np.roll(coulomb_field(L, R5, background=False)[0][..., 0].sum(axis=(1, 2)), shift5)
steps = np.append(poisson, poisson[-1])
ticks = np.arange(shift5 % 2, L + 1, 2)
fig, ax = plt.subplots(figsize=(5.6, 3.2), layout="constrained")
ax.axhline(0.0, color="0.6", linewidth=0.6, zorder=0)
ax.step(
    np.arange(L + 1),
    steps,
    where="post",
    color="0.45",
    linewidth=1.2,
    linestyle="--",
    label=r"Poisson, zero mean: $R/L - 1$ and $R/L$",
)
ax.step(
    np.arange(L + 1),
    steps - R5 / L,
    where="post",
    color="0.45",
    linewidth=1.2,
    label=r"Poisson plus background $-R/L^3$ per link",
)
ax.plot(
    np.arange(L) + 0.5,
    np.roll(flux.mean(axis=0), shift5),
    color=BLUE,
    linestyle="none",
    marker="o",
    markersize=5,
    label=f"measured, each of the {len(flux)} chains",
)
ax.axvline(shift5, color="0.5", linewidth=0.8, linestyle=":")
ax.axvline(shift5 + R5, color="0.5", linewidth=0.8, linestyle=":")
ax.text(shift5, 0.75, "−", ha="center", va="bottom")
ax.text(shift5 + R5, 0.75, "+", ha="center", va="bottom")
ax.set_ylim(-1.15, 0.75)
ax.set_xlim(0, L)
ax.set_xticks(ticks, [str((i - shift5) % L) for i in ticks])
ax.set_xlabel(r"plane $n$")
ax.set_ylabel(r"flux $\Phi_1(n)$")
ax.set_title(rf"$L = {L}$, $R = {R5}$, $\lambda / J = {LAM:g}$", pad=14)
ax.legend(frameon=False, loc="upper right", fontsize=8)
ax.grid(color="0.92", linewidth=0.6)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(HERE / "thesis" / "flux_planes.pdf")
plt.close(fig)

plt.style.use(["default", HERE / "presentation.mplstyle"])

# ---------------------------------------------------------------- presentation/field_line.png
# The field on the links of the line through the charges. A link is drawn at its middle.
ticks = np.arange(0, L + 1, 2)
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True, layout="constrained")
for ax, lam in zip(axes, LAMS, strict=True):
    ax.axhline(0.0, color="0.6", linewidth=0.6, zorder=0)
    ax.plot(
        np.arange(L) + 0.5,
        np.roll(coulomb[:, 0, 0, 0], x0),
        color="0.3",
        linewidth=1.6,
        label="lattice Coulomb",
    )
    ax.errorbar(
        np.arange(L) + 0.5,
        np.roll(mean[L, R, lam][:, 0, 0, 0], x0),
        yerr=np.roll(error[L, R, lam][:, 0, 0, 0], x0),
        color=COLORS[lam],
        linestyle="none",
        marker=MARKERS[lam],
        markersize=7,
        elinewidth=1.1,
        capsize=3,
        label=f"measured ({len(fields[L, R, lam])} chains)",
    )
    for position, sign in ((x0, "−"), (x0 + R, "+")):
        ax.axvline(position, color="0.5", linewidth=0.9, linestyle=":")
        ax.text(position, 0.215, sign, ha="center", va="bottom", fontsize=16)
    ax.set_ylim(-0.21, 0.21)
    ax.set_xlim(0, L)
    ax.set_xticks(ticks, [str((i - x0) % L) for i in ticks])
    ax.set_xlabel(r"$x$")
    ax.set_title(rf"$\lambda = {lam:g}$", loc="left", pad=22)
    ax.legend(frameon=False, loc="upper center", fontsize=12)
    ax.grid(color="0.92", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel(r"$E_x$")
fig.savefig(HERE / "presentation" / "field_line.png")
plt.close(fig)

# ---------------------------------------------------------------- presentation/potential_line.png
# The potential on the line through the charges. The field on a link is the difference of the
# potential at its two ends, so phi(x + 1) = phi(x) - E_x(x), for every chain. The uniform
# background is not the gradient of a potential and is taken out first. phi has zero mean.
expected = potential[:, 0, 0] - potential[:, 0, 0].mean()
ticks = np.arange(0, L, 2)
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True, layout="constrained")
for ax, lam in zip(axes, LAMS, strict=True):
    on_line = fields[L, R, lam][:, :, 0, 0, 0].astype(float)
    gradient = on_line - on_line.mean(axis=1, keepdims=True)
    phi = np.concatenate(
        [np.zeros((len(on_line), 1)), -np.cumsum(gradient, axis=1)[:, :-1]], axis=1
    )
    phi = phi - phi.mean(axis=1, keepdims=True)
    ax.axhline(0.0, color="0.6", linewidth=0.6, zorder=0)
    ax.plot(
        np.arange(L), np.roll(expected, x0), color="0.3", linewidth=1.6, label="lattice Coulomb"
    )
    ax.errorbar(
        np.arange(L),
        np.roll(phi.mean(axis=0), x0),
        yerr=np.roll(phi.std(axis=0, ddof=1) / np.sqrt(len(phi)), x0),
        color=COLORS[lam],
        linestyle="none",
        marker=MARKERS[lam],
        markersize=7,
        elinewidth=1.1,
        capsize=3,
        label=f"measured ({len(phi)} chains)",
    )
    for position, sign in ((x0, "−"), (x0 + R, "+")):
        ax.axvline(position, color="0.5", linewidth=0.9, linestyle=":")
        ax.text(position, 0.305, sign, ha="center", va="bottom", fontsize=16)
    ax.set_ylim(-0.3, 0.3)
    ax.set_xlim(-0.5, L - 0.5)
    ax.set_xticks(ticks, [str((i - x0) % L) for i in ticks])
    ax.set_xlabel(r"$x$")
    ax.set_title(rf"$\lambda = {lam:g}$", loc="left", pad=22)
    ax.legend(frameon=False, loc="upper left", fontsize=12)
    ax.grid(color="0.92", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel(r"$\varphi$")
fig.savefig(HERE / "presentation" / "potential_line.png")
plt.close(fig)
