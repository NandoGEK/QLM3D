"""All figures of the photon, from the chains without charges and flux.

The chains without charges and flux store the correlator C(tau) of the transverse electric mode
with the smallest momentum at tau = j beta / 16, j = 0 ... 8. Its gap follows from
C(beta / 4) / C(beta / 2) = cosh(Delta beta / 4). Errors are jackknife errors over the chains.

    uv run python photon.py

writes thesis/photon_correlator.pdf.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import import_data as d

HERE = Path(__file__).resolve().parent
LAMS = [0.25, 0.5]
BLUE = "#1f5fa8"
ORANGE = "#c2571a"
GREEN = "#3a8f5c"

# The chains without charges and flux at beta = 2 L, with their mean photon correlator.
chains = d.chains()
chains = chains[chains["start"] == "g"]

plt.style.use(HERE / "thesis.mplstyle")

# ---------------------------------------------------------------- thesis/photon_correlator.pdf
# Left: the weight cos(2 pi x_1 / L) of the links in the second direction in the mode, plane by
# plane. Right: the correlator relative to its value at tau = beta / 2, against the single cosh
# of the gap, which passes through the points at tau = beta / 4 and beta / 2 by construction.
SHOWN = 0.25  # the coupling of the right panel
SIZES = [4, 8, 16]
SHADES = {4: "#9ec3e6", 8: "#3f86c6", 16: "#0b3c7a"}
MARKERS = {4: "o", 8: "s", 16: "^"}
fig = plt.figure(figsize=(7.2, 3.3), layout="constrained")
size = 8
top = size - 1
box = fig.add_subplot(1, 2, 1, projection="3d")
for plane in range(size):
    weight = np.cos(2 * np.pi * plane / size)
    color = BLUE if weight > 1e-9 else ORANGE if weight < -1e-9 else "0.6"
    corners = [(plane, 0, 0), (plane, top, 0), (plane, top, top), (plane, 0, top)]
    box.add_collection3d(
        Poly3DCollection([corners], facecolor=color, edgecolor=color, alpha=0.10, linewidth=0.6)
    )
    if abs(weight) > 1e-9:
        for row in (1, 3, 5):
            for height in (1, 3, 5):
                length = 1.6 * weight
                start = (plane, row + 0.8 - length / 2, height)
                box.quiver(*start, 0, length, 0, color=color, linewidth=1.2)
    box.text(plane, 0, -1.2, str(plane), ha="center", va="top", fontsize=8, color="0.3")
box.text(top / 2, 0, -2.6, r"$x_1$", ha="center", va="top")
box.text(top + 0.8, top / 2, -0.6, r"$x_2$")
box.set_xlim(0, top)
box.set_ylim(0, top)
box.set_zlim(0, top)
box.set_box_aspect((1.8, 1, 1))
box.view_init(elev=16, azim=-62)
box.set_axis_off()
box.set_title(r"weight $\cos(2 \pi x_1 / L)$ of $E_{x,2}$ in $O$")

ax = fig.add_subplot(1, 2, 2)
for n in SIZES:
    part = chains[(chains["L"] == n) & (chains["lam"] == SHOWN)]
    c = np.array(part["correlator"].tolist())
    beta = 2 * n
    tau = np.arange(9) * beta / 16
    mean = c.mean(axis=0)
    left_out = (c.sum(axis=0) - c) / (len(c) - 1)
    ratio = mean / mean[-1]
    ratio_error = np.sqrt((len(c) - 1) * np.var(left_out / left_out[:, -1:], axis=0))
    gap = np.arccosh(mean[4] / mean[8]) / (beta / 4)
    print(f"lam = {SHOWN}, L = {n:2d}: {len(c)} chains, gap = {gap:.4f}")
    smooth = np.linspace(0, beta / 2, 100)
    ax.plot(smooth / beta, np.cosh(gap * (smooth - beta / 2)), color=SHADES[n], linewidth=1.2)
    ax.errorbar(
        tau / beta,
        ratio,
        yerr=ratio_error,
        label=rf"$L = {n}$",
        color=SHADES[n],
        linestyle="none",
        marker=MARKERS[n],
        markersize=5,
        elinewidth=0.9,
        capsize=2,
    )
ax.set_yscale("log")
ax.set_xlabel(r"$\tau / \beta$")
ax.set_ylabel(r"$C(\tau) / C(\beta / 2)$")
ax.set_title(rf"$\lambda / J = {SHOWN:g}$, $\beta = 2L$", loc="left")
ax.legend(frameon=False, loc="upper right")
ax.grid(color="0.92", linewidth=0.6)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(HERE / "thesis" / "photon_correlator.pdf")
plt.close(fig)
