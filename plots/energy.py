"""All figures of the energies of the sectors, from the mean energy of every chain.

The sectors are named by their start: "g" no charges and no flux, "w1_0_0" one unit of flux,
"q<R>" two static charges at the distance R. The energy of a sector is the mean over its chains,
and its error the spread between them. The first bins of every chain are left out.

    uv run python energy.py

writes thesis/flux_energy.pdf, potential.pdf, coupling.pdf and presentation/flux_energy_slide.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import import_data as d

HERE = Path(__file__).resolve().parent
LAMS = [0.25, 0.5]
BLUE = "#1f5fa8"
ORANGE = "#c2571a"
GREEN = "#3a8f5c"
MIN_L_FLUX = 6  # smallest lattice of the fits of the flux energy
MIN_L_PHOTON = 8  # smallest lattice of the mean over the coupling of the photon mode


def potential_shape(n):
    """f_L(R) = 2 [G_L(0) - G_L(R)] + R^2 / L^3 for R = 0 ... L - 1: the energy of the lattice Coulomb field per unit of a."""
    k = 2 * np.pi * np.fft.fftfreq(n)
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    k2 = 4 * (np.sin(kx / 2) ** 2 + np.sin(ky / 2) ** 2 + np.sin(kz / 2) ** 2)
    k2[0, 0, 0] = np.inf
    green = np.fft.ifftn(1 / k2).real[:, 0, 0]
    return 2 * (green[0] - green) + np.arange(n) ** 2 / n**3


chains = d.chains()
sizes = sorted({int(n) for n in chains["L"]})

# The flux energy E(one unit of flux) - E(no flux) per lam: (L, value, error).
flux = {}
for lam in LAMS:
    rows = []
    for n in sizes:
        of_size = chains[(chains["L"] == n) & (chains["lam"] == lam)]
        wound = np.asarray(of_size[of_size["start"] == "w1_0_0"]["energy"])
        zero = np.asarray(of_size[of_size["start"] == "g"]["energy"])
        if len(wound) > 1:
            spread = np.hypot(
                wound.std(ddof=1) / np.sqrt(len(wound)), zero.std(ddof=1) / np.sqrt(len(zero))
            )
            rows.append((n, wound.mean() - zero.mean(), spread))
    flux[lam] = tuple(np.array(column) for column in zip(*rows, strict=True))

plt.style.use(HERE / "thesis.mplstyle")

# ---------------------------------------------------------------- thesis/flux_energy.pdf
# The flux energy against 1 / L. In a Coulomb phase it is kappa / L, a straight line through the
# origin. In a confined phase it is sigma L, which grows towards 1 / L = 0. Weighted least squares
# with one parameter each, on the lattices with L >= MIN_L_FLUX. The smaller ones are drawn open.
kappa = {}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), layout="constrained", sharey=True)
smooth = np.linspace(1 / 40, 0.27, 200)  # 1 / L
for ax, lam in zip(axes, LAMS, strict=True):
    x, y, e = flux[lam]
    fitted = x >= MIN_L_FLUX
    w = 1 / e[fitted] ** 2
    for label, name, power, style in (
        ("kappa", r"Coulomb, $\kappa / L$", -1, "-"),
        ("constant", "constant", 0, ":"),
        ("sigma", r"confined, $\sigma L$", 1, "--"),
    ):
        basis = x[fitted] ** float(power)
        par = np.sum(w * basis * y[fitted]) / np.sum(w * basis**2)
        par_error = 1 / np.sqrt(np.sum(w * basis**2))
        chi2 = float(np.sum(w * (y[fitted] - par * basis) ** 2))
        print(
            f"lam = {lam}, L >= {MIN_L_FLUX}: {label} = {par:.4f} +- {par_error:.4f}, "
            f"chi^2 / dof = {chi2:.1f} / {fitted.sum() - 1}"
        )
        if label == "kappa":
            kappa[lam] = (par, par_error)
        ax.plot(
            smooth,
            par * smooth ** float(-power),
            color="0.3",
            linewidth=1.2,
            linestyle=style,
            label=rf"{name}: $\chi^2 / \mathrm{{dof}} = {chi2:.1f} / {fitted.sum() - 1}$",
        )
    ax.axhline(0.0, color="0.6", linewidth=0.6, zorder=0)
    for part, face in ((fitted, BLUE), (~fitted, "white")):
        ax.errorbar(
            1 / x[part],
            y[part],
            yerr=e[part],
            color=BLUE,
            markerfacecolor=face,
            linestyle="none",
            marker="o",
            markersize=5,
            elinewidth=0.9,
            capsize=2,
        )
    ax.set_title(rf"$\lambda / J = {lam:g}$", loc="left")
    ax.set_xlabel(r"$1 / L$")
    ax.set_xlim(0, 0.27)
    ax.set_xticks([0, 1 / 12, 1 / 8, 1 / 6, 1 / 4], ["0", "1/12", "1/8", "1/6", "1/4"])
    ax.set_ylim(-0.1, 0.3)
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    ax.grid(color="0.92", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel(r"$\Delta E = E_{W = 1} - E_{W = 0}$")
fig.savefig(HERE / "thesis" / "flux_energy.pdf")
plt.close(fig)

# ---------------------------------------------------------------- thesis/potential.pdf
# The energies of the sectors with charges are the potential plus an energy that depends on L but
# not on R. Two forms are fitted per lam to the energies of all L at once, with one free constant
# c_L per lattice size: E = c_L + a f_L(R) (Coulomb) and E = c_L + sigma R (linear).
# Shown: E - c_L of the Coulomb fit, averaged over the lattice sizes at every R with the weights
# of the fit, and the two fits in the same average.
a_potential = {}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), layout="constrained", sharey=True)
for ax, lam in zip(axes, LAMS, strict=True):
    rows = []
    for n in sizes:
        of_size = chains[(chains["L"] == n) & (chains["lam"] == lam)]
        for start in sorted({str(s) for s in of_size["start"] if str(s).startswith("q")}):
            values = np.asarray(of_size[of_size["start"] == start]["energy"])
            if len(values) > 1:
                distance = int(start[1:])
                spread = values.std(ddof=1) / np.sqrt(len(values))
                rows.append((n, distance, values.mean(), spread, potential_shape(n)[distance]))
    size, dist, value, spread, shape = (np.array(column) for column in zip(*rows, strict=True))
    weight = 1 / spread**2
    offsets = np.array([[float(n == m) for m in sorted(set(size))] for n in size])
    fits = {}
    for name, x in (("coulomb", shape), ("linear", dist)):
        design = np.column_stack([offsets, x])
        cov = np.linalg.inv(design.T @ (weight[:, None] * design))
        par = cov @ design.T @ (weight * value)
        chi2 = float(np.sum(weight * (value - design @ par) ** 2))
        fits[name] = (par, np.sqrt(cov[-1, -1]), chi2, len(value) - design.shape[1])
        print(
            f"lam = {lam}, {name:7s}: slope = {par[-1]:.3f} +- {np.sqrt(cov[-1, -1]):.3f}, "
            f"chi^2 / dof = {chi2:.1f} / {len(value) - design.shape[1]}"
        )
    a_potential[lam] = (fits["coulomb"][0][-1], fits["coulomb"][1])
    constant = offsets @ fits["coulomb"][0][:-1]
    constant_linear = offsets @ fits["linear"][0][:-1]
    a, sigma = fits["coulomb"][0][-1], fits["linear"][0][-1]
    distances = sorted(set(dist))
    points, errors, curve, line = [], [], [], []
    for r in distances:
        at = dist == r
        norm = weight[at].sum()
        points.append(np.sum(weight[at] * (value[at] - constant[at])) / norm)
        errors.append(1 / np.sqrt(norm))
        curve.append(a * np.sum(weight[at] * shape[at]) / norm)
        line.append(np.sum(weight[at] * (constant_linear[at] + sigma * r - constant[at])) / norm)
    ax.plot(distances, curve, color=BLUE, linewidth=1.2, label=r"Coulomb, $a\,f_L(R)$")
    ax.plot(
        distances, line, color="0.3", linewidth=1.2, linestyle="--", label=r"linear, $\sigma R$"
    )
    ax.errorbar(
        distances,
        points,
        yerr=errors,
        color=BLUE,
        linestyle="none",
        marker="o",
        markersize=5,
        elinewidth=0.9,
        capsize=2,
        label="measured",
    )
    notes = [
        rf"{label}: $\chi^2 / \mathrm{{dof}} = {fits[name][2]:.1f} / {fits[name][3]}$"
        for name, label in (("coulomb", "Coulomb"), ("linear", "linear"))
    ]
    ax.set_title("\n".join(notes), loc="right", fontsize=8)
    ax.set_title(rf"$\lambda / J = {lam:g}$", loc="left")
    ax.set_xlabel(r"$R$")
    ax.set_xticks(distances)
    ax.grid(color="0.92", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel(r"$E_R - c_L$, averaged over $L$")
axes[0].legend(frameon=False, loc="lower right", fontsize=8)
fig.savefig(HERE / "thesis" / "potential.pdf")
plt.close(fig)

# ---------------------------------------------------------------- thesis/coupling.pdf
# The electric coupling a from three parts of the field. Flux energy: a = kappa. Potential: the
# slope a of the Coulomb fit. Photon mode: the mode is two harmonic oscillators of frequency
# Delta, so C(0) = L^3 Delta coth(beta Delta / 2) / (4 a), with Delta from
# C(beta / 4) / C(beta / 2) = cosh(Delta beta / 4). Jackknife over the chains.
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True, layout="constrained")
for ax, lam in zip(axes, LAMS, strict=True):
    photon = []
    for n in sizes:
        vacuum = chains[(chains["L"] == n) & (chains["lam"] == lam) & (chains["start"] == "g")]
        c = np.array(vacuum["correlator"].tolist())
        beta = 2 * n
        left_out = (c.sum(axis=0) - c) / (len(c) - 1)
        estimates = []
        for corr in (c.mean(axis=0)[None, :], left_out):
            gap = np.arccosh(corr[:, 4] / corr[:, 8]) / (beta / 4)
            estimates.append(n**3 * gap / np.tanh(beta * gap / 2) / (4 * corr[:, 0]))
        photon.append((n, estimates[0][0], np.sqrt((len(c) - 1) * np.var(estimates[1]))))
    size, a_photon, a_error = (np.array(column) for column in zip(*photon, strict=True))
    large = size >= MIN_L_PHOTON
    weight = 1 / a_error[large] ** 2
    a_mean = np.sum(weight * a_photon[large]) / np.sum(weight)
    a_mean_error = 1 / np.sqrt(np.sum(weight))
    print(
        f"lam = {lam}: a = {a_mean:.3f} +- {a_mean_error:.3f} (photon mode, L >= {MIN_L_PHOTON}), "
        f"{kappa[lam][0]:.3f} +- {kappa[lam][1]:.3f} (flux energy), "
        f"{a_potential[lam][0]:.3f} +- {a_potential[lam][1]:.3f} (potential)"
    )
    for (value, spread), color, label in (
        (kappa[lam], ORANGE, "flux energy"),
        (a_potential[lam], GREEN, "potential of the charges"),
    ):
        ax.axhspan(value - spread, value + spread, color=color, alpha=0.18, linewidth=0)
        ax.axhline(
            value, color=color, linewidth=1.2, label=rf"{label}: ${value:.2f} \pm {spread:.2f}$"
        )
    ax.errorbar(
        size,
        a_photon,
        yerr=a_error,
        color=BLUE,
        linestyle="none",
        marker="o",
        markersize=5,
        elinewidth=0.9,
        capsize=2,
        label=rf"photon mode: ${a_mean:.3f} \pm {a_mean_error:.3f}$ ($L \geq {MIN_L_PHOTON}$)",
    )
    ax.set_title(rf"$\lambda / J = {lam:g}$", loc="left")
    ax.set_xlabel(r"$L$")
    ax.set_xticks(size)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.grid(color="0.92", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel(r"electric coupling $a = e^2 / 2$")
axes[0].set_ylim(0, 1.6)
fig.savefig(HERE / "thesis" / "coupling.pdf")
plt.close(fig)

plt.style.use(["default", HERE / "presentation.mplstyle"])

# ---------------------------------------------------------------- presentation/flux_energy_slide.png
# The flux energy against L for the lattices with small errors, with kappa / L and sigma L fitted
# to L >= MIN_L_FLUX as above.
MAX_L = 10  # largest lattice shown
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True, layout="constrained")
smooth = np.linspace(3.5, MAX_L + 0.5, 100)
for ax, lam in zip(axes, LAMS, strict=True):
    x, y, e = flux[lam]
    fitted = x >= MIN_L_FLUX
    w = 1 / e[fitted] ** 2
    for power, style in ((-1, "-"), (1, "--")):
        basis = x[fitted] ** float(power)
        par = np.sum(w * basis * y[fitted]) / np.sum(w * basis**2)
        ax.plot(smooth, par * smooth ** float(power), color="0.3", linewidth=1.2, linestyle=style)
    for part, face in ((fitted, BLUE), (~fitted, "white")):
        shown = part & (x <= MAX_L)
        ax.errorbar(
            x[shown],
            y[shown],
            yerr=e[shown],
            color=BLUE,
            markerfacecolor=face,
            linestyle="none",
            marker="o",
            markersize=7,
            elinewidth=0.9,
            capsize=2,
        )
    ax.set_xlabel(r"$L$")
    ax.set_xticks(x[x <= MAX_L])
    ax.set_ylim(0.0, 0.3)
    ax.set_title(rf"$\lambda / J = {lam:g}$", loc="left")
    ax.grid(color="0.92", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel(r"$\Delta U = U_{W = 1} - U_{W = 0}$")
axes[0].plot([], [], color="0.3", linewidth=1.2, label=r"Coulomb: $\Delta U = \kappa / L$")
axes[0].plot(
    [], [], color="0.3", linewidth=1.2, linestyle="--", label=r"confined: $\Delta U = \sigma L$"
)
axes[0].legend(frameon=False, loc="upper center", fontsize=11)
fig.savefig(HERE / "presentation" / "flux_energy_slide.png")
plt.close(fig)
