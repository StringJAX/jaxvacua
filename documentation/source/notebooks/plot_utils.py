"""Shared plotting utilities for JAXVacua tutorial notebooks.

Purpose
-------
Provide compact plotting helpers used by example notebooks to inspect sampled
flux-vacuum data.

Main public API
---------------
- ``make_overview_plots``: creates overview scatter plots for moduli,
  axio-dilaton, superpotential values and flux tadpoles.

Design notes
------------
Notebook helpers should remain lightweight and presentation-focused.  They
assume the supplied model exposes the standard ``FluxEFT`` superpotential and
tadpole methods.
"""
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from jax import vmap

cmap = sn.color_palette("viridis", as_cmap=True)


def make_overview_plots(model, moduli, tau, fluxes,
                        moduli_range=[[-0.5, 0.5], [0, 10]],
                        tau_range=[[-0.5, 0.5], [0, 10]],
                        W0_range=[-10, 10],
                        use_normal_w0=False):
    """Produce three overview plots for a set of SUSY flux vacua.

    Generates:
    - A 2×2 scatter grid: Im(z₁) vs Im(z₂), Re(z₁) vs Re(z₂),
      Re(τ) vs Im(τ), Re(W₀) vs Im(W₀), all coloured by N_flux.
    - A log-log scatter of |W₀| vs g_s = 1/Im(τ).
    - A log-log scatter of |W₀| vs N_flux.

    Parameters
    ----------
    model : FluxEFT
        The JAXVacua model (must expose ``superpotential``,
        ``superpotential_gauge_invariant``, and ``tadpole``).
    moduli : array, shape (N, h12)
        Complex structure moduli for each vacuum.
    tau : array, shape (N,)
        Axio-dilaton values for each vacuum.
    fluxes : array, shape (N, 2*(h12+1))
        Flux quanta for each vacuum.
    moduli_range : list of two [lo, hi] pairs
        Plot limits ``[Re(z) range, Im(z) range]``.
    tau_range : list of two [lo, hi] pairs
        Plot limits ``[Re(τ) range, Im(τ) range]``.
    W0_range : [lo, hi]
        Symmetric axis limits for the Re/Im(W₀) scatter.
    use_normal_w0 : bool
        If True, use the gauge-dependent ``superpotential``; otherwise use
        the gauge-invariant ``superpotential_gauge_invariant``.
    """
    W1v = vmap(model.superpotential)
    W0v = vmap(model.superpotential_gauge_invariant)
    tadpole_v = vmap(model.tadpole)

    if use_normal_w0:
        W0 = W1v(moduli, tau, fluxes)
    else:
        W0 = W0v(moduli, tau, fluxes)

    Nflux = tadpole_v(fluxes)

    fig, ax = plt.subplots(2, 2, dpi=200, figsize=(6, 4))

    fs = 8

    h = Nflux
    hlabel = r"$N_{\mathrm{flux}}$"

    norm = plt.Normalize(min(h), max(h))
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])

    ReZ_range = moduli_range[0]
    ImZ_range = moduli_range[1]

    c_range = tau_range[0]
    s_range = tau_range[1]

    sn.scatterplot(x=moduli[:, 0].imag, y=moduli[:, 1].imag, s=5, ax=ax[0, 0], hue=h, palette=cmap)
    ax[0, 0].set_xlim(ImZ_range[0], ImZ_range[1])
    ax[0, 0].set_ylim(ImZ_range[0], ImZ_range[1])
    ax[0, 0].legend_.remove()
    ax[0, 0].set_xlabel(r"Im$(z_1)$", fontsize=fs)
    ax[0, 0].set_ylabel(r"Im$(z_2)$", fontsize=fs)

    sn.scatterplot(x=moduli[:, 0].real, y=moduli[:, 1].real, s=5, ax=ax[0, 1], hue=h, palette=cmap)
    ax[0, 1].set_xlim(ReZ_range[0], ReZ_range[1])
    ax[0, 1].set_ylim(ReZ_range[0], ReZ_range[1])
    ax[0, 1].legend_.remove()
    ax[0, 1].set_xlabel(r"Re$(z_1)$", fontsize=fs)
    ax[0, 1].set_ylabel(r"Re$(z_2)$", fontsize=fs)

    sn.scatterplot(x=tau.real, y=tau.imag, s=5, ax=ax[1, 0], hue=h, palette=cmap)
    ax[1, 0].set_xlim(c_range[0], c_range[1])
    ax[1, 0].set_ylim(s_range[0], s_range[1])
    ax[1, 0].legend_.remove()
    ax[1, 0].set_xlabel(r"Re$(\tau)$", fontsize=fs)
    ax[1, 0].set_ylabel(r"Im$(\tau)$", fontsize=fs)

    sn.scatterplot(x=W0.real, y=W0.imag, s=4, ax=ax[1, 1], hue=h, palette=cmap)
    ax[1, 1].set_xlim(W0_range[0], W0_range[1])
    ax[1, 1].set_ylim(W0_range[0], W0_range[1])
    ax[1, 1].legend_.remove()
    ax[1, 1].set_xlabel(r"Re$(W_0)$", fontsize=fs)
    ax[1, 1].set_ylabel(r"Im$(W_0)$", fontsize=fs)

    for i in range(2):
        for j in range(2):
            ax[i, j].tick_params(labelsize=fs - 1)

    plt.tight_layout()

    clb = fig.colorbar(sm, label=hlabel, ax=ax.ravel().tolist())
    clb.ax.tick_params(labelsize=fs)

    plt.show()

    fs = 12

    fig = plt.figure(dpi=150, figsize=(6, 4))

    ax = sn.scatterplot(x=np.abs(W0), y=1 / tau.imag, hue=h, palette=cmap)
    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel(r"$|W_0|$", fontsize=fs)
    plt.ylabel(r"$g_s$", fontsize=fs)

    clb = fig.colorbar(sm, label=hlabel, ax=ax)
    clb.ax.tick_params(labelsize=fs)
    plt.legend("", frameon=False)

    plt.show()

    fig = plt.figure(dpi=150, figsize=(6, 4))

    ax = sn.scatterplot(x=np.abs(W0), y=Nflux, hue=h, palette=cmap)
    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel(r"$|W_0|$", fontsize=fs)
    plt.ylabel(r"$N_{\mathrm{flux}}$", fontsize=fs)

    plt.legend("", frameon=False)
    clb = fig.colorbar(sm, label=hlabel, ax=ax)
    clb.ax.tick_params(labelsize=fs)

    plt.show()
