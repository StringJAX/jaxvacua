# Copyright 2022-2026 Andreas Schachner
#
# This file is part of JAXVacua.
#
# JAXVacua is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# JAXVacua is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with JAXVacua. If not, see <https://www.gnu.org/licenses/>.

"""One-modulus hypergeometric Calabi-Yau model registry.

Purpose
-------
Provide hard-coded one-modulus models from Bastian, van de Heisteeg and
Schlechter, arXiv:2306.01059, together with closed-form prepotentials at
large-complex-structure, K-point and C-point limits.

Main public API
---------------
- ``HypergeometricModels`` and ``list_hypergeometric_models`` for listing and
  constructing supported models.
- Prepotential builders ``make_prepot_LCS``, ``make_prepot_Kpoint`` and
  ``make_prepot_Cpoint`` for low-level custom workflows.
- Registry dictionaries for hypergeometric, non-hypergeometric and special
  point data.

Design notes
------------
The K-point formula follows equations (1060-1063) of the reference.  The
C-point formula follows equations (846-849) with the package convention that
the instanton normalisation absorbs a factor of ``(8*pi)^-1`` into ``A1``.
Models built through this registry use the standard ``FluxVacuaFinder`` API.
"""

from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import gamma as gamma_func
from scipy.special import zeta as scipy_zeta

__all__ = [
    "HypergeometricModels",
    "list_hypergeometric_models",
    "make_prepot_LCS",
    "make_prepot_Kpoint",
    "make_prepot_Cpoint",
    "HYPERGEOMETRIC_MODELS",
    "NON_HYPERGEOMETRIC_MODELS",
    "ALL_MODELS",
    "KPOINT_DATA",
    "CPOINT_DATA",
]

# ─────────────────────────────────────────────────────────────────────────────
# L-function values (closed-form Γ-function expressions, Table 3–4 of [1])
# ─────────────────────────────────────────────────────────────────────────────

def _Lf27_1():
    r"""**Description:** :math:`L(27.3.b.a,\,1)` value used for the :math:`X_{3,3}` K-point (weight-3 form)."""
    return gamma_func(1/3)**6 / (8.0 * np.sqrt(3) * np.pi**3)

def _Lf16_1():
    r"""**Description:** :math:`L(16.3.c.a,\,1)` value used for the :math:`X_{4,4}` K-point (weight-3 form)."""
    return gamma_func(1/4)**4 * gamma_func(1/2)**2 / (32.0 * np.pi**3)

def _Lf12_1():
    r"""**Description:** :math:`L(12.3.c.a,\,1)` value used for the :math:`X_{6,6}` K-point (weight-3 form)."""
    return np.sqrt(3) * gamma_func(1/3)**6 / (32.0 * 2**(2/3) * np.pi**3)

def _Lf108K_1():
    r"""**Description:** :math:`L(108.3.c.a,\,1)` value used for the operator-4.47 K-point (weight-3 form)."""
    return np.sqrt(3) * gamma_func(1/3)**6 / (8.0 * 2**(1/3) * np.pi**3)

def _Lf36_1():
    r"""**Description:** :math:`L(36.3.d.a,\,1)` value used for the operator-3.7 K-point (weight-3 form)."""
    return gamma_func(1/4)**4 * gamma_func(1/2)**2 / (12.0 * np.sqrt(2) * 3**(1/4) * np.pi**3)

def _Lf32C_1():
    r"""**Description:** :math:`L(32.4.a.b,\,1)` value used for the :math:`X_{4,2}` and op-2.62 C-point (weight-4 form)."""
    return gamma_func(1/4)**6 * gamma_func(1/2)**3 / (16.0 * np.sqrt(2) * np.pi**5)

def _Lf9C_1():
    r"""**Description:** :math:`L(9.4.a.a,\,1)` value used for the :math:`X_{3,2,2}` C-point (weight-4 form)."""
    return gamma_func(1/3)**9 / (32.0 * np.sqrt(3) * np.pi**5)

def _Lf108C_1():
    r"""**Description:** :math:`L(108.4.a.b,\,1)` value used for the :math:`X_{6,2}` C-point (weight-4 form)."""
    return 3.0 * np.sqrt(3) * gamma_func(1/3)**9 / (16.0 * 2**(1/3) * np.pi**5)


# ─────────────────────────────────────────────────────────────────────────────
# Table 1: All 14 hypergeometric one-modulus CY models
#
# Columns: label, name, alpha (Picard-Fuchs exponents), mu (= 1/z_conifold),
#          kappa (triple intersection of mirror), c2 (integrated 2nd Chern
#          class of mirror), chi (Euler characteristic of mirror),
#          singularity_at_inf (type of singularity at z → ∞).
#
# Singularity types at z → ∞:
#   M = MUM (second large complex structure point, all α_i equal)
#   K = K-point (finite-order monodromy; α_1=α_2, α_3=α_4, α_2≠α_3)
#   C = conifold (middle two exponents equal; α_2=α_3, α_1≠α_2, α_3≠α_4)
#   F = F-point (all four exponents distinct; generic finite-order monodromy)
# ─────────────────────────────────────────────────────────────────────────────

HYPERGEOMETRIC_MODELS = {
    "M": {
        "name": "Self-mirror model (α=(1/2,1/2,1/2,1/2))",
        "alpha": (1/2, 1/2, 1/2, 1/2),
        "mu": 256,           # 2^8
        "kappa": 16, "c2": 64, "chi": -128,
        "singularity_at_inf": "M",
    },
    "F1": {
        "name": "F-type model (α=(1/4,1/3,2/3,3/4))",
        "alpha": (1/4, 1/3, 2/3, 3/4),
        "mu": 1728,          # 2^6 * 3^3
        "kappa": 6, "c2": 48, "chi": -156,
        "singularity_at_inf": "F",
    },
    "X42": {
        "name": "X_{4,2} (α=(1/4,1/2,1/2,3/4), conifold at infinity)",
        "alpha": (1/4, 1/2, 1/2, 3/4),
        "mu": 1024,          # 2^10
        "kappa": 8, "c2": 56, "chi": -176,
        "singularity_at_inf": "C",
    },
    "X5": {
        "name": "Quintic X_5 ⊂ P^4 (α=(1/5,2/5,3/5,4/5))",
        "alpha": (1/5, 2/5, 3/5, 4/5),
        "mu": 3125,          # 5^5
        "kappa": 5, "c2": 50, "chi": -200,
        "singularity_at_inf": "F",
    },
    "X33": {
        "name": "X_{3,3} CI (α=(1/3,1/3,2/3,2/3), K-point at infinity)",
        "alpha": (1/3, 1/3, 2/3, 2/3),
        "mu": 729,           # 3^6
        "kappa": 9, "c2": 54, "chi": -144,
        "singularity_at_inf": "K",
    },
    "X44": {
        "name": "X_{4,4} (α=(1/4,1/4,3/4,3/4), K-point at infinity)",
        "alpha": (1/4, 1/4, 3/4, 3/4),
        "mu": 4096,          # 2^12
        "kappa": 4, "c2": 40, "chi": -144,
        "singularity_at_inf": "K",
    },
    "X322": {
        "name": "X_{3,2,2} CI (α=(1/3,1/2,1/2,2/3), conifold at infinity)",
        "alpha": (1/3, 1/2, 1/2, 2/3),
        "mu": 432,           # 2^4 * 3^3
        "kappa": 12, "c2": 60, "chi": -144,
        "singularity_at_inf": "C",
    },
    "X62": {
        "name": "X_{6,2} (α=(1/6,1/2,1/2,5/6), conifold at infinity)",
        "alpha": (1/6, 1/2, 1/2, 5/6),
        "mu": 6912,          # 2^8 * 3^3
        "kappa": 4, "c2": 52, "chi": -256,
        "singularity_at_inf": "C",
    },
    "108a": {
        "name": "AESZ-108a (α=(1/6,1/3,2/3,5/6))",
        "alpha": (1/6, 1/3, 2/3, 5/6),
        "mu": 11664,         # 2^4 * 3^6
        "kappa": 3, "c2": 42, "chi": -204,
        "singularity_at_inf": "F",
    },
    "X8": {
        "name": "X_8 ⊂ WP^4(1,1,1,1,4) (α=(1/8,3/8,5/8,7/8))",
        "alpha": (1/8, 3/8, 5/8, 7/8),
        "mu": 65536,         # 2^16
        "kappa": 2, "c2": 44, "chi": -296,
        "singularity_at_inf": "F",
    },
    "144": {
        "name": "AESZ-144 (α=(1/6,1/4,3/4,5/6))",
        "alpha": (1/6, 1/4, 3/4, 5/6),
        "mu": 27648,         # 2^10 * 3^3
        "kappa": 2, "c2": 32, "chi": -156,
        "singularity_at_inf": "F",
    },
    "X10": {
        "name": "X_{10} ⊂ WP^4(1,1,1,2,5) (α=(1/10,3/10,7/10,9/10))",
        "alpha": (1/10, 3/10, 7/10, 9/10),
        "mu": 800000,        # 2^8 * 5^5
        "kappa": 1, "c2": 34, "chi": -288,
        "singularity_at_inf": "F",
    },
    "X66": {
        "name": "X_{6,6} (α=(1/6,1/6,5/6,5/6), K-point at infinity)",
        "alpha": (1/6, 1/6, 5/6, 5/6),
        "mu": 1679616,       # 2^8 * 3^6
        "kappa": 1, "c2": 22, "chi": -120,
        "singularity_at_inf": "K",
    },
    "864": {
        "name": "AESZ-864 (α=(1/12,5/12,7/12,11/12))",
        "alpha": (1/12, 5/12, 7/12, 11/12),
        "mu": 2985984,       # 2^12 * 3^6
        "kappa": 1, "c2": 46, "chi": -484,
        "singularity_at_inf": "F",
    },
}

# Non-hypergeometric models studied in [1] as additional examples
NON_HYPERGEOMETRIC_MODELS = {
    "2.62": {
        "name": "Operator 2.62 (C-point example, κ=42)",
        "kappa": 42, "c2": 84, "chi": -96,
        "singularity_at_inf": "C",
    },
    "2.17": {
        "name": "Operator 2.17 (C-point example with irrational τ, κ=4m family)",
        "kappa": None, "c2": None, "chi": None,
        "singularity_at_inf": "C",
        "note": "Parameter family; κ=4m, c2=4m, χ=24m for integer m.",
    },
    "4.47": {
        "name": "Operator 4.47 (K-point example, κ=4)",
        "kappa": 4, "c2": 28, "chi": -18,
        "singularity_at_inf": "K",
    },
    "3.7": {
        "name": "Operator 3.7 (K-point without semisimple monodromy, κ=9)",
        "kappa": 9, "c2": 30, "chi": 12,
        "singularity_at_inf": "K",
    },
}

ALL_MODELS = {**HYPERGEOMETRIC_MODELS, **NON_HYPERGEOMETRIC_MODELS}


# ─────────────────────────────────────────────────────────────────────────────
# K-point boundary data (Table 4 of [1])
#
# Near the K-point, the coordinate is s = (X[1] - τ) / X[0], with τ the rigid
# period. The prepotential expands as F = F₀ + s·F₁ + s²·F₂ + s³·F₃ + …
#
# Parameters stored per model:
#   tau   : rigid period τ ∈ H (upper half-plane)
#   gamma : extension datum γ ∈ ℝ
#   delta : extension datum δ ∈ ℝ
#   a,b,c : entries of the 2×2 log-monodromy submatrix N_K = [[a,b],[b,c]];
#           τ = (-b + i√(ac−b²)) / c
#   B1,B2,B3 : first three instanton coefficients B_k = B̂_k × Bnorm
#   Bnorm : common transcendental prefactor of the B-series
#   Lf1   : L(f,1) for the associated weight-3 modular form f
#   Bhat  : list of rational B̂_k values (index = k−1, 0 if absent)
#   modular_form : LMFDB label of the associated modular form
# ─────────────────────────────────────────────────────────────────────────────

def _build_kpoint_X33():
    """Build K-point boundary data for the X33 (degree-27) model."""
    Lf1  = _Lf27_1()
    Bnorm = -1.0 / (3.0 * Lf1**2)
    return dict(
        tau=complex(-0.5, np.sqrt(3)/2),
        gamma=1/6, delta=1/3,
        a=2, b=1, c=2,
        Bnorm=Bnorm, Lf1=Lf1,
        B1=1.0*Bnorm, B2=0.0, B3=0.0,
        # Non-zero: B̂₁=1, B̂₄=−1/2, B̂₇=501119/196000  (spacing 3)
        Bhat={1: 1.0, 4: -0.5, 7: 501119/196000},
        modular_form="27.3.b.a",
    )

def _build_kpoint_X44():
    """Build K-point boundary data for the X44 (degree-16) model."""
    Lf1  = _Lf16_1()
    Bnorm = -1.0 / (4.0 * Lf1**2)
    return dict(
        tau=complex(0.0, 1.0),
        gamma=0.5, delta=0.0,
        a=1, b=0, c=1,
        Bnorm=Bnorm, Lf1=Lf1,
        B1=1.0*Bnorm, B2=0.0, B3=12.0*Bnorm,
        # Non-zero: B̂₁=1, B̂₃=12, B̂₅=474122/675  (spacing 2)
        Bhat={1: 1.0, 3: 12.0, 5: 474122/675},
        modular_form="16.3.c.a",
    )

def _build_kpoint_X66():
    """Build K-point boundary data for the X66 (degree-12) model."""
    Lf1  = _Lf12_1()
    Bnorm = -9.0 / (8.0 * 2**(1/3) * Lf1**2)
    return dict(
        tau=complex(-0.5, np.sqrt(3)/2),
        gamma=1/3, delta=1/6,
        a=2, b=1, c=2,
        Bnorm=Bnorm, Lf1=Lf1,
        # Leading instanton at k=2 (B₁=0); standard formula assumes B₁≠0
        B1=0.0, B2=1.0*Bnorm, B3=0.0,
        Bhat={2: 1.0, 5: -640.0, 8: 1251305.0},
        modular_form="12.3.c.a",
        note="Leading instanton at k=2; use period_vector_Kpoint_X66 for this model.",
    )

def _build_kpoint_447():
    """Build K-point boundary data for the 4.47 (degree-108) model."""
    Lf1  = _Lf108K_1()
    Bnorm = 8.0 * 2**(1/3) * 3**(3/4) * np.pi**2 / Lf1**2
    return dict(
        tau=complex(0.0, 1.0/np.sqrt(3)),
        gamma=5/12, delta=5/6,
        a=2, b=0, c=6,
        Bnorm=Bnorm, Lf1=Lf1,
        B1=1.0*Bnorm, B2=0.0, B3=0.0,
        Bhat={1: 1.0, 4: 112/(3*3**(3/4)), 7: 5217112/(3375*np.sqrt(3))},
        modular_form="108.3.c.a",
    )

def _build_kpoint_37():
    """Build K-point boundary data for the 3.7 (degree-36) model."""
    Lf1  = _Lf36_1()
    Bnorm = -1.0 / (4.0 * np.sqrt(3) * Lf1**2)
    # Extension data γ=δ≈0.355955 given only numerically in [1]
    gd = 0.355955
    return dict(
        tau=complex(-0.5, 0.5),
        gamma=gd, delta=gd,
        a=1, b=1, c=2,
        Bnorm=Bnorm, Lf1=Lf1,
        B1=1.0*Bnorm, B2=0.0, B3=0.0,
        Bhat={1: 1.0},
        modular_form="36.3.d.a",
        note="γ=δ≈0.355955 are given numerically in [1]; exact form not provided.",
    )

KPOINT_DATA = {
    "X33":  _build_kpoint_X33(),
    "X44":  _build_kpoint_X44(),
    "X66":  _build_kpoint_X66(),
    "4.47": _build_kpoint_447(),
    "3.7":  _build_kpoint_37(),
}


# ─────────────────────────────────────────────────────────────────────────────
# C-point boundary data (Table 3 of [1])
#
# Near the C-point, s = (X[1] − τ) / X[0]. The prepotential is
# F = F₀ + s·F₁ + s²·F₂ + s³·F₃ + …
#
# Parameters stored per model:
#   tau   : rigid period τ (complexified gauge coupling of the heavy vector)
#   gamma : extension datum γ ∈ ℝ
#   delta : extension datum δ ∈ ℝ
#   k     : conifold order parameter
#   A1,A2 : first two instanton coefficients A_k = Â_k × Anorm
#   Anorm : common transcendental prefactor of the A-series
#   Lf1   : L(f,1) for the associated weight-4 modular form f
#   Ahat  : dict of rational Â_k values (only non-zero entries)
#   modular_form : LMFDB label of the associated modular form
# ─────────────────────────────────────────────────────────────────────────────

def _build_cpoint_X42():
    """Build C-point boundary data for the X42 (degree-32) model."""
    Lf1  = _Lf32C_1()
    Anorm = -1.0 / Lf1
    return dict(
        tau=complex(0.5, 0.5),
        gamma=0.0, delta=0.5,
        k=2,
        Anorm=Anorm, Lf1=Lf1,
        A1=1.0*Anorm, A2=0.0,
        # Non-zero: Â₁=1, Â₅=−7/30, Â₉=−65/3528  (spacing 4)
        Ahat={1: 1.0, 5: -7/30, 9: -65/3528},
        modular_form="32.4.a.b",
    )

def _build_cpoint_X322():
    """Build C-point boundary data for the X322 (degree-9) model."""
    Lf1  = _Lf9C_1()
    Anorm = 2.0 * 2**(1/3) * np.sqrt(3) / (27.0 * Lf1)
    return dict(
        tau=complex(0.5, 1.0/(2*np.sqrt(3))),
        gamma=0.0, delta=0.0,
        k=6,
        Anorm=Anorm, Lf1=Lf1,
        A1=1.0*Anorm, A2=0.0,
        Ahat={1: 1.0, 7: -272/14175, 13: -5256154/6450283125},
        modular_form="9.4.a.a",
    )

def _build_cpoint_X62():
    """Build C-point boundary data for the X62 (degree-108) model."""
    Lf1  = _Lf108C_1()
    Anorm = -32.0 / (6561.0 * Lf1)   # −2⁵ / (3⁸ · Lf1)
    return dict(
        tau=complex(0.5, np.sqrt(3)/2),
        gamma=2/3, delta=1/3,
        k=1,
        Anorm=Anorm, Lf1=Lf1,
        A1=1.0*Anorm, A2=0.0,
        Ahat={1: 1.0, 4: -70/9, 7: -314432/1148175},
        modular_form="108.4.a.b",
    )

def _build_cpoint_262():
    """Build C-point boundary data for the 2.62 (degree-32) model."""
    Lf1  = _Lf32C_1()
    Anorm = -1j / (2.0 * np.sqrt(2) * Lf1)
    return dict(
        tau=complex(0.5, 0.5),
        gamma=0.0, delta=0.5,
        k=42,
        Anorm=Anorm, Lf1=Lf1,
        A1=1.0*Anorm, A2=0.0,
        Ahat={1: 1.0, 5: -89/2560, 9: -19407/3670016},
        modular_form="32.4.a.b",
    )

CPOINT_DATA = {
    "X42":  _build_cpoint_X42(),
    "X322": _build_cpoint_X322(),
    "X62":  _build_cpoint_X62(),
    "2.62": _build_cpoint_262(),
}


# ─────────────────────────────────────────────────────────────────────────────
# LCS prepotential  (standard, valid for all 14 hypergeometric models)
#
# F(X₀,X₁) = κ/6·X₁³/X₀ − σ/2·X₁² − c₂/24·X₀X₁ − K₀·X₀²
# where σ = (κ/2) mod 1  and  K₀ = ζ(3)·χ / (2·(2πi)³)
# ─────────────────────────────────────────────────────────────────────────────

def make_prepot_LCS(kappa, c2, chi):
    r"""
    **Description:**
    Return an LCS prepotential callable ``F(X, conj=False)``.

    Args:
        kappa (int): Triple intersection number of the mirror CY.
        c2 (int): Integrated second Chern class (:math:`c_2 \cdot D`) of the mirror CY.
        chi (int): Euler characteristic of the mirror CY.

    Returns:
        callable: ``F(X, conj=False)`` where ``X`` has shape ``(2,)``. Returns the homogeneous degree-2 prepotential (scalar).
    """
    sigma = (kappa / 2) % 1
    K0 = scipy_zeta(3) * chi / (2.0 * (2j * np.pi)**3)
    K0_conj = np.conj(K0)

    @partial(jax.jit, static_argnames=("conj",))
    def F(X, conj=False):
        """Evaluate the LCS prepotential for periods X.

        Caller convention: at conj=True the user passes X̄ and the function
        returns conj(F(X)). Since kappa, sigma, c2 are real, the only piece
        that needs conjugation is K0 (which is purely imaginary).
        """
        z = X[1] / X[0]
        K0_use = K0_conj if conj else K0
        F_inh = (kappa/6) * z**3 - (sigma/2) * z**2 - (c2/24) * z - K0_use
        return X[0]**2 * F_inh

    return F


def make_period_vector_LCS(kappa, c2, chi):
    r"""
    **Description:**
    Return a period-vector callable :math:`\Pi(X)` near LCS.

    The ordering is :math:`\Pi = (\partial F/\partial X^0,\, \partial F/\partial X^1,\, X^0,\, X^1)`
    consistent with jaxvacua's prepotential-input interface.

    Args:
        kappa (int): Triple intersection number of the mirror CY.
        c2 (int): Integrated second Chern class (:math:`c_2 \cdot D`) of the mirror CY.
        chi (int): Euler characteristic of the mirror CY.

    Returns:
        callable: ``Pi(X, conj=False)`` where ``X`` has shape ``(2,)``. Returns the period vector of shape ``(4,)``.
    """
    F = make_prepot_LCS(kappa, c2, chi)

    def Pi(X, conj=False):
        """Evaluate the LCS period vector for periods X."""
        grad_F = jax.grad(lambda x: F(x, conj=conj), holomorphic=True)(X)
        return jnp.concatenate([grad_F, X])

    return Pi


# ─────────────────────────────────────────────────────────────────────────────
# K-point prepotential  (eqs. 1060–1063 of [1], matches notebook 14)
#
# s = (X[1] − τ) / X[0]
# F = F₀ + s·F₁ + s²·F₂ + s³·F₃ + …
#   F₀ = δ/2 + γ·τ
#   F₁ = γ − c·τ₂/(2π)·(log(i·s/(2·B₁·τ₂)) − 1)
#   F₂ = i·c/(4π²)·(log(i·s/(2·B₁·τ₂)) − 2 + B₂/(4·B₁²))
#   F₃ = c/(16π·τ₂)·(1 − B₂²/B₁⁴ + 2·B₃/(3·B₁³))
# ─────────────────────────────────────────────────────────────────────────────

def make_prepot_Kpoint(tau, gamma, delta, c, B1, B2, B3):
    r"""
    **Description:**
    Return a K-point prepotential callable ``F(X, conj=False)``.

    Args:
        tau (complex): Rigid period :math:`\tau` (:math:`\operatorname{Im}\tau > 0`).
        gamma (float): Extension datum :math:`\gamma`.
        delta (float): Extension datum :math:`\delta`.
        c (float): Log-monodromy coefficient (lower-right entry of :math:`N_K`).
        B1 (complex): Leading instanton coefficient (must be :math:`\neq 0`).
        B2 (complex): Second instanton coefficient.
        B3 (complex): Third instanton coefficient.

    Returns:
        callable: ``F(X, conj=False)`` where ``X`` has shape ``(2,)``. Returns the homogeneous degree-2 prepotential (scalar).
    """
    tau2 = float(tau.imag)

    @partial(jax.jit, static_argnames=("conj",))
    def F(X, conj=False):
        """Evaluate the K-point prepotential for periods X."""
        tau_use = jnp.conj(jnp.array(tau)) if conj else jnp.array(tau)
        s = (X[1] - tau_use) / X[0]
        F0 = delta / 2 + gamma * tau_use
        if conj:
            logarg = -1j * s / (2 * B1 * tau2)
            F1 = gamma - c * tau2 / (2 * jnp.pi) * (jnp.log(logarg) - 1)
            F2 = -1j * c / (4 * jnp.pi**2) * (jnp.log(logarg) - 2 + B2 / (4 * B1**2))
        else:
            logarg = 1j * s / (2 * B1 * tau2)
            F1 = gamma - c * tau2 / (2 * jnp.pi) * (jnp.log(logarg) - 1)
            F2 = 1j * c / (4 * jnp.pi**2) * (jnp.log(logarg) - 2 + B2 / (4 * B1**2))
        F3 = c / (16 * jnp.pi * tau2) * (1 - B2**2 / B1**4 + 2 * B3 / (3 * B1**3))
        return F0 + s * F1 + s**2 * F2 + s**3 * F3

    return F


# ─────────────────────────────────────────────────────────────────────────────
# C-point prepotential  (eqs. 846–849 of [1], matches notebook 14 convention)
#
# s = (X[1] − τ) / X[0]
# F = F₀ + s·F₁ + s²·F₂ + s³·F₃ + …
#   F₀ = τ/2
#   F₁ = δ − γ·τ
#   F₂ = i·k·(3 − 2·log(s) + 2·log(A₁)) − γ·F₁
#   F₃ = −i·k·(3·A₁²·γ − A₂) / (12·π·A₁²)
#
# Note: The coefficient of F₂ differs from the paper by a factor of (8π)⁻¹,
# absorbed into the instanton normalization Anorm.
# ─────────────────────────────────────────────────────────────────────────────

def make_prepot_Cpoint(tau, gamma, delta, k, A1, A2):
    r"""
    **Description:**
    Return a C-point prepotential callable ``F(X, conj=False)``.

    Args:
        tau (complex): Rigid period :math:`\tau`.
        gamma (float): Extension datum :math:`\gamma`.
        delta (float): Extension datum :math:`\delta`.
        k (int): Conifold order parameter.
        A1 (complex): Leading instanton coefficient (:math:`A_1 \neq 0`).
        A2 (complex): Second instanton coefficient.

    Returns:
        callable: ``F(X, conj=False)`` where ``X`` has shape ``(2,)``. Returns the homogeneous degree-2 prepotential (scalar).
    """
    A1_arr = jnp.array(A1) + 0j   # ensure complex dtype so jnp.log handles negative reals
    log_A1 = jnp.log(A1_arr)
    log_A1_conj = jnp.conj(log_A1)

    @partial(jax.jit, static_argnames=("conj",))
    def F(X, conj=False):
        """Evaluate the C-point prepotential for periods X."""
        tau_use = jnp.conj(jnp.array(tau)) if conj else jnp.array(tau)
        s = (X[1] - tau_use) / X[0]
        F0 = tau_use / 2
        F1 = delta - gamma * tau_use
        if conj:
            # log(A1) on the conjugate branch must equal conj(log(A1_orig)) for
            # the prepotential to satisfy F(conj(z), conj=True) = conj(F(z)).
            # For complex or negative-real A1, that's NOT the same as log(A1).
            F2 = -1j * k * (3 - 2 * jnp.log(s) + 2 * log_A1_conj) - gamma * F1
            F3 =  1j * k * (3 * A1**2 * gamma - A2) / (12 * jnp.pi * A1**2)
        else:
            F2 =  1j * k * (3 - 2 * jnp.log(s) + 2 * log_A1) - gamma * F1
            F3 = -1j * k * (3 * A1**2 * gamma - A2) / (12 * jnp.pi * A1**2)
        return F0 + s * F1 + s**2 * F2 + s**3 * F3

    return F


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

class HypergeometricModels:
    r"""
    **Description:**
    Static registry + factory for the hardcoded one-modulus CY models from
    arXiv:2306.01059 (quintics, hypergeometrics, etc.).  Construct ready-to-use
    :class:`~jaxvacua.flux_vacua_finder.FluxVacuaFinder` instances at LCS,
    K-point, or C-point boundaries via :meth:`build`.

    The class itself is never instantiated; all methods are classmethods.

    Examples
    --------
    >>> import jaxvacua as jvc
    >>> jvc.HypergeometricModels.list()           # discover available labels
    ['447', 'X10', 'X33', ...]
    >>> m = jvc.HypergeometricModels.build("X33", limit="LCS")
    >>> m.DW_x(x0, flux)                          # standard FluxVacuaFinder API
    """

    @classmethod
    def list(cls):
        r"""
        **Description:**
        Return all registered model labels (hypergeometric + additional).

        Returns:
            list[str]: Sorted labels.
        """
        return sorted(ALL_MODELS.keys())

    @classmethod
    def available_limits(cls, label):
        r"""
        **Description:**
        Return the moduli-space points at which this model has closed-form data.

        Args:
            label (str): Model label, e.g. ``"X33"``.

        Returns:
            list[str]: Subset of ``["LCS", "Kpoint", "Cpoint"]``.
        """
        out = ["LCS"] if label in ALL_MODELS else []
        if label in KPOINT_DATA and label != "X66":
            out.append("Kpoint")
        if label in CPOINT_DATA:
            out.append("Cpoint")
        return out

    @classmethod
    def lcs_data(cls, label):
        r"""
        **Description:**
        Return the raw LCS-side topological data for a model.

        Args:
            label (str): Model label.

        Returns:
            dict: Contains ``kappa``, ``c2``, ``chi``, ``alpha``, ``mu`` etc.

        Raises:
            KeyError: If ``label`` is not in the registry.
        """
        if label not in ALL_MODELS:
            raise KeyError(f"Unknown hypergeometric model: {label!r}")
        return dict(ALL_MODELS[label])

    @classmethod
    def boundary_data(cls, label, limit):
        r"""
        **Description:**
        Return the raw K- or C-point boundary parameters for a model
        (:math:`\tau`, :math:`\gamma`, :math:`\delta`, instanton coefficients,
        etc.).

        Args:
            label (str): Model label.
            limit (str): ``"Kpoint"`` or ``"Cpoint"``.

        Returns:
            dict: Boundary-data fields.

        Raises:
            KeyError: If no boundary data is registered for ``(label, limit)``.
            ValueError: If ``limit`` is not ``"Kpoint"`` or ``"Cpoint"``.
        """
        if limit == "Kpoint":
            if label not in KPOINT_DATA:
                raise KeyError(f"No K-point data for {label!r}")
            return dict(KPOINT_DATA[label])
        if limit == "Cpoint":
            if label not in CPOINT_DATA:
                raise KeyError(f"No C-point data for {label!r}")
            return dict(CPOINT_DATA[label])
        raise ValueError(f"limit must be 'Kpoint' or 'Cpoint'; got {limit!r}")

    @classmethod
    def resolve_prepotential(cls, model_ID, limit):
        r"""
        **Description:**
        Validate ``(model_ID, limit)`` against the hypergeometric registry and
        return the closed-form prepotential callable for the requested
        boundary.  Used by :class:`~jaxvacua.periods.periods` to auto-detect
        and resolve registered models without the user having to pass an
        explicit ``prepotential_input``.

        The ``h12 != 1`` invariant is checked by the caller.

        Args:
            model_ID (str): Hypergeometric model label (e.g. ``"X33"``).
            limit (str): ``"Kpoint"`` or ``"Cpoint"``.

        Returns:
            callable: ``F(X, conj=False)`` — the closed-form prepotential.

        Raises:
            ValueError: If ``model_ID`` is not in the registry, or if
                ``limit`` is not one of ``{"Kpoint", "Cpoint"}``.
            KeyError: If no boundary data is registered for ``(model_ID, limit)``.
            NotImplementedError: For ``X66`` at K-point.
        """
        if model_ID not in ALL_MODELS:
            raise ValueError(
                f"limit={limit!r} requires `model_ID=` to be one of the "
                f"registered one-modulus model labels (e.g. 'X33', 'X42', "
                f"'2.62'); got {model_ID!r}. For custom one-modulus models, "
                f"pass `prepotential_input=` directly."
            )
        if limit == "Kpoint":
            if model_ID == "X66":
                raise NotImplementedError(
                    "X66 K-point requires the degenerate B1=0 formula "
                    "(see arXiv:2306.01059 §5); not implemented yet."
                )
            if model_ID not in KPOINT_DATA:
                raise KeyError(f"No K-point data for {model_ID!r}.")
            bd = KPOINT_DATA[model_ID]
            return make_prepot_Kpoint(
                tau=bd["tau"], gamma=bd["gamma"], delta=bd["delta"],
                c=bd["c"], B1=bd["B1"], B2=bd["B2"], B3=bd["B3"],
            )
        if limit == "Cpoint":
            if model_ID not in CPOINT_DATA:
                raise KeyError(f"No C-point data for {model_ID!r}.")
            bd = CPOINT_DATA[model_ID]
            return make_prepot_Cpoint(
                tau=bd["tau"], gamma=bd["gamma"], delta=bd["delta"],
                k=bd["k"], A1=bd["A1"], A2=bd["A2"],
            )
        raise ValueError(f"limit must be 'Kpoint' or 'Cpoint'; got {limit!r}")

    @classmethod
    def build(cls, label, limit="LCS", **kwargs):
        r"""
        **Description:**
        Construct a :class:`~jaxvacua.flux_vacua_finder.FluxVacuaFinder` for the
        named model at the given moduli-space point.

        For ``limit="LCS"`` the factory builds a standard ``lcs_tree`` from
        ``(kappa, c2, chi)`` with no GV invariants — the closed-form LCS
        prepotential of arXiv:2306.01059 already includes the instanton sum
        resummed into modular forms.

        For ``limit="Kpoint"`` or ``"Cpoint"`` the factory just forwards
        ``model_ID=label`` and ``limit=`` to :class:`FluxVacuaFinder`; the
        ``periods``/``css`` constructors auto-detect the registry entry and
        resolve the closed-form prepotential.

        Extra ``**kwargs`` are forwarded to :class:`FluxVacuaFinder`
        (e.g. ``Q=``, ``gauge_choice=``, ``D3_tadpole=``).

        Args:
            label (str): Model label, e.g. ``"X33"``, ``"X42"``, ``"X5"``.
            limit (str, optional): ``"LCS"``, ``"Kpoint"``, or ``"Cpoint"``.
                Defaults to ``"LCS"``.
            **kwargs: Forwarded to :class:`FluxVacuaFinder`.

        Returns:
            FluxVacuaFinder: A standard finder instance.

        Raises:
            KeyError: If ``label`` is not registered.
            ValueError: If ``limit`` is not one of the three supported values.
            NotImplementedError: For X66 at K-point (degenerate :math:`B_1=0`).
        """
        from .flux_vacua_finder import FluxVacuaFinder
        if label not in ALL_MODELS:
            raise KeyError(
                f"Unknown hypergeometric model {label!r}. "
                f"Available: {cls.list()}"
            )

        if limit == "LCS":
            data = ALL_MODELS[label]
            # Build a minimal model_data dict carrying the closed-form LCS
            # topological data; no GV invariants — the LCS prepotential is
            # complete as the polynomial-only F_poly(z).
            model_data = {
                "h11":          1,
                "h12":          1,
                "intnums":      jnp.array([[[data["kappa"]]]], dtype=jnp.int_),
                "c2":           jnp.array([data["c2"]], dtype=jnp.int_),
                "chi":          int(data["chi"]),
                "model_type":   "hypergeometric",
                "model_ID":     label,
                "limit":        "LCS",
            }
            return FluxVacuaFinder(
                h12=1, model_type="hypergeometric", model_ID=label,
                model_data=model_data,
                limit="LCS",
                **kwargs,
            )

        if limit in ("Kpoint", "Cpoint"):
            # periods/css auto-detect model_ID + limit and resolve the
            # closed-form prepotential from the registry. We just forward.
            return FluxVacuaFinder(
                h12=1, model_ID=label,
                limit=limit,
                **kwargs,
            )

        raise ValueError(
            f"Unknown limit: {limit!r}. "
            f"Available for {label}: {cls.available_limits(label)}."
        )


def list_hypergeometric_models():
    r"""
    **Description:**
    Print a summary table of all available models and the moduli-space points
    at which each has closed-form data.  For programmatic use, prefer
    :meth:`HypergeometricModels.list`.
    """
    print("=" * 70)
    print("Hypergeometric one-modulus models (Table 1 of arXiv:2306.01059)")
    print("=" * 70)
    for label, d in HYPERGEOMETRIC_MODELS.items():
        avail = HypergeometricModels.available_limits(label)
        print(f"  {label:8s}  κ={d['kappa']:3d}  χ={d['chi']:6d}  "
              f"∞-type={d['singularity_at_inf']}  {avail}")
        print(f"           {d['name']}")
    print()
    print("Additional non-hypergeometric models")
    print("-" * 70)
    for label, d in NON_HYPERGEOMETRIC_MODELS.items():
        avail = HypergeometricModels.available_limits(label)
        print(f"  {label:8s}  {avail}  {d['name']}")
    print()
    print("K-point data available for:", list(KPOINT_DATA.keys()))
    print("C-point data available for:", list(CPOINT_DATA.keys()))
