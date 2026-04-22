# ==============================================================================
# one_modulus_models.py
#
# Standalone module for loading period data of one-modulus Calabi-Yau models
# from arXiv:2306.01059 (Bastian, van de Heisteeg, Schlechter).
#
# Provides model databases and prepotential/period functions near each singularity:
#   - LCS    (Large Complex Structure / MUM, at z → 0)
#   - Kpoint (finite-order monodromy, at z → ∞ for K-type models)
#   - Cpoint (conifold singularity)
#
# No dependency on other jaxvacua modules. All constants are computed from
# closed-form expressions involving Γ-function values (from arXiv:2306.01059,
# Tables 3–4).
#
# Prepotential conventions follow the companion notebook 14_period_input.ipynb.
# The K-point formula reproduces eqs. (1060–1063) of [1] exactly.
# The C-point formula reproduces eqs. (846–849) of [1] in a convention where
# the instanton normalization absorbs a factor of (8π)⁻¹ into A₁.
#
# Usage
# -----
# >>> from jaxvacua.one_modulus_models import get_model, list_models
# >>> list_models()
# >>> model = get_model("X33", "Kpoint")
# >>> model.prepotential(X)          # callable F(X, conj=False)
# >>> model.period_vector(X)         # Π(X) = (∂F/∂X^I, X^I)
#
# References
# ----------
# [1] Bastian, van de Heisteeg, Schlechter, arXiv:2306.01059
# [2] AESZ database: Almkvist, Enckevort, van Straten, Zudilin
# ==============================================================================

import functools

import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import gamma as gamma_func
from scipy.special import zeta as scipy_zeta

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

    def F(X, conj=False):
        """Evaluate the LCS prepotential for periods X."""
        if conj:
            X = jnp.conj(X)
        z = X[1] / X[0]
        F_inh = (kappa/6) * z**3 - (sigma/2) * z**2 - (c2/24) * z - K0
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
    def F(X, conj=False):
        """Evaluate the C-point prepotential for periods X."""
        tau_use = jnp.conj(jnp.array(tau)) if conj else jnp.array(tau)
        s = (X[1] - tau_use) / X[0]
        F0 = tau_use / 2
        F1 = delta - gamma * tau_use
        if conj:
            F2 = -1j * k * (3 - 2 * jnp.log(s) + 2 * jnp.log(A1)) - gamma * F1
            F3 =  1j * k * (3 * A1**2 * gamma - A2) / (12 * jnp.pi * A1**2)
        else:
            F2 =  1j * k * (3 - 2 * jnp.log(s) + 2 * jnp.log(A1)) - gamma * F1
            F3 = -1j * k * (3 * A1**2 * gamma - A2) / (12 * jnp.pi * A1**2)
        return F0 + s * F1 + s**2 * F2 + s**3 * F3

    return F


# ─────────────────────────────────────────────────────────────────────────────
# Model container
# ─────────────────────────────────────────────────────────────────────────────

class OneModulusModel:
    r"""**Description:** Container for a one-modulus CY model at a specific singularity.

    Attributes:
        label (str): Model identifier (e.g. ``"X33"``).
        singularity (str): Singularity type (``"LCS"``, ``"Kpoint"``, or ``"Cpoint"``).
        lcs_data (dict): LCS topological data (:math:`\kappa`, :math:`c_2`, :math:`\chi`, …).
        boundary (dict): Boundary data for the singularity (:math:`\tau`, :math:`\gamma`, …).
        prepotential (callable): ``F(X, conj=False)``, homogeneous degree-2 in :math:`X=(X_0,X_1)`.
    """

    def __init__(self, label, singularity, lcs_data, boundary, prepotential):
        """Initialise a OneModulusModel from its components."""
        self.label       = label
        self.singularity = singularity
        self.lcs_data    = lcs_data
        self.boundary    = boundary
        self.prepotential = prepotential

    def __repr__(self):
        r"""**Description:** Return a string representation of the model."""
        name = ALL_MODELS.get(self.label, {}).get("name", self.label)
        return (f"OneModulusModel(label='{self.label}', "
                f"singularity='{self.singularity}', "
                f"name='{name}')")

    @functools.lru_cache(maxsize=4)
    def _period_vector_jit(self, conj):
        r"""
        **Description:**
        Return a JIT-compiled period-vector function for a fixed ``conj`` flag.

        Args:
            conj (bool): If ``True``, return the function for the conjugate conventions.

        Returns:
            callable: JIT-compiled function ``_pv(X)`` returning the period vector of shape ``(4,)``.
        """
        F = self.prepotential

        @jax.jit
        def _pv(X):
            """Compute the period vector for a fixed conjugation flag."""
            # F is holomorphic in X (conj=False) or in X̄ (conj=True).
            # Use jax.grad with holomorphic=True for a single gradient call.
            grad_F = jax.grad(lambda x: F(x, conj=conj), holomorphic=True)(X)
            return jnp.concatenate([grad_F, X])
        return _pv

    def period_vector(self, X, conj=False):
        r"""
        **Description:**
        Compute :math:`\Pi = (\partial F/\partial X^0,\, \partial F/\partial X^1,\, X^0,\, X^1)`
        via JAX autodiff (JIT-compiled).

        Args:
            X (Array): Homogeneous coordinates, shape ``(2,)``.
            conj (bool, optional): If ``True``, use conjugate conventions. Defaults to ``False``.

        Returns:
            Array: Period vector of shape ``(4,)``.
        """
        return self._period_vector_jit(conj)(X)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def list_models():
    r"""**Description:** Print a summary of all available models and singularity types."""
    print("=" * 70)
    print("Hypergeometric one-modulus models (Table 1 of arXiv:2306.01059)")
    print("=" * 70)
    for label, d in HYPERGEOMETRIC_MODELS.items():
        kpoints  = "Kpoint " if label in KPOINT_DATA else ""
        cpoints  = "Cpoint " if label in CPOINT_DATA else ""
        avail = f"[LCS {kpoints}{cpoints}]"
        print(f"  {label:8s}  κ={d['kappa']:3d}  χ={d['chi']:6d}  "
              f"∞-type={d['singularity_at_inf']}  {avail}")
        print(f"           {d['name']}")
    print()
    print("Non-hypergeometric models (additional examples in [1])")
    print("-" * 70)
    for label, d in NON_HYPERGEOMETRIC_MODELS.items():
        kpoints = "Kpoint " if label in KPOINT_DATA else ""
        cpoints = "Cpoint " if label in CPOINT_DATA else ""
        avail = f"[LCS {kpoints}{cpoints}]"
        print(f"  {label:8s}  {avail}  {d['name']}")
    print()
    print("K-point data available for:", list(KPOINT_DATA.keys()))
    print("C-point data available for:", list(CPOINT_DATA.keys()))


def get_model(label, singularity="LCS"):
    r"""
    **Description:**
    Load a one-modulus model at a given singularity type.

    Args:
        label (str): Model label, e.g. ``"X33"``, ``"X44"``, ``"X42"``, ``"X5"``, …
        singularity (str, optional): One of ``"LCS"``, ``"Kpoint"``, ``"Cpoint"``.
            Defaults to ``"LCS"``.

    Returns:
        OneModulusModel: Model container with ``prepotential`` and ``period_vector`` attributes.

    Examples:
        >>> model = get_model("X33", "Kpoint")
        >>> model.prepotential(jnp.array([1.0+0j, 0.5+0.5j]))
        >>> model.period_vector(jnp.array([1.0+0j, 0.5+0.5j]))

        Using with jaxvacua ``flux_sector``::

            import jaxvacua as jvc
            fs = jvc.flux_sector(h12=1, moduli_space_limit=None,
                                 model_type=None,
                                 prepotential_input=model.prepotential)
    """
    if label not in ALL_MODELS:
        raise ValueError(
            f"Unknown model '{label}'. "
            f"Available: {list(ALL_MODELS.keys())}"
        )

    lcs_data = ALL_MODELS[label]
    kappa = lcs_data.get("kappa")
    c2    = lcs_data.get("c2")
    chi   = lcs_data.get("chi")

    if singularity == "LCS":
        if kappa is None:
            raise ValueError(f"Model '{label}' has no fixed LCS data (parameter family).")
        prepot = make_prepot_LCS(kappa, c2, chi)
        boundary = {"kappa": kappa, "c2": c2, "chi": chi,
                    "sigma": (kappa / 2) % 1,
                    "K0": scipy_zeta(3) * chi / (2.0 * (2j * np.pi)**3)}
        return OneModulusModel(label, "LCS", lcs_data, boundary, prepot)

    elif singularity == "Kpoint":
        if label not in KPOINT_DATA:
            available = list(KPOINT_DATA.keys())
            raise ValueError(
                f"No K-point data for '{label}'. "
                f"Available: {available}"
            )
        bd = KPOINT_DATA[label]
        if bd.get("B1", 1.0) == 0.0:
            raise ValueError(
                f"Model '{label}' has B₁=0 (leading instanton at k≥2). "
                f"The standard prepotential formula requires B₁≠0. "
                f"Use KPOINT_DATA['{label}'] directly and implement a "
                f"modified prepotential."
            )
        prepot = make_prepot_Kpoint(
            tau=bd["tau"], gamma=bd["gamma"], delta=bd["delta"],
            c=bd["c"], B1=bd["B1"], B2=bd["B2"], B3=bd["B3"],
        )
        return OneModulusModel(label, "Kpoint", lcs_data, bd, prepot)

    elif singularity == "Cpoint":
        if label not in CPOINT_DATA:
            available = list(CPOINT_DATA.keys())
            raise ValueError(
                f"No C-point data for '{label}'. "
                f"Available: {available}"
            )
        bd = CPOINT_DATA[label]
        prepot = make_prepot_Cpoint(
            tau=bd["tau"], gamma=bd["gamma"], delta=bd["delta"],
            k=bd["k"], A1=bd["A1"], A2=bd["A2"],
        )
        return OneModulusModel(label, "Cpoint", lcs_data, bd, prepot)

    else:
        raise ValueError(
            f"Unknown singularity '{singularity}'. "
            "Choose from 'LCS', 'Kpoint', 'Cpoint'."
        )
