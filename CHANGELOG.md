# Changelog

All notable changes to JAXVacua will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed
- `tests/test_css.py::TestMonodromyAsymmetric::test_monodromy_single_LCS_equals_numerical` compared the
  raw float matrix of `_monodromy_matrix_LCS` with the integer matrix of `_monodromy_matrix_numerical`
  by exact equality (2026-10-12). Under jax 0.11.2 one entry is 1.9999999999999998, so the scheduled
  CI run of 2026-10-11 failed (job "Full (geometry-medium)"; reproduced in a replica of the CI
  environment). The test now applies `monodromy_matrix_single`'s own rule: integral to 1e-8, then
  rounded. No package code changed.
- The docstring of `css._monodromy_matrix_LCS` described its result both as an integer (`jnp.int32`)
  matrix and as the raw float matrix; it returns the raw float matrix, which
  `monodromy_matrix_single` rounds, and which is integral only for the canonical (HKTY) `a_matrix`
  (2026-10-13; docstring only).

## [0.1.0] - 2026-04-30

First public release.

### Added
- **Periods module** (`periods.py`): evaluation of period integrals, prepotential, Kähler potential, and gauge kinetic matrix. Supports LCS, coniLCS, coniLCS-series, and coniLCS-bulk limits.
- **Complex structure sector** (`css.py`): Kähler geometry of the complex structure moduli space via JAX automatic differentiation (metric, Christoffel symbols, curvature).
- **Flux EFT** (`flux_eft.py`): flux superpotential, covariant derivatives, scalar potential, SL(2,ℤ) duality, tadpole accounting.
- **Flux vacua finder** (`flux_vacua_finder.py`): gradient-based and Newton-type vacuum searches, linearised stability analysis, exact Hessians and physical mass spectra.
- **ISD sampling** (`sampling.py`): Monte Carlo sampling of flux space and the Kähler cone interior, including ISD-biased flux generation (ISD+, ISD−, H, F modes).
- **Flux utilities** (`flux_utils.py`): standalone flux-vector algebra, PFV decomposition, symplectic operations.
- **Conifold utilities** (`conifold_utils.py`): conifold-sector tools for coni-LCS models.
- **Freezer** (`freezer.py`): reduced effective theories by integrating out heavy moduli; abstract base class + conifold implementation.
- **Flux bounding** (`flux_bounding.py`): systematic flux enumeration within a bounded region of moduli space.
- **Hypergeometric models** (`hypergeometric_models.py`): self-contained closed-form one-modulus Calabi-Yau families.
- **StringForge database integration**: the CY database and vacua-vault stack now lives in the sibling `stringforge` package, while JAXVacua consumes its loaded `lcs_tree` / `FluxVacuaFinder` models.
- **CYTools interface** (`cytools_interface.py`): construction of topological data from CYTools polytope triangulations.
- **`lcs_tree` pytree** (`lcs.py`): JAX-registered data container for all topological and geometric input data.
