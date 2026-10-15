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
- `tests/test_lcs.py::TestLcsTree::test_from_file_*` looked for `jaxvacua/models/KS/h12_2/model_1.p`, a
  path that never existed: it mixes the old layout `models/KS/h12_2/Model_1/`, removed in `1abf5bfb`
  (2026-04-21), with the new flat `models/h12_2/model_1.p` (2026-10-15). Both tests were skipped ("Model
  file not found") everywhere, CI included, since they were written (`ed7c2d56`, 2026-04-22); with the
  current path both pass.

### Changed
- CI: the fast job (`pytest -m "not slow"`) exceeded its 30-minute limit on Python 3.12 and took 20.6–27.8
  min on 3.13, against ~8 min before 2026-07-30 (2026-10-15). Two classes added then,
  `tests/test_freezer.py::TestPFVEFTLCS` and `TestPFVEFTConiLCS`, took 500 of the 688 s of the fast
  selection locally. Both are now `@pytest.mark.slow` as whole classes (the LCS class's `setUpClass` alone
  takes ~50 s); they still run on every push in the "Full (freezer)" job. Locally the fast selection now
  takes 170 s (433 passed, 7 skipped, 1 xfailed, with the `test_lcs.py` fix above).
- CI: `tests/test_vacuum.py` runs in the "Full (light-core)" job; it was the only test file in no
  full-suite job (2026-10-15).

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
