# Changelog

All notable changes to JAXVacua will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] — 0.1.3 (prepared 2026-10-28; not yet tagged)

### Fixed
- **The PFV tadpole check was a factor 2 too lax.** `pfv_conditions` compared −½ M·K with `Q()` = Q_O =
  h11 + h21 + 2, admitting −M·K up to 2 Q_O. The literature writes the bound either as −½ M·K ≤ Q_D3 = Q_O/2
  (arXiv:1912.10047, 2009.03312) or as Q_flux = f·Σ·h = −M·K ≤ Q_O (arXiv:2406.13751, eqs. (2.27)–(2.29));
  `FluxEFT.tadpole` already returned f·Σ·h = −M·K for PFV fluxes, so the check mixed the two. It now tests
  0 ≤ −M·K ≤ Q_O + 2 n_antiD3, and its reported value is −M·K (before: −M·K/2). Its docstring misquoted pfvs's
  condition #5 with a ½. PFVs with Q_O < −M·K ≤ 2 Q_O now fail the check.
- `Vacuum.diagnostics` capped the flux charge at Q_O, so it rejected the one-anti-D3 fluxes of arXiv:2406.13751
  (Q_flux = Q_O + 2, eq. (2.30)); it now allows for `n_antiD3` anti-D3-branes.
- A `PFV`'s diagnostics reused the PFV conditions stored with it, so records saved by 0.1.2 kept the lax tadpole
  verdict; the conditions are now recomputed from (M, K) (the stored ones only if that fails).
- `jaxvacua.__version__` and the introduction's Gauss law (`N_flux + N_D3 = Q_O`, which counted mobile D3-branes
  once against a flux charge counted on the covering space): now 2(N_D3 − N_antiD3) + N_flux = Q_O with
  N_flux = f·Σ·h (arXiv:2406.13751, eq. (2.29)); notebook 20's explanation of the `pfv:` tadpole key; notebook 07
  called Q_O = 276 "Q_D3".
- `util.py` imported `thread` first, which on Python 3 is not the standard library (an unrelated package of
  that name would be imported); it imports `_thread` directly.
- `FluxEFT.tadpole`'s docstring said it returns an `int`; it returns a float (integer-valued for integer fluxes).
  `FluxEFT.Q`'s docstring now says it is Q_O.
- Documentation: 12 in-notebook links pointed at targets MyST does not generate (raw-HTML `<a id>` anchors, and
  heading slugs written with the maths, which MyST's slugs omit); they point at MyST's anchors now. The docs build
  has no warning left, and CI's docs job and Read the Docs now fail on warnings (`-W`, `fail_on_warning: true`).

### Added
- `FluxEFT.tadpole_cap(n_antiD3=0)`: Q_O + 2 n_antiD3, the largest Q_flux allowed by Gauss's law
  (arXiv:2406.13751, eq. (2.29)); a negative or non-integer `n_antiD3` raises.
- `n_antiD3` in `pfv_conditions` (default 0), `PFVData.from_fluxes`, `FluxEFT.pfv_data`,
  `PFV.from_quantum_numbers` (stored as `PFVData.n_antiD3` and serialised; older records load with 0) and
  `Vacuum.diagnostics` (default `None`: the vacuum's own, a PFV's `data.n_antiD3`, else 0; validated). pfvs's
  conifold PFVs carry one anti-D3 (Q_O + 2): build them with `n_antiD3=1`. The samplers keep their explicit `Q`
  budget: pass `Q=model.tadpole_cap(p)` for p anti-D3-branes.
- `VacuumAnalysis.brane_balance`: N_D3 − N_antiD3 = (Q_O − Q_flux)/2 implied by Gauss's law, a reported payload
  (not a check; a half-integral value needs a brane stuck on an O-plane).
- Extras `bounding` (Gurobi, which remains a core dependency: the default moduli sampling needs it) and `all`
  (every optional dependency; pfvs itself still from GitHub). `tests/test_packaging.py`: `__version__` equals
  `setup.py`'s version.

### Changed
- `jaxpolylog>=0.4.0` (was `>=0.3.0`; also CI's pin, which was `>=0.1.0,!=0.2.0`).
- `Vacuum.diagnostics`' cache key includes `n_antiD3`, so a report cached by 0.1.2 is recomputed. The subclass hook
  `_extra_checks` receives `n_antiD3` (a hook written for 0.1.2, taking only the model, is still called without it).
- Tests: the licence headers of 21 test files said Apache-2.0; they now carry the package's GPL-3.0-or-later
  header.

## [0.1.2] - 2026-10-15

Tagged `v0.1.2` at `775f91b6`. The entries below were listed under [Unreleased] at the tag; the summary of the
other changes since 0.1.1 is taken from the commit subjects (`git log v0.1.1..v0.1.2`, 31 commits), not
re-derived from the code.

### Added (from the commit subjects)
- `jaxvacua.vacuum` (the vacuum data containers; `19f14d87`, 2026-07-30), the PFV utilities and the pfvs
  interface (`a92b8c13`, `421d32f9`), the reduced-EFT light mass spectrum (`7164ea9d`), the PFVEFT tangent
  reduction and pytree support (`b8b5a529`), `auto_vmap` utilities and stringjax-tools integration
  (`cad4a1db`, `dd52b068`, `0a129998`: `PytreePolicy`; `from_model` renamed `from_eft`), `CYData` export
  (`70989ee3`), `flux_bounding` exported (`92cd459a`), ISD-sampling ellipsoid helpers (`f8587606`).

### Changed (from the commit subjects)
- Period APIs refactored and period-derivative APIs renamed (`19f14d87`, `70989ee3`); monodromy routines
  migrated to JAX (`e88a293a`); the `a_matrix` basis of `lcs_tree` set to the symplectic choice (`effb5f21`);
  `compute_a_matrix` deprecated (`6134b898`); the freezer's mode-aware on-shell screen (`84b73b2c`); CI matrix
  and test splitting (`7151db30`, `5a8efa10`, `9dfb3979`, `b4c20769`); repository URLs (`31ffc153`);
  documentation and notebooks (`611e8028`, `5bcad1b9`, `538ec58c`).

### Fixed (listed under [Unreleased] at the tag)
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

### Changed (listed under [Unreleased] at the tag)
- CI: the fast job (`pytest -m "not slow"`) exceeded its 30-minute limit on Python 3.12 and took 20.6–27.8
  min on 3.13, against ~8 min before 2026-07-30 (2026-10-15). Two classes added then,
  `tests/test_freezer.py::TestPFVEFTLCS` and `TestPFVEFTConiLCS`, took 500 of the 688 s of the fast
  selection locally. Both are now `@pytest.mark.slow` as whole classes (the LCS class's `setUpClass` alone
  takes ~50 s); they still run on every push in the "Full (freezer)" job. Locally the fast selection now
  takes 170 s (433 passed, 7 skipped, 1 xfailed, with the `test_lcs.py` fix above).
- CI: `tests/test_vacuum.py` runs in the "Full (light-core)" job; it was the only test file in no
  full-suite job (2026-10-15).

## [0.1.1] - 2026-06-11

Tagged `v0.1.1` at `c8ca9e42`.

### Fixed
- Packaging: `find_packages()` and the bundled models (`9b4fd503`, `c8ca9e42`).

## [0.1.0] - 2026-04-30

*(Note added 2026-10-28: PyPI records the upload of 0.1.0 on 2026-06-11, with 0.1.1 the same day.)*

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
