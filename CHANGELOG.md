# Changelog

All notable changes to JAXVacua will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
- **One-modulus models** (`one_modulus_models.py`): self-contained hypergeometric one-modulus Calabi–Yau families.
- **Database** (`database.py`): CYDatabase / TDFDatabase / CICYDatabase classes for loading Calabi–Yau data from HuggingFace (`aschachner/cy-database`) or local parquet shards.
- **CYTools interface** (`cytools_interface.py`): construction of topological data from CYTools polytope triangulations.
- **`lcs_tree` pytree** (`lcs.py`): JAX-registered data container for all topological and geometric input data.
