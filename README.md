# JAXVacua - Flux Vacua in String Theory with JAX

<p align="center">
  <a href="https://github.com/AndreasSchachner/jaxvacua/actions/workflows/ci.yml"><img src="https://github.com/AndreasSchachner/jaxvacua/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
  <a href="https://www.gnu.org/licenses/gpl-3.0"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="License: GPL v3"/></a>
</p>


**JAXVacua** is a Python library designed for the systematic exploration of flux vacua in string theory through the numerical analysis of supergravity scalar potentials. It leverages automatic differentiation and just-in-time compilation tools provided by the [JAX library](https://github.com/google/jax) to efficiently locate and analyze critical points of highly non-linear potentials. The package is intended to be accessible both as a high-level user-facing library and as a flexible collection of modular components that can be reused or extended for custom workflows. In particular, JAXVacua is well suited for the study of flux compactifications in string theory, where large parameter spaces and intricate scalar potentials make traditional analytic approaches challenging.


The introduction gives a summary of the physical and mathematical context and aim of the library, which serves to give a broad overview to the structure and code of the library. The tutorials show how to use the library on a code level and give several examples.

## Installation

You may want to install the code in a new virtual environment. This can be created using `python -m venv jaxvacua-env` and activated using `source jaxvacua-env/bin/activate` from within the terminal at a desired working directory.

> [!NOTE]
> If a specific version of [JAX](https://github.com/jax-ml/jax) is required, e.g. with GPU support, follow the instructions [here](https://github.com/jax-ml/jax#installation). Otherwise, by default the CPU version of JAX will be installed.


The recommended way to install the package is via [pip](https://packaging.python.org/en/latest/key_projects/#pip). Before installing, make sure your packaging tools are up to date by running:

`pip install --upgrade pip setuptools`

Next, choose the installation method that best fits your use case:

- **Editable install directly from GitHub:** 
  If you only want to use the package or make light modifications, you can install it directly from the repository with:
  `pip install -e git+https://github.com/AndreasSchachner/jaxvacua.git#egg=jaxvacua`

- **Editable install from a local clone (recommended for development):**  
  If you plan to actively develop or experiment with the code, first download or clone the repository. Then navigate to the root directory of the project in a terminal and run:
  `pip install -e .`  
  The `-e` (editable) option ensures that any local code changes take effect immediately without requiring reinstallation.


## Requirements

Currently, the code supports Python versions `>= 3.12`, both with and without JAX GPU support.

The required packages, which are listed in `setup.py`, are automatically installed with the above installation process. Otherwise, they have to be installed manually.


> [!NOTE]
> JAXVacua defaults to `float64` (double precision) but also supports `float32` via `jvc.set_precision("float32")` or the `JAXVACUA_PRECISION=float32` environment variable. On Apple Silicon, [JAX Metal](https://developer.apple.com/metal/jax/) does not currently support complex-number operations, so `jaxvacua` detects a Metal backend on import and transparently falls back to the CPU backend for the complex-arithmetic code paths. A dedicated conda environment is provided in [environment_metal.yml](environment_metal.yml).



## Documentation

The documentation of this repository is generated with [sphinx](https://www.sphinx-doc.org/en/master/). Following the installation of the `jaxvacua` package, install the additional requirements in [documentation/requirements.txt](documentation/requirements.txt) by running `pip install -r documentation/requirements.txt` from the repository root. Then `cd documentation && make html` generates the html version, which is placed in [documentation/build/html](documentation/build/html).


## Quick start

A geometry can be loaded in several ways — pick whichever is most convenient:

**1. Bundled local models.** A small selection of Kreuzer-Skarke (`KS`) and complete-intersection (`CICY`) models ships with the package under [jaxvacua/models/](jaxvacua/models/), indexed by $h^{1,2}$ and a `model_ID`:

```python
import jaxvacua as jvc
model = jvc.FluxEFT(h12=2, model_ID=1)
```

**2. Directly from CYTools.** Given a [`cytools.CalabiYau`](https://cy.tools) object, the topological data at LCS is extracted automatically:

```python
# Assuming `cy` is a cytools.CalabiYau object
tree = jvc.lcs_tree.from_cytools(cy, maximum_degree=1)
model = jvc.FluxEFT(lcs_tree=tree)
```

**3. From the CY database (new).** The [`cy-database`](https://huggingface.co/datasets/aschachner/cy-database) HuggingFace dataset hosts precomputed topological data for millions of Calabi-Yau geometries. The database stack lives in the [`stringforge`](https://github.com/AndreasSchachner/stringforge) umbrella package: `TDFDatabase` / `CICYDatabase` for pure I/O, and `LCSDatabase` for ready-to-use `FluxVacuaFinder` instances. The interface downloads only the catalog (~10 MB) upfront and pulls individual model shards on demand:

```python
from stringforge.lcs_database import LCSDatabase
db = LCSDatabase(dataset="tdf")
df = db.query(h12=2, has_conifolds=True)             # catalog-only filter
model = db.load_model(ks_id=int(df.iloc[0]["ks_id"]),
                      triang_id=int(df.iloc[0]["triang_id"]),
                      include_gv=True, include_conifolds=True)
```

For offline / HPC use, pass `offline=True`; cached shards are then served locally. Flux-vacuum solutions can be stored in a local vacua vault and, optionally, pushed to the community [`vacua_vault`](https://huggingface.co/datasets/aschachner/vacua_vault) repository. See the tutorials under [`05_database_and_infrastructure/`](documentation/source/notebooks/05_database_and_infrastructure/) for a full walkthrough.


## Repository structure

    .
    ├── jaxvacua/                       # main package
    │   ├── __init__.py                 # backend detection, precision setup (float32/float64)
    │   ├── periods.py                  # period vector, prepotential, Kähler potential
    │   ├── css.py                      # complex structure sector (Kähler geometry, gauge kinetic matrix)
    │   ├── flux_eft.py                 # FluxEFT: superpotential, F-terms, scalar potential, Hessian
    │   ├── flux_vacua_finder.py        # FluxVacuaFinder: Newton solver, vacuum sampling
    │   ├── flux_bounding.py            # bounded_fluxes: flux enumeration, cluster parallelisation
    │   ├── flux_utils.py               # PFV algebra (flux ↔ PFV ↔ moduli)
    │   ├── critical_points.py          # CriticalPointFinder
    │   ├── sampling.py                 # moduli / flux sampling, ISD sampling
    │   ├── freezer.py                  # Freezer / ConifoldFreezer: light-field EFT
    │   ├── conifold_utils.py           # Conifold class, basis changes, find_conifolds
    │   ├── one_modulus_models.py       # closed-form hypergeometric models
    │   ├── lcs.py                      # lcs_tree: topological data container
    │   ├── database.py                 # CYDatabase / TDFDatabase / CICYDatabase, vacua vault API
    │   ├── cytools_interface.py        # CYTools interface
    │   ├── util.py, utils_jaxvacua.py  # utilities
    │   ├── vacua_vault/                # vacua vault subpackage + CLI (jaxvacua-vault)
    │   └── models/                     # bundled pre-computed model data (grouped by h12)
    ├── documentation/
    │   ├── source/
    │   │   ├── intro/                  # physics / maths background
    │   │   ├── applications/           # papers using jaxvacua
    │   │   ├── notebooks/              # tutorials (7 thematic subdirs, see below)
    │   │   └── jaxvacua.*.rst          # API reference pages
    │   ├── build/                      # generated html (gitignored)
    │   └── requirements.txt
    ├── tests/                          # pytest suite (test_periods, test_css, test_flux_eft, ...)
    ├── src/jaxpolylog/                 # companion package (poly-logarithm / hypergeometric utilities)
    ├── setup.py
    ├── environment.yml                 # conda env (CPU)
    ├── environment_metal.yml           # conda env (Apple Silicon, falls back to CPU for complex)
    ├── pytest.ini
    ├── CHANGELOG.md
    ├── CITATION.cff
    ├── LICENSE
    └── README.md

Tutorials are grouped by theme under [documentation/source/notebooks/](documentation/source/notebooks/):

    notebooks/
    ├── 01_basics/                      # JAX intro, jaxvacua overview, CYTools interface
    ├── 02_vacuum_finding/              # flux vacua finder, ISD sampling
    ├── 03_flux_bounding/               # bounded fluxes, cluster parallelisation
    ├── 04_geometry_and_limits/         # CICY, sampling, moduli limits, conifolds, monodromy, PFV
    ├── 05_database_and_infrastructure/ # database interface, vacua vault, performance benchmarks
    ├── 06_analysis_and_tools/          # threshold ISD, freezer, visualisation, Hessian analysis
    └── 07_physics_pipelines/           # hypergeometric models, landscape statistics, W0 scan



## Contact

For questions or feedback, please get in touch: <as3475@cornell.edu>.

## License

JAXVacua is released under the [GNU General Public License v3.0](LICENSE).

## Citation

If you find this software useful, please cite:

```bibtex
@article{Dubey:2023dvu,
    author = "Dubey, Abhishek and Krippendorf, Sven and Schachner, Andreas",
    title = "{JAXVacua \textemdash{} a framework for sampling string vacua}",
    eprint = "2306.06160",
    archivePrefix = "arXiv",
    primaryClass = "hep-th",
    doi = "10.1007/JHEP12(2023)146",
    journal = "JHEP",
    volume = "12",
    pages = "146",
    year = "2023"
}
```

