# JAXVacua - String Vacua with JAX

**JAXVacua** is a python library for numerically finding minima of supergravity scalar potentials
using automatic differentation tools implemented in the [JAX library](https://github.com/google/jax).
It is meant to be accessible both as a top-level library as well as a toolkit of modular functions.
As of now, this implementation is limited to Type IIB flux vacua.
A generalisation to a wider class of applications is work in progress.


The introduction gives a summary of the physiccal and mathematical context and aim of the library, which serves to give a broad overview to the structure and code of the library. The tutorials show how to use the library on a code level and give several examples.

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

Currently, the code works with python versions greater or equal version `3.7`, both with or without JAX GPU support.

The required packages, which are listed in `setup.py`, are automatically installed with the above installation process. Otherwise, they have to be installed manually.


> [!CAUTION]
> JAXVacua requires `float64` support. For this reason, using [JAX Metal](https://developer.apple.com/metal/jax/) on Mac computers is currently not suitable to run JAXVacua.


> [!CAUTION]
> When using the [CYTools](https://cy.tools) docker image, check support for different versions of numpy. 


## Documentation

The documentation of this repository is generated with [sphinx](https://www.sphinx-doc.org/en/master/). Following the installation of the `jaxvacua` package, install the additional requirements in [documentation/requirements.txt](documentation/requirements.txt) by running `pip install -r requirements.txt`. Inside the [documentation](documentation) folder, run `make html` to generate the html version of the documentation, which can afterwards be found in [documentation/build/html](documentation/build/html).


## Repository structure

    .
    ├── data
    │   └── models
    │       └── CICY              
    │       └── KS
    ├── jaxvacua
    │   └── complex_structure_sector.py
    │   └── flux_sector.py
    │   └── periods.py
    │   └── util.py
    ├── documentation
    │   └── build
    │   └── source
    ├── notebooks
    ├── tests
    ├── LICENSE
    ├── README.md
    ├── setup.py
    


## Contact

For questions or feedback, please get in touch: <as3475@cornell.edu> or <a.schachner@lmu.de>.

## Reference

If you find this work useful, please cite::

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



