JAXVacua -- String Vacua with JAX
===========================

**JAXVacua** is a python library for numerically finding minima of supergravity scalar potentials
using automatic differentation tools implemented in the `JAX library <https://github.com/google/jax>`_.
It is meant to be accessible both as a top-level library as well as a toolkit of modular functions.
As of now, this implementation is limited to Type IIB flux vacua.
A generalisation to a wider class of applications is work in progress.


The introduction gives a summary of the physiccal and mathematical context and aim of the library, which serves to give a broad overview to the structure and code of the library. The tutorials show how to use the library on a code level and give several examples.


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


Table of contents
-----------------

.. toctree::
    :maxdepth: 1
    :caption: Introduction

    intro/sugra
    intro/geometries
    
.. toctree::
    :maxdepth: 1
    :caption: Tutorials
    
    notebooks/jaxvacua_overview
    notebooks/jax_introduction
    notebooks/cytools_interface
    notebooks/cicy
    notebooks/finding_flux_vacua
    
    
.. toctree::
    :maxdepth: 1
    :caption: Applications

    applications/nonSUSY_vacua2023
    applications/W0_distribution2023
    applications/deep_observations2025

    
    

.. toctree::
    :maxdepth: 3
    :caption: API Documentation
    
    jaxvacua
    
    
    


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
