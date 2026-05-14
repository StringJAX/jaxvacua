JAXVacua -- String Vacua with JAX
==================================

**JAXVacua** is a python library for numerically finding minima of supergravity scalar potentials
using automatic differentation tools implemented in the `JAX library <https://github.com/google/jax>`_.
It is meant to be accessible both as a top-level library as well as a toolkit of modular functions.
As of now, this implementation is limited to Type IIB flux vacua.
A generalisation to a wider class of applications is work in progress.


The introduction gives a summary of the physical and mathematical context and aim of the library. The tutorials show how to use the library on a code level and give several examples.


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

    intro/index
    intro/sugra
    intro/geometries
    intro/flux_compactifications
    intro/moduli_stabilisation
    intro/periods
    intro/pfv

.. toctree::
    :maxdepth: 1
    :caption: Tutorials — Basics

    notebooks/01_basics/01_jax_introduction
    notebooks/01_basics/02_jaxvacua_overview
    notebooks/01_basics/03_cytools_interface
    notebooks/01_basics/04_sampling_module

.. toctree::
    :maxdepth: 1
    :caption: Tutorials — Vacuum Finding

    notebooks/02_vacuum_finding/05_finding_flux_vacua
    notebooks/02_vacuum_finding/06_ISD_sampling_principle
    notebooks/02_vacuum_finding/07_ISD_sampling
    notebooks/02_vacuum_finding/8_ISD_sampling_wrapper
    notebooks/02_vacuum_finding/9_sampling_vacua_from_fluxes
    notebooks/02_vacuum_finding/19_non_susy_sampling

.. toctree::
    :maxdepth: 1
    :caption: Tutorials — Flux Bounding

    notebooks/03_flux_bounding/10_flux_bounding
    notebooks/03_flux_bounding/10b_stochastic_flux_search
    notebooks/03_flux_bounding/10c_sample_bounded_fluxes_stepbystep
    notebooks/03_flux_bounding/26_recovering_dataset_B

.. toctree::
    :maxdepth: 1
    :caption: Tutorials — Geometry and Limits

    notebooks/04_geometry_and_limits/12_moduli_space_limits
    notebooks/04_geometry_and_limits/13_coniLCS
    notebooks/04_geometry_and_limits/14_period_input
    notebooks/04_geometry_and_limits/15_hypergeometric_models

.. note::

    The **database, vacua-vault, and cluster-parallelisation** tutorials
    have moved to the StringForge umbrella package, which now hosts the
    shared catalog I/O and storage layer (see ``stringforge.cy_io``,
    ``stringforge.lcs_database``, ``stringforge.vacua_writer``).
    The tutorials live at
    `<https://stringforge.readthedocs.io/en/latest/tutorials/database_and_infrastructure/>`_.

.. toctree::
    :maxdepth: 1
    :caption: Tutorials — Analysis and Tools

    notebooks/06_analysis_and_tools/11_threshold_ISD
    notebooks/06_analysis_and_tools/16_freezer
    notebooks/06_analysis_and_tools/18_sampling_comparison
    notebooks/06_analysis_and_tools/18_visualisation_cookbook

.. toctree::
    :maxdepth: 1
    :caption: Tutorials — Physics Pipelines

    notebooks/07_physics_pipelines/19_mass_spectrum_hessian
    notebooks/07_physics_pipelines/20_pfv_conifold_pipeline
    notebooks/07_physics_pipelines/21_landscape_statistics
    notebooks/07_physics_pipelines/22_performance_benchmarking
    notebooks/07_physics_pipelines/27_hessian_SUGRA


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
