Tutorials
=========

This page collects the executable notebooks in one place.  Use it as the
main entry point once you want to run code rather than read background
material.

Choosing a path
---------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: First working example
      :link: notebooks/quickstart
      :link-type: doc

      Run the quickstart notebook before diving into the longer workflows.

   .. grid-item-card:: End-to-end package tour
      :link: notebooks/01_basics/02_jaxvacua_overview
      :link-type: doc

      Follow the overview notebook to see geometry input, model
      construction, sampling, refinement, and analysis in one narrative.

   .. grid-item-card:: Vacuum finding
      :link: notebooks/02_vacuum_finding/05_finding_flux_vacua
      :link-type: doc

      Start here for Newton refinement, ISD-biased sampling, and bounded
      flux searches.

   .. grid-item-card:: Geometry and limits
      :link: notebooks/03_geometry_and_limits/09_moduli_limits
      :link-type: doc

      Use these notebooks when you need moduli-space limits, coni-LCS
      expansions, or one-modulus model data.

Tutorial catalogue
------------------

Quickstart and reference
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Notebook
     - Use it for
   * - :doc:`jaxvacua quickstart <notebooks/quickstart>`
     - Minimal setup and first run through the package.
   * - :doc:`Notebook glossary <notebooks/_glossary>`
     - Common notation used across the tutorials, including
       :math:`W_0`, :math:`z_{\rm cf}`, :math:`g_s`, PFV, AFV, ISD, and AISD.

Basics
~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Notebook
     - Use it for
   * - :doc:`Warm-up: string vacua and automatic differentiation in JAX <notebooks/01_basics/01_jax_introduction>`
     - A gentle JAX and automatic-differentiation warm-up in the string-vacua setting.
   * - :doc:`JAXVacua: overview <notebooks/01_basics/02_jaxvacua_overview>`
     - The main end-to-end orientation notebook for the package.
   * - :doc:`CYTools interface <notebooks/01_basics/03_cytools_interface>`
     - Moving from CYTools objects to JAXVacua model data.
   * - :doc:`Sampling module <notebooks/01_basics/04_sampling_module>`
     - Moduli, axio-dilaton, flux, and initial-guess sampling.

Vacuum finding
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Notebook
     - Use it for
   * - :doc:`Finding flux vacua <notebooks/02_vacuum_finding/05_finding_flux_vacua>`
     - Newton refinement and the main SUSY flux-vacuum workflow.
   * - :doc:`Flux vacua ensembles via ISD sampling <notebooks/02_vacuum_finding/06_ISD_sampling_principle>`
     - The principle behind ISD-completed flux seeds.
   * - :doc:`Sampling flux vacua <notebooks/02_vacuum_finding/07_ISD_sampling>`
     - Practical ISD sampling and ensemble generation.
   * - :doc:`Flux bounding: enumeration and stochastic sampling <notebooks/02_vacuum_finding/08_flux_bounding>`
     - Systematic bounded-flux searches and stochastic variants.

Geometry and limits
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Notebook
     - Use it for
   * - :doc:`Moduli-space limits and custom prepotentials <notebooks/03_geometry_and_limits/09_moduli_limits>`
     - Large-complex-structure limits, custom period input, and validity checks.
   * - :doc:`coni-LCS limit and PFV pipeline <notebooks/03_geometry_and_limits/10_coniLCS_pipeline>`
     - Conifold/LCS expansions and perturbatively flat vacuum pipelines.
   * - :doc:`One-modulus Calabi-Yau models <notebooks/03_geometry_and_limits/11_hypergeometric_models>`
     - Hypergeometric one-modulus examples and closed-form prepotentials.

Analysis and pipelines
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Notebook
     - Use it for
   * - :doc:`Freezer <notebooks/04_analysis_and_pipelines/12_freezer>`
     - Reduced EFTs with heavy fields solved away.
   * - :doc:`Visualisation cookbook <notebooks/04_analysis_and_pipelines/13_visualisation_cookbook>`
     - Plotting and inspecting vacuum data.
   * - :doc:`Hessian analysis <notebooks/04_analysis_and_pipelines/14_hessian_analysis>`
     - Mass spectra, Hessians, and stability diagnostics.
   * - :doc:`Landscape statistics <notebooks/04_analysis_and_pipelines/15_landscape_statistics>`
     - Distribution-level analysis of vacuum ensembles.

External workflow tutorials
---------------------------

The database, vacua-vault, and cluster-parallelisation tutorials have moved
to the StringForge umbrella package, which hosts the shared catalogue I/O
and storage layer.  See the
`StringForge database and infrastructure tutorials <https://stringforge.readthedocs.io/en/latest/tutorials/database_and_infrastructure/>`_.

.. toctree::
   :hidden:
   :maxdepth: 2

   tutorials/quickstart
   tutorials/basics
   tutorials/vacuum_finding
   tutorials/geometry_and_limits
   tutorials/analysis_and_pipelines
