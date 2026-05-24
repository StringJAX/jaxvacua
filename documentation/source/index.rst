JAXVacua -- String Vacua with JAX
==================================

**JAXVacua** is a Python library for constructing Type IIB flux-vacuum
models and numerically finding vacua with JAX-based automatic
differentiation.  It is intended both as a high-level workflow for
sampling flux vacua and as a modular toolkit for period computations,
effective field theory evaluation, flux sampling, and post-processing.

The documentation is organised by how users usually approach the package:
first the physics and geometry background, then executable tutorials, then
the API reference for individual modules.

How to navigate
---------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: New to the physics
      :link: intro/index
      :link-type: doc

      Start with the introduction chapters.  They explain the Type IIB
      compactification setup, Calabi-Yau input data, periods, moduli
      stabilisation, and perturbatively flat vacua.

   .. grid-item-card:: New to the code
      :link: tutorials
      :link-type: doc

      Use the tutorial catalogue.  It separates quickstart, basic usage,
      vacuum finding, geometry limits, and analysis workflows into a
      guided sequence of notebooks.

   .. grid-item-card:: Looking for a module
      :link: jaxvacua
      :link-type: doc

      Go to the API reference when you already know which object or
      function you need.  The module pages include workflow figures and
      curated member indexes.

   .. grid-item-card:: Looking for examples from papers
      :link: applications/nonSUSY_vacua2023
      :link-type: doc

      The application notes connect package workflows to concrete research
      use cases, including non-SUSY vacua, :math:`W_0` distributions, and
      deep-observation pipelines.

Recommended first path
----------------------

For a first pass through the documentation, read:

1. :doc:`Introduction <intro/index>` for the conceptual map.
2. :doc:`Tutorials <tutorials>` for executable notebooks.
3. :doc:`API documentation <jaxvacua>` once you need precise class and
   function signatures.

The :doc:`quickstart notebook <notebooks/quickstart>` is the shortest route
to a working example.  The :doc:`JAXVacua overview notebook
<notebooks/01_basics/02_jaxvacua_overview>` gives a broader end-to-end
walkthrough.

Citing JAXVacua
---------------

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

Reference lookup
----------------

* :ref:`genindex`
* :ref:`modindex`

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Start here

   intro/index
   tutorials
   jaxvacua

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Introduction

   intro/sugra
   intro/geometries
   intro/flux_compactifications
   intro/moduli_stabilisation
   intro/periods
   intro/pfv

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Applications

   applications/nonSUSY_vacua2023
   applications/W0_distribution2023
   applications/deep_observations2025
