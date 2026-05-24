jaxvacua.lcs
===============

.. currentmodule:: jaxvacua.lcs

.. automodule:: jaxvacua.lcs

Role in the package
-----------------------------------

``lcs_tree`` is the shared model-data container for large-complex-structure
workflows.  It carries the geometry and instanton data consumed by period
objects, complex-structure sector helpers, flux EFTs, samplers, and vacuum
finders.  Typical fields include intersection numbers, Chern data, cone data,
GV/GW invariants, basis changes, conifold descriptors, and small pieces of
metadata needed to rebuild a model consistently.

Construction routes
-----------------------------------

* ``from_dict`` builds a tree from already assembled model data.
* ``from_file`` loads a saved local model-data file.
* ``from_cytools`` starts from CYTools-derived geometry and normalises it into
  the data layout expected by JAXVacua.

Data lifecycle
-----------------------------------

An ``lcs_tree`` instance is the place where raw geometry is loaded,
normalised, and kept together with any basis-change or conifold information.
Downstream classes can then reuse the same prepotential and cone data without
repeating the conversion step.  The container remains pytree-friendly, so it
can be passed through JAX-aware workflows alongside numerical arrays.

LCS-tree class
-----------------------------------

.. autosummary::
    :toctree: _autosummary
    :template: custom-class-template.rst

    lcs_tree


Constructors
-----------------------------------

.. autosummary::

    lcs_tree.from_dict
    lcs_tree.from_file
    lcs_tree.from_cytools


Conversion helpers
-----------------------------------

.. autosummary::
    :toctree: _autosummary

    coo_to_intnums
