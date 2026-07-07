jaxvacua.util
=============

.. currentmodule:: jaxvacua.util

.. automodule:: jaxvacua.util


PRNG / random sampling
----------------------

.. autosummary::
    :toctree: _autosummary
    :template: custom-class-template.rst

    PRNGSequence

.. autosummary::
    :toctree: _autosummary

    random_uniform
    random_integer
    random_uniform_jit
    random_integer_jit


JIT / vmap helpers
------------------

The auto-vectorisation helper :func:`auto_vmap` fixes the shapes of the
standard JAXVacua inputs — the complex-structure moduli, the axio-dilaton and
the flux vector — so that the core methods can be ``vmap``-ped over batches of
those inputs in a controlled, testable way (the expected input shapes are set
in one place rather than re-specified at every call site).

.. autosummary::
    :toctree: _autosummary

    auto_vmap
    set_auto_vmap_defaults
    get_auto_vmap_defaults
    reset_auto_vmap_defaults
    set_auto_vmap_default_shapes
    get_auto_vmap_default_shapes
    reset_auto_vmap_default_shapes


Array / numerical helpers
-------------------------

.. autosummary::
    :toctree: _autosummary

    subsets
    flatten
    flatten_top
    check_nan
    compute_evs_hermitian
    rank_matrix


Pickle I/O
----------

.. autosummary::
    :toctree: _autosummary

    load_pickle
    load_zipped_pickle
    save_zipped_pickle


Dict / DataFrame helpers
------------------------

.. autosummary::
    :toctree: _autosummary

    mergeDictionary
    is_outlier


Timeout / progress
------------------

.. autosummary::
    :toctree: _autosummary

    progress_bar_jax
    quit_function
    exit_after


Model-data I/O
--------------

.. autosummary::
    :toctree: _autosummary

    save_model_data


Number-theoretic / lattice helpers
----------------------------------

.. autosummary::
    :toctree: _autosummary

    extended_euclidean
    orthogonal_lattice
