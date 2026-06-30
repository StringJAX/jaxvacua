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

.. autosummary::
    :toctree: _autosummary

    vmapping_func
    vmapping_func_cached
    jit_with_static_args
    jit_with_dynamic_static_args
    is_static


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


Pytree flatten / unflatten
--------------------------

Generic flatten / unflatten functions used by ``register_pytree_node`` for
the project's pytree-registered classes (``periods``, ``css``, ``FluxEFT``,
``FluxVacuaFinder`` and ``lcs_tree``).

.. autosummary::
    :toctree: _autosummary

    flatten_func
    unflatten_func_class


Number-theoretic / lattice helpers
----------------------------------

.. autosummary::
    :toctree: _autosummary

    extended_euclidean
    orthogonal_lattice
