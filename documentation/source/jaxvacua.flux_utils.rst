jaxvacua.flux_utils
=====================

.. currentmodule:: jaxvacua.flux_utils

.. automodule:: jaxvacua.flux_utils

Role in solver workflows
-----------------------------------

This module collects stateless helpers used by flux EFTs and vacuum finders.
The functions convert between integral flux vectors and PFV variables, derive
PFV-level moduli, and post-process candidate solutions after a numerical
search.

PFV and flux conversion workflow
-----------------------------------

``flux_to_pfv`` extracts the PFV variables from a full flux vector,
``pfv_to_flux`` reconstructs compatible flux data, and ``pfv_to_moduli``
computes the associated moduli at PFV level.  These helpers are useful when a
search alternates between a full flux description and the reduced PFV
description used by specialised conifold or ISD-inspired pipelines.

PFV algebra and conditions
-----------------------------------

``N_matrix`` returns the flux-contracted intersection form
:math:`N_{ab} = \kappa_{abc} M^c` and ``pfv_p_vector`` the associated p-vector
(:math:`p = N^{-1}K` for LCS; the bulk-only :math:`\hat p` for coniLCS).
``pfv_conditions`` evaluates the perturbatively-flat-vacuum conditions of
arXiv:2512.17095 (§6.2 for LCS, §6.4 for the conifold case), and
``pfv_racetrack`` returns the leading-order analytic 2-term-racetrack estimate
of the effective-theory vacuum (:math:`\tau_0`, :math:`W_0`, :math:`g_s`).  All
four accept batches of :math:`(M, K)` and are attached to
:class:`~jaxvacua.flux_eft.FluxEFT`.

Solution post-processing
-----------------------------------

``dedup_key`` builds stable keys for identifying repeated candidates,
``classify_solution`` labels the outcome of a solver run, ``is_physical``
applies the physicality checks used by the finder, and ``map_to_fd`` maps
solutions to a chosen fundamental domain before comparison or storage.

External ``pfvs`` interop
-----------------------------------

The jaxvacua :math:`\to` ``pfvs`` conversion lives on the topology container as
:meth:`jaxvacua.lcs.lcs_tree.to_cydata` (and the ungated
:meth:`jaxvacua.lcs.lcs_tree.to_cydata_kwargs`), which build a ``pfvs.CYData``
from an ``lcs_tree`` so the external PFV enumerator
(github.com/natemacfadden/pfvs) can analyse jaxvacua geometries.  ``pfvs`` is optional and **not on PyPI**: install it from source with
``pip install git+https://github.com/natemacfadden/pfvs``; its
dependencies come from ``pip install jaxvacua[pfvs]``.
``has_pfvs`` reports whether it is importable.  Note pfvs' ``h21`` is jaxvacua's ``h11`` (its ``h11``
is the kappa dimension = jaxvacua's ``h12``) and pfvs' ``PFV(data, K, M)`` takes
``K`` before ``M``.

PFV algebra
-----------------------------------

.. autosummary::
    :toctree: _autosummary

    flux_to_pfv
    pfv_to_flux
    pfv_to_moduli
    N_matrix
    pfv_p_vector
    pfv_conditions
    pfv_racetrack


Solution processing
-----------------------------------

.. autosummary::
    :toctree: _autosummary

    dedup_key
    classify_solution
    is_physical
    map_to_fd


pfvs interop
-----------------------------------

.. autosummary::
    :toctree: _autosummary

    has_pfvs

See :meth:`jaxvacua.lcs.lcs_tree.to_cydata` /
:meth:`jaxvacua.lcs.lcs_tree.to_cydata_kwargs` for the conversion itself.
