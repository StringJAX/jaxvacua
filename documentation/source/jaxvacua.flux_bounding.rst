jaxvacua.flux_bounding
========================

.. currentmodule:: jaxvacua.flux_bounding

.. automodule:: jaxvacua.flux_bounding

Pipeline overview
-----------------

Systematic flux enumeration is the heaviest pipeline in the package.
Given an upstream :class:`jaxvacua.flux_eft.FluxEFT` instance and a
flux radius :math:`N_{\max}`, ``bounded_fluxes`` enumerates every
integer flux

.. math::
   :label: eq:jaxvacua-flux-bounding-01

   G_3 = F_3 - \tau H_3, \qquad
   (F_3, H_3) \in \mathbb{Z}^{2(h^{1,2}+1)},

inside a bounding box subject to physical constraints — the s-bound
:math:`s_{\max}`, the D3-tadpole budget :math:`N_{\text{flux}} \leq Q_{\text{O3}}`,
and modular reduction along the
:math:`\mathrm{SL}(2,\mathbb{Z})` orbit. Phases 1–4 are filters and
lattice operations; phase 5 produces the actual vacua via per-batch
ISD completion and multi-start Newton refinement.

In the diagram, the inherited :class:`jaxvacua.flux_eft.FluxEFT` instance is shown
light grey, the user-supplied :math:`N_{\max}` and the moduli
samples for the eigenvalue bounds are shown dashed (external), and
the deduplicated output catalogue is highlighted in orange.

.. raw:: html
   :file: _static/figures/f7_flux_bounding.html


Cluster-scale execution
-----------------------

Large flux-bounding runs can be split into independent chunk files and
processed on a cluster.  ``export_cluster_job`` prepares the search once,
serialises the pipeline state to ``pipeline.npz`` and ``config.json``, writes
pre-filtered ``h``-flux chunks under ``chunks/``, and creates a worker script.
With ``generate_slurm=True`` it also writes a SLURM array submission script.

Each worker calls ``process_chunk_from_disk`` for one chunk.  The worker
reconstructs the saved pipeline, ISD-completes and filters that chunk, and
writes a compressed result file under ``results/``.  Since chunks are
independent, failed or slow array tasks can be rerun without repeating the
export step.

After the array finishes, ``merge_cluster_results`` reads the result files,
reports missing chunk IDs, deduplicates flux vectors, and optionally maps
solutions to the fundamental domain before comparison.  If a model is supplied
it can Newton-refine the merged candidates; if a database handle is supplied
it can write the merged batch through the database integration hooks.

.. raw:: html
   :file: _static/figures/f11_cluster_parallel.html


Bounded fluxes class
-----------------------------------

.. autosummary::
    :toctree: _autosummary
    :template: custom-class-template.rst

    bounded_fluxes


Bounding box computation
-----------------------------------

.. autosummary::

    bounded_fluxes.compute_bounding_box
    bounded_fluxes.get_h_box
    bounded_fluxes.compute_evs
    bounded_fluxes.compute_evs_vmap
    bounded_fluxes.update_global
    bounded_fluxes.update_evs
    bounded_fluxes.update_local


Flux enumeration
-----------------------------------

.. autosummary::

    bounded_fluxes.enumerate_fluxes
    bounded_fluxes.sample_bounded_fluxes
    bounded_fluxes.get_h_candidates


Cluster execution API
-----------------------------------

.. autosummary::

    bounded_fluxes.export_cluster_job
    bounded_fluxes.process_chunk_from_disk
    bounded_fluxes.merge_cluster_results


Batch flux checking
-----------------------------------

.. autosummary::

    bounded_fluxes.check_bounds_batch
    bounded_fluxes.compute_tadpole_batch
    bounded_fluxes.newton_refine_batch
    bounded_fluxes.in_patch_batch


Flux bounds
-----------------------------------

.. autosummary::

    bounded_fluxes.bound_h1_local
    bounded_fluxes.bound_h1_global
    bounded_fluxes.bound_h2_local
    bounded_fluxes.bound_h2_global
    bounded_fluxes.bound_f1_local
    bounded_fluxes.bound_f1_global
    bounded_fluxes.bound_f2_local
    bounded_fluxes.bound_f2_global
    bounded_fluxes.bound_s_local
    bounded_fluxes.bound_s_global
    bounded_fluxes.bound_h_local
    bounded_fluxes.bound_h_global
    bounded_fluxes.bound_f_local
    bounded_fluxes.bound_f_global


Flux utilities
-----------------------------------

.. autosummary::

    bounded_fluxes.get_nflux
    bounded_fluxes.get_fh
    bounded_fluxes.get_flux_split
    bounded_fluxes.get_subvector
    bounded_fluxes.compute_norm
    bounded_fluxes.check_bounds
    bounded_fluxes.check_bounds_flat
