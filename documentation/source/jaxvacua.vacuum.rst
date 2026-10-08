jaxvacua.vacuum
===============

.. currentmodule:: jaxvacua.vacuum

.. automodule:: jaxvacua.vacuum

Overview
-----------------------------------

This module holds lightweight, pure-data containers for flux-vacuum quantities,
in two layers.

:class:`PFVData` is the light algebra layer of a perturbatively flat vacuum
(PFV): it bundles the flux quanta :math:`(M, K)` with the derived N-matrix,
p-vector, full flux vector and PFV-condition results, and exposes thin accessors
that delegate to the model.

:class:`Vacuum` and its subclass :class:`PFV` are the solver-state layer.  A
``Vacuum`` means *a solved point in moduli space*: it is defined by its **flux**
and its **location** — supplied either as the interleaved real vector ``x`` or as
``(z, tau)``, whichever is missing being derived — plus the solved diagnostics
:math:`W_0`, :math:`D W`, the residual and :math:`g_s`.  The conifold quantities
:math:`z_{\rm cf}` and the bulk/conifold residual split are **gated on the
geometry limit**: for a plain LCS vacuum there is no conifold direction, so they
are not merely unknown but undefined, and :attr:`Vacuum.has_conifold` reports
``False`` while the fields stay ``None``.

:class:`PFV` composes a :class:`PFVData` and exposes the analytic
2-term-racetrack *estimate* as properties (distinct from the solved fields); its
:meth:`~Vacuum.diagnostics` additionally checks that the record actually
satisfies the PFV algebra.

Promotion provenance is deliberately **not** here: a vacuum from a plain Newton
solve has no promotion history, so ``genealogy`` / ``success`` / ``trajectory``
live on the ``afvs`` subclasses (``AFV``, ``PromotedPFV``) together with
``is_solved()``.  The orchestration that *populates* these objects lives in that
same separate ``afvs`` package; this module never imports from it, and
:func:`register_vacuum_kind` is how such downstream types make themselves
loadable here.

Three notions of sameness are available, and the choice matters:
:meth:`Vacuum.equals` is exact and field-by-field (its purpose is to prove that a
round trip lost nothing); :meth:`Vacuum.equivalent_to` without a model compares
the *points*, rounded, which is what solver output needs; with a model it also
identifies :math:`SL(2, \mathbb{Z}) \times` monodromy images, as
:func:`unique_vacua` does in bulk.

PFV data container
-----------------------------------

.. autosummary::
    :toctree: _autosummary
    :template: custom-class-template.rst

    PFVData

Vacuum hierarchy
-----------------------------------

.. autosummary::
    :toctree: _autosummary
    :template: custom-class-template.rst

    Vacuum
    PFV
    VacuumAnalysis

Consistency checks and derived quantities
-----------------------------------------

:meth:`Vacuum.diagnostics` returns a named, explained report
``{check: (ok, value, reason)}`` and caches it — together with any eigenvalues
computed on the way — in :class:`VacuumAnalysis`.  :meth:`Vacuum.is_consistent`
is its reduction, so the verdict and the explanation can never disagree.  A check
whose inputs are unavailable reports ``"skipped: …"`` and is *excluded* from the
verdict, so a missing input can never silently pass.

.. autosummary::
    :toctree: _autosummary

    conifold_alignment
    resolve_hyperplanes

Coordinates
-----------------------------------

The interleaved convention ``x = [Re z1, Im z1, …, Re tau, Im tau]`` stated once,
model-free (the jitted, model-bound equivalent is
``FluxEFT._convert_real_to_complex``).

.. autosummary::
    :toctree: _autosummary

    real_to_complex
    complex_to_real

Serialization and deduplication
-----------------------------------

Two storage tiers.  :func:`save_vacua` gzips *pickled dicts* — convenient
locally, and safe because the payload is plain dicts of arrays rather than class
instances, so a stored file survives a class refactor.  For anything others
download, prefer :func:`vacuum_to_json`: unpickling is a remote-code-execution
vector, whereas the tagged-JSON payload is inert, human-inspectable and diffable.
Either tier can drop the whole derived block with ``analysis=False``.

.. autosummary::
    :toctree: _autosummary

    register_vacuum_kind
    unique_vacua
    dedup_vacua
    save_vacua
    load_vacua
    vacuum_to_json
    vacuum_from_json
    encode_json
    decode_json
