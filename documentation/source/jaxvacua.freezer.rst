jaxvacua.freezer
=================

.. currentmodule:: jaxvacua.freezer

.. automodule:: jaxvacua.freezer

When to use this module
-----------------------------------

Use a freezer after constructing a full flux effective theory, when one or
more complex-structure fields are treated as heavy and should be solved away
before scanning the remaining light directions.  ``Freezer`` provides the
base reduced-EFT interface; ``ConifoldFreezer`` is the specialised
implementation for integrating out a conifold modulus with the coniLCS
``z_cf`` equation of motion; ``PFVEFT`` is the perturbatively-flat-vacuum
effective theory, in which every complex-structure modulus is slaved to the
axio-dilaton along the flat direction :math:`z = p\,\tau` (LCS) or
:math:`z_{\rm bulk} = \hat p\,\tau` with :math:`z_{\rm cf}` integrated out
(coniLCS), leaving a one-dimensional theory in :math:`\tau`.

Reduced-EFT workflow
-----------------------------------

1. Wrap the full model with ``ConifoldFreezer`` and specify the conifold
   modulus through ``conifold_index``.
2. Call ``solve_heavy`` to determine the heavy modulus for fixed light
   moduli, axio-dilaton, and fluxes.
3. Use ``reconstruct_full_moduli`` when a full moduli vector is needed again,
   for example before evaluating quantities defined on the original model.
4. Evaluate reduced quantities with ``superpotential``, ``DW_light``,
   ``DW_x_light``, ``dDW_x_light``, ``V_x_light``, ``dV_x_light``, and
   ``ddV_x_light``.
5. Compute the reduced light-field mass spectrum with ``light_mass_spectrum``
   (aliased as ``bulk_mass_spectrum`` on ``ConifoldFreezer``), passing the
   stored vacuum point through ``x_full`` so the reduced Hessian is evaluated
   with the heavy modulus on-shell.

Evaluation cost: what is expensive, and why
-------------------------------------------

The heavy solve dominates everything else in a reduced-EFT evaluation, so the
cost of a call is decided almost entirely by *how many times it is performed* and
*whether it is differentiated through*.

.. list-table::
   :header-rows: 1
   :widths: 34 22 44

   * - Quantity
     - Heavy solves
     - Notes
   * - ``ddV_x_light`` (``frozen``/``schur``/``tangent``) with ``x_full``
     - **0**
     - first-order contractions only; sub-millisecond
   * - the same with ``x_full=None``
     - 1
     - reconstructs; use :meth:`Freezer.full_real_point` once instead
   * - ``G_x_light`` (``method="pullback"``, default)
     - 0 with ``x_full``
     - first-order pullback of the full metric
   * - ``ddV_x_light(reduction="autodiff")``, ``G_x_light(method="autodiff")``
     - differentiates *through* the solve
     - the assumption-free references; orders of magnitude dearer
   * - ``DW_light``
     - 2
     - one per holomorphic/antiholomorphic branch -- see below

``DW_light`` takes :math:`(z, \bar z, \tau, \bar\tau)` as **independent**
arguments so that a caller can differentiate holomorphically, which is what
:math:`D_\tau W = \partial_\tau W + (\partial_\tau K)W` requires.  It therefore
solves for the heavy moduli twice, once per branch.  For a pure *evaluation* at
:math:`\bar\tau = \overline{\tau}` the second solve is redundant and
``assume_conjugate=True`` skips it -- but that option is **invalid under
differentiation with respect to the conjugate arguments**, because it drops the
:math:`\bar\tau` dependence flowing through the conjugate solve while keeping the
direct one, yielding a silently *incomplete* derivative rather than an obviously
zero one.  The regression suite pins this behaviour.

Freezers are registered JAX pytrees
-----------------------------------

``Freezer`` and every subclass (including user-defined ones -- registration is
automatic via ``__init_subclass__``) are registered pytree nodes, so a freezer can
be passed as a traced argument to compiled code.  Arrays and the bound model are
traced children; ``str``/``bool`` configuration travels as static auxiliary data;
integer configuration must be declared in ``_pytree_static_keys``.  Consequences
worth knowing:

* the model's arrays are **runtime inputs, not compile-time constants**, so
  editing a model is never served stale from a compiled kernel, and one compiled
  graph is reusable across geometries of equal shape;
* subclasses should keep their attributes to arrays, registered pytrees and
  simple static scalars -- see the subclass contract in the :class:`Freezer`
  docstring.

Index conventions
-----------------------------------

``heavy_indices`` names the frozen complex-structure moduli and
``light_indices`` is its complement.  The counters ``n_heavy`` and
``n_light`` refer to complex moduli, while real light-field vectors used by
the ``*_x_light`` methods contain real and imaginary parts of the light
moduli plus the real and imaginary parts of ``tau``.

.. raw:: html
   :file: _static/figures/f9_freezer.html


Freezer base class
-----------------------------------

.. autosummary::
    :toctree: _autosummary
    :template: custom-class-template.rst

    Freezer


Light-field EFT interface
-----------------------------------

.. autosummary::

    Freezer.solve_heavy
    Freezer.reconstruct_full_moduli
    Freezer.superpotential
    Freezer.DW_light
    Freezer.DW_x_light
    Freezer.dDW_x_light
    Freezer.V_x_light
    Freezer.dV_x_light
    Freezer.ddV_x_light


Reduced mass spectrum
-----------------------------------

The reduced light-field masses follow from integrating out the heavy moduli
and solving the generalised eigenproblem
:math:`H_{\rm eff}\,v = \lambda\,K_{\rm eff}\,v` in the real interleaved basis,
which avoids the ill-conditioning of the full ``FluxEFT.mass_matrix`` near a
conifold.  ``ddV_x_light`` supplies the reduced Hessian :math:`H_{\rm eff}`
through four ``reduction`` schemes: ``"frozen"`` (the bare leading-order block, a
diagnostic), ``"schur"`` (the Schur complement — integrating the moduli out at
their **V-minimum**, the right reduction when a genuinely heavy modulus is
integrated out, as for ``ConifoldFreezer``), ``"autodiff"`` (differentiated
through the on-shell heavy solve) and ``"tangent"`` (the same **F-flat** reduced
Hessian as ``"autodiff"`` at a vacuum, from the first-order on-shell tangent —
fast).  The two families answer different questions: ``"schur"`` extremises over
the heavy directions, whereas ``"tangent"``/``"autodiff"`` follow the physical
slaving, so they coincide only when the slaved direction is itself a V-valley (it
is for a heavy conifold modulus; it is *not* for a perturbatively-flat direction).
``K_x_light`` and ``G_x_light`` give the substituted Kähler potential and reduced
metric :math:`K_{\rm eff}`, and ``light_mass_spectrum`` returns the spectrum
together with its stability diagnostics as a ``LightSpectrum``; its default
reduction is per-class (``"schur"`` for ``ConifoldFreezer``, ``"tangent"`` for
``PFVEFT``).

The reduced metric ``G_x_light`` defaults to ``method="pullback"``: because
F-flatness makes the heavy solve *holomorphic* in the light fields, the reduced
metric is the pullback :math:`J^{T} G_{\rm full} J` of the full Kähler metric
along the same first-order on-shell tangent that ``reduction="tangent"`` uses --
no second derivative, and no differentiation through the heavy solve.  The
original route survives as ``method="autodiff"`` (``jax.hessian`` of the
substituted Kähler potential through the solve); it makes no holomorphy
assumption and is kept as the cross-check.  The two agree to a relative
``5e-8`` on the LCS reference PFV (masses to 8 significant figures), while the
pullback is some five orders of magnitude cheaper.

.. admonition:: Evaluating efficiently -- reconstruct once
    :class: tip

    The heavy solve dominates every reduced-EFT evaluation (a Newton iteration
    for ``PFVEFT(mode="eom")``).  Call :meth:`Freezer.full_real_point` **once**
    and pass the result as ``x_full`` to :meth:`Freezer.ddV_x_light`,
    :meth:`Freezer.G_x_light` and :meth:`Freezer.light_mass_spectrum` -- or pass
    a stored certified vacuum directly.  The reduced Hessian then costs one
    ``dDW_x`` plus one ``ddV_x`` evaluation (sub-millisecond), because the
    implicit function theorem needs only the *point*, not a re-derivation of it.
    ``reduction="autodiff"`` and ``G_x_light(method="autodiff")`` are the
    exceptions: they differentiate *through* the solve, so they cannot reuse it.

    These kernels are JIT-compiled, so a first call pays a one-off XLA compile
    (tens of seconds, dominated by the Newton solve and its nested ``jacrev``)
    and later calls are sub-millisecond.  Enable JAX's persistent on-disk
    compilation cache (``stringjax_tools.configure_compilation_cache``) in a
    script or notebook preamble to pay that once rather than once per process,
    and always warm up before timing anything -- otherwise you measure the
    compiler.

.. autosummary::

    Freezer.full_real_point
    Freezer.K_x_light
    Freezer.G_x_light
    Freezer.light_mass_spectrum

.. autosummary::
    :toctree: _autosummary
    :template: custom-class-template.rst

    LightSpectrum


Conifold freezer
-----------------------------------

.. autosummary::
    :toctree: _autosummary
    :template: custom-class-template.rst

    ConifoldFreezer


Conifold EOM and spectrum
-----------------------------------

.. autosummary::

    ConifoldFreezer.solve_heavy
    ConifoldFreezer.reconstruct_full_moduli
    ConifoldFreezer.light_mass_spectrum
    ConifoldFreezer.bulk_mass_spectrum


PFV effective theory
-----------------------------------

``PFVEFT`` slaves every complex-structure modulus to the axio-dilaton along the
perturbatively-flat direction, leaving a complex one-dimensional theory in
:math:`\tau` (``n_light = 0``).  For LCS the ansatz :math:`z = p\,\tau` is linear,
so *in* ``mode="ansatz"`` the ``"frozen"`` reduced Hessian equals the
``"autodiff"`` one exactly; for coniLCS the bulk moduli are slaved and
:math:`z_{\rm cf}` is integrated out analytically, so its :math:`\tau`-dependence
is nonlinear.  For **any** PFV mass use ``mode="eom"`` with
``reduction="tangent"`` (fast and exact at a vacuum) or ``"autodiff"`` (robust
off-shell) — these follow the physical F-flat slaving and reproduce the racetrack
:math:`\tau`-mass, and ``light_mass_spectrum`` therefore defaults to ``"tangent"``
on a ``PFVEFT``.  Two traps: ``reduction="frozen"`` keeps the *constant ansatz*
tangent, so outside ``mode="ansatz"`` it is unreliable (even for LCS it can be
:math:`O(10^2)` off on the exponentially small mass), and the *default*
``mode="ansatz"`` is off-shell for coniLCS — both emit a warning.
``reduction="schur"`` is a correct Schur complement but integrates the moduli out
at their **V-minimum**, a different reduction from the flat-direction slaving, so
for a PFV it lands a few :math:`\times` off the racetrack mass (it remains the
right choice for ``ConifoldFreezer``).  Build one from PFV quantum numbers with
:meth:`PFVEFT.from_fluxes` or from a :class:`jaxvacua.vacuum.PFVData` with
:meth:`PFVEFT.from_pfv_data`.

Two reconstruction modes select how the moduli are slaved to :math:`\tau`:

* ``mode="ansatz"`` (default) — the leading-order flat direction
  (:math:`z = p\,\tau`, plus the analytic :math:`z_{\rm cf}` throat solve for
  coniLCS).
* ``mode="eom"`` — the moduli are Newton-solved from their F-terms
  :math:`\partial_x W = 0` at fixed :math:`\tau` (seeded at the ansatz),
  capturing the subleading corrections: the exponentially-small instanton
  correction to :math:`z = p\,\tau` for LCS, and for coniLCS the exact
  :math:`z_{\rm cf}` back-reaction (the conifold modulus becomes a genuine Newton
  variable rather than the leading-order throat solve).  In ``mode="eom"`` prefer
  ``reduction="tangent"`` — a single implicit-function-theorem linear solve on the
  ``dDW_x`` moduli block (the reduced Hessian's on-shell tangent), fast and exact
  at a vacuum — or ``"autodiff"`` (a full ``jax.hessian`` through the solve) when
  you need the exact off-shell Hessian.  Do **not** pair ``mode="eom"`` with
  ``reduction="frozen"``: the frozen tangent is the *ansatz* one, so it does not
  see the eom slaving.

.. autosummary::
    :toctree: _autosummary
    :template: custom-class-template.rst

    PFVEFT

.. autosummary::

    PFVEFT.from_fluxes
    PFVEFT.from_pfv_data
    PFVEFT.reconstruct_full_moduli
    PFVEFT.DW_light
