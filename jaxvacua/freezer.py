# Copyright 2022-2026 Andreas Schachner
#
# This file is part of JAXVacua.
#
# JAXVacua is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# JAXVacua is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with JAXVacua. If not, see <https://www.gnu.org/licenses/>.

r"""Reduced EFT interfaces for integrating out heavy moduli.

Purpose
-------
Provide abstractions for solving heavy-field equations of motion and
evaluating a reduced flux EFT on the remaining light fields.

Main public API
---------------
- ``Freezer``: abstract base class defining the reduced-EFT interface,
  including heavy/light index bookkeeping, reconstruction and light-field
  derivatives.
- ``ConifoldFreezer``: concrete implementation for freezing the conifold
  modulus ``z_cf`` in coniLCS models.
- ``Freezer.light_mass_spectrum`` / ``LightSpectrum``: the reduced light-field
  mass spectrum, obtained by integrating out the heavy moduli and solving the
  generalised eigenproblem :math:`H_{\rm eff}\,v = \lambda\,K_{\rm eff}\,v` in
  the real interleaved basis -- which avoids the ill-conditioning of the full
  ``FluxEFT.mass_matrix`` near a conifold.

Design notes
------------
Freezers wrap an existing flux model.  They do not own the underlying
geometry; instead they solve heavy fields as functions of light moduli,
axio-dilaton and fluxes, then reuse the model's superpotential and derivative
methods on the reconstructed full field point.

The reduced light-field theory exposes four Hessian-reduction schemes (the
``reduction`` argument of :meth:`Freezer.ddV_x_light` and
:meth:`Freezer.light_mass_spectrum`): ``"frozen"`` (the bare leading-order block,
no back-reaction -- a diagnostic), ``"schur"`` (the Schur complement of the heavy
block -- integrating the moduli out at their V-minimum, the right reduction for a
genuinely heavy modulus), ``"autodiff"`` (the Hessian of the scalar potential
differentiated through the on-shell heavy solve) and ``"tangent"`` (the same
F-flat reduced Hessian as ``"autodiff"`` at a vacuum, from the first-order
on-shell tangent -- fast).  For a :class:`PFVEFT` the moduli are slaved along the
flat direction, not held at a V-minimum, so ``"tangent"``/``"autodiff"`` give the
racetrack mass while ``"schur"`` gives a different V-minimum value.  Masses are
formed in the real interleaved basis from a generalised eigenproblem rather than
through ``mass_matrix``.  The mass-spectrum entry points are eager host-side
helpers (NumPy/SciPy at the eigensolve), i.e. not ``jit``/``vmap``-able; batch
over vacua with a Python loop.
"""

import warnings
from functools import partial
from dataclasses import dataclass
from typing import Tuple, Any, Dict, Optional
from abc import ABC, abstractmethod

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import solve_triangular
from jax.tree_util import register_pytree_node
from scipy.linalg import eigh as _geigh
from stringjax_tools import PytreePolicy

__all__ = ["Freezer", "ConifoldFreezer", "PFVEFT", "LightSpectrum"]


# ----------------------------------------------------------------------------
# Pytree registration
# ----------------------------------------------------------------------------
# Freezers are registered as JAX pytrees, exactly as ``FluxEFT`` and friends are
# (see ``jaxvacua.util._PYTREE_POLICY``).  This is what allows ``self`` to be a
# *traced argument* of the jitted reduced-EFT kernels below rather than a Python
# closure variable: a closed-over model has its arrays baked into the compiled
# HLO as constants, so a later model edit would silently keep serving results
# computed from the pre-edit data, and every geometry would compile its own
# executable (defeating JAX's persistent compilation cache).
#
# Integer configuration must be declared static explicitly -- ``str``/``bool``
# are covered by ``static_types`` -- because the kernels branch on it or use it
# as a Python trip count / tuple index (``_eom_iters`` in ``fori_loop``,
# ``_conifold_index`` in ``heavy_indices``).
_FREEZER_STATIC_KEYS: Tuple[str, ...] = (
    "_conifold_index",     # ConifoldFreezer: indexes heavy_indices
    "_eom_iters",          # PFVEFT: fori_loop trip count
)

#: Classes already handed to :func:`jax.tree_util.register_pytree_node`.  JAX
#: rejects a duplicate registration, and ``__init_subclass__`` can fire more than
#: once for the same name (module reload, a class defined inside a function).
_REGISTERED_FREEZER_TYPES: set = set()


def _register_freezer_pytree(cls: type) -> None:
    r"""
    **Description:**
    Register ``cls`` with the JAX pytree registry, so that ``self`` can be a
    traced argument of the compiled reduced-EFT kernels.

    .. admonition:: Details
        :class: dropdown

        Pytree registration is keyed on the *exact* type, so every concrete
        ``Freezer`` subclass needs its own entry -- including user-defined ones,
        since subclassing ``Freezer`` is a documented extension point.
        :meth:`Freezer.__init_subclass__` calls this automatically, so an external
        subclass needs no boilerplate.

        The policy is built per class so that a subclass can extend it through
        :attr:`Freezer._pytree_static_keys` (an ``int`` attribute used as a trip
        count or index **must** be static; ``str``/``bool`` are covered
        automatically by ``static_types``).

    Args:
        cls (type): The ``Freezer`` subclass to register.

    Returns:
        None
    """
    if cls in _REGISTERED_FREEZER_TYPES:
        return
    static_keys = tuple(_FREEZER_STATIC_KEYS) + tuple(
        getattr(cls, "_pytree_static_keys", ()))
    policy = PytreePolicy(
        static_keys=static_keys,
        static_types=(str, bool),
        validate_static=True,
    )
    flatten, unflatten = policy.make_flatteners(cls)
    try:
        register_pytree_node(cls, flatten, unflatten)
    except ValueError as exc:              # pragma: no cover - reload only
        # The only expected cause is a duplicate registration surviving a module
        # reload (our own set is reset by the reload, JAX's registry is not).
        # Anything else is a real policy error and must not be swallowed.
        if "duplicate" not in str(exc).lower():
            raise
    _REGISTERED_FREEZER_TYPES.add(cls)


#: Names of keyword arguments that must be treated as JAX-static by the jitted
#: reduced-EFT kernels: the reduction scheme, the potential flavour, and the
#: heavy-solve configuration forwarded through ``**kwargs``.  A *string* or
#: *bool* keyword not listed here cannot be passed to those kernels (``jax.jit``
#: raises), which is deliberate -- it surfaces a typo instead of silently
#: ignoring the setting.
_STATIC_KERNEL_KWARGS: Tuple[str, ...] = (
    "noscale", "reduction", "return_heavy_block",
    "mode", "apply_correction", "conj",
)


# ----------------------------------------------------------------------------
# Implicitly-differentiated eom reconstruction
# ----------------------------------------------------------------------------

@jax.custom_jvp
def _eom_reconstruct(freezer: Any, x_light: Array, fluxes: Array) -> Array:
    r"""
    **Description:**
    ``PFVEFT(mode="eom")`` reconstruction :math:`x_{\rm light} \mapsto x_{\rm
    full}`, carrying an **implicit-function derivative rule** instead of being
    differentiated through.

    .. admonition:: Why a custom rule
        :class: dropdown

        The primal is a Newton iteration on the moduli F-terms
        :math:`F \equiv \partial_x W|_{\rm moduli} = 0` at fixed :math:`\tau`.
        Differentiating *through* that iteration (what ``jax.hessian`` did) tapes
        every step, so ``reduction="autodiff"`` and the ``"autodiff"`` reduced
        metric cost minutes to compile.  At the root the implicit function theorem
        gives the same derivative from **one linear solve** on the moduli block the
        Newton step already builds:

        .. math::
            \frac{\partial F}{\partial x_{\rm mod}}\,\mathrm{d}x_{\rm mod}
            + \mathrm{d}F\big|_{x_{\rm mod}\ \rm fixed} = 0 .

        The external differential :math:`\mathrm{d}F|_{x_{\rm mod}}` is obtained by
        a single :func:`jax.jvp` of the residual with respect to **every**
        differentiable input -- ``x_light``, ``fluxes`` *and* the freezer's own
        arrays.  Deriving only the :math:`\tau` term (all the reduced Hessian
        needs) would leave flux derivatives silently wrong, so the rule is written
        input-agnostically on purpose.

        Real in, real out: the complex :math:`\tau` lives strictly inside the
        primal, which keeps complex tangent conventions out of the rule.

    Args:
        freezer (Any): The :class:`PFVEFT` (a registered pytree, so its arrays are
            differentiable inputs).
        x_light (Array): Real light coordinates ``[Re tau, Im tau]``.
        fluxes (Array): Full flux vector.

    Returns:
        Array: Full real coordinate vector with the moduli on-shell.
    """
    tau = x_light[0] + 1j * x_light[1]
    return freezer._solve_eom(tau, fluxes)


@_eom_reconstruct.defjvp
def _eom_reconstruct_jvp(primals: Tuple[Any, ...],
                         tangents: Tuple[Any, ...]) -> Tuple[Array, Array]:
    r"""
    **Description:**
    Implicit-function JVP for :func:`_eom_reconstruct` (see its ``Details``).

    Args:
        primals (Tuple): ``(freezer, x_light, fluxes)``.
        tangents (Tuple): Matching tangents.

    Returns:
        Tuple[Array, Array]: ``(x_full, dx_full)``.
    """
    freezer, x_light, fluxes = primals
    d_freezer, d_x_light, d_fluxes = tangents
    x_full = _eom_reconstruct(freezer, x_light, fluxes)
    n_z = 2 * int(freezer.model.h12)
    x_mod = x_full[:n_z]                       # held FIXED: this is the IFT

    def _residual(fr: Any, xl: Array, fx: Array) -> Array:
        # The axio-dilaton block of x_full IS x_light (n_light = 0 for PFVEFT).
        return fr.model.DW_x(jnp.concatenate([x_mod, xl]), fx)[:n_z]

    _, dF_ext = jax.jvp(_residual, (freezer, x_light, fluxes),
                        (d_freezer, d_x_light, d_fluxes))
    A = freezer.model.dDW_x(x_full, fluxes)[:n_z, :n_z]
    dx_mod = -jnp.linalg.solve(A, dF_ext)
    return x_full, jnp.concatenate([dx_mod, d_x_light])


# ----------------------------------------------------------------------------
# Real <-> complex Kähler-metric helpers (interleaved layout)
# ----------------------------------------------------------------------------

def _G_from_real_hessian(H_K: Array) -> Array:
    r"""
    **Description:**
    Extract the complex Hermitian Kähler metric :math:`G_{A\bar B}` from the
    real symmetric Hessian of a real Kähler potential in the interleaved
    ``(Re_0, Im_0, Re_1, Im_1, ...)`` layout (modulus :math:`z = a + \mathrm{i} b`):

    .. math::
        \mathrm{Re}\,G_{AB} = \tfrac14\bigl(H[2A,2B] + H[2A+1,2B+1]\bigr),\quad
        \mathrm{Im}\,G_{AB} = \tfrac14\bigl(H[2A,2B+1] - H[2A+1,2B]\bigr).

    The holomorphic-holomorphic :math:`K_{AB}` pieces cancel, so the result is
    Hermitian by construction whenever ``H_K`` is symmetric.

    Args:
        H_K (Array): Real symmetric Hessian of a real scalar, of shape
            ``(2N, 2N)`` in the interleaved layout.

    Returns:
        Array: Complex Hermitian metric of shape ``(N, N)``.
    """
    ReG = 0.25 * (H_K[0::2, 0::2] + H_K[1::2, 1::2])
    ImG = 0.25 * (H_K[0::2, 1::2] - H_K[1::2, 0::2])
    G = ReG + 1j * ImG
    return 0.5 * (G + jnp.conj(G.T))


def _kahler_metric_real_interleaved(G: Array) -> Array:
    r"""
    **Description:**
    Rebuild the real ``(2N, 2N)`` Kähler metric in the interleaved basis
    ``(Re_0, Im_0, ..., Re_{N-1}, Im_{N-1})`` from a complex Hermitian metric
    :math:`G_{A\bar B}`.

    For :math:`\phi = (a + \mathrm{i} b)/\sqrt2` the kinetic term carries the 2x2
    block :math:`\left[\begin{smallmatrix}\mathrm{Re}\,G & -\mathrm{Im}\,G\\
    \mathrm{Im}\,G & \mathrm{Re}\,G\end{smallmatrix}\right]` for each
    :math:`(A, B)` pair.

    Args:
        G (Array): Complex Hermitian metric of shape ``(N, N)``.

    Returns:
        Array: Real metric of shape ``(2N, 2N)``.
    """
    ReG = jnp.real(G)
    ImG = jnp.imag(G)
    n = G.shape[0]
    Kr = jnp.zeros((2 * n, 2 * n), dtype=ReG.dtype)
    Kr = Kr.at[0::2, 0::2].set(ReG)
    Kr = Kr.at[1::2, 1::2].set(ReG)
    Kr = Kr.at[0::2, 1::2].set(-ImG)
    Kr = Kr.at[1::2, 0::2].set(ImG)
    return Kr


@dataclass
class LightSpectrum:
    r"""
    **Description:**
    Container for the light (reduced) mass spectrum returned by
    :meth:`Freezer.light_mass_spectrum`.

    Attributes:
        masses (np.ndarray): Signed light masses
            :math:`\mathrm{sign}(m^2)\sqrt{|m^2|}`, sorted ascending.
        eigenvalues (np.ndarray): Raw generalised eigenvalues
            :math:`\lambda = 2 m^2` of
            :math:`H_{\rm eff}\,v = \lambda\,K_{\rm eff}\,v`.
        m2_min (float): Smallest light mass squared, :math:`0.5\,\min\lambda`.
        stable (bool): ``True`` if ``m2_min >= -rel_tol * max|m^2|`` (flat
            directions allowed, tachyons rejected).  Note the tolerance is
            scaled by the *largest* mass: a tachyon smaller than
            ``rel_tol * max|m^2|`` is reported stable, so consult ``m2_min`` and
            ``info["m2_dynamic_range"]`` directly for a strongly hierarchical
            spectrum.
        dw_residual (float): On-shell screen value ``max|DW_x|`` of the *full*
            F-terms (heavy component included) at the evaluation point.  When
            ``x_full`` is supplied this is the stored-vacuum residual; otherwise
            it is the analytic-reconstructed point's residual (the heavy solve's
            accuracy).
        reduction (str): Reduction scheme used
            (``"frozen"`` / ``"schur"`` / ``"autodiff"`` / ``"tangent"``).
        info (Dict[str, Any]): Auxiliary diagnostics -- the reduced-metric
            condition number ``cond_Keff``, the reduced-spectrum dynamic range
            ``m2_dynamic_range`` (``max|m^2| / min|m^2|``, flat directions
            excluded; large values mean the lightest masses are precision-limited
            relative to the heaviest), the mode count ``n_modes``, and -- on an
            early return -- a ``reason`` string.
    """
    masses: np.ndarray
    eigenvalues: np.ndarray
    m2_min: float
    stable: bool
    dw_residual: float
    reduction: str
    info: Dict[str, Any]


class Freezer(ABC):
    r"""
    **Description:**
    Abstract base class for a reduced effective field theory obtained by
    integrating out a set of heavy moduli.

    Given a full model with moduli :math:`(z_{\text{heavy}}, z_{\text{light}}, \tau)`
    and fluxes, the reduced EFT expresses the heavy moduli as functions of the
    light fields via their leading-order EOM:

    .. math::
        z_{\text{heavy}} = z_{\text{heavy}}(z_{\text{light}}, \tau, \text{fluxes})

    and provides the superpotential, its derivatives, etc. evaluated on this
    solution.

    Subclasses must implement:
        - ``heavy_indices``: which moduli are frozen out
        - ``solve_heavy``: solve for heavy moduli given light fields
        - ``_real_light_to_full``: convert real light-field coordinates to full array

    .. admonition:: Subclass contract -- freezers are JAX pytrees
        :class: dropdown

        Every subclass is registered as a pytree automatically
        (:meth:`__init_subclass__`), because the compiled reduced-EFT kernels take
        ``self`` as a *traced* argument.  Instance attributes are therefore
        classified when the freezer crosses a ``jit`` boundary:

        - ``str`` / ``bool`` values travel as static auxiliary data;
        - arrays and registered pytrees (e.g. the bound ``model``) become traced
          children;
        - **anything else -- notably ``int`` -- becomes a traced child**, which
          breaks it if the value is used as a Python trip count, a tuple index, or
          a branch condition.  List such attributes in
          :attr:`_pytree_static_keys`.

        Keep attributes to arrays, registered pytrees, and simple static scalars.
        Mutable scratch state (caches, open files) does not belong on a freezer
        that is passed to compiled code.
    """

    #: Extra instance-attribute names a subclass needs treated as JAX-static (see
    #: the subclass contract above).  ``str``/``bool`` attributes are already
    #: static; this is for ``int`` (or other hashable) configuration that the
    #: kernels use as a trip count, index, or branch condition.  Merged with
    #: :data:`_FREEZER_STATIC_KEYS` at registration.
    _pytree_static_keys: Tuple[str, ...] = ()

    #: Default reduction for :meth:`light_mass_spectrum` when ``reduction=None``.
    #: The base/``ConifoldFreezer`` default is ``"schur"`` (the V-minimum Schur
    #: complement — the correct reduction when a genuinely heavy modulus is
    #: integrated out at its potential minimum).  :class:`PFVEFT` overrides this
    #: to the F-flat ``"tangent"`` reduction, because a perturbatively-flat
    #: vacuum slaves its moduli along the flat direction (:math:`\partial_z W=0`),
    #: which is *not* a V-minimum, so ``"schur"`` would return a different mass.
    _default_light_reduction: str = "schur"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        r"""
        **Description:**
        Register every ``Freezer`` subclass as a JAX pytree.

        The jitted reduced-EFT kernels take ``self`` as a *traced* argument, which
        requires the concrete class to be in the pytree registry; registration is
        keyed on the exact type, so subclasses (including user-defined ones --
        subclassing ``Freezer`` is a documented extension point) must each be
        registered.  Doing it here means an external subclass needs no
        boilerplate.

        Args:
            **kwargs: Forwarded to ``super().__init_subclass__``.

        Returns:
            None
        """
        super().__init_subclass__(**kwargs)
        _register_freezer_pytree(cls)

    def __init__(self, model: Any) -> None:
        r"""
        **Description:**
        Initialise the Freezer base class.

        Args:
            model: A flux EFT model object providing ``superpotential``,
                ``DW``, ``DW_x``, ``dDW_x``, ``_convert_real_to_complex``,
                and period data via ``lcs_tree``.
        """
        self.model = model

    @property
    def lcs_tree(self) -> Any:
        r"""
        Description:
        The bound model's period tree, :attr:`model.lcs_tree`.

        A delegating property rather than a stored attribute on purpose.  Storing
        it would (i) duplicate the entire period/GV payload as a second set of
        traced children in every compiled kernel (freezers are registered
        pytrees, see :func:`_register_freezer_pytree`), and (ii) freeze a snapshot
        of a *mutable* object, so an in-place ``lcs_tree`` edit would leave the
        freezer disagreeing with its own model.

        Returns:
            Any: The model's ``lcs_tree``.
        """
        return self.model.lcs_tree

    @property
    @abstractmethod
    def heavy_indices(self) -> Tuple[int, ...]:
        r"""
        Description:
        Indices of the heavy moduli within the full moduli array.

        Returns:
            tuple[int, ...]: The heavy-modulus indices.
        """
        ...

    @property
    def light_indices(self) -> Tuple[int, ...]:
        r"""
        Description:
        Indices of the light moduli (complement of ``heavy_indices``).

        Returns:
            tuple[int, ...]: The light-modulus indices.
        """
        all_idx = set(range(self.model.h12))
        return tuple(sorted(all_idx - set(self.heavy_indices)))

    @property
    def n_heavy(self) -> int:
        r"""
        Description:
        Number of heavy moduli.

        Returns:
            int: The number of heavy moduli.
        """
        return len(self.heavy_indices)

    @property
    def n_light(self) -> int:
        r"""
        Description:
        Number of light moduli.

        Returns:
            int: The number of light moduli.
        """
        return len(self.light_indices)

    @abstractmethod
    def solve_heavy(
        self,
        z_light: Array,
        tau: complex,
        fluxes: Array,
        **kwargs,
    ) -> Array:
        r"""
        **Description:**
        Solve the leading-order EOM for the heavy moduli as functions of
        the light moduli, axio-dilaton, and fluxes.

        Args:
            z_light (Array): Values of the light complex structure moduli.
            tau (complex): Axio-dilaton value.
            fluxes (Array): Full flux vector.

        Returns:
            Array: Values of the heavy moduli.
        """
        ...

    def full_real_point(self, x_light: Array, fluxes: Array, **kwargs) -> Array:
        r"""
        **Description:**
        Full real coordinate vector with the heavy moduli on-shell -- the value
        accepted by the ``x_full`` argument of :meth:`ddV_x_light` and
        :meth:`light_mass_spectrum`.

        .. admonition:: Why you want this
            :class: dropdown

            The heavy solve is by far the most expensive part of a reduced-EFT
            evaluation (for ``PFVEFT(mode="eom")`` it is a Newton iteration).
            Reconstruct the point **once** and pass it as ``x_full`` to evaluate
            several reductions -- or a whole mass spectrum -- at the same vacuum
            without repeating the solve.  When you already have a certified vacuum
            (e.g. from a root find, or a stored :class:`jaxvacua.vacuum.Vacuum`),
            pass that instead of reconstructing at all.

        Args:
            x_light (Array): Real coordinates for light moduli and axio-dilaton.
            fluxes (Array): Full flux vector.
            **kwargs: Forwarded to the heavy solve.

        Returns:
            Array: Full real coordinate vector of length ``2*(h12+1)``.
        """
        return self._real_light_to_full(x_light, fluxes, **kwargs)

    def reconstruct_full_moduli(
        self,
        z_light: Array,
        tau: complex,
        fluxes: Array,
        **kwargs,
    ) -> Array:
        r"""
        **Description:**
        Reconstruct the full moduli array by solving for the heavy moduli
        and inserting them at the correct positions.

        Args:
            z_light (Array): Light moduli values.
            tau (complex): Axio-dilaton value.
            fluxes (Array): Full flux vector.

        Returns:
            Array: Full moduli array of length ``h12``.
        """
        z_heavy = self.solve_heavy(z_light, tau, fluxes, **kwargs)
        z_full = jnp.zeros(self.model.h12, dtype=complex)
        z_full = z_full.at[jnp.array(self.light_indices)].set(z_light)
        z_full = z_full.at[jnp.array(self.heavy_indices)].set(z_heavy)
        return z_full

    def superpotential(
        self,
        z_light: Array,
        tau: complex,
        fluxes: Array,
        **kwargs,
    ) -> complex:
        r"""
        **Description:**
        Superpotential of the reduced theory.

        Args:
            z_light (Array): Light moduli values.
            tau (complex): Axio-dilaton.
            fluxes (Array): Full flux vector.

        Returns:
            complex: :math:`W(z_{\text{light}}, \tau)` with heavy moduli on-shell.
        """
        z_full = self.reconstruct_full_moduli(z_light, tau, fluxes, **kwargs)
        return self.model.superpotential(z_full, tau, fluxes)

    def DW_light(
        self,
        z_light: Array,
        z_light_c: Array,
        tau: complex,
        tau_c: complex,
        fluxes: Array,
        assume_conjugate: bool = False,
        **kwargs,
    ) -> Array:
        r"""
        **Description:**
        Covariant derivatives :math:`D_i W` with respect to the light moduli,
        with heavy moduli on-shell.

        .. admonition:: Two heavy solves, and why that is the default
            :class: dropdown

            The holomorphic and antiholomorphic arguments are deliberately
            **independent**: the heavy moduli are solved once from
            :math:`(z_{\rm light}, \tau)` and once from
            :math:`(\bar z_{\rm light}, \bar\tau)`.  That is the whole purpose of
            the four-argument signature -- it lets a caller differentiate
            holomorphically, :math:`\partial_\tau` at fixed :math:`\bar\tau`, which
            is what a Kähler-covariant derivative
            :math:`D_\tau W = \partial_\tau W + (\partial_\tau K)W` requires.

            When you merely *evaluate* at :math:`\bar\tau = \overline{\tau}`, the
            second solve is redundant and ``assume_conjugate=True`` reuses
            :math:`\overline{z_{\rm full}}` instead, halving the cost.

        .. warning::
            Treat ``assume_conjugate=True`` as **evaluation only**.

            :math:`\bar\tau` reaches the result by two routes: directly, as the
            ``tau_c`` argument of ``model.DW``, and indirectly, through the
            conjugate heavy solve :math:`\overline{z_{\rm full}}(\bar z_{\rm
            light}, \bar\tau)`.  Reusing :math:`\overline{z_{\rm full}}` keeps the
            first and **re-parents the second** onto
            :math:`(z_{\rm light}, \tau)`, so derivative correctness with respect
            to the conjugate arguments is not guaranteed in general.  Since
            ``jnp.conj`` is itself differentiable, JAX will propagate through it
            and return *something* rather than raising, so an error of this kind
            would be silent.

            Measured on the LCS reference PFV (``n_light = 0``, real
            :math:`\vec p`, linear reconstruction) the ``d/d tau_c`` derivatives of
            the two routes agree **exactly** -- there, the :math:`\bar\tau`
            dependence of :math:`D_\tau W` happens to flow entirely through the
            direct argument.  That is a property of this case, not a general
            guarantee: a reconstruction whose conjugate branch genuinely depends on
            :math:`\bar\tau` (e.g. a :math:`z_{\rm cf}` throat solve) has not been
            checked.  Leave it ``False`` inside a ``grad``/``jacfwd``/``jacrev``
            over the conjugate arguments unless you have verified your own case.

        Args:
            z_light (Array): Light moduli values.
            z_light_c (Array): Conjugate light moduli values.
            tau (complex): Axio-dilaton.
            tau_c (complex): Conjugate axio-dilaton.
            fluxes (Array): Full flux vector.
            assume_conjugate (bool, optional): Reuse
                :math:`\overline{z_{\rm full}}` instead of performing the second
                heavy solve.  Valid only when ``z_light_c``/``tau_c`` really are the
                conjugates **and** no derivative is taken with respect to them --
                see the warning. Defaults to ``False`` (two independent solves).

        Returns:
            Array: :math:`D_i W` for the light moduli and :math:`D_\tau W`.
        """
        z_full = self.reconstruct_full_moduli(z_light, tau, fluxes, **kwargs)
        z_full_c = (jnp.conj(z_full) if assume_conjugate else
                    self.reconstruct_full_moduli(
                        z_light_c, tau_c, fluxes, **kwargs))
        DW_full = self.model.DW(z_full, z_full_c, tau, tau_c, fluxes)
        # Extract DW for light moduli + tau
        light_idx = jnp.array(self.light_indices)
        DW_z_light = DW_full[light_idx]
        DW_tau = DW_full[-1]
        return jnp.append(DW_z_light, DW_tau)

    def DW_x_light(
        self,
        x_light: Array,
        fluxes: Array,
        **kwargs,
    ) -> Array:
        r"""
        **Description:**
        Gradient of the superpotential :math:`\partial_{x^a} W` in
        real coordinates for the light moduli, with heavy moduli on-shell.

        This is the analogue of ``model.DW_x`` but restricted to the light
        degrees of freedom.

        Args:
            x_light (Array): Real variables for light moduli and axio-dilaton.
            fluxes (Array): Full flux vector.

        Returns:
            Array: Real gradient restricted to light directions.
        """
        x_full = self._real_light_to_full(x_light, fluxes, **kwargs)
        DW_x_full = self.model.DW_x(x_full, fluxes)
        return self._real_light_jacobian.T @ DW_x_full

    def dDW_x_light(
        self,
        x_light: Array,
        fluxes: Array,
        **kwargs,
    ) -> Array:
        r"""
        **Description:**
        Hessian :math:`\partial_{x^a}\partial_{x^b} W` in real coordinates
        for the light moduli.

        Args:
            x_light (Array): Real variables for light moduli and axio-dilaton.
            fluxes (Array): Full flux vector.

        Returns:
            Array: Hessian restricted to light directions.
        """
        x_full = self._real_light_to_full(x_light, fluxes, **kwargs)
        dDW_x_full = self.model.dDW_x(x_full, fluxes)
        J = self._real_light_jacobian
        return J.T @ dDW_x_full @ J

    def V_x_light(
        self,
        x_light: Array,
        fluxes: Array,
        noscale: bool = True,
        **kwargs,
    ) -> float:
        r"""
        **Description:**
        Scalar potential :math:`V` evaluated at the light-field coordinates,
        with heavy moduli on-shell.

        Args:
            x_light (Array): Real coordinates for light moduli and axio-dilaton.
            fluxes (Array): Full flux vector.
            noscale (bool, optional): If ``True``, uses the no-scale scalar
                potential :math:`V = e^K K^{I\bar J} D_I W D_{\bar J}\bar W`.
                Defaults to ``True``.

        Returns:
            float: Value of :math:`V` with heavy moduli at their on-shell values.
        """
        x_full = self._real_light_to_full(x_light, fluxes, **kwargs)
        return self.model.V_x(x_full, fluxes, noscale=noscale)

    def dV_x_light(
        self,
        x_light: Array,
        fluxes: Array,
        noscale: bool = True,
        **kwargs,
    ) -> Array:
        r"""
        **Description:**
        Gradient of the scalar potential :math:`\nabla_\phi V` with respect to
        the real light-field coordinates, with heavy moduli on-shell.

        .. admonition:: Details
            :class: dropdown

            Let :math:`\phi^\alpha = (a^1, v^1, \ldots, a^{n_{\rm light}},
            v^{n_{\rm light}}, c_0, s)` denote the real light-field coordinates,
            where :math:`z^i = a^i + \mathrm{i}\,v^i` and
            :math:`\tau = c_0 + \mathrm{i}\,s`. This function returns the
            restriction

            .. math::
                \nabla_\phi V \big|_{\phi^\alpha}
                = \partial_{\phi^\alpha} V(x_{\rm full}(\phi))

            where :math:`x_{\rm full}(\phi)` substitutes the on-shell heavy
            moduli via :meth:`_real_light_to_full`.

        Args:
            x_light (Array): Real coordinates for light moduli and axio-dilaton.
            fluxes (Array): Full flux vector.
            noscale (bool, optional): If ``True``, uses the no-scale scalar
                potential. Defaults to ``True``.

        Returns:
            Array: Gradient :math:`\partial_{\phi^\alpha} V`, restricted to
            light directions, of shape ``(2 * n_light + 2,)``.
        """
        x_full = self._real_light_to_full(x_light, fluxes, **kwargs)
        dV_full = self.model.dV_x(x_full, fluxes, noscale=noscale)
        return self._real_light_jacobian.T @ dV_full

    def ddV_x_light(
        self,
        x_light: Array,
        fluxes: Array,
        noscale: bool = True,
        reduction: str = "frozen",
        x_full: Optional[Array] = None,
        **kwargs,
    ) -> Array:
        r"""
        **Description:**
        Reduced Hessian of the scalar potential
        :math:`\partial_{\phi^\alpha}\partial_{\phi^\beta} V` with respect to
        the real light-field coordinates, with the heavy moduli on-shell.

        .. admonition:: Details
            :class: dropdown

            The real light-field coordinates are
            :math:`\phi^\alpha = (a^1, v^1, \ldots, a^{n_{\rm light}},
            v^{n_{\rm light}}, c_0, s)` (with :math:`z^i = a^i + \mathrm{i}\,v^i`
            and :math:`\tau = c_0 + \mathrm{i}\,s`).  Four reduction schemes are
            provided via ``reduction``:

            - ``"frozen"`` (default): the *selection* block
              :math:`J^T (\nabla\nabla V) J`, with the heavy moduli held at their
              on-shell values but WITHOUT the back-reaction of the light fields
              on the heavy solution.

            - ``"schur"``: the Schur complement on the heavy block,

              .. math::
                  H_{\rm eff} = H_{\ell\ell}
                      - H_{\ell h} H_{hh}^{-1} H_{h\ell}\, ,

              which integrates the heavy moduli out **at their V-minimum**
              (:math:`\partial_{z_{\rm heavy}} V = 0`, obtained by extremising the
              quadratic form over the heavy directions).  This is the right
              reduction when a genuinely heavy modulus really is integrated out at
              its potential minimum (:class:`ConifoldFreezer`); it is exact only
              where that heavy direction is on-shell, so pass ``x_full`` (the
              stored full point) — otherwise the heavy field is reconstructed from
              the *analytic* solve, which is on-shell only deep in the throat, and
              ``schur`` can return a spurious tachyon at moderate throats.

              .. important::
                 For a :class:`PFVEFT` the moduli are **not** at a V-minimum: they
                 are slaved along the flat direction (:math:`\partial_z W = 0`),
                 which is not a V-valley.  ``"schur"`` then computes a genuinely
                 *different* reduction from the racetrack :math:`\tau`-mass and
                 lands a few :math:`\times` off it (verified against a
                 finite-difference reference and the racetrack).  Use ``"tangent"``
                 / ``"autodiff"`` (the F-flat slaving) for a PFV mass; ``"schur"``
                 there is a V-minimum diagnostic, not the physical mass.

              Only the heavy block :math:`H_{hh}` is inverted, so it is well
              conditioned at a genuine vacuum, but the back-reaction is a
              difference of large nearly cancelling terms once the heavy/light mass
              hierarchy approaches ``1/eps`` (deep in a conifold throat); there it
              loses precision.

            - ``"autodiff"``: the Hessian of :math:`V(x_{\rm full}(\phi))`
              differentiated directly through the on-shell heavy solve, so the
              back-reaction enters automatically via the chain rule.  This builds
              a fresh ``jax.hessian`` trace on each call (the inner kernels are
              cached, the outer transform is not), so jit/loop accordingly for
              repeated use.  Exact everywhere but the most expensive option.

            - ``"tangent"``: the *same* physical reduced Hessian as ``"autodiff"``
              at a genuine vacuum, built from the first-order on-shell tangent
              :math:`J = \partial x_{\rm full}/\partial\phi` (one ``jax.jacfwd`` of
              the heavy solve): :math:`H_{\rm eff} = J^T(\nabla\nabla V)J`.  Avoiding
              the second-order ``jax.hessian`` makes it orders of magnitude cheaper
              (a warm call is ~instant).  Unlike ``"schur"`` — which integrates the
              moduli out at their V-minimum — it follows the physical F-flat slaving
              (:math:`\partial_z W = 0`, the racetrack direction) and carries no
              large-cancellation precision loss.  It differs from ``"frozen"`` only
              in that :math:`J` is the true on-shell tangent (with the heavy
              back-reaction), not the constant leading-order one.  Drops the
              :math:`O(\partial V)` term, so it is exact only at a critical point —
              the recommended choice for **masses at a vacuum**.

        .. warning::
            ``reduction="frozen"`` omits the integrate-out back-reaction
            :math:`-H_{\ell h} H_{hh}^{-1} H_{h\ell}`, which can dominate (or flip
            the sign of) the lightest light mass in a conifold throat.  It equals
            ``"autodiff"`` only for LCS **in** ``mode="ansatz"`` (the linear ansatz
            map, where :math:`\partial^2 x_{\rm full}/\partial\phi^2 = 0`); in
            ``mode="eom"`` it keeps the *constant ansatz* tangent at the on-shell
            point, so on an exponentially small mass it can be :math:`O(10^2)` off.
            For any **vacuum mass** prefer ``reduction="tangent"`` (fast + exact)
            or ``"autodiff"`` (robust off-shell); ``"schur"`` computes the
            V-minimum reduction (right for a genuinely heavy modulus, but off the
            racetrack mass for a PFV) and also loses precision when the heavy/light
            hierarchy is large.  Note :meth:`light_mass_spectrum` pairs the frozen
            Hessian with the *substituted* reduced metric, so its eigenvalues are a
            hybrid (a no-back-reaction Hessian against a with-back-reaction metric),
            not the naive frozen masses.

        Args:
            x_light (Array): Real coordinates for light moduli and axio-dilaton.
            fluxes (Array): Full flux vector.
            noscale (bool, optional): If ``True``, uses the no-scale scalar
                potential. Defaults to ``True``.
            reduction (str, optional): Reduction scheme, one of
                ``{"frozen", "schur", "autodiff", "tangent"}``. Defaults to
                ``"frozen"`` (the backwards-compatible leading-order block; note
                :meth:`light_mass_spectrum` resolves its own, per-class default
                instead).  For a mass **at a vacuum** ``"tangent"`` is the fast +
                exact choice; ``"schur"`` gives the V-minimum reduction (right for
                a genuinely heavy modulus, not for a PFV flat direction).
                Distinct from the ``mode`` keyword (forwarded via ``**kwargs`` to
                the heavy solve).
            x_full (Array, optional): Full real point at which to evaluate the
                Hessian for ``"frozen"``/``"schur"`` (e.g. the stored vacuum,
                with the heavy field on-shell).  If ``None`` (default) the heavy
                field is reconstructed from the analytic solve via
                :meth:`_real_light_to_full`.  Ignored by ``"autodiff"`` (which
                differentiates through the solve).

        Returns:
            Array: Reduced Hessian restricted to light directions, of shape
            ``(2 * n_light + 2, 2 * n_light + 2)``.
        """
        # Guardrail: the coniLCS PFVEFT tau mass needs the z_cf(tau) back-reaction
        # AND an on-shell point, so the *default* ``mode="ansatz"`` +
        # ``reduction="frozen"`` returns garbage (an O(1) or tachyonic mass).  Warn
        # and point to the reliable combination.  Scoped to PFVEFT (it carries a
        # ``mode``); ConifoldFreezer's frozen/schur bulk spectrum is a documented
        # building block and is not flagged here.
        self._warn_reduction_traps(reduction)

        if reduction not in ("frozen", "schur", "autodiff", "tangent"):
            raise ValueError(
                "`reduction` must be one of {'frozen', 'schur', 'autodiff', "
                f"'tangent'}}, got {reduction!r}."
            )

        # The reduction itself is a handful of small contractions, but eager
        # tracing of the light->full reconstruction (a Newton solve in
        # ``mode="eom"``) dominates the runtime by ~3 orders of magnitude, so the
        # work happens inside the jitted :meth:`_ddV_x_light_impl`.
        return self._ddV_x_light_impl(
            x_light, fluxes, noscale=noscale, reduction=reduction,
            x_full=x_full, **kwargs)

    def _warn_reduction_traps(self, reduction: str) -> None:
        r"""
        **Description:**
        Emit the coniLCS ``PFVEFT`` reduction guardrails: the :math:`\tau` mass
        needs the :math:`z_{\rm cf}(\tau)` back-reaction **and** an on-shell point,
        so the *default* ``mode="ansatz"`` + ``reduction="frozen"`` returns an
        :math:`O(1)` or tachyonic value.

        Scoped to :class:`PFVEFT` (it carries a ``mode``); ``ConifoldFreezer``'s
        frozen/schur bulk spectrum is a documented building block and is not
        flagged.  Called by :meth:`ddV_x_light` and by the ``schur`` fast path in
        :meth:`light_mass_spectrum`, so both warn identically.

        Args:
            reduction (str): The requested reduction scheme.

        Returns:
            None
        """
        if getattr(self, "mode", None) is None:
            return
        _limit = str(getattr(getattr(self.model, "lcs_tree", None), "limit", ""))
        if "coni" not in _limit.lower():
            return
        if reduction == "frozen":
            warnings.warn(
                "PFVEFT.ddV_x_light(reduction='frozen') omits the z_cf "
                "back-reaction for a coniLCS model and gives a wrong light "
                "mass. Use reduction='tangent' (fast, exact at a vacuum) or "
                "'autodiff'.", stacklevel=3)
        if self.mode == "ansatz":
            warnings.warn(
                "A coniLCS PFVEFT Hessian with mode='ansatz' is evaluated at "
                "an off-shell point (bulk moduli only leading-order), so the "
                "eigenvalues are unreliable. Rebuild with mode='eom'.",
                stacklevel=3)

    def _ddV_x_light_impl(
        self,
        x_light: Array,
        fluxes: Array,
        noscale: bool = True,
        reduction: str = "frozen",
        x_full: Optional[Array] = None,
        return_heavy_block: bool = False,
        **kwargs,
    ) -> Any:
        r"""
        **Description:**
        Compiled implementation of the four reduced-Hessian schemes behind
        :meth:`ddV_x_light`, which validates and warns before dispatching here.

        ``self`` is a traced pytree argument (see :func:`_register_freezer_pytree`),
        so the bound model's arrays are runtime inputs rather than HLO constants:
        a model edit cannot be served stale, and one compiled graph is reusable
        across geometries of equal shape.  Static configuration travels through
        ``static_argnames``; ``reduction`` and ``return_heavy_block`` select the
        branch at trace time.

        Args:
            x_light (Array): Real coordinates for light moduli and axio-dilaton.
            fluxes (Array): Full flux vector.
            noscale (bool, optional): Use the no-scale potential. Defaults to
                ``True``.
            reduction (str, optional): One of ``{"frozen", "schur", "autodiff",
                "tangent"}``. Defaults to ``"frozen"``.  Assumed pre-validated.
            x_full (Array, optional): Full real point; ``None`` reconstructs it.
            return_heavy_block (bool, optional): ``schur`` only -- also return the
                heavy block :math:`H_{hh}`, so that
                :meth:`light_mass_spectrum`'s ``rcond`` diagnostic does not have to
                rebuild the full model Hessian. Defaults to ``False``.
            **kwargs: Forwarded to the heavy solve; string/bool values must be
                named in :data:`_STATIC_KERNEL_KWARGS`.

        Returns:
            Array: Reduced Hessian in the light directions -- or, when
            ``return_heavy_block`` is ``True`` (``schur`` only), the pair
            ``(H_eff, H_hh)``.
        """
        if reduction == "autodiff":
            # Delegated to a COMPILED helper: this branch builds a fresh
            # second-order AD trace, so left eager it re-traces on every call.
            # (The remaining branches are first-order contractions over kernels
            # the model already compiles, and must NOT be wrapped in an outer jit
            # -- that would inline and recompile those kernels per variant.)
            return self._ddV_x_light_autodiff(x_light, fluxes, noscale=noscale,
                                              **kwargs)

        # One reconstruction, shared by every remaining scheme (and reused for the
        # on-shell tangent below -- it used to be solved a second time inside
        # ``_onshell_tangent``, doubling the cost of ``reduction="tangent"``).
        xf = (self._real_light_to_full(x_light, fluxes, **kwargs)
              if x_full is None else x_full)

        if reduction == "frozen":
            return self._hvp_contract(xf, fluxes, self._real_light_jacobian,
                                      noscale=noscale)

        if reduction == "schur":
            # ``schur`` genuinely needs the *wide* heavy block, so here the full
            # Hessian really is the cheapest route (an HVP per heavy direction
            # would be no saving).
            ddV_full = self.model.ddV_x(xf, fluxes, noscale=noscale)
            J_l = self._real_light_jacobian
            J_h = self._real_heavy_jacobian
            H_ll = J_l.T @ ddV_full @ J_l
            H_hh = J_h.T @ ddV_full @ J_h
            H_lh = J_l.T @ ddV_full @ J_h
            H_eff = H_ll - H_lh @ jnp.linalg.solve(H_hh, H_lh.T)
            # ``light_mass_spectrum`` reports rcond(H_hh) as a precision
            # diagnostic; returning the block here means it does not have to
            # recompute the (expensive) full ``ddV_x`` to rebuild it.
            return (H_eff, H_hh) if return_heavy_block else H_eff

        # reduction == "tangent": reduced Hessian along the on-shell slice via the
        # *exact* on-shell tangent J = dx_full/dx_light: H = J^T (ddV_x) J.  At a
        # genuine vacuum (dV_x = 0) this equals the full "autodiff" Hessian but is
        # only a *first*-order derivative, so it is far cheaper than jax.hessian
        # through the solve and, unlike "schur", follows the F-flat slaving.  It
        # differs from "frozen" only in that J is the true on-shell tangent (with
        # the heavy back-reaction).  Off a vacuum it drops the O(dV_x) term; pass
        # the on-shell point (or use "autodiff") for the exact off-shell Hessian.
        J = self._onshell_tangent(x_light, fluxes, x_full=xf, **kwargs)
        return self._hvp_contract(xf, fluxes, J, noscale=noscale)

    @partial(jax.jit, static_argnames=("noscale",))
    def _hvp_contract(self, x_full: Array, fluxes: Array, J: Array,
                      noscale: bool = True) -> Array:
        r"""
        **Description:**
        The reduced Hessian :math:`J^{T}(\nabla\nabla V)J` computed by
        **Hessian-vector products**, without ever forming
        :math:`\nabla\nabla V`.

        .. admonition:: Why not build the full Hessian
            :class: dropdown

            Only :math:`J`'s columns are ever contracted, so the full
            :math:`(2h_{12}+2)^2` object is not needed: one forward-over-reverse
            product per column suffices,
            :math:`w_b = \nabla\nabla V \cdot J_{:,b}` via :func:`jax.jvp` of
            ``model.dV_x``, then :math:`H_{ab} = J_{:,a}\cdot w_b`.

            The motivation is **accuracy**, not throughput.  For a PFV the full
            Hessian is a violently cancelling object -- :math:`|\nabla\nabla V|
            \sim 30` collapsing to :math:`|H_{\rm eff}| \sim 10^{-11}` -- and
            forming it before contracting loses those digits.  Measured on the LCS
            reference PFV: this route sits :math:`2.0\times10^{-5}` from a
            finite-difference reference and matches ``reduction="autodiff"`` to
            :math:`2\times10^{-8}`, where building the full Hessian first sits
            :math:`1.3\times10^{-4}` away -- roughly a sixfold improvement on the
            lightest mass.  Away from such cancellation the two agree to
            :math:`\sim10^{-15}`, i.e. they are the same computation.

            The speed gain is modest (measured 1.15--1.93x for
            :math:`h_{12} = 2\ldots6`), far below the naive
            "one VJP per row" estimate: ``model.ddV_x`` is a single compiled kernel
            whose expensive forward work (periods, GV sums) is shared across all
            output rows, so the marginal cost per row is small.

            Compiled, because :func:`jax.jvp` builds an AD trace that would
            otherwise be re-traced on every call.  ``J`` is a traced argument, so
            ``"frozen"`` and ``"tangent"`` share one executable.  Not used by
            ``"schur"``, which needs the wide heavy block.

        Args:
            x_full (Array): Full real point at which to evaluate.
            fluxes (Array): Full flux vector.
            J (Array): Tangent of shape ``(2*(h12+1), 2*n_light+2)``.
            noscale (bool, optional): Use the no-scale potential. Defaults to
                ``True``.

        Returns:
            Array: Reduced Hessian of shape ``(2*n_light+2, 2*n_light+2)``.
        """
        cols = J.T                                   # (n_light_real, dim_full)

        def _hv(v: Array) -> Array:
            return jax.jvp(lambda y: self.model.dV_x(y, fluxes, noscale=noscale),
                           (x_full,), (v,))[1]

        return cols @ jax.vmap(_hv)(cols).T

    @partial(jax.jit, static_argnames=_STATIC_KERNEL_KWARGS)
    def _ddV_x_light_autodiff(self, x_light: Array, fluxes: Array,
                              noscale: bool = True, **kwargs) -> Array:
        r"""
        **Description:**
        Reduced Hessian by differentiating the substituted potential
        :math:`V(x_{\rm full}(\phi))` straight through the heavy solve --
        ``ddV_x_light(reduction="autodiff")``.

        Compiled deliberately: a fresh ``jax.hessian`` trace per call is what made
        this cost minutes when left eager.  It cannot reuse a supplied ``x_full``,
        because the whole point is to differentiate the solve; prefer
        ``reduction="tangent"`` at a vacuum, where the two agree.

        Args:
            x_light (Array): Real coordinates for light moduli and axio-dilaton.
            fluxes (Array): Full flux vector.
            noscale (bool, optional): Use the no-scale potential. Defaults to
                ``True``.
            **kwargs: Forwarded to the heavy solve; string/bool values must be
                named in :data:`_STATIC_KERNEL_KWARGS`.

        Returns:
            Array: Reduced Hessian in the light directions.
        """
        def _V_light(xl: Array) -> float:
            xf = self._real_light_to_full(xl, fluxes, **kwargs)
            return self.model.V_x(xf, fluxes, noscale=noscale)
        return jax.hessian(_V_light)(x_light)

    def K_x_light(
        self,
        x_light: Array,
        fluxes: Array,
        **kwargs,
    ) -> float:
        r"""
        **Description:**
        Real Kähler potential :math:`\mathrm{Re}\,K` evaluated at the
        light-field coordinates, with the heavy moduli integrated out on-shell.

        .. admonition:: Details
            :class: dropdown

            The heavy field is substituted at the level of the *potential*: the
            light real coordinates are mapped to the full point via
            :meth:`_real_light_to_full` (heavy moduli on-shell), converted to
            complex moduli, and inserted into the model's Kähler potential.
            Using the *actual* conjugate (rather than an independent
            :math:`\bar\phi`) keeps the result a genuinely real scalar, so its
            real Hessian — the reduced Kähler metric of
            :meth:`G_x_light` — is symmetric and the extracted
            metric Hermitian by construction.

        Args:
            x_light (Array): Real coordinates for light moduli and axio-dilaton.
            fluxes (Array): Full flux vector.

        Returns:
            float: :math:`\mathrm{Re}\,K(z_{\rm heavy}^\ast(\phi), \phi)`.
        """
        x_full = self._real_light_to_full(x_light, fluxes, **kwargs)
        moduli, moduli_c, tau, tau_c = self.model._convert_real_to_complex(x_full)
        return jnp.real(self.model.kahler_potential(moduli, moduli_c, tau, tau_c))

    def G_x_light(
        self,
        x_light: Array,
        fluxes: Array,
        x_full: Optional[Array] = None,
        method: str = "pullback",
        **kwargs,
    ) -> Array:
        r"""
        **Description:**
        Reduced Kähler metric of the light fields in the real interleaved basis,
        obtained by integrating out the heavy moduli at the level of the Kähler
        potential.

        .. admonition:: Details
            :class: dropdown

            The reduced metric is the mixed second derivative of the
            *substituted* Kähler potential
            :math:`K(z_{\rm heavy}^\ast(\phi), \phi)`, NOT the light submatrix of
            the full metric.  The substitution couples the light moduli through
            the heavy solution, so the reduced metric carries chain-rule terms
            (e.g. complex-structure–dilaton mixing) absent from the bare bulk
            block.

            Concretely, the real symmetric Hessian
            :math:`H_K = \partial_{\phi^\alpha}\partial_{\phi^\beta}
            \mathrm{Re}\,K` of :meth:`K_x_light` is taken by
            automatic differentiation, the complex Hermitian metric
            :math:`G_{A\bar B}` is extracted via :func:`_G_from_real_hessian`,
            and the real metric is rebuilt with
            :func:`_kahler_metric_real_interleaved`.

        .. admonition:: Two equivalent routes -- ``method``
            :class: dropdown

            ``"pullback"`` (default): the reduced metric is the **pullback** of the
            full Kähler metric along the on-shell tangent,
            :math:`G_{\rm eff} = J^T G_{\rm full} J`, with
            :math:`J = \partial x_{\rm full}/\partial\phi` from
            :meth:`_onshell_tangent`.  This is exact whenever the heavy solve is
            *holomorphic* in the light fields -- which F-flatness provides, since
            :math:`W` is holomorphic, so :math:`z^\ast(\phi)` is too and the mixed
            derivative :math:`\partial\bar\partial K` picks up no second-derivative
            term.  Cost: one metric evaluation plus one first-order tangent.

            ``"autodiff"``: the original route, ``jax.hessian`` of the *substituted*
            :meth:`K_x_light` straight through the heavy solve.  Kept as the
            reference -- it makes no holomorphy assumption -- but for
            ``PFVEFT(mode="eom")`` it differentiates through a Newton iteration,
            which is ~5 orders of magnitude more expensive.

            **Measured equivalence, and its one caveat.**  The two routes were
            compared directly:

            - ``PFVEFT`` (LCS reference PFV, 2x2 reduced metric): relative
              ``4.9e-08``, masses agreeing to 8 significant figures.
            - ``ConifoldFreezer`` (``"aule"`` coniLCS, ``n_light=4``, 10x10 reduced
              metric) with ``apply_correction=False`` -- i.e. a *pure F-term* z_cf
              solve, which is strictly holomorphic: relative ``2.9e-16``, machine
              precision, as the argument predicts exactly.
            - the same with ``apply_correction=True`` (the ``ConifoldFreezer``
              default): relative ``1.7e-07``.

            That last case is **not** exact, and the reason is physical: the
            Kähler-covariant z_cf correction depends on :math:`\bar z` as well as
            :math:`z`, so the heavy solve is no longer strictly holomorphic and the
            second-derivative term no longer cancels identically.  The residual is
            nine orders of magnitude larger than the holomorphic case, yet still
            far below the EFT's own truncation error -- so ``"pullback"`` remains
            the sensible default.  If you need the assumption-free value (for a
            convergence study, or a deep throat where the correction is large),
            use ``method="autodiff"``.

        Args:
            x_light (Array): Real coordinates for light moduli and axio-dilaton.
            fluxes (Array): Full flux vector.
            x_full (Array, optional): Full real point with the heavy moduli
                on-shell (see :meth:`full_real_point`).  Supplying it skips the
                heavy solve entirely; ``"autodiff"`` ignores it, since it must
                differentiate through that solve.
            method (str, optional): ``"pullback"`` (default, first-order) or
                ``"autodiff"`` (the through-the-solve reference).
            **kwargs: Forwarded to the heavy solve / :meth:`K_x_light`.

        Returns:
            Array: Reduced Kähler metric in the real interleaved basis, of shape
            ``(2 * n_light + 2, 2 * n_light + 2)``.

        Raises:
            ValueError: If ``method`` is not one of ``{"pullback", "autodiff"}``.
        """
        if method == "autodiff":
            return self._G_x_light_autodiff(x_light, fluxes, **kwargs)
        if method != "pullback":
            raise ValueError(
                "`method` must be one of {'pullback', 'autodiff'}, "
                f"got {method!r}.")
        xf = (self.full_real_point(x_light, fluxes, **kwargs)
              if x_full is None else x_full)
        J = self._onshell_tangent(x_light, fluxes, x_full=xf, **kwargs)
        return J.T @ self._G_x_full(xf) @ J

    @jax.jit
    def _G_x_full(self, x_full: Array) -> Array:
        r"""
        **Description:**
        Full (unreduced) Kähler metric in the real interleaved basis at a full
        real field point -- the object the ``"pullback"`` route of
        :meth:`G_x_light` contracts with the on-shell tangent.

        Compiled: this is a ``jax.hessian`` of the Kähler potential, so leaving it
        eager would re-trace on every call.  Unlike :meth:`_G_x_light_autodiff` the
        differentiation does **not** pass through the heavy solve, so the graph is
        small and compiles quickly.

        Args:
            x_full (Array): Full real coordinate vector.

        Returns:
            Array: Full Kähler metric in the real interleaved basis.
        """
        def _K_full(x: Array) -> float:
            moduli, moduli_c, tau, tau_c = self.model._convert_real_to_complex(x)
            return jnp.real(
                self.model.kahler_potential(moduli, moduli_c, tau, tau_c))
        H_K = jax.hessian(_K_full)(x_full)
        return _kahler_metric_real_interleaved(_G_from_real_hessian(H_K))

    @partial(jax.jit, static_argnames=_STATIC_KERNEL_KWARGS)
    def _G_x_light_autodiff(self, x_light: Array, fluxes: Array,
                            **kwargs) -> Array:
        r"""
        **Description:**
        Reference reduced Kähler metric: ``jax.hessian`` of the *substituted*
        :meth:`K_x_light`, differentiated straight through the heavy solve
        (``G_x_light(..., method="autodiff")``).

        Compiled deliberately -- building a fresh second-order AD trace on every
        call is what made this cost ~200 s *per call* when left eager.  It remains
        far more expensive than the ``"pullback"`` route because the differentiation
        passes through the Newton iteration; it exists as the assumption-free
        cross-check.

        Args:
            x_light (Array): Real coordinates for light moduli and axio-dilaton.
            fluxes (Array): Full flux vector.
            **kwargs: Forwarded to :meth:`K_x_light`; string/bool values must be
                named in :data:`_STATIC_KERNEL_KWARGS`.

        Returns:
            Array: Reduced Kähler metric in the real interleaved basis.
        """
        H_K = jax.hessian(self.K_x_light)(x_light, fluxes, **kwargs)
        return _kahler_metric_real_interleaved(_G_from_real_hessian(H_K))

    def light_mass_spectrum(
        self,
        x_light: Array,
        fluxes: Array,
        reduction: Optional[str] = None,
        noscale: bool = True,
        dw_tol: float = 1e-4,
        rel_tol: float = 1e-8,
        eig_backend: str = "scipy",
        warn_dynamic_range: float = 1e14,
        x_full: Optional[Array] = None,
        **kwargs,
    ) -> LightSpectrum:
        r"""
        **Description:**
        Mass spectrum of the light fields with the heavy moduli integrated out.

        Solves the generalised eigenvalue problem
        :math:`H_{\rm eff}\,v = \lambda\,K_{\rm eff}\,v`, where
        :math:`H_{\rm eff}` is the reduced Hessian (:meth:`ddV_x_light`) and
        :math:`K_{\rm eff}` the reduced Kähler metric (:meth:`G_x_light`).
        Routing the masses through a generalised eigenproblem in the real basis
        avoids the ill-conditioning and basis artefacts of the full
        :meth:`mass_matrix`.

        .. admonition:: Details
            :class: dropdown

            The light masses follow the supergravity real-field normalisation
            :math:`\phi = (a + \mathrm{i} b)/\sqrt2`, so
            :math:`m^2 = \tfrac12\,\lambda`.

            **On-shell evaluation and screening (two modes).** The reduced
            Hessian (``schur``/``frozen``) equals the Hessian of the reduced
            potential only where the heavy direction is on-shell, so the screen
            and the recommended usage depend on which heavy solution you trust:

            - **Analytic reduced EFT** (``x_full=None``, the default): the heavy
              field is reconstructed from the analytic :meth:`compute_zcf` solve,
              which IS its on-shell value in the EFT.  The vacuum condition is then
              the *light* F-terms, so the point is screened on
              ``max|DW_x_light| <= dw_tol``.  The full residual ``max|DW_x|``
              (heavy component included) is the controlled EFT truncation: it is
              reported as ``dw_residual`` but does NOT reject the point, so a
              legitimate moderate-throat vacuum is no longer spuriously flagged.
            - **Certified numerical vacuum** (``x_full`` given): evaluate the
              Hessian at a stored full vacuum and screen the **full** residual
              ``max|DW_x(x_full)| <= dw_tol`` -- the heavy direction must itself be
              on-shell, which also catches a wrong heavy solve / wrong
              ``apply_correction``.

            In both modes ``dw_residual`` is the full residual, and an off-shell
            point returns an empty spectrum with ``reason="off-shell"``.  The
            reduced metric always uses the analytic substituted potential (it
            differentiates through the solve).  For coniLCS the
            ``apply_correction=True`` z_cf-solve default (set by
            :class:`ConifoldFreezer`) gives a usable analytic seed; it belongs to
            the *z_cf solve* and is unrelated to ``reduction="autodiff"``.

        .. note::
            Eager host-side helper (NumPy/SciPy at the eigensolve): **not**
            ``jit``/``vmap``-able; batch over vacua with a Python loop.

        Args:
            x_light (Array): Real coordinates for light moduli and axio-dilaton.
            fluxes (Array): Full flux vector.
            reduction (str, optional): Hessian reduction scheme, one of
                ``{"frozen", "schur", "autodiff", "tangent"}``. Defaults to
                ``None``, which resolves to :attr:`_default_light_reduction` --
                ``"schur"`` for the base class / :class:`ConifoldFreezer` (the
                V-minimum Schur complement, correct for a genuinely heavy modulus
                integrated out at its potential minimum) and ``"tangent"`` for
                :class:`PFVEFT` (the F-flat racetrack mass; ``"schur"`` there gives
                the *different* V-minimum mass -- see :meth:`ddV_x_light`).
                ``"frozen"`` omits the back-reaction (and is paired with the
                *substituted* reduced metric, so its eigenvalues are a hybrid);
                it is diagnostic only.  Orthogonal to the z_cf-solve ``mode``
                forwarded via ``**kwargs``.
            noscale (bool, optional): If ``True``, uses the no-scale scalar
                potential. Defaults to ``True``.
            dw_tol (float, optional): On-shell tolerance on the F-term residual --
                the light ``max|DW_x_light|`` when ``x_full=None`` (analytic EFT),
                or the full ``max|DW_x(x_full)|`` when ``x_full`` is given.
                Defaults to ``1e-4``.
            rel_tol (float, optional): Relative tolerance for the stability flag
                and the flat-direction floor of the dynamic-range diagnostic.
                Defaults to ``1e-8``.
            eig_backend (str, optional): ``"scipy"`` (default, generalised
                ``scipy.linalg.eigh``; tolerates an indefinite ``H_eff`` but
                requires a positive-definite ``K_eff``) or ``"jax"`` (Cholesky
                whitening, also requires a positive-definite ``K_eff``). Defaults
                to ``"scipy"``.
            warn_dynamic_range (float, optional): Warn when the reduced
                spectrum's dynamic range ``max|m^2|/min|m^2|`` (flat directions
                excluded) exceeds this -- a large range means the lightest masses
                are precision-limited relative to the heaviest (float64
                ``1/eps ~ 4.5e15``). Defaults to ``1e14``.
            x_full (Array, optional): Full real point (the stored vacuum, heavy
                field on-shell) at which to evaluate the reduced Hessian and the
                on-shell screen.  If ``None`` the heavy field is reconstructed
                from the analytic solve via ``**kwargs``.  Note
                ``reduction="autodiff"`` IGNORES ``x_full`` for the Hessian (it
                always differentiates through the analytic heavy solve); only the
                on-shell screen and the reduced metric use ``x_full`` in that
                case.  Pass ``reduction="schur"`` to evaluate the masses at a
                stored ``x_full``.

        Returns:
            LightSpectrum: The reduced spectrum and stability diagnostics.
        """
        if reduction is None:
            reduction = self._default_light_reduction
        if reduction not in ("frozen", "schur", "autodiff", "tangent"):
            raise ValueError(
                "`reduction` must be one of {'frozen', 'schur', 'autodiff', "
                f"'tangent'}}, got {reduction!r}."
            )
        if eig_backend not in ("scipy", "jax"):
            raise ValueError(
                f"`eig_backend` must be one of {{'scipy', 'jax'}}, got {eig_backend!r}."
            )
        f_j = jnp.asarray(fluxes)
        xf = (jnp.asarray(x_full) if x_full is not None
              else self._real_light_to_full(x_light, f_j, **kwargs))

        # The full F-term residual (heavy component included) is ALWAYS the
        # reported on-shell diagnostic ``dw_residual``; the screen is mode-aware.
        #  * x_full given    -> certify a stored numerical vacuum: the heavy
        #    direction must be on-shell, so screen the full residual (this also
        #    catches a wrong heavy solve / wrong ``apply_correction``).
        #  * x_full is None  -> analytic reduced EFT: the heavy direction is
        #    on-shell BY the z_cf formula, so the EFT vacuum condition is the
        #    LIGHT F-terms; screen those.  The full residual is then the
        #    (controlled) EFT truncation -- reported, not a rejection -- so a
        #    legitimate moderate-throat vacuum is not spuriously flagged.
        # One F-term evaluation serves both diagnostics: the light residual is the
        # projection ``J_l^T . DW_x`` of the full one (exactly what DW_x_light
        # returns), so calling DW_x_light here would redundantly repeat both the
        # heavy reconstruction and this DW_x evaluation.
        dw_vec = self.model.DW_x(xf, f_j)
        dw_full = float(jnp.max(jnp.abs(dw_vec)))
        screen = (dw_full if x_full is not None else
                  float(jnp.max(jnp.abs(self._real_light_jacobian.T @ dw_vec))))
        if not np.isfinite(screen) or screen > dw_tol:
            return LightSpectrum(np.array([]), np.array([]), np.nan, False, dw_full,
                                 reduction, {"reason": "off-shell"})

        # For ``schur`` take the reduced Hessian AND its heavy block from a single
        # compiled kernel: the rcond diagnostic below needs H_hh, and rebuilding it
        # afterwards would repeat the full ``model.ddV_x`` -- the single most
        # expensive object in the spectrum.  Same impl (and same guardrails) as
        # ``ddV_x_light``, so the Hessian is identical either way.
        H_hh_dev = None
        if reduction == "schur":
            self._warn_reduction_traps(reduction)
            _H_eff_dev, H_hh_dev = self._ddV_x_light_impl(
                x_light, f_j, noscale=noscale, reduction="schur", x_full=xf,
                return_heavy_block=True, **kwargs)
            H_eff = np.asarray(_H_eff_dev)
        else:
            H_eff = np.asarray(self.ddV_x_light(
                x_light, f_j, noscale=noscale, reduction=reduction, x_full=xf,
                **kwargs))
        # Reuse the reconstruction: the reduced metric is the pullback of the full
        # metric along the on-shell tangent at this same point.
        K_eff = np.asarray(self.G_x_light(x_light, f_j, x_full=xf, **kwargs))
        H_eff = 0.5 * (H_eff + H_eff.T)
        K_eff = 0.5 * (K_eff + K_eff.T)
        if not (np.all(np.isfinite(H_eff)) and np.all(np.isfinite(K_eff))):
            return LightSpectrum(np.array([]), np.array([]), np.nan, False, dw_full,
                                 reduction, {"reason": "nan-hessian"})

        try:
            lam = self._generalised_eigvals(H_eff, K_eff, eig_backend)
        except np.linalg.LinAlgError as exc:   # non-PD / singular reduced metric
            return LightSpectrum(
                np.array([]), np.array([]), np.nan, False, dw_full, reduction,
                {"reason": f"eig-failed: {type(exc).__name__}"})
        if not np.all(np.isfinite(lam)):       # e.g. jax Cholesky on a non-PD K
            return LightSpectrum(np.array([]), np.array([]), np.nan, False, dw_full,
                                 reduction, {"reason": "nan-eig"})

        m2 = 0.5 * lam
        masses = np.where(m2 > 0, np.sqrt(np.abs(m2)), -np.sqrt(np.abs(m2)))
        m2_min = float(np.min(m2))
        absm2 = np.abs(m2)
        scale = max(1.0, float(np.max(absm2)))
        stable = bool(m2_min >= -rel_tol * scale)
        # Dynamic range of the REDUCED (light) spectrum, flat directions (|m2|
        # below the roundoff floor) excluded.  This is a light-sector property,
        # not the heavy/light precision wall (the heavy mode is integrated out);
        # a large range means the lightest masses are precision-limited.
        big = absm2[absm2 > rel_tol * scale]
        dyn_range = float(np.max(absm2) / np.min(big)) if big.size else float("inf")
        if np.isfinite(dyn_range) and dyn_range > warn_dynamic_range:
            warnings.warn(
                f"light_mass_spectrum: reduced-spectrum dynamic range "
                f"{dyn_range:.2e} exceeds {warn_dynamic_range:.0e}; the lightest "
                "masses are precision-limited relative to the heaviest.",
                RuntimeWarning, stacklevel=2,
            )
        info = {"cond_Keff": float(np.linalg.cond(K_eff)),
                "m2_dynamic_range": dyn_range, "n_modes": int(len(lam))}
        if reduction == "schur":
            # Heavy-block conditioning: deep in a throat H_hh is near-singular
            # and the Schur back-reaction is a difference of large, nearly
            # cancelling terms, so the lightest mass loses precision.  Surface
            # rcond(H_hh) and warn; reduction="autodiff" avoids the inversion.
            H_hh = np.asarray(H_hh_dev)          # reused from the kernel above
            rcond = float(1.0 / np.linalg.cond(H_hh)) if H_hh.shape[0] else 1.0
            info["H_hh_rcond"] = rcond
            if rcond < 1e-12:
                warnings.warn(
                    f"light_mass_spectrum(reduction='schur'): heavy block H_hh "
                    f"is ill-conditioned (rcond={rcond:.1e}); the Schur "
                    "back-reaction loses precision — prefer reduction='autodiff'.",
                    RuntimeWarning, stacklevel=2,
                )
        return LightSpectrum(
            np.sort(masses), lam, m2_min, stable, dw_full, reduction, info,
        )

    @staticmethod
    def _generalised_eigvals(
        H: np.ndarray,
        K: np.ndarray,
        backend: str = "scipy",
    ) -> np.ndarray:
        r"""
        **Description:**
        Solve the generalised symmetric eigenvalue problem
        :math:`H\,v = \lambda\,K\,v` and return the eigenvalues.

        Args:
            H (np.ndarray): Symmetric reduced Hessian.
            K (np.ndarray): Symmetric reduced (kinetic) metric.
            backend (str, optional): ``"scipy"`` uses ``scipy.linalg.eigh``
                (tolerates an indefinite ``H`` but requires ``K`` positive
                definite -- it raises ``LinAlgError`` otherwise); ``"jax"`` uses a
                Cholesky whitening ``L = chol(K)``,
                :math:`\lambda = \mathrm{eigvalsh}(L^{-1} H L^{-T})` and requires
                ``K`` positive-definite. Defaults to ``"scipy"``.

        Returns:
            np.ndarray: The generalised eigenvalues.
        """
        if backend == "scipy":
            return _geigh(H, K, eigvals_only=True)
        if backend == "jax":
            L = jnp.linalg.cholesky(jnp.asarray(K))
            # cholesky returns NaN (does not raise) on a non-PD metric; surface
            # it as a LinAlgError so callers handle it like the scipy backend.
            if not bool(jnp.all(jnp.isfinite(L))):
                raise np.linalg.LinAlgError(
                    "reduced metric is not positive-definite (Cholesky failed)."
                )
            Y = solve_triangular(L, jnp.asarray(H), lower=True)   # L^{-1} H
            M = solve_triangular(L, Y.T, lower=True).T            # L^{-1} H L^{-T}
            return np.asarray(jnp.linalg.eigvalsh(0.5 * (M + M.T)))
        raise ValueError(
            f"`eig_backend` must be one of {{'scipy', 'jax'}}, got {backend!r}."
        )

    @property
    def _real_light_slice(self) -> Array:
        r"""
        Description:
        Indices into the real coordinate array ``x`` corresponding to the
        light moduli. The real array is ordered as
        ``[Re(z_0), Im(z_0), ..., Re(z_{h-1}), Im(z_{h-1}), Re(tau), Im(tau)]``,
        so modulus ``i`` maps to real indices ``2*i`` and ``2*i+1``.

        Returns:
            Array: Integer indices selecting the light real coordinates.
        """
        light = list(self.light_indices)
        real_idx = []
        for i in light:
            real_idx.extend([2 * i, 2 * i + 1])
        # tau is always the last two entries
        real_idx.extend([2 * self.model.h12, 2 * self.model.h12 + 1])
        return jnp.array(real_idx)

    @property
    def _real_light_jacobian(self) -> Array:
        r"""
        Description:
        Real Jacobian :math:`J = \partial x_{\rm full}/\partial x_{\rm light}`
        (heavy moduli held fixed) mapping the light real coordinates into the
        full real array.  The light real gradient is :math:`J^T\cdot(\text{full
        gradient})` and the Hessian block :math:`J^T\cdot(\text{full
        Hessian})\cdot J`.

        For an axis-aligned light/heavy split this is the selection matrix that
        picks :attr:`_real_light_slice` (so :math:`J^T v = v[\text{slice}]` and
        :math:`J^T M J = M[\text{slice},\text{slice}]`, bit-identical to plain
        slicing).  Subclasses whose light directions are *not* coordinate axes
        — e.g. a conifold modulus that is a generic charge combination
        (``conifold_basis=False``) — override this with the corresponding
        embedding Jacobian.

        Returns:
            Array: Real Jacobian of shape ``(2*(h12+1), 2*n_light+2)``.
        """
        dim_full = 2 * (self.model.h12 + 1)
        return jnp.eye(dim_full)[:, self._real_light_slice]

    @property
    def _real_heavy_slice(self) -> Array:
        r"""
        Description:
        Indices into the real coordinate array ``x`` corresponding to the heavy
        moduli (``2*i`` and ``2*i+1`` for each ``i`` in :attr:`heavy_indices`).

        Unlike :attr:`_real_light_slice` this contains no axio-dilaton entry,
        since :math:`\tau` is always a light field.

        Returns:
            Array: Integer indices selecting the heavy real coordinates.
        """
        real_idx = []
        for i in self.heavy_indices:
            real_idx.extend([2 * i, 2 * i + 1])
        return jnp.array(real_idx)

    @property
    def _real_heavy_jacobian(self) -> Array:
        r"""
        Description:
        Real Jacobian :math:`J_h = \partial x_{\rm full}/\partial x_{\rm heavy}`
        selecting the heavy directions — the complement of
        :attr:`_real_light_jacobian`.  Together ``[J_h | J_l]`` form an
        invertible change of basis on the full real field space, so the Schur
        complement in :meth:`ddV_x_light` (``reduction="schur"``) is the exact
        on-shell reduced Hessian expressed in the light coordinates.

        For an axis-aligned heavy/light split this is the selection matrix that
        picks :attr:`_real_heavy_slice`.  Subclasses whose heavy direction is
        *not* a coordinate axis — e.g. a conifold modulus that is a generic
        charge combination (``conifold_basis=False``) — override this with the
        corresponding embedding Jacobian.

        Returns:
            Array: Real Jacobian of shape ``(2*(h12+1), 2*n_heavy)``.
        """
        dim_full = 2 * (self.model.h12 + 1)
        return jnp.eye(dim_full)[:, self._real_heavy_slice]

    @partial(jax.jit, static_argnames=_STATIC_KERNEL_KWARGS)
    def _onshell_tangent(self, x_light: Array, fluxes: Array,
                         x_full: Optional[Array] = None, **kwargs) -> Array:
        r"""
        **Description:**
        On-shell field-space tangent :math:`J = \partial x_{\rm full}/\partial
        x_{\rm light}` **with the heavy back-reaction** — the derivative of the
        heavy solve, as opposed to the frozen leading-order
        :attr:`_real_light_jacobian`.  Used by ``ddV_x_light(reduction="tangent")``.

        Compiled: this builds a forward-mode AD trace, so left eager it would
        re-trace on every call (the dominant cost for a ``ConifoldFreezer``, which
        uses this base implementation).  :meth:`PFVEFT._onshell_tangent` overrides
        it with a closed form that is *not* AD and is therefore deliberately left
        eager, so it does not inline and recompile the model's own kernels.

        ``x_full`` is accepted for signature symmetry and **ignored** here: this
        base implementation differentiates the reconstruction map itself, so the
        point alone does not determine the tangent.  :meth:`PFVEFT._onshell_tangent`
        overrides it with a closed form that *does* only need the point, and uses
        ``x_full`` to skip the heavy solve entirely.

        The base implementation is one forward-mode :func:`jax.jacfwd` of
        :meth:`_real_light_to_full`, which is cheap whenever the heavy solve is
        analytic (e.g. the ConifoldFreezer ``z_cf`` throat solve).  Subclasses
        whose heavy solve is an iterative Newton loop should override this with the
        closed-form implicit-function tangent, to avoid differentiating through the
        loop (see :meth:`PFVEFT._onshell_tangent`).

        Args:
            x_light (Array): Real light-field coordinates.
            fluxes (Array): Full flux vector.
            x_full (Array, optional): Accepted for signature symmetry with
                :meth:`PFVEFT._onshell_tangent` and **ignored** here (this
                implementation differentiates the reconstruction map, so a point
                alone does not determine the tangent).
            **kwargs: Forwarded to :meth:`_real_light_to_full`.

        Returns:
            Array: Tangent of shape ``(2*(h12+1), 2*n_light+2)``.
        """
        return jax.jacfwd(
            lambda xl: self._real_light_to_full(xl, fluxes, **kwargs))(x_light)

    @abstractmethod
    def _real_light_to_full(
        self,
        x_light: Array,
        fluxes: Array,
        **kwargs,
    ) -> Array:
        r"""
        **Description:**
        Convert real light-field coordinates to the full real coordinate
        array by solving for and inserting the heavy moduli.

        Args:
            x_light (Array): Real coordinates for light moduli + tau.
            fluxes (Array): Full flux vector.

        Returns:
            Array: Full real coordinate array.
        """
        ...


class ConifoldFreezer(Freezer):
    r"""
    **Description:**
    Integrates out the conifold modulus :math:`z_{\text{cf}}` (index 0) in
    coniLCS models.

    Near the conifold locus, the conifold modulus acquires a parametrically
    large mass from the flux superpotential. Its leading-order EOM gives

    .. math::
        z_{\text{cf}} = -\frac{1}{2\pi i}
            \exp\!\Bigl(-\frac{2\pi i\,\widetilde{W}_1}{n_{\text{cf}}(M_1 - \tau H_1)}\Bigr)

    where :math:`\widetilde{W}_1` is the effective superpotential contribution
    from the bulk moduli and :math:`n_{\text{cf}}` is the conifold degree.

    Args:
        model: A flux EFT model with ``"coniLCS"`` in ``model.periods.limit``.
        conifold_index (int, optional): Index of the conifold modulus in the
            moduli array. Defaults to ``0``.

    Note:
        The conifold degree :math:`n_{\text{cf}}` is *not* a constructor
        argument; it is exposed read-only through the :attr:`ncf` property
        (sourced from ``model.lcs_tree.conifold.ncf``).
    """

    def __init__(
        self,
        model: Any,
        conifold_index: int = 0,
    ) -> None:
        r"""
        **Description:**
        Initialise the ConifoldFreezer.

        Args:
            model: A flux EFT model with ``"coniLCS"`` in ``model.periods.limit``.
            conifold_index (int, optional): Index of the conifold modulus in the
                moduli array. Defaults to ``0``.

        Attributes:
            _conifold_index (int): Stored index of the conifold modulus.
        """
        super().__init__(model)
        self._conifold_index = conifold_index

    @property
    def heavy_indices(self) -> Tuple[int, ...]:
        r"""
        Description:
        Indices of the heavy (conifold) modulus; always a length-1 tuple.

        Returns:
            tuple[int, ...]: The single conifold-modulus index.
        """
        return (self._conifold_index,)

    @property
    def ncf(self) -> int:
        r"""
        Description:
        Conifold degree :math:`n_{\text{cf}}`, sourced from
        ``self.model.lcs_tree.conifold.ncf`` (single source of truth).

        Returns:
            int: The conifold degree :math:`n_{\text{cf}}`.
        """
        return int(self.model.lcs_tree.conifold.ncf)

    # ------------------------------------------------------------------ #
    # conifold_basis=False reconstruction.
    #
    # When the geometry is NOT rotated into the conifold-aligned frame the
    # conifold modulus is the charge combination z_cf = q·z, not a coordinate
    # axis, and the light (bulk) directions span ker(q).  The light↔full map is
    # then z_full = z_cf·e_q + bulk_embedding·z_light (instead of an index
    # scatter), and the light F-terms / Jacobian project through ``bulk_embedding``
    # (= Λ[1:]ᵀ).  In the aligned basis (``conifold_basis=True``) every method
    # below defers to the base-class index-based implementation, bit-identical.
    # ------------------------------------------------------------------ #

    def reconstruct_full_moduli(self, z_light, tau, fluxes, **kwargs):
        r"""
        **Description:**
        Reconstruct the full modulus vector from the light (bulk) moduli with
        the conifold modulus on-shell.  Aligned: index scatter (base class).
        General: :math:`z_{\rm full} = z_{\rm cf}\,e_q + \text{bulk\_embedding}\,z_{\rm light}`.

        Args:
            z_light (Array): Complex light (bulk) moduli, length ``n_light``.
            tau (complex): Axio-dilaton.
            fluxes (Array): Full flux vector.
            **kwargs: Forwarded to :meth:`solve_heavy` (e.g. the ``z_cf`` solve
                ``mode`` and ``apply_correction``).

        Returns:
            Array: Full complex modulus vector of length ``h12`` with
            :math:`z_{\rm cf}` on-shell.
        """
        if self.model.lcs_tree.conifold_basis:
            return super().reconstruct_full_moduli(z_light, tau, fluxes, **kwargs)
        z_cf = self.solve_heavy(z_light, tau, fluxes, **kwargs)[0]
        coni = self.model.lcs_tree.conifold
        e_q = jnp.asarray(coni.embedding,      dtype=z_light.dtype)
        be  = jnp.asarray(coni.bulk_embedding, dtype=z_light.dtype)
        return z_cf * e_q + be @ z_light

    def DW_light(self, z_light, z_light_c, tau, tau_c, fluxes,
                 assume_conjugate: bool = False, **kwargs):
        r"""
        **Description:**
        Covariant derivatives :math:`D_i W` for the light moduli (+ :math:`D_\tau W`)
        with the conifold modulus on-shell.  General basis: project the full
        :math:`D_i W` onto the bulk directions, :math:`D_a W = D_i W\,
        \text{bulk\_embedding}^{i}{}_{a}` (the conifold component
        :math:`D_i W\,e_q^i = \partial_{z_{\rm cf}}W \approx 0` on-shell).

        Args:
            z_light (Array): Complex light (bulk) moduli, length ``n_light``.
            z_light_c (Array): Complex conjugate of *z_light*.
            tau (complex): Axio-dilaton.
            tau_c (complex): Complex conjugate of *tau*.
            fluxes (Array): Full flux vector.
            assume_conjugate (bool, optional): Reuse
                :math:`\overline{z_{\rm full}}` instead of a second heavy solve --
                **evaluation only**, see the warning on
                :meth:`Freezer.DW_light`. Defaults to ``False``.
            **kwargs: Forwarded to :meth:`reconstruct_full_moduli` /
                :meth:`solve_heavy`.

        Returns:
            Array: Complex vector ``[D_a W (light moduli), D_tau W]`` of length
            ``n_light + 1``.
        """
        if self.model.lcs_tree.conifold_basis:
            return super().DW_light(z_light, z_light_c, tau, tau_c, fluxes,
                                    assume_conjugate=assume_conjugate, **kwargs)
        z_full   = self.reconstruct_full_moduli(z_light,   tau,   fluxes, **kwargs)
        z_full_c = (jnp.conj(z_full) if assume_conjugate else
                    self.reconstruct_full_moduli(z_light_c, tau_c, fluxes, **kwargs))
        DW_full = self.model.DW(z_full, z_full_c, tau, tau_c, fluxes)
        be = jnp.asarray(self.model.lcs_tree.conifold.bulk_embedding, dtype=DW_full.dtype)
        DW_z_light = DW_full[:self.model.h12] @ be
        DW_tau = DW_full[-1]
        return jnp.append(DW_z_light, DW_tau)

    @property
    def _real_light_jacobian(self) -> Array:
        r"""
        Description:
        Real Jacobian :math:`\partial x_{\rm full}/\partial x_{\rm light}` for the
        ConifoldFreezer.  General basis: the bulk embedding lifted to real
        coordinates, :math:`\text{bulk\_embedding}\otimes \mathbb{1}_2` on the
        moduli block plus :math:`\mathbb{1}_2` for :math:`\tau`.  Aligned basis:
        the selection matrix (base class).

        Returns:
            Array: Real Jacobian of shape ``(2*h12+2, 2*n_light+2)``.
        """
        if self.model.lcs_tree.conifold_basis:
            return super()._real_light_jacobian
        be = jnp.asarray(self.model.lcs_tree.conifold.bulk_embedding)
        J_mod = jnp.kron(be, jnp.eye(2, dtype=be.dtype))   # (2*h12, 2*(h12-1))
        nm, nl = J_mod.shape
        J = jnp.zeros((nm + 2, nl + 2), dtype=be.dtype)
        J = J.at[:nm, :nl].set(J_mod)
        J = J.at[nm:, nl:].set(jnp.eye(2, dtype=be.dtype))   # tau block
        return J

    @property
    def _real_heavy_jacobian(self) -> Array:
        r"""
        Description:
        Real Jacobian selecting the heavy (conifold) direction — the complement
        of :attr:`_real_light_jacobian`.  General basis (``conifold_basis=False``):
        the conifold charge direction lifted to real coordinates,
        :math:`e_q\otimes\mathbb{1}_2` on the moduli block (no :math:`\tau`
        entry, as :math:`\tau` is light).  Aligned basis: the selection matrix
        (base class).

        Returns:
            Array: Real Jacobian of shape ``(2*h12+2, 2*n_heavy)`` selecting the
            heavy (conifold) direction.
        """
        if self.model.lcs_tree.conifold_basis:
            return super()._real_heavy_jacobian
        e_q = jnp.asarray(self.model.lcs_tree.conifold.embedding, dtype=jnp.float_)
        J_mod = jnp.kron(e_q.reshape(-1, 1), jnp.eye(2, dtype=e_q.dtype))   # (2*h12, 2)
        nm = J_mod.shape[0]
        J = jnp.zeros((nm + 2, 2), dtype=e_q.dtype)
        J = J.at[:nm, :].set(J_mod)
        return J

    def light_mass_spectrum(
        self,
        x_light: Array,
        fluxes: Array,
        **kwargs,
    ) -> LightSpectrum:
        r"""
        **Description:**
        Conifold-aware override of :meth:`Freezer.light_mass_spectrum`: defaults
        the z_cf solve to ``apply_correction=True`` (the Kähler-covariant
        correction needed for the analytic seed to reproduce the stored vacuum),
        then defers to the base implementation.

        Args:
            x_light (Array): Real light-field coordinates (moduli + axio-dilaton).
            fluxes (Array): Full flux vector.
            **kwargs: Forwarded to :meth:`Freezer.light_mass_spectrum`
                (``reduction``, ``noscale``, ``dw_tol``, ``x_full``,
                ``eig_backend``, ...); ``apply_correction`` defaults to ``True``.

        Returns:
            LightSpectrum: The reduced light-field spectrum with its stability
            diagnostics.

        .. seealso:: :meth:`Freezer.light_mass_spectrum`
        """
        kwargs.setdefault("apply_correction", True)
        return super().light_mass_spectrum(x_light, fluxes, **kwargs)

    def bulk_mass_spectrum(
        self,
        x_light: Array,
        fluxes: Array,
        **kwargs,
    ) -> LightSpectrum:
        r"""
        **Description:**
        Bulk mass spectrum of a coniLCS vacuum with the conifold modulus
        integrated out.  Identical to :meth:`light_mass_spectrum` (here the
        "bulk" fields *are* the base-class "light" fields); the alias provides
        the conifold/throat vocabulary used in the literature.  The on-shell
        ``apply_correction=True`` z_cf-solve default is applied.

        Args:
            x_light (Array): Real bulk-field coordinates (moduli + axio-dilaton).
            fluxes (Array): Full flux vector.
            **kwargs: Forwarded to :meth:`light_mass_spectrum` (``reduction``,
                ``dw_tol``, ``x_full``, ``eig_backend``, ...).

        Returns:
            LightSpectrum: The reduced bulk-field spectrum with its stability
            diagnostics.

        .. seealso:: :meth:`light_mass_spectrum`, :meth:`Freezer.light_mass_spectrum`
        """
        return self.light_mass_spectrum(x_light, fluxes, **kwargs)

    def solve_heavy(
        self,
        z_light: Array,
        tau: complex,
        fluxes: Array,
        conj: bool = False,
        mode: str = "manual",
        apply_correction: bool = False,
    ) -> Array:
        r"""
        **Description:**
        Solve for :math:`z_{\text{cf}}` from its leading-order EOM by
        delegating to :func:`jaxvacua.conifold.zcf_solver.compute_zcf` (the unified
        complex-coord dispatcher attached to the model).

        Args:
            z_light (Array): Bulk (light) moduli values.
            tau (complex): Axio-dilaton.
            fluxes (Array): Full flux vector.
            conj (bool, optional): Conjugate conventions. Defaults to ``False``.
            mode (str, optional): One of ``{"manual", "autodiff", "pfv"}``.
                Routes through ``model.W_log_coeff(..., mode=mode)``.
                Defaults to ``"manual"`` (closed-form ``kappa`` /
                ``a_matrix`` / ``b_vector`` + ``Li`` assembly).
            apply_correction (bool, optional): If ``True``, add the
                Kähler-covariant correction ``log_coeff_K_corr`` to the log
                coefficient before exponentiating. Defaults to ``False``.

        Returns:
            Array: Value of :math:`z_{\text{cf}}` (length-1 array).
        """
        cz_light = jnp.conj(z_light)
        ctau = jnp.conj(tau)
        zcf = self.model.compute_zcf(
            z_light, cz_light, tau, ctau, fluxes,
            mode=mode, apply_correction=apply_correction, conj=conj,
        )
        return jnp.array([zcf])

    def _real_light_to_full(
        self,
        x_light: Array,
        fluxes: Array,
        mode: str = "manual",
        conj: bool = False,
        apply_correction: bool = False,
    ) -> Array:
        r"""
        **Description:**
        Convert real light-field coordinates to the full real array by solving
        for :math:`z_{\text{cf}}` and prepending it.  Delegates to
        :func:`jaxvacua.conifold.zcf_solver.zcf_handling`, which expects ``x_light``
        to already be the bulk-only real vector (length ``2 * h12``, no
        conifold direction).

        Args:
            x_light (Array): Real coordinates for bulk moduli + tau
                (length ``2 * n_light + 2``).
            fluxes (Array): Full flux vector.
            mode (str, optional): Solving mode forwarded to ``zcf_handling``.
                Defaults to ``"manual"``.
            conj (bool, optional): Conjugate conventions, forwarded to
                ``zcf_handling``.
            apply_correction (bool, optional): If ``True``, include the
                Kähler-covariant correction in the z_cf solve.

        Returns:
            Array: Full real coordinate array.
        """
        return self.model.zcf_handling(
            x_light, fluxes,
            mode=mode, apply_correction=apply_correction, conj=conj,
        )


# ---------------------------------------------------------------------------
# PFVEFT — the perturbatively-flat-vacuum effective theory.
#
# The single light field is the axio-dilaton tau; the complex-structure moduli
# are all heavy.  Two limits:
#   * LCS     : every modulus is slaved to the flat direction  z = p * tau
#               (p = pfv_p_vector, length h12).  The ANSATZ map is LINEAR, so in
#               mode="ansatz" the tangent is constant and frozen == autodiff
#               exactly; in mode="eom" frozen keeps that constant ansatz tangent
#               at the on-shell point and is unreliable (use tangent/autodiff).
#   * coniLCS : the BULK moduli are slaved  z_bulk = p_hat * tau  (p_hat =
#               pfv_p_vector, length h12-1) AND the conifold modulus z_cf is
#               integrated out ANALYTICALLY via its throat EOM (model.compute_zcf
#               / zcf_handling — the same machinery ConifoldFreezer uses).
#               z_cf(tau) is NONLINEAR, so the physical (racetrack) reduced Hessian
#               is reduction="tangent" (fast) or "autodiff" (robust), which carry
#               the z_cf back-reaction along the F-flat slaving; the constant
#               "frozen" tangent is first-order-only, and "schur" gives the
#               DIFFERENT V-minimum reduction.  Requires conifold_basis=True.
#
# mode="eom" (both limits): instead of the leading-order ansatz, Newton-solve the
#   full 2*h12 moduli F-terms  DW_x[:2*h12] = 0  at fixed tau, seeded at the
#   ansatz.  For LCS this captures the exponentially-small instanton correction to
#   z = p*tau; for coniLCS it promotes z_cf to a genuine Newton variable, so its
#   coupling to the bulk moduli is captured exactly (not just the leading-order
#   throat solve of the ansatz).  Differentiable (jax.lax.fori_loop), so
#   ddV_x_light(reduction="autodiff") differentiates through the solve.
# ---------------------------------------------------------------------------

# Single definition, shared with ``jaxvacua.vacuum`` (which gates its conifold
# diagnostics on the same set) -- see ``jaxvacua.util.CONI_LIMITS``.
from .util import CONI_LIMITS  # noqa: E402  (shared with jaxvacua.vacuum)


class PFVEFT(Freezer):
    r"""
    **Description:**
    Perturbatively-flat-vacuum effective theory: the ``h12`` complex-structure
    moduli are integrated out, leaving a one-dimensional theory in the
    axio-dilaton :math:`\tau` (``n_light = 0``).

    * **LCS** — every modulus is slaved to the flat direction
      :math:`z^a = p^a\,\tau`.
    * **coniLCS** — the bulk moduli are slaved :math:`z_{\rm bulk} = \hat p\,\tau`
      and the conifold modulus :math:`z_{\rm cf}` is integrated out analytically
      via its throat EOM (``conifold_basis=True`` only).

    The physical reduced :math:`\tau`-Hessian (the racetrack mass) is the F-flat
    reduction ``reduction="tangent"`` (fast, exact at a vacuum) or ``"autodiff"``
    (robust) — the moduli are slaved along the flat direction
    (:math:`\partial_z W = 0`).  Two traps: ``"frozen"`` equals ``"autodiff"``
    *only* in ``mode="ansatz"`` (where the light->full map is the linear ansatz);
    in the recommended ``mode="eom"`` it keeps using the constant ansatz tangent
    at the on-shell point and is unreliable (for LCS it can be :math:`O(10^2)` off
    on the exponentially small mass).  And ``"schur"`` computes the *V-minimum*
    Schur complement (:math:`\partial_z V = 0`), a genuinely different reduction
    that for a PFV misses the flat-direction slaving and lands a few :math:`\times`
    off the racetrack mass.  Accordingly :meth:`light_mass_spectrum` defaults to
    ``"tangent"`` for a :class:`PFVEFT` (see :attr:`_default_light_reduction`).

    Args:
        model: An LCS or coniLCS :class:`~jaxvacua.flux_eft.FluxEFT` /
            :class:`~jaxvacua.flux_vacua_finder.FluxVacuaFinder`.
        p (Array): Flat direction — length ``h12`` (LCS) or ``h12-1`` (coniLCS,
            bulk-only :math:`\hat p`).
        flux (Array, optional): The bound PFV flux vector.
        mode (str): ``"ansatz"`` (default) slaves the moduli to the leading-order
            flat direction; ``"eom"`` Newton-solves the full moduli F-terms at
            fixed :math:`\tau` (seeded at the ansatz), capturing the subleading
            corrections — for coniLCS the exact :math:`z_{\rm cf}` back-reaction —
            so prefer ``ddV_x_light(reduction="tangent")`` (fast) or ``"autodiff"``
            (robust) there.
        zcf_mode (str): coniLCS z_cf solve mode (``"manual"`` / ``"autodiff"`` /
            ``"pfv"``); forwarded to ``model.compute_zcf`` / ``zcf_handling``.
        apply_correction (bool): coniLCS — include the Kähler-covariant z_cf
            correction (matches :class:`ConifoldFreezer` on-shell).
        eom_iters (int): Newton iterations for ``mode="eom"`` (default 15).

    See also: :class:`ConifoldFreezer`, :func:`jaxvacua.flux_utils.pfv_p_vector`,
        :class:`jaxvacua.vacuum.PFVData`.
    """

    #: PFV masses default to the F-flat ``"tangent"`` reduction (the racetrack
    #: mass), overriding the base ``"schur"`` (V-minimum) default — see the class
    #: docstring and :meth:`ddV_x_light`.
    _default_light_reduction: str = "tangent"

    def __init__(self, model: Any, p: Array, *, flux: Optional[Array] = None,
                 mode: str = "ansatz", zcf_mode: str = "manual",
                 apply_correction: bool = True, eom_iters: int = 15) -> None:
        r"""
        **Description:**
        Initialise a :class:`PFVEFT` and validate the model + flat direction.

        Args:
            model: LCS or coniLCS model (``model.periods.limit``).
            p (Array): Flat direction — length ``h12`` (LCS) or ``h12-1``
                (coniLCS, bulk-only :math:`\hat p`).
            flux (Array, optional): Bound PFV flux.
            mode (str): ``"ansatz"`` or ``"eom"`` (both limits).
            zcf_mode (str): coniLCS z_cf solve mode.
            apply_correction (bool): coniLCS Kähler-covariant z_cf correction.
            eom_iters (int): Newton iterations for ``mode="eom"``.

        Raises:
            NotImplementedError: On a non-(coni)LCS limit, ``h12 < 2``, a
                ``conifold_basis=False`` coniLCS model, or an unknown ``mode``.
            ValueError: If ``p`` has the wrong length for the limit.
        """
        super().__init__(model)
        limit = getattr(getattr(model, "periods", None), "limit", None)
        self._is_coni = limit in CONI_LIMITS
        if not (limit == "LCS" or self._is_coni):
            raise NotImplementedError(
                f"PFVEFT supports limit='LCS' or coniLCS; got limit={limit!r}."
            )
        if self._is_coni and not getattr(model.lcs_tree, "conifold_basis", True):
            raise NotImplementedError(
                "coniLCS PFVEFT requires conifold_basis=True (aligned frame; "
                "conifold modulus at index 0)."
            )
        if int(model.h12) < 2:
            raise NotImplementedError(
                "PFVEFT requires h12 >= 2 (one-modulus / hypergeometric models "
                "have no non-trivial flat direction)."
            )
        if mode not in ("ansatz", "eom"):
            raise NotImplementedError(f"PFVEFT mode={mode!r} not implemented "
                                      "(supported: 'ansatz', 'eom').")
        self.p = jnp.asarray(p, dtype=float)
        # the flat direction is full (LCS) or bulk-only h12-1 (coniLCS)
        n_expected = int(model.h12) - (1 if self._is_coni else 0)
        if self.p.shape != (n_expected,):
            raise ValueError(
                f"PFVEFT: p must have shape ({n_expected},) for "
                f"{'coniLCS' if self._is_coni else 'LCS'} h12={int(model.h12)}, "
                f"got {tuple(self.p.shape)}."
            )
        self.flux = None if flux is None else jnp.asarray(flux)
        self.mode = mode
        self._zcf_mode = zcf_mode
        self._apply_correction = bool(apply_correction)
        self._eom_iters = int(eom_iters)

    # --- constructors --------------------------------------------------------
    @classmethod
    def from_fluxes(cls, model: Any, M: Array, K: Array, *, mode: str = "ansatz",
                    zcf_mode: str = "manual", apply_correction: bool = True,
                    eom_iters: int = 15) -> "PFVEFT":
        r"""
        **Description:**
        Build a :class:`PFVEFT` from PFV flux quanta :math:`(M, K)`.  The flat
        direction is the model's ``pfv_p_vector`` (full for LCS, bulk-only
        :math:`\hat p` for coniLCS).

        Args:
            model: LCS or coniLCS model.
            M (Array): M-vector.
            K (Array): K-vector.
            mode / zcf_mode / apply_correction / eom_iters: See :class:`PFVEFT`.

        Returns:
            PFVEFT: The bound EFT.
        """
        p = model.pfv_p_vector(M, K)
        flux = model.pfv_to_flux(M, K)
        return cls(model, p, flux=flux, mode=mode, zcf_mode=zcf_mode,
                   apply_correction=apply_correction, eom_iters=eom_iters)

    @classmethod
    def from_pfv_data(cls, data: Any, *, mode: str = "ansatz",
                      zcf_mode: str = "manual", apply_correction: bool = True,
                      eom_iters: int = 15) -> "PFVEFT":
        r"""
        **Description:**
        Build a :class:`PFVEFT` from a :class:`jaxvacua.vacuum.PFVData` carrying
        an attached model.

        Args:
            data (PFVData): PFV algebra object (must have ``_model`` and a
                non-singular ``p``).
            mode / zcf_mode / apply_correction / eom_iters: See :class:`PFVEFT`.

        Returns:
            PFVEFT: The bound EFT.

        Raises:
            RuntimeError: If ``data`` has no attached model.
            ValueError: If ``data.p`` is ``None`` (singular N — not a PFV).
        """
        if getattr(data, "_model", None) is None:
            raise RuntimeError("PFVData has no attached model; re-attach a finder.")
        if data.p is None:
            raise ValueError("PFVData.p is None (singular N — not a PFV).")
        return cls(data._model, data.p, flux=data.flux, mode=mode,
                   zcf_mode=zcf_mode, apply_correction=apply_correction,
                   eom_iters=eom_iters)

    # --- Freezer contract ----------------------------------------------------
    @property
    def heavy_indices(self) -> Tuple[int, ...]:
        r"""
        Description:
        Every complex-structure modulus is heavy (the bulk moduli are slaved to
        :math:`\tau`; the conifold modulus is integrated out).

        Returns:
            tuple[int, ...]: All ``h12`` modulus indices.
        """
        return tuple(range(int(self.model.h12)))

    def solve_heavy(self, z_light: Array, tau: complex, fluxes: Array,
                    **kwargs) -> Array:
        r"""
        **Description:**
        The on-shell heavy moduli — here the full moduli vector, since every
        modulus is heavy (see :meth:`reconstruct_full_moduli`).

        Args:
            z_light (Array): Light moduli (empty at ``n_light = 0``).
            tau (complex): Axio-dilaton.
            fluxes (Array): Full flux vector.
            **kwargs: Forwarded to :meth:`reconstruct_full_moduli`.

        Returns:
            Array: Full complex modulus vector of length ``h12`` on-shell.
        """
        return self.reconstruct_full_moduli(z_light, tau, fluxes, **kwargs)

    def reconstruct_full_moduli(self, z_light: Array, tau: complex, fluxes: Array,
                                **kwargs) -> Array:
        r"""
        **Description:**
        Full moduli vector on the flat direction.  LCS: :math:`z = p\,\tau`.
        coniLCS: :math:`z_{\rm bulk} = \hat p\,\tau` with :math:`z_{\rm cf}`
        on-shell (analytic throat solve), conifold at index 0.

        Overrides the base index-scatter, which fails at ``n_light = 0``.
        With ``mode="eom"`` (both limits) the moduli are Newton-solved from their
        F-terms at fixed tau instead of the leading-order ansatz.

        Args:
            z_light (Array): Light moduli (empty at ``n_light = 0``).
            tau (complex): Axio-dilaton.
            fluxes (Array): Full flux vector.
            **kwargs: Forwarded to the ``z_cf`` solve where applicable.

        Returns:
            Array: Full complex modulus vector of length ``h12`` (conifold at
            index 0 for coniLCS).
        """
        if self.mode == "eom":
            z, _cz, _t, _ct = self.model._convert_real_to_complex(
                self._solve_eom(tau, fluxes))
            return z
        if not self._is_coni:
            return self.p * tau
        z_bulk = self.p * tau
        z_cf = self.model.compute_zcf(
            z_bulk, jnp.conj(z_bulk), tau, jnp.conj(tau), fluxes,
            mode=self._zcf_mode, apply_correction=self._apply_correction, conj=False)
        return jnp.concatenate([jnp.reshape(z_cf, (1,)), z_bulk])

    def DW_light(self, z_light: Array, z_light_c: Array, tau: complex,
                 tau_c: complex, fluxes: Array,
                 assume_conjugate: bool = False, **kwargs) -> Array:
        r"""
        **Description:**
        The single light F-term :math:`D_\tau W` on-shell (the base indexes an
        empty *float* array at ``n_light = 0`` -> ``TypeError``).

        .. note::
            This is the complex-coordinate covariant derivative.  For coniLCS the
            conjugate point is reconstructed with the same (``conj=False``)
            analytic z_cf solve at ``(z_light_c, tau_c)``; the physically
            validated F-terms use the real-coordinate :meth:`DW_x_light`
            (verified on-shell), which goes through :meth:`_real_light_to_full`.

        Args:
            z_light (Array): Light moduli (empty at ``n_light = 0``).
            z_light_c (Array): Complex conjugate of *z_light*.
            tau (complex): Axio-dilaton.
            tau_c (complex): Complex conjugate of *tau*.
            fluxes (Array): Full flux vector.
            assume_conjugate (bool, optional): Reuse
                :math:`\overline{z_{\rm full}}` instead of a second heavy solve --
                **evaluation only**, see the warning on
                :meth:`Freezer.DW_light`.  For ``mode="eom"`` this is the more
                valuable of the two savings, since each solve is a Newton
                iteration. Defaults to ``False``.
            **kwargs: Forwarded to :meth:`reconstruct_full_moduli`.

        Returns:
            Array: Length-1 array holding the light F-term :math:`D_\tau W`.
        """
        z_full = self.reconstruct_full_moduli(z_light, tau, fluxes, **kwargs)
        z_full_c = (jnp.conj(z_full) if assume_conjugate else
                    self.reconstruct_full_moduli(z_light_c, tau_c, fluxes,
                                                 **kwargs))
        DW_full = self.model.DW(z_full, z_full_c, tau, tau_c, fluxes)
        return jnp.array([DW_full[-1]])

    @property
    def _real_light_jacobian(self) -> Array:
        r"""
        Description:
        Constant "frozen" tangent :math:`\partial x_{\rm full}/\partial
        x_{\rm light} = \mathrm{kron}(v^T, \mathbb{1}_2)` with
        :math:`v = (\vec p, 1)` [LCS] or :math:`v = (0, \hat p, 1)` [coniLCS,
        conifold held fixed].  For coniLCS the true tangent is
        :math:`\tau`-dependent (z_cf is nonlinear); use
        ``ddV_x_light(reduction="tangent")`` (fast, exact at a vacuum) or
        ``"autodiff"`` (robust) for the reduced Hessian.

        Returns:
            Array: Constant real tangent of shape ``(2*h12+2, 2)``.
        """
        if self._is_coni:
            v = jnp.concatenate([jnp.zeros(1), self.p, jnp.ones(1)])
        else:
            v = jnp.append(self.p, 1.0)
        return jnp.kron(v[:, None], jnp.eye(2))

    def _onshell_tangent(self, x_light: Array, fluxes: Array,
                         x_full: Optional[Array] = None, **kwargs) -> Array:
        r"""
        **Description:**
        On-shell tangent :math:`J = \partial x_{\rm full}/\partial x_\tau` for the
        PFVEFT.  In ``mode="ansatz"`` the heavy solve is analytic, so the base
        forward-mode :func:`jax.jacfwd` is already cheap.  In ``mode="eom"`` the
        moduli are Newton-solved, so rather than differentiating through the loop
        this returns the **closed-form implicit-function tangent**: from the
        on-shell condition :math:`DW_x|_{\rm moduli} = 0` at fixed :math:`\tau`,

        .. math::
            \frac{\partial x_{\rm mod}}{\partial x_\tau}
              = -\bigl(\partial_{x_{\rm mod}} DW_x\bigr)^{-1}
                 \bigl(\partial_{x_\tau} DW_x\bigr)\, ,

        a single linear solve on the moduli block of ``dDW_x`` (the very block the
        Newton step already builds) — no autodiff through the solve.

        Because the right-hand side depends only on the *point*, passing ``x_full``
        (e.g. a stored vacuum, or the reconstruction :meth:`ddV_x_light` has
        already performed) skips the Newton solve altogether: the tangent then
        costs a single ``dDW_x`` evaluation plus one small linear solve.

        Args:
            x_light (Array): Real light coordinates ``[Re tau, Im tau]``.
            fluxes (Array): Full flux vector.
            x_full (Array, optional): Full real point with the moduli already
                on-shell.  If ``None`` it is reconstructed via
                :meth:`_real_light_to_full` (a Newton solve in ``mode="eom"``).
            **kwargs: Accepted for signature symmetry (unused).

        Returns:
            Array: Tangent of shape ``(2*h12 + 2, 2)``.
        """
        if self.mode != "eom":
            return super()._onshell_tangent(x_light, fluxes, x_full=x_full,
                                            **kwargs)
        n_z = 2 * int(self.model.h12)
        xf = (self._real_light_to_full(x_light, fluxes) if x_full is None
              else x_full)
        dDW = self.model.dDW_x(xf, fluxes)
        dxmod_dtau = -jnp.linalg.solve(dDW[:n_z, :n_z], dDW[:n_z, n_z:])
        return jnp.concatenate(
            [dxmod_dtau, jnp.eye(2, dtype=dxmod_dtau.dtype)], axis=0)

    def _real_light_to_full(self, x_light: Array, fluxes: Array,
                            **kwargs) -> Array:
        r"""
        **Description:**
        Real light coordinates ``[Re tau, Im tau]`` -> the full real array.
        ``mode="ansatz"`` returns the leading-order flat direction (see
        :meth:`_ansatz_real`); ``mode="eom"`` returns the Newton-solved moduli at
        fixed :math:`\tau` (see :meth:`_solve_eom`).

        Args:
            x_light (Array): Real light coordinates ``[Re tau, Im tau]``.
            fluxes (Array): Full flux vector.
            **kwargs: Accepted for signature symmetry (unused).

        Returns:
            Array: Full real coordinate vector of length ``2*h12 + 2``.
        """
        if self.mode == "eom":
            # Via the implicitly-differentiated wrapper: the Newton solve is never
            # differentiated *through*, so `reduction="autodiff"` and the
            # "autodiff" reduced metric cost a linear solve rather than a tape
            # over every iteration.
            return _eom_reconstruct(self, x_light, fluxes)
        tau = x_light[0] + 1j * x_light[1]
        return self._ansatz_real(tau, fluxes)

    def _ansatz_real(self, tau: complex, fluxes: Array) -> Array:
        r"""
        **Description:**
        Full real coordinate vector on the leading-order ansatz (both limits).
        LCS: ``[Re(p*tau), Im(p*tau), Re tau, Im tau]``.  coniLCS: the bulk-only
        real vector ``[Re(p_hat*tau), Im(p_hat*tau), Re tau, Im tau]`` with
        ``z_cf`` solved + prepended by ``model.zcf_handling`` (throat solve).

        Also the seed for :meth:`_solve_eom`.

        Args:
            tau (complex): Axio-dilaton.
            fluxes (Array): Full flux vector.

        Returns:
            Array: Full real coordinate vector of length ``2*h12 + 2`` on the
            leading-order ansatz.
        """
        if not self._is_coni:
            z = self.p * tau
            return self.model._convert_complex_to_real(z, jnp.conj(z), tau, jnp.conj(tau))
        z_bulk = self.p * tau
        x_bulk = self.model._convert_complex_to_real(
            z_bulk, jnp.conj(z_bulk), tau, jnp.conj(tau))   # bulk-only (length 2*h12)
        return self.model.zcf_handling(
            x_bulk, fluxes, mode=self._zcf_mode,
            apply_correction=self._apply_correction, conj=False)

    # --- EOM mode ------------------------------------------------------------
    @jax.jit
    def _solve_eom(self, tau: Array, fluxes: Array) -> Array:
        r"""
        **Description:**
        (``mode="eom"``, both limits) Newton-solve the moduli F-terms
        :math:`\partial_{x} W = 0` restricted to the ``2*h12`` moduli directions,
        at fixed :math:`\tau`, seeded at the leading-order ansatz
        (:meth:`_ansatz_real`).  For coniLCS the seed already places
        :math:`z_{\rm cf}` on its throat solve; the full-moduli Newton then treats
        :math:`z_{\rm cf}` as a genuine variable, so its coupling to the bulk is
        captured exactly (the ansatz throat solve is leading-order only).

        Uses a fixed-iteration :func:`jax.lax.fori_loop`, so it is jit-compatible
        AND differentiable — ``ddV_x_light(reduction="tangent")`` takes one
        forward-mode ``jax.jacfwd`` of this solve for the on-shell tangent (fast +
        exact at a vacuum), and ``reduction="autodiff"`` differentiates a full
        Hessian through it (robust off-shell, expensive).  The leading-order
        ``_real_light_jacobian`` is only the ansatz tangent, so prefer
        ``"tangent"`` / ``"autodiff"`` over ``"frozen"`` in ``mode="eom"``.

        Args:
            tau (complex): Axio-dilaton (held fixed during the solve).
            fluxes (Array): Full flux vector.

        Returns:
            Array: Full real coordinate vector of length ``2*h12 + 2`` with the
            moduli Newton-solved on-shell and the axio-dilaton block held fixed.
        """
        n_z = 2 * int(self.model.h12)
        x0 = self._ansatz_real(tau, fluxes)          # seed (throat z_cf for coni)
        tau_block = x0[n_z:]                          # fixed [Re tau, Im tau]

        def step(_, x):
            R = self.model.DW_x(x, fluxes)[:n_z]              # moduli F-terms
            J = self.model.dDW_x(x, fluxes)[:n_z, :n_z]       # moduli-moduli block
            xz = x[:n_z] - jnp.linalg.solve(J, R)
            return jnp.concatenate([xz, tau_block])

        return jax.lax.fori_loop(0, self._eom_iters, step, x0)

    # --- flux-binding safety -------------------------------------------------
    def flux_matches(self, fluxes: Array, *, atol: float = 0.0) -> bool:
        r"""
        **Description:**
        Whether ``fluxes`` equals the bound PFV flux.  The ansatz ``z = p*tau``
        ignores the per-call flux, so a mismatched flux silently mis-embeds; a
        caller can guard with this.

        Args:
            fluxes (Array): Flux vector to compare against the bound PFV flux.
            atol (float): Absolute tolerance for the comparison. Defaults to
                ``0.0`` (exact).

        Returns:
            bool: ``True`` iff *fluxes* matches the bound flux (or no flux is
            bound).
        """
        if self.flux is None:
            return True
        return bool(np.allclose(np.asarray(fluxes), np.asarray(self.flux),
                                rtol=0, atol=atol))
