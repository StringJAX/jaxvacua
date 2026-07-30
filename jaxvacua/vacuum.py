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

"""Vacuum data containers.

Purpose
-------
Lightweight, pure-data containers for flux-vacuum quantities.  This module
ships two layers:

* :class:`PFVData` — the light algebra layer of a perturbatively flat vacuum
  (PFV): it bundles the flux quanta ``(M, K)`` with the derived N-matrix,
  p-vector, full flux vector and the PFV-condition results, and exposes thin
  accessors that delegate to the model.
* :class:`Vacuum` and its subclass :class:`PFV` — the solver-state vacuum
  hierarchy.  A ``Vacuum`` means *a solved point in moduli space*: it is defined
  by its **flux** and its **location** (either the interleaved real vector ``x``
  or the pair ``(z, tau)``, whichever is missing being derived), and carries the
  solved diagnostics ``W0``, ``DW``, ``residual``, ``gs``.  The conifold
  quantities ``zcf`` and the bulk/conifold residual split are **gated on**
  :attr:`Vacuum.limit`: outside a conifold limit there is no conifold direction,
  so they are undefined rather than unknown and stay ``None``.
  :class:`PFV` composes a :class:`PFVData`.
* :class:`VacuumAnalysis` — the *derived* tier: the consistency-check report from
  :meth:`Vacuum.diagnostics`, the eigenvalues computed to produce it, and the
  conifold alignment scalar.  Everything in it is reproducible from the core
  record given a model, so ``to_dict(analysis=False)`` drops the whole tier for
  bulk ensembles while published datasets keep it.

Promotion provenance is deliberately absent: a vacuum from a plain Newton solve
has no promotion history, so ``genealogy`` / ``success`` / ``trajectory`` and
``is_solved()`` live on the ``afvs`` subclasses (``AFV``, ``PromotedPFV``),
alongside the orchestration that populates them (``promote`` / ``sample_afvs``).
``afvs`` depends on ``jaxvacua``; the arrow never points the other way, so this
module imports nothing from it — :func:`register_vacuum_kind` is how such
downstream types make themselves loadable here.

Design notes
------------
Every array-bearing dataclass here is declared ``eq=False`` because the
auto-generated dataclass ``__eq__`` raises ``ValueError`` on array-valued
fields; use the explicit array-aware :meth:`~PFVData.equals` instead.
:class:`PFVData` keeps a ``_model`` handle only for the convenience methods;
that handle is excluded from equality and dropped by :meth:`~PFVData.to_dict`,
so a serialised ``PFVData`` is a pure-data record (re-attach a model for
:meth:`~PFVData.moduli`).  The :class:`Vacuum` (de)serialisers walk fields
explicitly rather than using :func:`dataclasses.asdict`, which would recurse
into and deep-copy the composed finder held by ``PFVData._model``.

Three notions of sameness coexist, and the choice matters: :meth:`Vacuum.equals`
is exact and field-by-field, because its job is to prove a round trip lost
nothing; :meth:`Vacuum.equivalent_to` compares the *points* with the rounding of
:func:`~jaxvacua.flux_utils.dedup_key`, which is what solver output needs; given a
model it additionally identifies duality images.

Persistence is dict-first — never pickle an instance.  :func:`save_vacua` gzips
``to_dict()`` payloads, so a stored file survives a refactor of these classes;
:func:`vacuum_to_json` is the tier for data others download, since unpickling is
a remote-code-execution vector.
"""

import json
import math
import warnings
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

import numpy as np
import jax.numpy as jnp
from jax import Array

from .util import CONI_LIMITS, load_zipped_pickle, save_zipped_pickle

if TYPE_CHECKING:  # pragma: no cover - type hints only; avoids an import cycle
    from .flux_vacua_finder import FluxVacuaFinder

__all__ = [
    "PFVData", "Vacuum", "PFV",
    "VacuumAnalysis", "real_to_complex", "complex_to_real",
    "register_vacuum_kind", "conifold_alignment", "resolve_hyperplanes",
    "encode_json", "decode_json", "vacuum_to_json", "vacuum_from_json",
    "unique_vacua", "dedup_vacua", "save_vacua", "load_vacua",
]


# ----------------------------------------------------------------------------
# Display helpers
# ----------------------------------------------------------------------------
# ``None`` means "not applicable" (e.g. the conifold split of an LCS vacuum) and
# NaN means "not computed"; both render as an em dash rather than as a misleading
# number.  Module level so the one-line and multi-line summaries share them.

def _num_or_dash(v: Any, fmt: str = "{:.3e}") -> str:
    r"""
    **Description:**
    Format a real scalar, or an em dash when it is ``None``/NaN.

    Args:
        v (Any): Value to format.
        fmt (str, optional): Format spec. Defaults to ``"{:.3e}"``.

    Returns:
        str: Formatted value, or ``"—"``.
    """
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    return "—" if math.isnan(f) else fmt.format(f)


def _abs_or_dash(v: Any, fmt: str = "{:.3e}") -> str:
    r"""
    **Description:**
    Format ``|v|`` for a complex scalar, or an em dash when ``None``/NaN.

    Args:
        v (Any): Value whose magnitude is formatted.
        fmt (str, optional): Format spec. Defaults to ``"{:.3e}"``.

    Returns:
        str: Formatted magnitude, or ``"—"``.
    """
    if v is None:
        return "—"
    try:
        a = abs(complex(v))
    except (TypeError, ValueError):
        return "—"
    return "—" if math.isnan(a) else fmt.format(a)


#: Universal numerical factor in the alignment scalar (arXiv:2009.03312).
_ALIGNMENT_KAPPA = 89.5643


def conifold_alignment(model: Any, z: Any, zcf: complex, gs: float,
                       W0: complex, M0: float) -> Optional[float]:
    r"""
    **Description:**
    Dimensionless alignment scalar :math:`\Xi` of a (PFV-style) flux vacuum near
    the conifold.

    .. admonition:: Details
        :class: dropdown

        :math:`\Xi` quantifies how close a PFV sits to the idealised AdS-vacuum
        window:

        .. math::

            \Xi \;=\; \frac{2\,\kappa\,V_{\tilde X}^{\,1/3}\,(2Q)\,
                             |z_{\rm cf}|^{\,4/3}}
                            {(g_s\, M_0\, |W_0|)^{\,2}},

        with :math:`\kappa = 89.5643` (arXiv:2009.03312), :math:`V_{\tilde X}` the
        mirror-CY volume ``model.mirror_volume``, :math:`Q` the D3 tadpole cap
        ``model.Q()``, :math:`M_0` the leading :math:`M`-flux entry, :math:`g_s`
        the string coupling and :math:`W_0` the rescaled superpotential.  (The
        :math:`(2Q)` factor is ``(2Q)**1.5`` raised to the power ``2/3``, written
        out that way in the source to match its origin as a volume proxy.)

        Deliberately **total**: any malformed input, a model without
        ``mirror_volume``, or a missing / non-positive :math:`Q` returns ``None``
        rather than raising or producing ``NaN``, so it can be evaluated on
        partial seeds without breaking a diagnostics snapshot.  ``None`` (never
        ``NaN``) also keeps :meth:`Vacuum.equals` well-behaved.

    Args:
        model (Any): Model supplying ``mirror_volume`` and ``Q()``.
        z (Any): Complex moduli vector.
        zcf (complex): Conifold modulus :math:`z_{\rm cf}`.
        gs (float): String coupling.
        W0 (complex): Rescaled superpotential.
        M0 (float): Leading :math:`M`-flux entry.

    Returns:
        float or None: :math:`\Xi`, or ``None`` when undefined.
    """
    try:
        Q_fn = getattr(model, "Q", None)
        Q_raw = Q_fn() if callable(Q_fn) else Q_fn
        if Q_raw is None:
            return None
        Q = float(Q_raw)
        if not (Q > 0):
            return None
        vol_proxy = (2 * Q) ** 1.5
        za = jnp.asarray(z)
        Vtilde = complex(model.mirror_volume(za, jnp.conj(za))).real
        gsM = float(gs) * float(M0)
        xi = float(2 * _ALIGNMENT_KAPPA * (Vtilde ** (1 / 3))
                   * (vol_proxy ** (2 / 3)) * (abs(complex(zcf)) ** (4 / 3))
                   / (gsM * abs(complex(W0))) ** 2)
        return xi if np.isfinite(xi) else None
    except Exception:
        return None


def resolve_hyperplanes(model: Any) -> Optional[Any]:
    r"""
    **Description:**
    Locate a model's Kähler-cone hyperplane matrix, tolerating the three places
    it has historically lived.

    ``flux_utils.is_physical`` looks only at ``model.lcs_tree.hyperplanes`` and so
    silently skips the cone test on models that expose it elsewhere; the ``afvs``
    promotion pipeline grew a more forgiving resolver.  This is that resolver, in
    one place.

    Args:
        model (Any): A ``FluxEFT`` / ``FluxVacuaFinder`` (or ``None``).

    Returns:
        Any or None: The hyperplane matrix, or ``None`` when unavailable.
    """
    for obj, attr in ((model, "hyperplanes"),
                      (getattr(model, "lcs_tree", None), "hyperplanes"),
                      (getattr(model, "periods", None), "hyperplanes")):
        hp = getattr(obj, attr, None) if obj is not None else None
        if hp is not None:
            return hp
    return None


# ----------------------------------------------------------------------------
# Real <-> complex coordinate layout (model-free)
# ----------------------------------------------------------------------------
# The interleaved convention ``x = [Re z_1, Im z_1, ..., Re tau, Im tau]`` is the
# one JAXVacua's solvers consume.  These two helpers are the single, model-free
# statement of it: ``FluxEFT._convert_real_to_complex`` is the jitted, model-bound
# equivalent, and both ``Vacuum.from_dict`` and StringForge's finder-free vacuum
# adapter route through here rather than re-deriving the layout.

def real_to_complex(x: Any) -> tuple:
    r"""
    **Description:**
    Split an interleaved real coordinate vector into complex moduli and
    axio-dilaton.

    The inverse of :func:`complex_to_real`, and the model-free counterpart of
    ``FluxEFT._convert_real_to_complex`` (which is jitted and additionally
    returns the conjugates).

    Args:
        x (Any): Real vector ``[Re z_1, Im z_1, ..., Re tau, Im tau]`` of length
            ``2*(h12+1)``.

    Returns:
        tuple: ``(z, tau)`` -- a complex NumPy array of length ``h12`` and a
        Python ``complex``.
    """
    xa = np.asarray(x, dtype=float).ravel()
    if xa.size < 2 or xa.size % 2:
        raise ValueError(
            "`x` must have even length >= 2 ([Re z, Im z, ..., Re tau, Im tau]), "
            f"got size {xa.size}."
        )
    z = xa[0:-2:2] + 1j * xa[1:-2:2]
    tau = complex(xa[-2], xa[-1])
    return z, tau


def complex_to_real(z: Any, tau: complex) -> np.ndarray:
    r"""
    **Description:**
    Pack complex moduli and axio-dilaton into the interleaved real vector.

    The inverse of :func:`real_to_complex`.

    Args:
        z (Any): Complex moduli, length ``h12`` (an empty array is allowed, e.g.
            a ``PFVEFT`` with ``n_light = 0``).
        tau (complex): Axio-dilaton.

    Returns:
        np.ndarray: Real vector of length ``2*(len(z)+1)``.
    """
    za = np.asarray(z, dtype=complex).ravel()
    t = complex(tau)
    x = np.empty(2 * za.size + 2, dtype=float)
    x[0:-2:2], x[1:-2:2] = za.real, za.imag
    x[-2], x[-1] = t.real, t.imag
    return x


# ----------------------------------------------------------------------------
# Tagged-JSON codec (the pickle-free storage tier)
# ----------------------------------------------------------------------------
# ``save_vacua`` gzips *pickled dicts*, which is fine locally but unacceptable for
# data other people download: unpickling is a remote-code-execution vector.  This
# codec is the public/vault tier -- JSON with two tags so NumPy arrays and complex
# scalars survive a round trip losslessly, while staying human-inspectable and
# diffable.  It lives here (rather than privately in StringForge, where it grew)
# so both repos share one implementation.

#: Tag marking an encoded ``np.ndarray`` (payload: nested lists plus ``dtype``).
_ARRAY_TAG = "__nd__"
#: Tag marking an encoded complex scalar (payload: ``[re, im]``).
_COMPLEX_TAG = "__c__"


def encode_json(obj: Any) -> Any:
    r"""
    **Description:**
    Recursively encode a :meth:`Vacuum.to_dict` payload into JSON-safe data.

    .. admonition:: Details
        :class: dropdown

        ``np.ndarray`` becomes ``{"__nd__": <nested list>, "dtype": <str>}`` and
        complex scalars become ``{"__c__": [re, im]}``; NumPy scalars are demoted
        to Python scalars; ``dict``/``list``/``tuple`` recurse.  Anything else that
        is not JSON-native is ``str()``-ed -- mirroring ``to_dict``'s
        exception-to-string handling, so a stray ``metadata`` value can never break
        a write.  Note that ``tuple`` round-trips as ``list`` (JSON has no tuple),
        which is why :meth:`Vacuum.diagnostics` reports are compared by content and
        not by type after a reload.

    Args:
        obj (Any): The (possibly nested) value to encode.

    Returns:
        Any: A JSON-serialisable representation.
    """
    if isinstance(obj, (np.ndarray, jnp.ndarray)):
        arr = np.asarray(obj)
        # Recurse into ``tolist()`` so complex elements get tagged rather than
        # left un-serialisable.
        return {_ARRAY_TAG: encode_json(arr.tolist()), "dtype": str(arr.dtype)}
    if isinstance(obj, (complex, np.complexfloating)):
        return {_COMPLEX_TAG: [float(obj.real), float(obj.imag)]}
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): encode_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [encode_json(v) for v in obj]
    if obj is None or isinstance(obj, (int, float, str)):
        return obj
    return str(obj)


def decode_json(obj: Any) -> Any:
    r"""
    **Description:**
    Inverse of :func:`encode_json`: restore NumPy arrays and complex scalars so
    :meth:`Vacuum.from_dict` re-hydrates unchanged.

    Args:
        obj (Any): A value produced by :func:`encode_json`.

    Returns:
        Any: The decoded value.
    """
    if isinstance(obj, dict):
        if _ARRAY_TAG in obj:
            # Decode the (possibly complex-tagged) nested list first, then rebuild
            # the array with its original dtype.
            return np.array(decode_json(obj[_ARRAY_TAG]), dtype=obj.get("dtype"))
        if _COMPLEX_TAG in obj:
            re_, im_ = obj[_COMPLEX_TAG]
            return complex(re_, im_)
        return {k: decode_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode_json(v) for v in obj]
    return obj


#: Maps the ``_kind`` discriminator written by ``to_dict`` back to a class.
#: ``afvs`` extends this with its own types via :func:`register_vacuum_kind`, which
#: is why a stored ``AFV`` no longer silently degrades to a base ``Vacuum``.
_VACUUM_KINDS: Dict[str, type] = {}


def register_vacuum_kind(cls: type) -> type:
    r"""
    **Description:**
    Register a :class:`Vacuum` subclass so :meth:`Vacuum.from_dict` can rebuild it
    from its ``_kind`` tag.  Usable as a class decorator.

    Downstream packages (``afvs``) call this at import time; without it their
    records deserialise as a base ``Vacuum``, losing both the type and any
    subclass-only fields.

    Args:
        cls (type): The subclass to register (keyed on ``cls.__name__``).

    Returns:
        type: ``cls`` unchanged, so this can be used as a decorator.
    """
    _VACUUM_KINDS[cls.__name__] = cls
    return cls


def _warn_if_pfv_conditions_fail(data: Any) -> None:
    r"""
    **Description:**
    Warn when a reconstructed :class:`PFVData` carries stored PFV conditions that
    do not hold -- i.e. a record labelled a PFV that is not one.

    A warning rather than an exception on purpose: raising here would make a bad
    record unreadable, whereas a warning plus a ``False`` from
    :meth:`Vacuum.is_consistent` is actionable.

    Args:
        data (Any): A :class:`PFVData` (or anything else, which is ignored).

    Returns:
        None
    """
    cond = getattr(data, "conditions", None)
    if not cond:
        return
    bad = []
    for name, entry in cond.items():
        if name == "p":          # a value, not a verdict (same filter as NB19)
            continue
        try:
            if not bool(np.asarray(entry[0]).all()):
                bad.append(name)
        except Exception:         # not a (ok, value) pair -- skip quietly
            continue
    if bad:
        warnings.warn(
            "Reconstructed PFV carries failing PFV conditions: "
            f"{', '.join(bad)}. It is labelled a PFV but does not satisfy the "
            "PFV algebra; check `diagnostics(model)`.",
            RuntimeWarning, stacklevel=3)


@dataclass(eq=False)
class VacuumAnalysis:
    r"""
    **Description:**
    Derived quantities for a :class:`Vacuum` -- the consistency-check report and
    whatever eigenvalues were computed to produce it.

    .. admonition:: Why this is a separate record
        :class: dropdown

        Everything here is *reproducible* from the core vacuum (flux + location)
        given a model, so it is optional payload rather than identity.  Grouping it
        lets an entire tier be dropped with one switch: ``to_dict(analysis=False)``
        for bulk ensembles pushed to the vault, the default for published datasets
        where the mass spectrum should travel with the record.

    Args:
        checks (dict, optional): ``{name: (ok, value, reason)}`` as returned by
            :meth:`Vacuum.diagnostics`.
        kahler_metric_eigenvalues (Array, optional): Eigenvalues of the Kähler
            metric, from the ``kahler_metric_pd`` check.
        hessian_eigenvalues (Array, optional): Real-basis Hessian eigenvalues,
            populated only when ``stability=True``.
        mass_eigenvalues (Array, optional): Physical mass-squared eigenvalues.
            **Producer-set**: :meth:`Vacuum.diagnostics` never computes these, so
            the field stays ``None`` unless a caller attaches a spectrum it
            obtained elsewhere (e.g. ``Freezer.light_mass_spectrum``).  The
            omission is deliberate -- ``mass_matrix`` has open defects in its
            ``mode=None`` / ``"SUSY"`` branches (see ``worklog/ERRORS.md``), so
            stability is checked through ``classify_solution`` instead and
            reported as ``hessian_eigenvalues``.
        alignment (float, optional): Conifold alignment scalar :math:`\Xi`.
            Requires a model **and** PFV quantum numbers, hence derived here rather
            than stored as a core field.
        args (dict, optional): The tolerances actually used, so a cached report can
            be invalidated when they change.
    """

    checks: Optional[Dict[str, Any]] = None
    kahler_metric_eigenvalues: Optional[Array] = None
    hessian_eigenvalues: Optional[Array] = None
    mass_eigenvalues: Optional[Array] = None
    alignment: Optional[float] = None
    args: Optional[Dict[str, Any]] = None

    def summary(self) -> str:
        r"""
        **Description:**
        One-line digest -- how many checks passed, and which are failing.

        Returns:
            str: e.g. ``"7/8 ok (failing: tadpole)"``, or ``"not computed"``.
        """
        if not self.checks:
            return "not computed"
        graded = [(k, v) for k, v in self.checks.items()
                  if not str(v[2]).startswith("skipped")]
        bad = [k for k, v in graded if not bool(v[0])]
        head = f"{len(graded) - len(bad)}/{len(graded)} ok"
        return head if not bad else f"{head} (failing: {', '.join(bad)})"

    def to_dict(self) -> Dict[str, Any]:
        r"""
        **Description:**
        Plain-data form; arrays become NumPy so the record is picklable and
        JSON-encodable.

        Returns:
            dict: Serialisable mapping of the populated entries.
        """
        out: Dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if v is None:
                continue
            out[f.name] = (np.asarray(v)
                           if isinstance(v, (jnp.ndarray, np.ndarray)) else v)
        return out

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VacuumAnalysis":
        r"""
        **Description:**
        Rebuild from :meth:`to_dict` output, ignoring unknown keys.

        Args:
            d (dict): Mapping produced by :meth:`to_dict`.

        Returns:
            VacuumAnalysis: The reconstructed record.
        """
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in dict(d).items() if k in valid})


@dataclass(eq=False)
class PFVData:
    r"""
    **Description:**
    Light, pure-data container for a perturbatively flat vacuum (PFV): the flux
    quanta ``(M, K)`` and the derived algebra (N-matrix, p-vector, full flux
    vector, PFV-condition results).  No solver state.

    Construct with :meth:`from_fluxes` (or ``model.pfv_data(M, K)``).
    """

    M: Any
    K: Any
    N: Any = None
    p: Any = None
    flux: Any = None
    conditions: Optional[dict] = None
    _model: Any = field(default=None, repr=False)

    @classmethod
    def from_fluxes(cls, model, M, K):
        r"""
        **Description:**
        Build a :class:`PFVData` from a model and integer flux quanta ``(M, K)``.

        Args:
            model (FluxEFT): Model providing ``N_matrix``, ``pfv_conditions``,
                ``pfv_p_vector``, ``pfv_to_flux``.
            M (Array): M-vector.
            K (Array): K-vector.

        Returns:
            PFVData: The populated container.  ``p`` is ``None`` when ``N`` is
            singular (i.e. the flux does not define a PFV).
        """
        M = jnp.asarray(M, dtype=float)
        K = jnp.asarray(K, dtype=float)
        N = model.N_matrix(M)
        conditions = model.pfv_conditions(M, K)
        det_ok = bool(conditions["det N!=0"][0])  # concrete jnp bool -> Python bool
        p = model.pfv_p_vector(M, K) if det_ok else None
        flux = model.pfv_to_flux(M, K)
        return cls(M=M, K=K, N=N, p=p, flux=flux, conditions=conditions, _model=model)

    # --- accessors (model-independent after construction, except moduli) ------
    def to_flux(self):
        r"""Description: Full flux vector implied by ``(M, K)``."""
        return self.flux

    def check(self):
        r"""Description: The PFV-condition results (see ``pfv_conditions``)."""
        return self.conditions

    def moduli(self, tau):
        r"""
        **Description:**
        Complex-structure moduli on the flat direction at axio-dilaton ``tau``.

        Args:
            tau (complex): Axio-dilaton value.

        Returns:
            Array: Moduli ``z`` (delegates to ``model.pfv_to_moduli``).
        """
        if self._model is None:
            raise RuntimeError(
                "PFVData has no attached model; re-attach a finder to compute moduli."
            )
        return self._model.pfv_to_moduli(self.M, self.K, tau)

    # --- serialization: explicit field walk; drop _model; jax -> numpy --------
    def to_dict(self):
        r"""Description: Pure-data ``dict`` (drops the ``_model`` handle; casts
        arrays to NumPy)."""
        def _np(x):
            return None if x is None else np.asarray(x)
        conditions = None
        if self.conditions is not None:
            conditions = {k: (_np(v[0]), _np(v[1])) for k, v in self.conditions.items()}
        return {
            "M": _np(self.M), "K": _np(self.K), "N": _np(self.N),
            "p": _np(self.p), "flux": _np(self.flux), "conditions": conditions,
        }

    @classmethod
    def from_dict(cls, d):
        r"""Description: Rebuild from :meth:`to_dict` output (no ``_model``;
        re-attach a model for :meth:`moduli`)."""
        def _jx(x):
            return None if x is None else jnp.asarray(x)
        return cls(
            M=jnp.asarray(d["M"], dtype=float), K=jnp.asarray(d["K"], dtype=float),
            N=_jx(d.get("N")), p=_jx(d.get("p")), flux=_jx(d.get("flux")),
            conditions=d.get("conditions"), _model=None,
        )

    # --- array-aware equality (the auto __eq__ would raise on array fields) ---
    def equals(self, other):
        r"""Description: Array-aware equality on the flux quanta ``(M, K)``."""
        if not isinstance(other, PFVData):
            return NotImplemented
        return bool(
            np.array_equal(np.asarray(self.M), np.asarray(other.M))
            and np.array_equal(np.asarray(self.K), np.asarray(other.K))
        )

    def __repr__(self):
        det = None
        if self.conditions is not None and "det N!=0" in self.conditions:
            det = float(np.asarray(self.conditions["det N!=0"][1]))
        return (f"PFVData(M={np.asarray(self.M).tolist()}, "
                f"K={np.asarray(self.K).tolist()}, detN={det})")


#: Bumped when the stored layout changes, so a future field move can be
#: migrated rather than guessed at.  Written by ``to_dict``, ignored by
#: ``from_dict`` today (nothing to migrate yet).
_SCHEMA_VERSION = 2

#: Fields derived from the core record: excluded from equality (they are payload,
#: not identity) and from ``to_dict`` where re-derivable.
_DERIVED_FIELDS = ("z", "tau", "analysis")


# Fields stored as JAX arrays that must be re-cast to ``jax.Array`` on
# deserialisation (the ``to_dict`` step downcasts them to NumPy).  A subclass with
# further array fields lists them in a ``_extra_array_fields`` class attribute
# (a ``ClassVar``, so the dataclass machinery ignores it) rather than overriding
# ``from_dict``.
_ARRAY_FIELDS = ("x", "flux", "DW")


@register_vacuum_kind
@dataclass(eq=False)
class Vacuum:
    r"""
    **Description:**
    Solver-state flux vacuum: real coordinates ``x``, the full flux vector and
    the solved diagnostics.  Base class of :class:`PFV` (and the private
    ``afvs.AFV`` proxy-seed type).

    .. admonition:: Details
        :class: dropdown

        A ``Vacuum`` stores only the real-coordinate representation of the
        moduli/axio-dilaton (``x``) and the integer flux vector, not a model;
        the duality-equivalence helpers (:meth:`canonical_key`,
        :meth:`equivalent_to`), :meth:`diagnostics` and :meth:`is_consistent`
        therefore all take an explicit model argument.  The solved diagnostic
        fields (``W0``, ``DW``, ``residual``, ``gs``) are set by whichever solver
        produced the point and default to ``NaN`` when not recorded; a ``NaN``
        ``residual`` makes the ``residual`` check report *skipped*, never *passed*.

    Args:
        flux (Array): Full integer flux vector of length
            :math:`2 n_{\text{fluxes}}`.  **Required** -- a vacuum is defined by
            its flux quanta and its location.
        x (Array, optional): Full real coordinate vector (moduli axions/saxions
            plus the axio-dilaton), as consumed by ``FluxVacuaFinder`` solvers.
            Required *unless* both ``z`` and ``tau`` are given, from which it is
            derived (see :meth:`__post_init__`).
        z (Array, optional): Complex-structure moduli -- an alternative to ``x``.
            Always populated after construction.
        tau (complex, optional): Axio-dilaton -- an alternative to ``x``.  Always
            populated after construction.
        limit (str): Moduli-space limit the vacuum lives in (``"LCS"``,
            ``"coniLCS"``, ...).  Gates the conifold fields below -- see
            :attr:`has_conifold`.
        W0 (complex): Flux superpotential :math:`W_0` at the solved point, in the
            :math:`\sqrt{2/\pi}` normalisation of
            ``FluxEFT.superpotential(..., normalise=True)`` (the same convention
            ``pfv_racetrack`` reports).
        DW (Array, optional): F-term vector :math:`D W`.
        residual (float): Solver residual :math:`\max|DW|`.
        gs (float): String coupling :math:`g_s = 1/\mathrm{Im}\,\tau`.
        metadata (dict): Free-form key/value store.
        residual_bulk (float, optional): Residual restricted to the bulk
            directions.  **Conifold limits only** -- ``None`` otherwise.
        residual_conifold (float, optional): Residual restricted to the conifold
            direction.  **Conifold limits only** -- ``None`` otherwise.
        zcf (complex, optional): Conifold modulus :math:`z_{\text{cf}}`.
            **Conifold limits only** -- ``None`` otherwise.
        analysis (VacuumAnalysis, optional): Derived quantities produced by
            :meth:`diagnostics` (check report, eigenvalues, alignment).  ``None``
            until computed; droppable as a whole via ``to_dict(analysis=False)``
            so bulk ensembles stay small.

    .. note::
        Promotion provenance (``genealogy``, ``success``, ``trajectory``) is
        **not** part of a core vacuum -- a vacuum from a plain Newton solve has
        no promotion history.  Those fields live on the ``afvs`` subclasses
        (``afvs.AFV``, ``afvs.PromotedPFV``), together with ``is_solved()``.
    """

    # --- core: what a solver produces -------------------------------------
    # A vacuum is defined by its FLUX and its LOCATION, so ``flux`` is a required
    # field (no default -- omitting it is a TypeError, not a silent NaN record).
    # The location may be given either as the interleaved real vector ``x`` or as
    # ``(z, tau)``; whichever is missing is derived in ``__post_init__``, with
    # ``x`` canonical since it is what the solvers consume.  ``flux`` must come
    # first: dataclass fields without defaults precede those with.
    flux: Array
    x: Optional[Array] = None
    z: Optional[Any] = None
    tau: Optional[complex] = None
    limit: str = "LCS"
    W0: complex = complex("nan")
    DW: Optional[Array] = None
    residual: float = float("nan")
    gs: float = float("nan")
    metadata: Dict[str, Any] = field(default_factory=dict)

    # --- conifold-only: meaningful iff ``limit`` is a conifold limit -------
    # ``None`` (not NaN) means "not applicable": NaN would compare unequal to
    # itself and has already broken ``equals`` once (see afvs ``alignment=nan``).
    residual_bulk: Optional[float] = None
    residual_conifold: Optional[float] = None
    zcf: Optional[complex] = None

    # --- derived, computed on demand by ``diagnostics(model)`` -------------
    analysis: Optional["VacuumAnalysis"] = None

    def __post_init__(self) -> None:
        r"""
        **Description:**
        Enforce the defining content of a vacuum and reconcile its two
        coordinate representations.

        A vacuum requires a **flux** and a **location**.  The location may be
        supplied either as the interleaved real vector ``x`` or as ``(z, tau)``;
        the missing one is derived here so both are always populated and
        consistent.  ``x`` is canonical: when both are given, ``z``/``tau`` are
        re-derived from it (and a mismatch raises rather than silently
        preferring one).

        Returns:
            None

        Raises:
            ValueError: If ``flux`` is ``None``, if neither ``x`` nor a complete
                ``(z, tau)`` pair is given, if ``x`` has an invalid length, or if
                ``x`` and ``(z, tau)`` disagree.
        """
        if self.flux is None:
            # ``flux`` has no default, so omitting it is already a TypeError;
            # this catches an explicit ``flux=None``.
            raise ValueError(
                f"{type(self).__name__} requires `flux`: a vacuum is defined by "
                "its flux quanta and its location in moduli space."
            )
        if self.x is None:
            if self.z is None or self.tau is None:
                raise ValueError(
                    f"{type(self).__name__} requires a location: pass either `x` "
                    "(the interleaved real vector) or both `z` and `tau`."
                )
            self.x = jnp.asarray(complex_to_real(self.z, self.tau))
            self.z, self.tau = real_to_complex(self.x)
            return
        z_from_x, tau_from_x = real_to_complex(self.x)   # also validates length
        if self.z is not None or self.tau is not None:
            z_given = z_from_x if self.z is None else np.asarray(
                self.z, dtype=complex).ravel()
            tau_given = tau_from_x if self.tau is None else complex(self.tau)
            if (z_given.shape != z_from_x.shape
                    or not np.allclose(z_given, z_from_x, rtol=1e-9, atol=1e-12)
                    or not np.isclose(tau_given, tau_from_x,
                                      rtol=1e-9, atol=1e-12)):
                raise ValueError(
                    "`x` and `(z, tau)` describe different points; pass only one "
                    "(`x` is canonical, so omit `z`/`tau` unless `x` is omitted)."
                )
        self.z, self.tau = z_from_x, tau_from_x

    @property
    def has_conifold(self) -> bool:
        r"""
        Description:
        Whether this vacuum's :attr:`limit` carries a conifold modulus -- i.e.
        whether :attr:`zcf`, :attr:`residual_bulk` and :attr:`residual_conifold`
        are meaningful at all.

        For a plain LCS vacuum there is no conifold direction, so the bulk /
        conifold split of the residual is not merely unknown but undefined; those
        fields stay ``None`` and are omitted from the summaries.

        Returns:
            bool: ``True`` for the limits in :data:`jaxvacua.util.CONI_LIMITS`.
        """
        return str(self.limit) in CONI_LIMITS

    # -- physical equivalence (tolerant; optionally up to duality) ------------
    def canonical_key(self, finder: Optional["FluxVacuaFinder"] = None, *,
                      n_digits: int = 6) -> tuple:
        r"""
        **Description:**
        Hashable key identifying this vacuum **up to numerical noise**, and -- when
        a model is supplied -- up to :math:`SL(2, \mathbb{Z}) \times` monodromy.

        .. admonition:: Details
            :class: dropdown

            The location is reduced to a hashable tuple by
            :func:`jaxvacua.flux_utils.dedup_key`, which rounds the moduli and
            :math:`\tau` to ``n_digits`` decimals and the flux to integers -- so
            this key, unlike :meth:`equals`, absorbs the last-digit differences
            between two independent solves of the same vacuum.

            With a ``finder`` the coordinates are first mapped into the
            fundamental domain by :func:`jaxvacua.flux_utils.map_to_fd`, so
            duality images collapse onto one key.  **Without** one, only the point
            itself is keyed: images related by :math:`\tau \to \tau + 1` (say) get
            *different* keys.  The model-free form exists because a ``Vacuum``
            stores no model and records reloaded from the vault are finder-free.

            For a coniLCS vacuum ``map_to_fd`` leaves the conifold modulus
            untouched, so conifold-direction monodromy images are not identified
            either way.

        Args:
            finder (FluxVacuaFinder, optional): Model supplying the
                fundamental-domain map.  Omit for the model-free, same-point key.
            n_digits (int, optional): Decimals retained for the continuous
                coordinates. Defaults to ``6``.

        Returns:
            tuple: Hashable key.
        """
        from .flux_utils import dedup_key, map_to_fd

        # Model-free split of the interleaved vector; verified equivalent to the
        # jitted ``finder._convert_real_to_complex``.
        z, tau = real_to_complex(self.x)
        if finder is None:
            return dedup_key(z, tau, self.flux, n_digits=n_digits)
        moduli_fd, tau_fd, flux_fd = map_to_fd(
            finder, jnp.asarray(z), tau, self.flux, enabled=True)
        return dedup_key(moduli_fd, tau_fd, flux_fd, n_digits=n_digits)

    def equivalent_to(self, other: "Vacuum",
                      finder: Optional["FluxVacuaFinder"] = None, *,
                      n_digits: int = 6) -> bool:
        r"""
        **Description:**
        Whether two vacua are the same vacuum, comparing with a tolerance.

        .. admonition:: Details
            :class: dropdown

            Three notions of sameness live on this class; choose deliberately:

            ==============================  =====================================  =======
            call                            identifies                             model
            ==============================  =====================================  =======
            ``a.equals(b)``                 *records*, every field, **exactly**    no
            ``a.equivalent_to(b)``          the same **point** (rounded)           no
            ``a.equivalent_to(b, finder)``  points related by **duality**          yes
            ==============================  =====================================  =======

            :meth:`equals` is bit-exact by design -- its purpose is to prove that
            a serialisation round trip or a parity re-run lost *nothing*, which a
            tolerance would silently defeat.  For solver output, where the last
            digits always differ, use this method.

            Note the two vacua need not be the same subclass: a base ``Vacuum``
            re-solved from scratch and a promoted ``PFV`` at the same point with
            the same flux are the same vacuum.  This is a rounded (binned)
            comparison, so a pair straddling a rounding boundary can miss; widen
            ``n_digits`` if that matters.

        Args:
            other (Vacuum): Vacuum to compare against.
            finder (FluxVacuaFinder, optional): Model supplying the transforms.
                Omit to compare the points directly, without duality.
            n_digits (int, optional): Decimals retained. Defaults to ``6``.

        Returns:
            bool: Whether the two vacua agree.
        """
        return (self.canonical_key(finder, n_digits=n_digits)
                == other.canonical_key(finder, n_digits=n_digits))

    # -- serialisation (explicit field walk; never asdict-recurses a finder) --
    def to_dict(self, *, analysis: bool = True) -> Dict[str, Any]:
        r"""
        **Description:**
        Serialise to a pure-data ``dict`` with a ``"_kind"`` discriminator.

        .. admonition:: Details
            :class: dropdown

            Fields are walked explicitly rather than via
            :func:`dataclasses.asdict`, which would recurse into and deep-copy
            the finder held by a composed :class:`PFVData`.  A composed
            ``.data`` is serialised through :meth:`PFVData.to_dict`, JAX arrays
            are downcast to NumPy, and any exception stored in ``metadata`` is
            replaced by its string message so the payload stays picklable.

        Args:
            analysis (bool, optional): Include the derived :attr:`analysis` block.
                Pass ``False`` for bulk ensembles (the vault tier): everything in
                ``analysis`` is reproducible from ``flux`` + location given a
                model, so it is payload rather than identity. Defaults to ``True``.

        Returns:
            dict: Serialisable representation (see :meth:`from_dict`).
        """
        d: Dict[str, Any] = {"_kind": type(self).__name__,
                             "_schema_version": _SCHEMA_VERSION}
        for f in fields(self):
            if f.name in ("z", "tau"):
                # Derived from ``x`` in ``__post_init__``; storing them would
                # duplicate the location and could go stale.
                continue
            v = getattr(self, f.name)
            if f.name == "analysis":
                if v is None or not analysis:
                    continue
                d[f.name] = v.to_dict()
            elif isinstance(v, PFVData):
                d[f.name] = v.to_dict()
            elif isinstance(v, (jnp.ndarray, np.ndarray)):
                d[f.name] = np.asarray(v)
            elif f.name == "metadata" and isinstance(v, dict):
                d[f.name] = {k: (str(x) if isinstance(x, BaseException) else x)
                             for k, x in v.items()}
            else:
                d[f.name] = v
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Vacuum":
        r"""
        **Description:**
        Rebuild a :class:`Vacuum` (or the correct subclass) from
        :meth:`to_dict` output.

        .. admonition:: Details
            :class: dropdown

            The ``_kind`` tag is looked up in the subclass registry (see
            :func:`register_vacuum_kind`), so a stored ``PFV`` -- or a downstream
            type such as ``afvs.AFV`` once ``afvs`` has been imported -- rebuilds
            as itself, with its subclass-only fields.  An *unregistered* kind
            degrades to a base :class:`Vacuum` with a ``RuntimeWarning`` rather
            than failing the load.

            A minimal ``{moduli, tau, flux}`` payload (without a full real ``x``)
            is also accepted -- ``x`` is built via :func:`complex_to_real`.
            Unknown keys are dropped, a composed ``analysis``/``data`` sub-dict is
            rebuilt into its record class, and the array fields are re-cast to
            :class:`jax.Array`.

        Args:
            d (dict): Payload produced by :meth:`to_dict`.

        Returns:
            Vacuum: Reconstructed instance of the appropriate subclass.
        """
        d = dict(d)
        d.pop("_schema_version", None)
        kind = d.pop("_kind", None)
        # Registry lookup: ``afvs`` registers its own types on import, so a stored
        # ``AFV``/``PromotedPFV`` rebuilds faithfully when afvs is available and
        # degrades to a base ``Vacuum`` (with a warning) when it is not.
        target = _VACUUM_KINDS.get(kind) if kind else Vacuum
        if target is None:
            warnings.warn(
                f"Unknown vacuum kind {kind!r}; rebuilding as a base Vacuum. "
                "If this is an afvs record, import afvs first so it can register "
                "its types.", RuntimeWarning, stacklevel=2)
            target = Vacuum
        # Permissive minimal input: accept ``{moduli, tau, flux}`` (complex form)
        # without a full real ``x`` by building ``x`` from the fixed real/imag
        # interleaving (model-independent, matching ``_convert_real_to_complex``).
        if "x" not in d and "moduli" in d and "tau" in d:
            # Minimal payload: derive the canonical interleaved vector.
            d["x"] = complex_to_real(d.pop("moduli"), d.pop("tau"))
        # ``z``/``tau`` are derived in ``__post_init__``, so a stored copy would be
        # redundant; drop them rather than risk a stale pair overriding ``x``.
        if "x" in d:
            d.pop("z", None)
            d.pop("tau", None)

        valid = {f.name for f in fields(target)}
        kwargs = {k: v for k, v in d.items() if k in valid}
        if "analysis" in kwargs and isinstance(kwargs["analysis"], dict):
            kwargs["analysis"] = VacuumAnalysis.from_dict(kwargs["analysis"])
        if "data" in kwargs and isinstance(kwargs["data"], dict):
            kwargs["data"] = PFVData.from_dict(kwargs["data"])
        # Base array fields plus any the subclass declares, so a downstream type
        # round-trips its own arrays without re-implementing this method.
        for k in _ARRAY_FIELDS + tuple(getattr(target, "_extra_array_fields", ())):
            if k in kwargs and isinstance(kwargs[k], np.ndarray):
                kwargs[k] = jnp.asarray(kwargs[k])
        _warn_if_pfv_conditions_fail(kwargs.get("data"))
        return target(**kwargs)

    # -- array-aware equality (the auto __eq__ would raise on array fields) ---
    def equals(self, other: "Vacuum") -> bool:
        r"""
        **Description:**
        Array-aware field-by-field equality (the dataclass ``__eq__`` is
        disabled because it raises on array fields).

        Args:
            other (Vacuum): Vacuum to compare against.

        Returns:
            bool: ``True`` iff both are the same subclass with equal fields
            (``NaN`` defaults compare equal; arrays via ``np.array_equal``).
        """
        if type(self) is not type(other):
            return False
        for f in fields(self):
            if f.name in _DERIVED_FIELDS:
                # Derived payload, not identity: two vacua at the same point with
                # the same flux are the same vacuum whether or not one of them
                # happens to carry a cached diagnostics report.  (``z``/``tau``
                # are re-derived from ``x``, which *is* compared.)
                continue
            a, b = getattr(self, f.name), getattr(other, f.name)
            if isinstance(a, PFVData) or isinstance(b, PFVData):
                if not (isinstance(a, PFVData) and isinstance(b, PFVData) and a.equals(b)):
                    return False
            elif isinstance(a, (jnp.ndarray, np.ndarray)) or isinstance(b, (jnp.ndarray, np.ndarray)):
                if a is None or b is None or not np.array_equal(np.asarray(a), np.asarray(b)):
                    return False
            elif isinstance(a, (float, complex)) and isinstance(b, (float, complex)):
                if np.isnan(a) and np.isnan(b):
                    continue
                if a != b:
                    return False
            elif a != b:
                return False
        return True

    # -- consistency checks ----------------------------------------------------
    def diagnostics(self, model, *, moduli_max: Optional[float] = None,
                    s_min: Optional[float] = None, residual_tol: float = 1e-8,
                    stability: bool = False,
                    refresh: bool = False) -> Dict[str, tuple]:
        r"""
        **Description:**
        Run the consistency checks and return a **named, explained** report:
        ``{check_name: (ok, value, reason)}``.

        .. admonition:: Details
            :class: dropdown

            This is the structured counterpart of :func:`jaxvacua.flux_utils.is_physical`,
            which returns a bare ``bool`` and names the failing check only through
            ``print(verbose=True)`` -- so callers cannot act on *why* a point was
            rejected.  Here every check reports its own verdict, the underlying
            value, and a human-readable reason when it fails.

            **Skipped checks.** A check whose inputs are unavailable (no sampler
            for the dilaton floor, no ``Q`` for the tadpole, ``moduli_max`` not
            given) reports ``reason`` starting with ``"skipped: "`` and is
            **excluded** from :meth:`is_consistent` -- a missing input can never
            silently *pass*.

            **Results are cached** on the vacuum in :attr:`analysis`, so the
            expensive pieces are computed once and remain inspectable afterwards.
            A second call reuses the cache unless ``refresh=True`` or the
            tolerances differ from those recorded in ``analysis.args``.

            ``Im(z) > 0`` is deliberately **not** a check: the moduli constraint is
            Kähler-cone membership plus metric positivity, and ``Im z`` need not be
            positive componentwise.  This is an intentional divergence from
            ``is_physical``'s last-resort check.

        Args:
            model: A ``FluxEFT`` or ``FluxVacuaFinder``.  Required -- every
                non-trivial check needs the geometry.  Sampler-dependent checks are
                skipped when given a bare ``FluxEFT``.
            moduli_max (float, optional): Runaway bound on
                ``max(|z|, |tau|)``.  Skipped when ``None``.
            s_min (float, optional): Explicit weak-coupling / dilaton floor on
                ``Im tau``.  Falls back to the model's sampler when available,
                else skipped.
            residual_tol (float, optional): F-term tolerance. Defaults to ``1e-8``.
            stability (bool, optional): Also check Hessian positivity (expensive).
                Defaults to ``False``.
            refresh (bool, optional): Recompute even if a cached report matches.

        Returns:
            Dict[str, tuple]: ``{name: (ok, value, reason)}``, also stored in
            ``self.analysis.checks``.
        """
        args = {"moduli_max": moduli_max, "s_min": s_min,
                "residual_tol": residual_tol, "stability": bool(stability)}
        cached = self.analysis
        if (not refresh and cached is not None and cached.checks is not None
                and cached.args == args):
            return cached.checks

        checks: Dict[str, tuple] = {}
        z, tau = self.z, self.tau

        # --- residual -------------------------------------------------------
        res = self.residual
        if res is None or (isinstance(res, float) and math.isnan(res)):
            checks["residual"] = (True, None, "skipped: residual not recorded")
        else:
            r = float(res)
            checks["residual"] = (
                r < residual_tol, r,
                "" if r < residual_tol else
                f"max|DW| = {r:.3e} exceeds residual_tol = {residual_tol:.1e}")

        # --- Im(tau) > 0 (the ONLY positivity requirement on the location) --
        im_tau = float(np.imag(tau))
        checks["im_tau_positive"] = (
            im_tau > 0.0, im_tau,
            "" if im_tau > 0.0 else f"Im(tau) = {im_tau:.3e} is not positive")

        # --- flux integrality ------------------------------------------------
        f = np.asarray(self.flux)
        dev = float(np.max(np.abs(f - np.round(f)))) if f.size else 0.0
        checks["flux_integrality"] = (
            dev < 1e-9, dev,
            "" if dev < 1e-9 else f"flux deviates from integers by {dev:.3e}")

        # --- runaway bound ---------------------------------------------------
        if moduli_max is None:
            checks["runaway_bound"] = (True, None, "skipped: no moduli_max given")
        else:
            big = float(max(np.max(np.abs(z)) if z.size else 0.0, abs(tau)))
            checks["runaway_bound"] = (
                big <= moduli_max, big,
                "" if big <= moduli_max else
                f"max(|z|,|tau|) = {big:.3e} exceeds moduli_max = {moduli_max:.3e}")

        # --- dilaton floor (sampler-dependent) -------------------------------
        floor = s_min
        if floor is None:
            # Only a FluxVacuaFinder carries a sampler; a bare FluxEFT does not.
            floor = getattr(getattr(model, "sampler", None), "s_lower", None)
        if floor is None:
            checks["dilaton_floor"] = (
                True, None,
                "skipped: no s_min given and model has no sampler")
        else:
            checks["dilaton_floor"] = (
                im_tau > float(floor), im_tau,
                "" if im_tau > float(floor) else
                f"Im(tau) = {im_tau:.3e} is below the floor {float(floor):.3e}")

        # --- Kaehler cone ----------------------------------------------------
        hp = resolve_hyperplanes(model)
        if hp is None:
            checks["kahler_cone"] = (
                True, None, "skipped: model exposes no Kaehler-cone hyperplanes")
        else:
            dots = np.asarray(hp, dtype=float) @ np.imag(z)
            worst = float(np.min(dots)) if dots.size else 0.0
            checks["kahler_cone"] = (
                worst > 0.0, worst,
                "" if worst > 0.0 else
                f"Im(z) violates {int(np.sum(dots <= 0))} of {dots.size} cone "
                f"hyperplanes (most negative dot = {worst:.3e})")

        # --- D3 tadpole (SIGNED) ---------------------------------------------
        # The signed window ``0 < f.Sigma.h <= Q`` is required: taking |.| first
        # admits negative D3 charge, which ERRORS.md records as a real bug.
        tad = None
        try:
            tad = float(np.real(model.tadpole(self.flux)))
        except Exception as exc:                       # no model / no tadpole
            checks["tadpole"] = (True, None,
                                 f"skipped: tadpole unavailable ({type(exc).__name__})")
        if tad is not None:
            try:
                cap = model.Q()
            except Exception:
                cap = None
            if cap is None:
                checks["tadpole"] = (
                    True, tad, "skipped: model exposes no D3 cap Q")
            else:
                cap = float(cap)
                ok = (tad > 0.0) and (tad <= cap)
                checks["tadpole"] = (
                    ok, tad, "" if ok else
                    (f"N_flux = {tad:.6g} is not in the signed window "
                     f"0 < N_flux <= Q = {cap:.6g}"
                     + (" (negative D3 charge)" if tad <= 0 else "")))

        # --- Kaehler-metric positive definiteness -----------------------------
        # The check that matters most in practice: omitting it inflated a
        # campaign's "physical" fraction from 65.9% to 89.2% (see ERRORS.md).
        km_eigs = None
        try:
            KM = model.kahler_metric(z, np.conj(z), tau, np.conj(tau))
            km_eigs = np.asarray(np.linalg.eigvalsh(np.asarray(KM)))
        except Exception as exc:
            checks["kahler_metric_pd"] = (
                True, None,
                f"skipped: Kaehler metric unavailable ({type(exc).__name__})")
        if km_eigs is not None:
            worst = float(np.min(km_eigs))
            checks["kahler_metric_pd"] = (
                worst > 0.0, worst,
                "" if worst > 0.0 else
                f"Kaehler metric is not positive definite "
                f"(smallest eigenvalue {worst:.3e})")

        # --- Hessian positivity (opt-in: expensive) ---------------------------
        hess_eigs = None
        if not stability:
            checks["hessian_min_eig"] = (
                True, None, "skipped: pass stability=True to check the Hessian")
        else:
            # Routed through ``classify_solution``, which diagonalises the
            # REAL-basis ``ddV_x``.  Deliberately NOT ``mass_matrix``: its
            # ``mode=None``/``"SUSY"`` branches return spurious negative/complex
            # eigenvalues on complex-KM LCS vacua and its ``mode="real"`` branch has
            # a layout mismatch -- three defects still open in worklog/ERRORS.md.
            try:
                from . import flux_utils as _fu   # local: avoids an import cycle
                info = _fu.classify_solution(model, self.x, self.flux)
                hess_eigs = np.asarray(info["eigenvalues"])
            except Exception as exc:
                checks["hessian_min_eig"] = (
                    True, None,
                    f"skipped: Hessian unavailable ({type(exc).__name__})")
            if hess_eigs is not None:
                worst = float(np.min(hess_eigs))
                checks["hessian_min_eig"] = (
                    worst > 0.0, worst,
                    "" if worst > 0.0 else
                    f"not a minimum: smallest Hessian eigenvalue {worst:.3e} "
                    "(tachyonic or flat direction)")

        # --- subclass extensions (the PFV algebra conditions, e.g.) -----------
        checks.update(self._extra_checks(model))

        # --- conifold alignment scalar (derived payload, not a check) ---------
        # Needs a model *and* M0 from the flux quanta, so it cannot be a stored
        # core field; and it is only defined where there is a conifold modulus.
        alignment = None
        if self.has_conifold and self.zcf is not None:
            alignment = conifold_alignment(model, z, self.zcf, self.gs, self.W0,
                                           self._alignment_M0(model))

        self._store_analysis(checks, args,
                             kahler_metric_eigenvalues=km_eigs,
                             hessian_eigenvalues=hess_eigs,
                             alignment=alignment)
        return checks

    def _extra_checks(self, model) -> Dict[str, tuple]:
        r"""
        **Description:**
        Subclass hook: extra ``{name: (ok, value, reason)}`` entries merged into
        the :meth:`diagnostics` report.

        Overridden by :class:`PFV` to add the PFV algebra conditions, so a record
        labelled a PFV that does not satisfy them reports ``False`` from
        :meth:`is_consistent` with the offending condition named.

        Args:
            model: The model passed to :meth:`diagnostics`.

        Returns:
            Dict[str, tuple]: Empty for a core vacuum.
        """
        return {}

    def _alignment_M0(self, model) -> float:
        r"""
        **Description:**
        The :math:`M_0` entry used by :func:`conifold_alignment`.

        For a core vacuum the only available source is the flux vector itself, so
        :math:`M_0` is read from the second RR-flux entry (the convention the
        ``afvs`` promotion pipeline has always used for non-PFV seeds).
        :class:`PFV` overrides this to use its quantum numbers ``data.M[0]``.

        Args:
            model: Model supplying ``n_fluxes`` to isolate the RR block.

        Returns:
            float: :math:`M_0`, or ``NaN`` when the flux is too short.
        """
        f = np.asarray(self.flux).ravel()
        n = getattr(model, "n_fluxes", None)
        if n is not None:
            f = f[: int(n)]
        return float(f[1]) if f.size >= 2 else float("nan")

    def _store_analysis(self, checks: Dict[str, tuple], args: Dict[str, Any],
                        **extra) -> None:
        r"""
        **Description:**
        Attach a diagnostics report (and any computed eigenvalues) to
        :attr:`analysis`, creating the container if needed.

        Args:
            checks (dict): The report from :meth:`diagnostics`.
            args (dict): Tolerances used, for cache invalidation.
            **extra: Additional :class:`VacuumAnalysis` fields to set.

        Returns:
            None
        """
        an = self.analysis if self.analysis is not None else VacuumAnalysis()
        an.checks, an.args = checks, args
        for k, v in extra.items():
            setattr(an, k, v)
        self.analysis = an

    def is_consistent(self, model, **kwargs) -> bool:
        r"""
        **Description:**
        Whether every applicable consistency check passes.

        A thin reduction over :meth:`diagnostics`, so the verdict and the
        explanation can never disagree -- call ``diagnostics`` for the reasons.
        Checks reported as skipped are excluded; if *nothing* could be graded the
        result is ``False``, since an unverifiable vacuum is not a verified one.

        Args:
            model: A ``FluxEFT`` or ``FluxVacuaFinder``.
            **kwargs: Forwarded to :meth:`diagnostics`.

        Returns:
            bool: ``True`` iff at least one check ran and all of them passed.
        """
        report = self.diagnostics(model, **kwargs)
        graded = [v for v in report.values()
                  if not str(v[2]).startswith("skipped")]
        return bool(graded) and all(bool(v[0]) for v in graded)

    # -- human-readable summaries (display helpers) ---------------------------
    def _short_summary_str(self) -> str:
        r"""
        **Description:**
        Build the one-line summary string.  Conifold quantities appear only when
        :attr:`has_conifold`, so an LCS vacuum is not padded with ``nan``s for a
        bulk/conifold split that does not exist for it.

        Returns:
            str: The one-line summary.
        """
        name = self.metadata.get("model_name", type(self).__name__)
        parts = [f"{type(self).__name__}[{name}]",
                 f"limit={self.limit}",
                 f"|W0|={_abs_or_dash(self.W0)}",
                 f"res={_num_or_dash(self.residual)}"]
        if self.has_conifold:
            parts.append(f"|zcf|={_abs_or_dash(self.zcf)}")
            parts.append(f"(bulk={_num_or_dash(self.residual_bulk, '{:.2e}')}"
                         f"/cf={_num_or_dash(self.residual_conifold, '{:.2e}')})")
        parts.append(f"gs={_num_or_dash(self.gs, '{:.4f}')}")
        return " ".join(parts)

    def short_summary(self) -> None:
        r"""
        **Description:**
        Print the one-line summary (kind, limit, ``|W0|``, residual, the conifold
        split where applicable, ``gs``) to ``stdout``.

        Returns:
            None
        """
        print(self._short_summary_str())

    def long_summary(self) -> None:
        r"""
        **Description:**
        Print the multi-line state summary to ``stdout``.

        The ``afvs`` promotion subclasses extend this with the per-step
        optimisation trajectory table; a core vacuum has no promotion history to
        show.

        Returns:
            None
        """
        print(self._long_summary_str())

    def _long_summary_str(self) -> str:
        r"""
        **Description:**
        Build the multi-line state summary.  Conifold rows are included only when
        :attr:`has_conifold`; subclasses override to append their own sections.

        Returns:
            str: The multi-line summary.
        """
        name = self.metadata.get("model_name", type(self).__name__)
        lines = [f"{type(self).__name__}[{name}] — state", ""]
        lines.append(f"    limit             = {self.limit}")
        lines.append(f"    |W0|              = {_abs_or_dash(self.W0)}")
        lines.append(f"    residual          = {_num_or_dash(self.residual)}")
        if self.has_conifold:
            lines.append(f"    |zcf|             = {_abs_or_dash(self.zcf)}")
            lines.append(f"    residual_bulk     = {_num_or_dash(self.residual_bulk)}")
            lines.append(f"    residual_conifold = {_num_or_dash(self.residual_conifold)}")
        lines.append(f"    gs                = {_num_or_dash(self.gs, '{:.4f}')}")
        if self.analysis is not None:
            lines.append(f"    analysis          = {self.analysis.summary()}")
        return "\n".join(lines)


@register_vacuum_kind
@dataclass(eq=False)
class PFV(Vacuum):
    r"""
    **Description:**
    A vacuum seeded from a perturbatively flat vacuum.  Composes a
    :class:`PFVData` (the light PFV algebra) in :attr:`data` and records the
    seed axio-dilaton in :attr:`tau_input`.

    .. admonition:: Details
        :class: dropdown

        The analytic 2-term-racetrack *estimate* is exposed via the
        :attr:`tau0` / :attr:`W0_estimate` / :attr:`gs_estimate` /
        :attr:`log10_W0_estimate` properties (mirroring ``pfvs.PFV.tau0`` /
        ``W0`` / ``gs``).  These are leading-order seed estimates and are
        deliberately named apart from the inherited :attr:`~Vacuum.W0` /
        :attr:`~Vacuum.gs` *fields*, which hold the full-Newton-solved values
        after :func:`promotion <afvs>`.

    Args:
        data (PFVData, optional): The composed PFV algebra object.
        tau_input (complex): Seed axio-dilaton used to build the initial guess.
    """

    data: Optional[PFVData] = None
    tau_input: complex = complex("nan")

    @classmethod
    def from_quantum_numbers(cls, finder: "FluxVacuaFinder", M: Array, K: Array,
                             tau: complex) -> "PFV":
        r"""
        **Description:**
        Build a :class:`PFV` seed from the PFV quantum numbers
        :math:`(\vec M, \vec K, \tau)`.

        .. admonition:: Details
            :class: dropdown

            Constructs the :class:`PFVData` and seeds the real coordinates ``x``
            from the flat-direction moduli :math:`z = p\,\tau`.

            **A singular N-matrix raises.** :math:`p = N^{-1}K` is the only thing
            that fixes the location, so without it there is nothing to seed from
            and the fluxes do not define a PFV at all.  Earlier versions returned
            an instance with ``x=None``; that object could not be used for
            anything and silently violated the invariant that a vacuum has a
            location, so the failure is now reported where it happens.  Test a
            candidate with ``model.pfv_conditions(M, K)["det N!=0"]`` first if it
            may be singular.

        Args:
            finder (FluxVacuaFinder): Model providing the PFV algebra and the
                coordinate transform.
            M (Array): M-vector.
            K (Array): K-vector.
            tau (complex): Seed axio-dilaton.

        Returns:
            PFV: Seed instance with ``data``, ``flux`` and ``x`` set.

        Raises:
            ValueError: If ``N`` is singular, i.e. ``(M, K)`` is not a PFV.
        """
        data = PFVData.from_fluxes(finder, M, K)
        if data.p is None:
            det = None
            if data.conditions is not None and "det N!=0" in data.conditions:
                det = float(np.asarray(data.conditions["det N!=0"][1]))
            raise ValueError(
                "these fluxes do not define a PFV: the N-matrix is singular "
                f"(det N = {det if det is None else f'{det:.3e}'}), so the flat "
                "direction p = N^-1 K -- and hence the location to seed from -- "
                f"does not exist. M={np.asarray(M).tolist()}, "
                f"K={np.asarray(K).tolist()}."
            )
        z0 = finder.pfv_to_moduli(M, K, tau)
        x = finder._convert_complex_to_real(z0, jnp.conj(z0), tau, jnp.conj(tau))
        return cls(flux=data.flux, x=x, data=data, tau_input=tau)

    # -- analytic 2-term-racetrack estimate (seed; distinct from solved fields)
    def racetrack_estimate(self) -> Dict[str, Any]:
        r"""
        **Description:**
        The full analytic 2-term-racetrack estimate ``{tau0, W0, log10_W0, gs,
        valid}`` from :meth:`~jaxvacua.flux_eft.FluxEFT.pfv_racetrack`.

        Returns:
            dict: The racetrack estimate; requires an attached model on
            :attr:`data`.
        """
        if self.data is None or self.data._model is None:
            raise RuntimeError(
                "PFV.data has no attached model for the racetrack estimate."
            )
        return self.data._model.pfv_racetrack(self.data.M, self.data.K)

    def _extra_checks(self, model) -> Dict[str, tuple]:
        r"""
        **Description:**
        Extend the diagnostics report with the **PFV algebra conditions**, so a
        record labelled a PFV is actually tested against being one.

        .. admonition:: Details
            :class: dropdown

            The conditions come from
            :meth:`~jaxvacua.flux_eft.FluxEFT.pfv_conditions` (arXiv:2512.17095
            Eq. 6.27 / 7.28): :math:`\det N \neq 0`, :math:`K_a p^a = 0`,
            :math:`p \in \mathcal{K}_X` (or :math:`\mathcal{K}_{\rm cf}` plus
            :math:`M_{\rm cf} \neq 0` in a conifold limit), the two integrality
            conditions, and the PFV tadpole window.  Keys are prefixed ``pfv:``
            both to mark their origin and to avoid colliding with the base
            ``tadpole`` check, which tests a different quantity
            (:math:`f\cdot\Sigma\cdot h` against ``Q``, not :math:`-M\cdot K/2`).

            The stored :attr:`PFVData.conditions` are reused when present
            (``PFVData.from_fluxes`` computes them at construction); otherwise
            they are recomputed from ``(M, K)``.  ``ok`` values are ``jnp``
            boolean arrays because ``pfv_conditions`` is ``auto_vmap``-ped, so
            each is reduced with ``.all()`` and coerced to a Python ``bool``.
            The ``"p"`` entry is skipped: it carries the p-vector, not a verdict.

        Args:
            model: Model providing ``pfv_conditions`` if they are not stored.

        Returns:
            Dict[str, tuple]: ``{"pfv:<condition>": (ok, value, reason)}``, empty
            when no ``data`` is attached.
        """
        if self.data is None:
            return {}
        cond = self.data.conditions
        if cond is None:
            try:
                cond = model.pfv_conditions(self.data.M, self.data.K)
            except Exception as exc:
                return {"pfv:conditions": (
                    True, None,
                    f"skipped: PFV conditions unavailable ({type(exc).__name__})")}
        out: Dict[str, tuple] = {}
        for name, entry in cond.items():
            if name == "p":       # the p-vector, not a verdict (same filter as NB19)
                continue
            try:
                ok = bool(np.asarray(entry[0]).all())
                val = np.asarray(entry[1])
                val = val.item() if val.size == 1 else val
            except Exception:
                continue          # not an (ok, value) pair -- skip quietly
            out[f"pfv:{name}"] = (
                ok, val, "" if ok else f"PFV condition `{name}` is violated")
        return out

    def _alignment_M0(self, model) -> float:
        r"""
        **Description:**
        :math:`M_0` from the PFV quantum numbers, i.e. ``data.M[0]`` -- the
        authoritative source, used in preference to the base class's flux-vector
        fallback.

        Args:
            model: Model (used only by the inherited fallback).

        Returns:
            float: :math:`M_0`.
        """
        if self.data is not None and self.data.M is not None:
            M = np.asarray(self.data.M).ravel()
            if M.size:
                return float(M[0])
        return super()._alignment_M0(model)

    def _long_summary_str(self) -> str:
        r"""
        **Description:**
        The base state summary plus the PFV algebra inputs.

        Overrides the base rather than having the base test ``isinstance(self, PFV)``
        -- a base class should not need to know about its subclasses.

        Returns:
            str: The multi-line summary.
        """
        out = super()._long_summary_str()
        if self.data is not None:
            out += (f"\n    PFV input         : M={np.asarray(self.data.M).tolist()}"
                    f"  K={np.asarray(self.data.K).tolist()}"
                    f"  tau_in={self.tau_input}")
        return out

    @property
    def tau0(self) -> complex:
        r"""Description: Estimated axio-dilaton VEV :math:`\tau_0` (racetrack)."""
        return self.racetrack_estimate()["tau0"]

    @property
    def W0_estimate(self) -> complex:
        r"""Description: Estimated :math:`W_0` (racetrack; distinct from the
        solved :attr:`~Vacuum.W0` field)."""
        return self.racetrack_estimate()["W0"]

    @property
    def log10_W0_estimate(self) -> float:
        r"""Description: :math:`\log_{10}|W_0|` estimate (racetrack, log-safe)."""
        return self.racetrack_estimate()["log10_W0"]

    @property
    def gs_estimate(self) -> float:
        r"""Description: Estimated string coupling :math:`g_s` (racetrack;
        distinct from the solved :attr:`~Vacuum.gs` field)."""
        return self.racetrack_estimate()["gs"]


def unique_vacua(vacua: Sequence[Vacuum],
                 finder: Optional["FluxVacuaFinder"] = None, *,
                 n_digits: int = 6) -> List[Vacuum]:
    r"""
    **Description:**
    Deduplicate a list of vacua, keeping one representative per class.

    .. admonition:: Details
        :class: dropdown

        A single pass over the hashable :meth:`Vacuum.canonical_key`, so this is
        :math:`O(n)` rather than :math:`O(n^2)`; insertion order is preserved (the
        first occurrence of each class is kept).  It never uses ``v in list`` --
        the array-bearing ``__eq__`` is disabled.

        With a ``finder`` the classes are :math:`SL(2, \mathbb{Z}) \times`
        monodromy orbits; without one, distinct points (rounded to ``n_digits``),
        which is what a finder-free consumer such as a vault read-back can do.

    Args:
        vacua (sequence[Vacuum]): Vacua to deduplicate.
        finder (FluxVacuaFinder, optional): Model supplying the
            fundamental-domain map.  Omit to deduplicate by point only.
        n_digits (int, optional): Decimals retained. Defaults to ``6``.

    Returns:
        list[Vacuum]: One representative per class.
    """
    seen: Dict[tuple, Vacuum] = {}
    for v in vacua:
        key = v.canonical_key(finder, n_digits=n_digits)
        if key not in seen:
            seen[key] = v
    return list(seen.values())


#: Alias of :func:`unique_vacua`.
dedup_vacua = unique_vacua


def vacuum_to_json(vacuum: Vacuum, *, analysis: bool = True,
                   indent: Optional[int] = None) -> str:
    r"""
    **Description:**
    Serialise a vacuum to a **tagged JSON string** -- the pickle-free storage tier.

    .. admonition:: Details
        :class: dropdown

        Prefer this over :func:`save_vacua` for anything others will download: a
        gzipped pickle executes arbitrary code on load, whereas this payload is
        inert, human-inspectable and diffable.  NumPy arrays and complex scalars
        survive via the tags documented in :func:`encode_json`.

    Args:
        vacuum (Vacuum): The vacuum to serialise.
        analysis (bool, optional): Include the derived
            :attr:`~Vacuum.analysis` block. Defaults to ``True``.
        indent (int, optional): ``json.dumps`` indent (``None`` for compact).

    Returns:
        str: The JSON document.
    """
    return json.dumps(encode_json(vacuum.to_dict(analysis=analysis)),
                      indent=indent, sort_keys=True)


def vacuum_from_json(text: str) -> Vacuum:
    r"""
    **Description:**
    Rebuild a vacuum from a :func:`vacuum_to_json` document, routed to the right
    subclass by its ``_kind`` tag.

    Args:
        text (str): The JSON document.

    Returns:
        Vacuum: The reconstructed instance.
    """
    return Vacuum.from_dict(decode_json(json.loads(text)))


def save_vacua(vacua: Sequence[Vacuum], path: str) -> None:
    r"""
    **Description:**
    Persist a list of :class:`Vacuum` instances to a gzipped pickle file.

    .. admonition:: Details
        :class: dropdown

        Each vacuum is serialised through :meth:`Vacuum.to_dict` (JAX→NumPy
        downcast plus the ``"_kind"`` discriminator), then the list of dicts is
        pickled and gzipped by
        :func:`jaxvacua.util.save_zipped_pickle`.

    Args:
        vacua (sequence[Vacuum]): Vacua to save.
        path (str): Output filename (typically ``.pkl.gz`` or ``.p``).
    """
    save_zipped_pickle([v.to_dict() for v in vacua], path)


def load_vacua(path: str) -> List[Vacuum]:
    r"""
    **Description:**
    Reload a list of :class:`Vacuum` instances saved via :func:`save_vacua`.

    Args:
        path (str): Input filename produced by :func:`save_vacua`.

    Returns:
        list[Vacuum]: Reconstructed instances of the appropriate subclasses
        (routed by each entry's ``"_kind"`` discriminator).
    """
    return [Vacuum.from_dict(d) for d in load_zipped_pickle(path)]