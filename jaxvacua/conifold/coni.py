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

"""Conifold geometry and discovery routines.

Purpose
-------
Represent conifold limits in complex-structure moduli space and discover
candidate conifold transitions from CYTools triangulations.

Main public API
---------------
- ``Conifold``: stores the conifold curve, canonical-basis charge, GV
  invariant, basis-change data and optional CYTools geometry references.
- ``find_conifolds``: searches triangulations for conifold/flop data.
- JAX pytree flattening and reconstruction helpers for ``Conifold`` objects.

Design notes
------------
The class separates persistent numerical state from optional CYTools objects.
After a JAX pytree round-trip the numerical data survives, while heavy CYTools
references may be absent.  Basis-change and projection routines live in
``jaxvacua.conifold.conifold_utils``.
"""

# Defer annotation evaluation (PEP 563) so the optional cytools type hints
# (Polytope / Triangulation) need not be importable at module-import time.
from __future__ import annotations

from typing import List, Optional

import numpy as np
import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node

# cytools is an optional dependency: only needed to construct a Conifold from
# toric geometry (Polytope / Triangulation).  Guard the import so importing the
# conifold subpackage (and hence `jaxvacua`) works without cytools installed;
# the clear error is raised at the point of use.
try:
    from cytools import Polytope
    from cytools.triangulation import Triangulation
except ImportError:
    Polytope = None
    Triangulation = None

from jaxvacua.conifold.conifold_utils import (
    get_basis_change, get_bulk_embedding, get_bulk_projection, get_embedding,
)


class Conifold:
    r"""
    **Description:**
    Represents a conifold singularity in complex structure moduli space dual
    to a flop transition in Kähler moduli space.

    .. admonition:: Details
        :class: dropdown

        A *conifold singularity* appears at the boundary of complex structure
        moduli space where a rational curve in the mirror Calabi-Yau shrinks
        to zero size.  On the mirror side, this corresponds to a topological
        transition — a *flop* — in Kähler moduli space.

        The flop is encoded in the triangulation of the toric polytope: a
        specific edge inside a 2-face of the dual polytope can be flipped
        (Whitehouse flip), and the corresponding curve is an extremal generator
        of the Mori cone.  The class stores the data needed to characterise
        this transition:

        * **Flop edge** :math:`e` — the edge in the dual triangulation that
          undergoes the flip.
        * **One-face divisors** — divisor indices interior to the 1-face of
          the polytope dual to the conifold (these become light in the
          coniLCS limit).
        * **Gopakumar-Vafa invariant** :math:`n_{\rm cf}` — the genus-0 GV
          invariant of the flopping curve.
        * **Conifold curve** :math:`q \in \mathbb{Z}^{h^{1,1}}` (original-basis
          charge) and **conifold_curve0** = :math:`q\,\Lambda^{T} = (1,0,\ldots,0)`
          (canonical-basis charge after the basis transformation
          :math:`\Lambda`); both are eagerly materialised at construction in
          geometric mode.

    The class supports two construction modes via the
    :meth:`from_geometry` and :meth:`from_data` classmethods.  After a JAX
    pytree round-trip, the cytools refs (:meth:`polytope`,
    :meth:`dual_triangulation`) return ``None``; the numerical state survives
    unchanged.
    """

    def __init__(self, *,
                 ncf: int = 0,
                 conifold_curve:     Optional[Array]         = None,
                 conifold_curve0:    Optional[Array]         = None,
                 basis_change:       Optional[Array]         = None,
                 flop_edge:          Optional[Array]         = None,
                 one_face_divisors:  Optional[Array]         = None,
                 polytope:           Optional[Polytope]      = None,
                 dual_triangulation: Optional[Triangulation] = None,
                 bulk_embedding: Optional[Array] = None,
                 bulk_projection: Optional[Array] = None,
                 embedding: Optional[Array] = None):
        r"""
        **Description:**
        Low-level keyword-only constructor; prefer :meth:`from_geometry` or
        :meth:`from_data` in user code.
        """
        # Eager / hot-path attributes (plain names, in pytree).
        self.ncf                 = int(ncf)
        self.conifold_curve      = conifold_curve     # original-basis charge
        self.conifold_curve0     = conifold_curve0    # canonical-basis charge
        # Lazy-getter backing storage (underscore-prefixed).
        self._basis_change       = basis_change
        self._flop_edge          = flop_edge
        self._one_face_divisors  = one_face_divisors
        # Cytools refs — dropped at the JAX pytree boundary by the custom flatten.
        self._polytope           = polytope
        self._dual_triangulation = dual_triangulation
        # Conifold/bulk split objects for the conifold_basis=False code paths.
        # In a general basis the bulk EMBEDDING (build a vector from its bulk
        # components / project a charge covector onto the bulk; = Λ[1:]ᵀ) and the
        # bulk PROJECTION (extract bulk components of a period vector;
        # = Λ⁻¹[:,1:]) are DIFFERENT matrices — they coincide only in the
        # conifold-aligned basis.  The conifold direction of a vector is the
        # embedding ``embedding`` (e_q = Λ[0], q·e_q = 1); the conifold modulus
        # is extracted by ``conifold_curve`` (z_cf = q·z).
        self.bulk_embedding = bulk_embedding
        self.bulk_projection = bulk_projection
        self.embedding = embedding

    def __repr__(self):
        r"""
        **Description:**
        Returns a human-readable description of the :class:`Conifold` instance,
        including the Gopakumar-Vafa invariant.

        Returns:
            str: String representation.
        """

        return f"A conifold limit in complex structure moduli space dual to a flop transition with GV = {self.ncf}"

    # ------------------------------------------------------------------ #
    # Lazy / on-demand getter methods
    # ------------------------------------------------------------------ #


    def basis_change(self):
        r"""
        **Description:**
        Returns the unimodular basis change matrix :math:`\Lambda` that maps
        the :attr:`conifold_curve` to :math:`(1, 0, \ldots, 0)`.  Computed
        lazily on first call when the cache is empty and a
        ``conifold_curve`` is available.

        Returns:
            Array | None: Integer matrix
            :math:`\Lambda \in \mathrm{GL}(h^{1,1},\mathbb{Z})` of shape
            :math:`(h^{1,1}, h^{1,1})`, or ``None`` if neither cache nor
            ``conifold_curve`` is available.

        See also: :func:`get_basis_change`
        """
        if self._basis_change is None and self.conifold_curve is not None:
            self._basis_change = jnp.asarray(get_basis_change(self.conifold_curve))
        return self._basis_change

    def flop_edge(self):
        r"""
        **Description:**
        Returns the pair of point indices forming the flop edge in the dual
        triangulation.

        Returns:
            Array | None: Pair of integer indices defining the flop edge,
            or ``None`` if not available.
        """
        return self._flop_edge

    def one_face_divisors(self, as_index: bool = True):
        r"""
        **Description:**
        Returns the divisors interior to the 1-face dual to the conifold
        transition.

        Args:
            as_index (bool, optional): If ``True`` (default), returns the
                integer indices of the interior points.  If ``False``,
                returns the corresponding rows of the GLSM charge matrix
                (only valid when the cytools polytope is loaded).

        Returns:
            Array | None: Interior-point indices or GLSM charges, depending
            on ``as_index``.
        """
        if as_index:
            return self._one_face_divisors
        if self._polytope is None:
            raise RuntimeError(
                "polytope not loaded; cannot compute GLSM charge rows. "
                "Reconstruct via Conifold.from_geometry to use as_index=False."
            )
        return self._polytope.glsm_charge_matrix().T[self._one_face_divisors]

    def polytope(self):
        r"""
        **Description:**
        Returns the cytools polytope, or ``None`` after a JAX pytree
        round-trip (geometric data is dropped at the pytree boundary).

        Returns:
            cytools.Polytope | None: The polytope, or ``None`` if not loaded.
        """
        return self._polytope

    def dual_triangulation(self):
        r"""
        **Description:**
        Returns the cytools fine, regular, star triangulation of the dual
        polytope, or ``None`` after a JAX pytree round-trip.

        Returns:
            cytools.Triangulation | None: The dual triangulation, or ``None``
            if not loaded.
        """
        return self._dual_triangulation

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_geometry(cls,
                      polytope: Polytope,
                      dual_triangulation: Triangulation,
                      one_face_divisors: Array,
                      flop_edge: Array,
                      ncf: int) -> "Conifold":
        r"""
        **Description:**
        Construct a :class:`Conifold` from cytools geometry.  Eagerly
        computes :attr:`conifold_curve`, :attr:`conifold_curve0` and the
        basis-change matrix from the supplied polytope, triangulation and
        flop edge.

        Args:
            polytope (cytools.Polytope): Toric polytope of the CY model.
            dual_triangulation (cytools.Triangulation): Fine, regular, star
                triangulation of the dual polytope.
            one_face_divisors (Array): Indices of divisors interior to the
                1-face dual to the conifold transition.
            flop_edge (Array): Pair of point indices forming the flop edge.
            ncf (int): Gopakumar-Vafa invariant :math:`n_{\rm cf}` of the
                flopping curve (used to normalise the conifold charge).

        Returns:
            Conifold: A geometric-mode Conifold with all numerical fields
            populated and the cytools refs attached.
        """
        # Compute the mirror CY intersection numbers and conifold charge.
        dual_cy    = dual_triangulation.get_cy()
        intnums    = dual_cy.intersection_numbers()
        dual_basis = polytope.dual().glsm_basis()
        edge_list  = list(np.asarray(flop_edge))
        charge_np  = np.rint(np.array([
            intnums.get(tuple(np.sort(edge_list + [b])), 0)
            for b in dual_basis
        ]) / ncf).astype(int)

        bc_np  = np.asarray(get_basis_change(charge_np))   # int matrix
        curve  = jnp.asarray(charge_np)
        bc     = jnp.asarray(bc_np)
        curve0 = curve @ bc.T                              # = (1,0,...,0)

        return cls(
            ncf=int(ncf),
            conifold_curve=curve,
            conifold_curve0=curve0,
            basis_change=bc,
            flop_edge=jnp.asarray(flop_edge),
            one_face_divisors=jnp.asarray(one_face_divisors),
            polytope=polytope,
            dual_triangulation=dual_triangulation,
            bulk_embedding=get_bulk_embedding(curve),
            bulk_projection=get_bulk_projection(curve),
            embedding=get_embedding(curve),
        )

    @classmethod
    def from_data(cls, **kwargs) -> "Conifold":
        r"""
        **Description:**
        Construct a numerical-only :class:`Conifold` from kwargs.  If
        ``conifold_curve`` and ``basis_change`` are supplied but
        ``conifold_curve0`` is omitted, it is derived as
        ``conifold_curve @ basis_change.T``.

        Args:
            **kwargs: Any subset of the :meth:`__init__` keyword arguments
                excluding ``polytope`` and ``dual_triangulation``.

        Returns:
            Conifold: A numerical-mode Conifold (cytools refs ``None``).
        """
        cc  = kwargs.get("conifold_curve")
        bc  = kwargs.get("basis_change")
        cc0 = kwargs.get("conifold_curve0")
        if cc0 is None and cc is not None and bc is not None:
            kwargs["conifold_curve0"] = jnp.asarray(cc) @ jnp.asarray(bc).T

        kwargs["bulk_embedding"]  = get_bulk_embedding(cc)
        kwargs["bulk_projection"] = get_bulk_projection(cc)
        kwargs["embedding"]       = get_embedding(cc)
        return cls(**kwargs)



    def to_data(self) -> "Conifold":
        r"""
        **Description:**
        Return a numerical-only copy of this :class:`Conifold` suitable for
        carrying across a JAX pytree boundary.  The cytools refs are
        stripped; the lazy basis-change cache is warmed first if a
        ``conifold_curve`` is available.

        Returns:
            Conifold: Numerical-mode copy.
        """
        # Warm the basis_change cache while geometry is still loaded.
        _ = self.basis_change()

        return Conifold.from_data(
            ncf=self.ncf,
            conifold_curve=self.conifold_curve,
            conifold_curve0=self.conifold_curve0,
            basis_change=self._basis_change,
            flop_edge=self._flop_edge,
            one_face_divisors=self._one_face_divisors,
        )


# --------------------------------------------------------------------------
# JAX pytree registration for Conifold.
#
# Cytools objects (Polytope, Triangulation) are not hashable and cannot live
# in aux_data, so the custom flatten silently drops them.  After a round-trip,
# ``polytope()`` and ``dual_triangulation()`` return ``None``.
# --------------------------------------------------------------------------

_CONIFOLD_DYNAMIC_KEYS = (
    # Hot-path attributes (plain names)
    'conifold_curve', 'conifold_curve0',
    'bulk_embedding', 'bulk_projection', 'embedding',
    # Lazy-getter backing storage (underscore-prefixed)
    '_basis_change', '_flop_edge', '_one_face_divisors',
)
_CONIFOLD_STATIC_KEYS = ('ncf',)


def _flatten_conifold(obj):
    children = tuple(getattr(obj, k, None) for k in _CONIFOLD_DYNAMIC_KEYS)
    aux_data = tuple((k, getattr(obj, k, None)) for k in _CONIFOLD_STATIC_KEYS)
    return children, aux_data


def _unflatten_conifold(aux_data, children):
    obj = object.__new__(Conifold)
    for k, v in aux_data:
        object.__setattr__(obj, k, v)
    for k, v in zip(_CONIFOLD_DYNAMIC_KEYS, children):
        object.__setattr__(obj, k, v)
    # Cytools refs are not part of the pytree — set to None on the way back.
    object.__setattr__(obj, '_polytope', None)
    object.__setattr__(obj, '_dual_triangulation', None)
    return obj


# Pytree registration must run exactly once per process.  Guard with an
# attribute on the class so re-imports (or simultaneous registration from the
# legacy ``jaxvacua.conifold_utils`` shim) don't trip jax's "already
# registered" check.
if not getattr(Conifold, "_pytree_registered", False):
    register_pytree_node(Conifold, _flatten_conifold, _unflatten_conifold)
    Conifold._pytree_registered = True


def find_conifolds(polytope: Polytope,
                   FRSTs: Optional[List[Triangulation]] = None,
                   n_conifolds: int = 2,
                   verbosity: int = 0):
    r"""
    **Description:**
    Identifies conifold limits in complex structure moduli space by searching
    for flop transitions in the triangulations of a toric polytope.

    .. admonition:: Details
        :class: dropdown

        A conifold transition corresponds to a *flop*: an edge inside a 2-face
        of the dual polytope that (i) can be flipped in the triangulation
        (Whitehouse flip) and (ii) whose associated curve is an extremal
        generator of the Mori cone.

        The search proceeds in three stages:

        1. **Face-pair selection** — pairs :math:`(f_1, f_2)` of primal 1-faces
           and dual 2-faces are retained when:

           * :math:`f_1` has exactly :math:`n_{\rm cf}-1` interior points
             (the conifold divisors).
           * :math:`f_2` has no interior points (for :math:`n_{\rm cf}>1`).
           * :math:`f_2` is not a simplex (:math:`\geq 4` vertices), so a
             flip is geometrically possible.

        2. **Triangulation loop** — for each FRST of the dual polytope, the
           simplices restricted to each candidate 2-face are extracted.
           Interior edges (shared by exactly two simplices) are enumerated,
           and flippable ones are identified by checking whether the convex
           hull of the four involved points has four vertices (a quadrilateral).

        3. **Mori cone check** — a candidate flop edge produces a charge
           vector :math:`q` (see :meth:`Conifold.conifold_charge`); only
           edges for which :math:`q` is an extremal ray of the Mori cone cap
           are accepted as genuine conifolds.

    Args:
        polytope (cytools.Polytope): Toric polytope of the CY model.
        FRSTs (list of cytools.Triangulation | None, optional): Pre-computed
            fine, regular, star triangulations of the dual polytope.  If
            ``None``, all NTFE classes are computed automatically. Defaults
            to ``None``.
        n_conifolds (int, optional): Number of conifold divisors — i.e. the
            number of interior points of the 1-face :math:`f_1`.  Defaults
            to ``2``.
        verbosity (int, optional): ``0`` — silent; ``1`` — prints candidate
            count; ``2`` — prints each conifold found. Defaults to ``0``.

    Returns:
        list of Conifold: All :class:`Conifold` instances found across all
        provided triangulations.
    """

    # ----------------------------------------
    # Basic polytope data
    # ----------------------------------------
    p     = polytope
    pdual = p.dual()                         # Dual polytope
    dpts  = pdual.points()                   # Coordinates of dual polytope points

    # All 1-faces of primal polytope and corresponding dual 2-faces
    one_faces      = p.faces(1)
    dual_two_faces = pdual.faces(2)

    # ----------------------------------------
    # Step 1: Select candidate face pairs
    # ----------------------------------------
    # We select (f1, f2) pairs satisfying:
    # - f1 has exactly (n_conifolds - 1) interior points
    # - f2 has no interior points (if n_conifolds > 1)
    # - f2 has at least 4 vertices (so a flip is possible)
    candidate_face_indices = []

    for i, (dual, one) in enumerate(zip(dual_two_faces, one_faces)):

        cond1 = (len(dual.interior_points()) == 0) if n_conifolds > 1 else True
        cond2 = (len(one.interior_points()) == n_conifolds - 1)
        cond3 = (len(dual.points()) > 3)

        if cond1 and cond2 and cond3:
            candidate_face_indices.append(i)

    if verbosity >= 1:
        print(f"Found {len(candidate_face_indices)} candidate face pairs for conifold limits.")

    # ----------------------------------------
    # Step 2: Prepare triangulations (FRSTs)
    # ----------------------------------------
    if FRSTs is None:
        if verbosity >= 1:
            print("No FRSTs provided. Computing all NTFEs...")

        # Warn user if computation may be expensive
        if pdual.h11("N") > 15:
            print("Warning: Computing all NTFEs for a polytope with h11 > 15 can be very time-consuming.")

        # Compute all fine, regular, star triangulations
        FRSTs = pdual.ntfe_frsts()

    # List of detected conifold objects
    conifolds = []

    # ----------------------------------------
    # Step 3: Loop over triangulations
    # ----------------------------------------
    for t in FRSTs:
        # Get simplices restricted to each 2-face (grouped by face)
        two_face_simplices = t.simplices(on_faces_dim=2, split_by_face=True)

        # Mori cone extremal rays (as tuples for fast lookup)
        Mcap = {
            tuple(q)
            for q in t.get_cy().mori_cone_cap(in_basis=True).extremal_rays()
        }

        # CY data associated to this triangulation
        dual_cy = t.get_cy()
        intnums = dual_cy.intersection_numbers()   # Intersection numbers dictionary
        dual_basis = p.dual().glsm_basis()         # Basis for charges

        # ----------------------------------------
        # Loop over candidate faces
        # ----------------------------------------
        for faceID in candidate_face_indices:
            # Interior lattice points of the 1-face (conifold divisors)
            one_face_divisors = np.array(
                one_faces[faceID].interior_points(as_indices=True)
            )

            # Simplices forming triangulation of this 2-face
            face_simplices = two_face_simplices[faceID]

            # ----------------------------------------
            # Extract edges in the 2-face
            # ----------------------------------------
            # Each triangle contributes 3 edges (remove one vertex at a time)
            edges_in_face = [
                np.delete(s, i, 0)
                for s in face_simplices
                for i in range(3)
            ]

            # Remove duplicate edges
            edges_in_face = np.unique(np.array(edges_in_face), axis=0)

            # ----------------------------------------
            # Identify interior edges
            # ----------------------------------------
            # Interior edges are those shared by exactly two simplices
            interior_edges = edges_in_face[
                np.where([
                    len(np.where([set(e).issubset(s) for s in face_simplices])[0]) == 2
                    for e in edges_in_face
                ])[0]
            ]

            # ----------------------------------------
            # Determine opposite vertices for each edge
            # ----------------------------------------
            # For each interior edge, find the two vertices opposite to it
            connected_points = np.array([
                [
                    [j for j in set(face_simplices[i]) - set(e)][0]
                    for i in np.where([
                        set(e).issubset(s) for s in face_simplices
                    ])[0]
                ]
                for e in interior_edges
            ])

            # ----------------------------------------
            # Identify flippable edges (Whitehouse flips)
            # ----------------------------------------
            # A flip is possible if the 4 involved points form a quadrilateral
            flop_indices = np.where([
                len(
                    Polytope(
                        np.array(
                            list(dpts[connected_points[i]]) +
                            list(dpts[interior_edges[i]])
                        )
                    ).vertices()
                ) == 4
                for i in range(len(interior_edges))
            ])[0]

            # Extract flippable edges
            flop_edges = interior_edges[flop_indices]

            # ----------------------------------------
            # Step 4: Mori cone check
            # ----------------------------------------
            for e in flop_edges:
                # Compute charge vector associated to the flop curve
                flop_charge = np.rint(np.array([
                    intnums.get(tuple(np.sort(list(e) + [b])), 0)
                    for b in dual_basis
                ]) / n_conifolds).astype(int)

                # Keep only if this is an extremal ray of the Mori cone
                if tuple(flop_charge) in Mcap:
                    conifolds.append(
                        Conifold.from_geometry(
                            polytope=p,
                            dual_triangulation=t,
                            one_face_divisors=one_face_divisors,
                            flop_edge=e,
                            ncf=n_conifolds,
                        )
                    )

                    if verbosity >= 2:
                        print(f"Found conifold: {conifolds[-1]}")

    # ----------------------------------------
    # Return all detected conifolds
    # ----------------------------------------
    return conifolds
