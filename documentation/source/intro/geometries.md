# Construction of Calabi-Yau geometries

We are mostly concerned with Calabi-Yau threefolds, which we abbreviate by *CY3s* in
what follows.
JAXVacua currently supports two classes of CY3 geometries: hypersurfaces in toric
varieties from the Kreuzer-Skarke database and complete intersection Calabi-Yau
threefolds (CICYs).
In the future, support for F-theory flux compactifications on Calabi-Yau fourfolds is
desirable.


## CY hypersurfaces from reflexive polytopes

Smooth Calabi-Yau threefold hypersurfaces can be obtained from triangulations of
reflexive polytopes in four dimensions.
These were enumerated by Kreuzer and Skarke
[[hep-th/0002240](https://arxiv.org/abs/hep-th/0002240)],
finding 473,800,776 such polytopes.
With the development of [*CYTools*](https://cy.tools)
[[2303.00757](https://arxiv.org/abs/2303.00757)],
efficient and targeted construction of such CY geometries became feasible on large
scales, opening a new window to study string theory compactifications.
Our implementation supports [*CYTools*](https://cy.tools) capabilities for the
construction of the EFT from compactifications on such Calabi-Yau threefold
hypersurfaces.


### Batyrev construction

We begin by setting notation and terminology for Calabi-Yau threefold hypersurfaces
in toric varieties, obtained from triangulations of four-dimensional reflexive polytopes
from the Kreuzer-Skarke list.
More details on this construction can be found in
[[2008.01730](https://arxiv.org/abs/2008.01730)].

Suppose that $\Delta \subset \mathbb{Z}^4$ is a four-dimensional reflexive polytope and
denote by $\Delta^{\circ}$ its polar dual.

Let $\mathcal{T}$ be a regular, star triangulation of $\Delta^{\circ}$ in which every
point of $\Delta^{\circ}$ that is not interior to a facet is a vertex of a simplex of
$\mathcal{T}$.
Such a triangulation is not in general a **fine**, regular, star triangulation (FRST),
because of the omission of points interior to facets, but the associated subvarieties
of $V$ do not intersect a generic hypersurface $X$, and so are immaterial for our
analysis.
With this understanding we refer to such triangulations $\mathcal{T}$ as FRSTs.
The toric fan associated to $\mathcal{T}$ defines a four-dimensional toric variety $V$
in which the generic anticanonical hypersurface $X$ is a smooth Calabi-Yau threefold
[[alg-geom/9310003](https://arxiv.org/abs/alg-geom/9310003)].

Regular triangulations of polytopes can be represented by a vector of heights which
lifts the point collection of $\Delta^\circ$ to one higher dimension.
As is well known, there is a massive redundancy when going from polytope triangulations
to Calabi-Yau hypersurfaces.
Wall's theorem asserts that Hodge numbers, triple intersection numbers, and second
Chern classes completely determine the homotopy type of a compact, simply connected
Calabi-Yau threefold with torsion-free homology.
For Calabi-Yau hypersurfaces, Hodge numbers are fixed by polytope data, while triple
intersection numbers and second Chern classes are determined purely by the induced
triangulations of two-faces.
Therefore, FRSTs of $\Delta^{\circ}$ with identical restrictions to two-faces give rise
to topologically equivalent Calabi-Yau threefolds.


### Orientifolds

We say that a polytope $\Delta^{\circ}$ is **trilayer** if the points of $\Delta^{\circ}$
lie in exactly three distinct affine sub-lattices of codimension one
[[2106.05084](https://arxiv.org/abs/2106.05084)].
Calabi-Yau threefold hypersurfaces in toric varieties $V$ resulting from triangulations
of trilayer polytopes admit very convenient orientifold involutions.
In each case there exists a toric coordinate $x_1$ such that the involution defined by

$$
    x_1 \to -x_1
$$ (eq:odef)

yields, when restricted to the generic invariant hypersurface $X \subset V$, an
orientifold with $h^{1,1}_-=h^{2,1}_+=0$, which we refer to as a **trilayer orientifold**.
All orientifolds considered in this work are of this type.

Given a Calabi-Yau orientifold $X/\mathcal{I}$, the D3-brane tadpole charge is

$$
Q_{\text{O}} =  2+h^{1,1}+h^{2,1}\,.
$$ (eq:qdef)

A larger tadpole $Q_O$ admits more flux configurations and thus a richer landscape of
vacua (see {doc}`flux_compactifications`).
In practice, we restrict to geometries with $3\leq h^{2,1}\leq 5$ and
$Q_{\text{O}}\ge 100$, or $6\leq h^{2,1}\leq 8$ and $Q_{\text{O}}\ge 150$.


### Prime toric divisors

A consequence of $\Delta^{\circ}$-favorability is that $h^{1,1}(V) = h^{1,1}(X)$,
with $h^{1,1}(V)+4$ toric coordinates $x_I$ generating the Cox ring.
We define the prime toric divisors of $V$ as $\hat{D}_I = \{x_I = 0\}$, and we refer
to $D_I = \hat{D}_I \cap X$ as the prime toric divisors of $X$.
The $D_I$ are all effective divisors, and (again using $\Delta^{\circ}$-favorability)
they generate $H_4(X,\mathbb{Z})$.
In general there also exist effective divisors $D$ that are non-positive integer linear
combinations of the $D_I$, which are relevant for non-perturbative contributions to the
superpotential (see {doc}`moduli_stabilisation`).


## Complete Intersection CYs

The JAXVacua package also supports **Complete Intersection Calabi-Yau threefolds**
(CICYs), defined as complete intersections of hypersurfaces in products of projective
spaces.
The 7,890 topologically distinct CICY threefolds were classified in
[[hep-th/8802033](https://arxiv.org/abs/hep-th/8802033)].
The package provides topological data (Hodge numbers, triple intersection numbers,
second Chern class) for all CICYs, enabling flux compactification studies on this class
of geometries.


### Background on CICYs

A CICY threefold $X$ is specified by a **configuration matrix** $[n_r \,|\, k_{ra}]$,
where $\mathbb{P}^{n_r}$ are the ambient projective factors and $k_{ra}$ are the degrees
of the defining polynomials.
The Hodge numbers $h^{1,1}$ and $h^{2,1}$, the triple intersection numbers
$\kappa_{ijk}$, and the second Chern class $c_2(X)$ — which enter the Kähler potential
and the D3-brane tadpole — are stored in the package data.
Kähler cone data are also included, enabling the construction of the prepotential.

For details on using CICY data in JAXVacua, see {doc}`../jaxvacua.database`.


## Calabi-Yau fourfolds

In the future, we hope to include a similar implementation for F-theory flux
compactifications on Calabi-Yau fourfolds.
