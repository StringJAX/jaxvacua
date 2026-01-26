# Construction of Calabi-Yau geometries

We are mostly concerned with Calabi-Yau threefolds which we abriviate by *CYs* in what follows.
In the future, support for F-theory flux compactifications on Calabi-Yau fourfolds is desirable, see the end of this section.

## CY hypersurfaces from reflexive polytopes


Smooth Calabi-Yau threefold hypersurfaces can be obtained from triangulations of reflexive polytopes in four dimensions. These were enumerated by Kreuzer and Skarke in \cite{Kreuzer:2000xy} finding 473,800,776 such polytopes. With the development of [*CYTools*](https://cy.tools), efficient and targeted construction of such CY geometries became feasible on large scales opening a new window to study string theory compactifications. Our implementation supports [*CYTools*](https://cy.tools) capabilities for the construction of the EFT from compactifications on such Calabi-Yau threefold hypersurfaces.

### Batyrev construction

We begin by setting notation and terminology for Calabi-Yau threefold hypersurfaces in toric varieties, which we will obtain from triangulations of four-dimensional reflexive polytopes from the Kreuzer-Skarke list \cite{Kreuzer:2000xy}. More details on this construction can be found e.g. in \cite{Demirtas:2020dbm,Braun:2017nhi}.


Suppose that $\Delta \subset \mathbb{Z}^4$ is a four-dimensional reflexive polytope, and denote by $\Delta^{\circ}$ its polar dual. We call a dual pair $(\Delta,\Delta^{\circ})$ **LOOK AT SMALL CCs!!!**

Let $\mathcal{T}$ be a regular, star triangulation of $\Delta^{\circ}$ in which every point of $\Delta^{\circ}$ that is not interior to a facet is a vertex of a simplex of $\mathcal{T}$. (We note that such a $\mathcal{T}$ is not in general a **fine**, regular, star triangulation (FRST), because of the omission of points interior to facets, but the associated subvarieties of $V$ do not intersect a generic hypersurface $X$, and so are immaterial for our analysis.  With this understanding, we refer to such triangulations $\mathcal{T}$ as FRSTs, because they are fine enough for our needs.) The toric fan associated to $\mathcal{T}$  defines a four-dimensional toric variety $V$ in which the generic anticanonical hypersurface $X$ is a smooth Calabi-Yau threefold \cite{Batyrev:1993oya}.  

Regular triangulations of polytopes can be represented by a vector of heights which lifts the point collection of $\Delta^\circ$ to one higher dimension, see e.g. \cite{Demirtas:2020dbm}. As is well known, there is a massive redundancy when going from polytope triangulations to Calabi-Yau hypersurfaces. Indeed, Wall's theorem \cite{wall} which asserts that Hodge numbers, triple intersection numbers, and second Chern classes completely determine the homotopy type of a compact, simply connected Calabi-Yau threefold with torsion-free homology. For Calabi-Yau hypersurfaces, Hodge numbers are fixed by polytope data, while triple intersection numbers and second Chern classes are determined purely by the induced triangulations of two-faces. Therefore, FRSTs of $\Delta^{\circ}$ with identical restrictions to two-faces give rise to topologically equivalent Calabi-Yau threefolds. When performing optimization on the space of FRSTs, it turns out to be beneficial to find an **encoding** for FRSTs which avoids such trivial redundancies as was studied in \cite{GApaper}. 

### Orientifolds


We say that a polytope $\Delta^{\circ}$ is **trilayer** if the points of $\Delta^{\circ}$ lie in exactly three distinct affine sub-lattices of codimension one, in a sense made precise in \cite{orientifolds}. Calabi-Yau threefold hypersurfaces in toric varieties $V$ resulting from triangulations of trilayer polytopes admit very convenient orientifold actions \cite{orientifolds}. In each case there exists a certain toric coordinate, which we denote by $x_1$, such that the involution of $V$ defined by

$$
    x_1 \to -x_1\,
$$ (eq:odef)

yields, when restricted to the generic invariant hypersurface $X \subset V$, an orientifold with $h^{1,1}_-=h^{2,1}_+=0$, which we will refer to as a **trilayer orientifold**. All  orientifolds considered in this work will be of this type.


Given a Calabi-Yau orientifold $X/\mathcal{I}$, the D3-brane tadpole, defined in \eqref{eq:D3-charges}, is a useful measure of the richness of flux vacua that one can expect in compactification on $X/\mathcal{I}$. For the orientifolds considered here,  we have 

$$
Q_{\text{O}} =  2+h^{1,1}+h^{2,1}\, .
$$ (eq:qdef)

Thus, the D3-brane tadpole is large if either Hodge number is large. However,  the construction of large ensembles of PFVs becomes expensive for $h^{2,1} \gtrsim 10$, so we restrict ourselves to the range $3 \le h^{2,1} \le 8$. The D3-brane tadpole can then still be large for sufficiently large values of $h^{1,1}$. In practice, we restrict to polytopes with either $3\leq h^{2,1}\leq 5$ and $Q_{\text{O}}\ge 100$, or $6\leq h^{2,1}\leq 8$ and $Q_{\text{O}}\ge 150$. 

**mention orientifolds**


### Prime toric divisors

A consequence of $\Delta^{\circ}$-favorability is that $h^{1,1}(V) = h^{1,1}(X)$, with $h^{1,1}(V)+4$ toric coordinates $x_I$ generating the Cox ring.  We define the prime toric divisors of $V$ as $\hat{D}_I =  \{x_I = 0\}$, and we refer to $D_I =  \hat{D}_I \cap X$ as the prime toric divisors of $X$. The $D_I$ are all effective divisors, and (again using $\Delta^{\circ}$-favorability) they generate $H_4(X,\mathbb{Z})$.  Even so, in general there exist effective divisors $D$ that are non-positive integer linear combinations of the $D_I$, which we term **autochthonous divisors**. **MAYBE RATHER IMPORTANT FOR KÄHLER STABILISATION!**


## Complete Intersection CYs

Our opensource package contains topological information about all **Complete Intersection Calabi-Yau threefolds** (CICYs).
The implementation provides the following functionality:
* grab CICY by ID
* ?

**describe the different databases for CICYs and mention the equivalences that we also upload as file?!**


### Background on CICYs

**see paper by Carta et al. on winding uplift as summary?**

**Describe the Kähler cones?**


## Calabi-Yau fourfolds


In the future, we hope to include a similar implementation for F-theory flux compactifications on Calabi-Yau fourfolds.



