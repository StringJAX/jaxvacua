# Perturbatively Flat Vacua

Perturbatively flat vacua (PFVs) are a mechanism for achieving exponentially small
values of the flux superpotential $|W_0| \ll 1$, as required by the KKLT scenario
(see {doc}`moduli_stabilisation`).
The idea, introduced in [[1903.00596](https://arxiv.org/abs/1903.00596)], is to choose
quantized fluxes such that all perturbative contributions to $W_{\text{flux}}$ vanish
exactly, leaving only exponentially suppressed worldsheet instanton corrections.
This note follows the review in [[2512.17095](https://arxiv.org/abs/2512.17095)],
sections 6.2 and 6.4.


## The PFV mechanism

We split the period vector into polynomial and exponential contributions,

$$
\vec{\Pi} = \vec{\Pi}_{\text{poly}} + \vec{\Pi}_{\text{exp}}\;,
$$

so that the flux superpotential decomposes as

$$
W_{\text{flux}} = W_{\text{poly}} + W_{\text{exp}}\;,
$$ (eq:PFVWsplit)

where

$$
W_{\text{poly}} = \vec{\Pi}_{\text{poly}}^T \cdot \Sigma \cdot (\vec{f} - \tau \vec{h})\;, \quad
W_{\text{exp}} = \vec{\Pi}_{\text{exp}}^T \cdot \Sigma \cdot (\vec{f} - \tau \vec{h})\;.
$$

The goal is to engineer flux choices for which $\langle W_{\text{poly}} \rangle = 0$
in the vacuum, so that $|W_0| \sim |\langle W_{\text{exp}} \rangle| \ll 1$.


### Flux ansatz and Diophantine conditions

Working with the LCS prepotential {eq}`eq:fpoly`, we make the flux ansatz

$$
\vec{f} = (R_0, R_a, 0, M^a)^\top\;, \quad \vec{h} = (0, K_a, 0, 0^a)^\top\;,
$$ (eq:PFVflux)

with all entries integer-valued and $a = 1, \ldots, h^{2,1}$.
The polynomial superpotential becomes

$$
W_{\text{poly}}(z^a, \tau) = \frac{1}{2} M^a \widetilde{\kappa}_{abc} z^b z^c - \tau K_a z^a + (R_a - \tilde{a}_{ab} M^b) z^a + \left(R_0 - \frac{M^a \tilde{c}_a}{24}\right)\;,
$$ (eq:WfluxPoly)

while the exponential part is

$$
W_{\text{exp}}(z^a) = -\frac{1}{(2\pi)^2} \sum_{\tilde{\mathbf{q}} \in \mathcal{M}_{\widetilde{X}}} \mathscr{N}_{\tilde{\mathbf{q}}} \, \tilde{\mathbf{q}}_a M^a \, \text{Li}_2\!\left(e^{2\pi i \, \tilde{\mathbf{q}}_a z^a}\right)\;.
$$ (eq:WfluxInst)

To cancel the constant and linear terms in {eq}`eq:WfluxPoly`, we impose the
*Diophantine conditions*

$$
R_a = \tilde{a}_{ab} M^b \in \mathbb{Z}\;, \quad R_0 = \frac{M^a \tilde{c}_a}{24} \in \mathbb{Z}\;.
$$ (eq:PFVPconst)

These can only be satisfied for particular choices of $M^a$ for which the right-hand
sides take integer values.


### Flat direction and PFV conditions

With {eq}`eq:PFVPconst` imposed, the polynomial superpotential simplifies to

$$
W_{\text{poly}}(z^a, \tau) = \frac{1}{2} N_{ab} z^a z^b - \tau K_a z^a\;,
$$

where $N_{ab} \coloneqq \kappa_{abc} M^c$.
Assuming $N_{ab}$ is invertible, $\partial_{z^a} W_{\text{poly}} = 0$ is solved along the
one-dimensional locus

$$
z^a = p^a \tau\;, \quad p^a \coloneqq N^{ab} K_b\;.
$$ (eq:znp)

One further requires $K_a p^a = K_a N^{ab} K_b = 0$, which ensures $W_{\text{poly}}$
and all its derivatives vanish along {eq}`eq:znp`.

In summary, the **PFV conditions** are:

$$
\det N \neq 0\;, \quad \vec{p} \in \mathcal{K}_{\widetilde{X}}\;, \quad K_a p^a = 0\;, \quad \tilde{a}_{ab} M^b \in \mathbb{Z}\;, \quad \tilde{c}_a M^a \in 24\mathbb{Z}\;.
$$ (eq:PFV)


## Racetrack stabilization

Along the flat direction $z^a = p^a \tau$, the flux superpotential reduces to
$W_{\text{exp}}$ in {eq}`eq:WfluxInst`, which takes the form

$$
W_{\text{eff}}(\tau) = c \left(e^{2\pi i p^1 \tau} + A \, e^{2\pi i p^2 \tau}\right) + \ldots\;,
$$ (eq:weffis)

where $c$ and $A$ depend on $\vec{M}$ and $\vec{K}$ but not on $\tau$.
When $|p^1 - p^2| \ll p^2$, the two exponential terms compete — this is the
*racetrack mechanism* [[hep-th/0404257](https://arxiv.org/abs/hep-th/0404257)].

Solving $\partial_\tau W_{\text{eff}} = 0$ gives

$$
\langle \tau \rangle = \frac{1}{2\pi i} \frac{1}{p^1 - p^2} \ln\!\left(-A \frac{p^2}{p^1}\right)\;,
$$

and the stabilized superpotential is

$$
W_{\text{eff}}(\langle \tau \rangle) = c \, \frac{p^2 - p^1}{p^2} \left(-A \frac{p^2}{p^1}\right)^{\frac{p^1}{p^1 - p^2}}\;,
$$

which is small precisely when $|p^1 - p^2| \ll p^2$.


### Example: degree-18 hypersurface

For the degree-18 hypersurface in $\mathbb{CP}_{[1,1,1,6,9]}$
[[hep-th/9309013](https://arxiv.org/abs/hep-th/9309013)] with $h^{2,1} = 2$ (on the
$\mathbb{Z}_6 \times \mathbb{Z}_{18}$-invariant locus) and $Q_{\text{D3}} = 138$,
one can choose

$$
\vec{M} = \begin{pmatrix} -16 \\ 50 \end{pmatrix}\;, \quad
\vec{K} = \begin{pmatrix} 3 \\ -4 \end{pmatrix}\;,
$$

yielding $Q_{\text{flux}} = 124$ and a PFV racetrack minimum at

$$
\langle \tau \rangle = 6.856\, i\;, \quad
\langle z^1 \rangle = 2.742\, i\;, \quad
\langle z^2 \rangle = 2.057\, i\;,
$$

with $|W_0| = 2.037 \times 10^{-8}$.


## Conifold PFVs

The PFV mechanism can be extended to stabilize moduli near a *conifold singularity*,
engineering a warped Klebanov-Strassler throat in a compact flux compactification
[[2004.10740](https://arxiv.org/abs/2004.10740)].


### Analytic continuation near the conifold

Conifold singularities arise at loci in $\mathcal{M}_{\text{cs}}(X)$ where a collection
of three-cycles shrink to zero volume.
The conifold modulus $z_{\text{cf}} = \tilde{\mathbf{q}}_{\text{cf}} \cdot \mathbf{z}$
controls the size of the shrinking cycle.

Near $z_{\text{cf}} \to 0$, the instanton prepotential must be analytically continued.
For a nilpotent conifold class with GV invariant
$n_{\text{cf}} = \mathscr{N}_{\tilde{\mathbf{q}}_{\text{cf}}}$, one finds

$$
\mathcal{F}(z_{\text{cf}}, z^\alpha) = n_{\text{cf}} \frac{z_{\text{cf}}^2}{4\pi i} \ln(-2\pi i \, z_{\text{cf}}) + \sum_{n=0}^{\infty} \frac{\mathcal{F}^{(n)}(z^\alpha)}{n!} z_{\text{cf}}^n\;,
$$ (eq:FconiLCS)

where the logarithmic term encodes the characteristic conifold monodromy and the
coefficients $\mathcal{F}^{(n)}(z^\alpha)$ depend on the polynomial prepotential and
the remaining (bulk) instanton corrections evaluated at $z_{\text{cf}} = 0$.


### Bulk and conifold superpotential

With quantized fluxes $\vec{f} = (P_0, P_a, 0, M^a)^\top$ and
$\vec{h} = (0, K_a, 0, 0^a)^\top$, the superpotential expands at leading order as

$$
W(z^\alpha, z_{\text{cf}}, \tau) = W_{\text{bulk}}(z^\alpha, \tau) + z_{\text{cf}} \, W^{(1)}(z^\alpha, z_{\text{cf}}, \tau) + \mathcal{O}(z_{\text{cf}}^2)\;.
$$ (eq:WExpConi)

The bulk superpotential $W_{\text{bulk}}$ takes the same form as {eq}`eq:WfluxPoly`
but with a shifted constant term $\tilde{c}'_a = \tilde{c}_a + n_{\text{cf}} \delta_{a,1}$.
The linear coefficient is

$$
W^{(1)} = -M \frac{n_{\text{cf}}}{2\pi i} \left(\ln(-2\pi i \, z_{\text{cf}}) - 1\right) - \tau K + \widetilde{\kappa}_{1a\gamma} M^a z^\gamma + \ldots\;,
$$ (eq:W1coniLCS)

where $M \coloneqq M^1$ and $K \coloneqq K_1$.


### Exponential hierarchy from the conifold

The $F$-flatness condition for $z_{\text{cf}}$ is satisfied at

$$
\langle |z_{\text{cf}}| \rangle = \frac{1}{2\pi} \exp\!\left(-\frac{2\pi K'}{(g_s M) \, n_{\text{cf}}}\right)\;,
$$ (eq:conifold_vev)

where

$$
K' = K - g_s \widetilde{\kappa}_{1a\beta} M^a \text{Im}(z^\beta)\;.
$$ (eq:Kprime)

Provided $K'/M > 0$, the conifold modulus is stabilized at an exponentially small value,
giving rise to a warped throat region.
The D3-brane charge stored in the throat is

$$
Q_{\text{flux}}^{\text{throat}} = K' M > 0\;.
$$

Importantly, $K'$ differs from the naive product $K \cdot M$ by corrections from the
bulk moduli {eq}`eq:Kprime` — a crucial effect absent in the non-compact
Klebanov-Strassler solution.

The bulk moduli are stabilized via the **conifold PFV conditions**:

$$
\det N \neq 0\;, \quad \vec{p} \in \mathcal{K}_{\text{cf}}\;, \quad K_\alpha p^\alpha = 0\;, \quad \tilde{a}_{\alpha b} M^b \in \mathbb{Z}\;, \quad \tilde{c}'_a M^a \in 24\mathbb{Z}\;,
$$ (eq:coniPFV)

with $N_{\alpha\beta} = M^a \kappa_{a\alpha\beta}$ and $p^\alpha = N^{\alpha\beta} K_\beta$
running over the *bulk* indices $\alpha$ only.
Along $z^\alpha = p^\alpha \tau$, the remaining flat direction is lifted by the bulk
racetrack superpotential

$$
W_{\text{bulk}}^{\text{eff}}(\tau) = -\frac{1}{(2\pi)^2} \sum_{\tilde{\mathbf{q}} \neq \tilde{\mathbf{q}}_{\text{cf}}} \mathscr{N}_{\tilde{\mathbf{q}}} \, \tilde{\mathbf{q}}_a M^a \, \text{Li}_2\!\left(e^{2\pi i \, \tilde{\mathbf{q}}_\alpha p^\alpha \tau}\right)\;.
$$ (eq:WfluxBulk3)


### Implementation in `jaxvacua`

```{eval-rst}
.. currentmodule:: jaxvacua.flux_bounding

.. autosummary::

    bounded_fluxes.enumerate_fluxes
    bounded_fluxes.sample_bounded_fluxes

```

The conifold period computation is handled through the `jaxvacua.conifold`
subpackage and the freezer module `jaxvacua.freezer`, which implements the
light-field EFT for conifold vacua.
