# Period Calculations

In this note we review how the period vector $\Pi$ of a Calabi-Yau threefold $X$
is computed in practice.
This extends the conceptual introduction in {doc}`sugra`, where the period vector
and prepotential were defined, to the concrete computational procedure used in JAXVacua.
The discussion follows [[hep-th/9308122](https://arxiv.org/abs/hep-th/9308122)],
[[hep-th/9406055](https://arxiv.org/abs/hep-th/9406055)], and the review in
[[2512.17095](https://arxiv.org/abs/2512.17095)].


## Mirror symmetry and period computation

Consider a mirror pair of Calabi-Yau threefolds $(X, \widetilde{X})$, where we
compactify Type IIB string theory on $X$.
The complex structure moduli space $\mathcal{M}_{\text{cs}}(X)$ is classically exact
and is identified, via mirror symmetry, with the complexified K&auml;hler moduli space
$\mathcal{K}_{\widetilde{X}}$ of the mirror $\widetilde{X}$:

$$
\vec{\Pi}_{\text{IIB}} = \vec{\Pi}_{\text{IIA}} \equiv \vec{\Pi}\;.
$$ (eq:mirrorsays)

In the symplectic basis $\{\alpha_A, \beta^A\}$, $A = 0, \ldots, h^{2,1}$, the periods
of the holomorphic $(3,0)$-form $\Omega$ are

$$
\vec{\Pi}_{\text{IIB}} \coloneqq \begin{pmatrix}
\int_X \Omega \wedge \beta_A \\
\int_X \Omega \wedge \alpha^A
\end{pmatrix}\;.
$$ (eq:iibperdef)

Computing these periods directly is difficult in general.
However, when $X$ is a hypersurface in a toric variety, one can obtain the periods
around the *large complex structure* (LCS) point following a procedure due to
Hosono, Klemm, Theisen, and Yau
[[hep-th/9308122](https://arxiv.org/abs/hep-th/9308122)].


## The fundamental period

Let $(X, \widetilde{X})$ be a mirror pair of Calabi-Yau hypersurfaces in toric
varieties $(\widetilde{V}, V)$, defined by fine, regular, star triangulations
$(\widetilde{\mathcal{T}}, \mathcal{T})$ of a dual pair of reflexive polytopes
$(\Delta, \Delta^\circ)$.
We take $X$ to be the vanishing locus of a generic anticanonical polynomial

$$
f(\mathbbm{t}) = \Psi^0 S_0(\mathbbm{t}) - \sum_{I=1}^m \Psi^I S_I(\mathbbm{t})\;,
$$ (eq:DefFX)

specified in terms of $m+1$ complex parameters $\Psi^I$ and monomials $S_I(\mathbbm{t})$
of the toric coordinates $(\mathbbm{t}^1, \ldots, \mathbbm{t}^4)$.

We introduce the *gauged linear sigma model (GLSM) matrix* with entries $Q^a_{~I}$,
$a = 1, \ldots, h^{1,1}$, $I = 1, \ldots, h^{1,1} + 4$, recording the divisor class
decomposition $[\widehat{D}_I] = \sum_a Q^a_{~I} \widehat{H}_a$.
The anticanonical charge is $Q^a_{~0} \coloneqq \sum_I Q^a_{~I}$.

Writing the holomorphic $3$-form as

$$
\Omega = \oint_{f=0} \frac{d\mathbbm{t}^1 \wedge d\mathbbm{t}^2 \wedge d\mathbbm{t}^3 \wedge d\mathbbm{t}^4}{(2\pi i)^4 \cdot f(\mathbbm{t})}\;,
$$

the *fundamental period* $\varpi_0(\psi)$ is defined by integrating $\Omega$ over the
SYZ $T^3$ fiber:

$$
\varpi_0(\psi) \coloneqq \Psi^0 \int_{T^3} \Omega = \Psi^0 \oint_{|\mathbbm{t}^1|=\epsilon} \frac{d\mathbbm{t}^1}{2\pi i} \cdots \oint_{|\mathbbm{t}^4|=\epsilon} \frac{d\mathbbm{t}^4}{2\pi i} \frac{1}{f(\mathbbm{t})}\;.
$$ (eq:fundef)

Applying the residue theorem and multinomial expansion, one arrives at the expression

$$
\varpi_0(\psi) = \sum_{\tilde{\mathbf{q}} \in \mathcal{M}_{\widetilde{V}} \cap H^2(\widetilde{V}, \mathbb{Z})} \frac{\Gamma(1 + \tilde{q}_a Q^a_{~0})}{\prod_I \Gamma(1 + \tilde{q}_a Q^a_{~I})} \, \psi^{\tilde{\mathbf{q}}} =: \sum_{\tilde{\mathbf{q}}} c_{\tilde{\mathbf{q}}} \, \psi^{\tilde{\mathbf{q}}}\;,
$$ (eq:fundamental_period)

where the sum runs over the Mori cone $\mathcal{M}_{\widetilde{V}}$ of the mirror toric
variety $\widetilde{V}$, and $\psi^{\tilde{\mathbf{q}}} \coloneqq \prod_a (\psi^a)^{\tilde{q}_a}$.


## Higher periods

All remaining periods are determined by the fundamental period
$\varpi_0(\psi) = \sum_{\tilde{\mathbf{q}}} c_{\tilde{\mathbf{q}}} \, \psi^{\tilde{\mathbf{q}}}$
via $\rho$-derivatives [[hep-th/9308122](https://arxiv.org/abs/hep-th/9308122)]:

$$
\varpi^a(\psi) = \sum_{\tilde{\mathbf{q}}} \left. \frac{\partial_{\rho_a}}{2\pi i} \left( c_{\tilde{\mathbf{q}}+\vec{\rho}} \, \psi^{\tilde{\mathbf{q}}+\vec{\rho}} \right) \right|_{\vec{\rho}=0}\;, \quad
\varpi^{ab}(\psi) = \sum_{\tilde{\mathbf{q}}} \left. \frac{\partial_{\rho_a} \partial_{\rho_b}}{(2\pi i)^2} \left( c_{\tilde{\mathbf{q}}+\vec{\rho}} \, \psi^{\tilde{\mathbf{q}}+\vec{\rho}} \right) \right|_{\vec{\rho}=0}\;,
$$ (eq:pfrelations1)

$$
\varpi^{abc}(\psi) = \sum_{\tilde{\mathbf{q}}} \left. \frac{\partial_{\rho_a} \partial_{\rho_b} \partial_{\rho_c}}{(2\pi i)^3} \left( c_{\tilde{\mathbf{q}}+\vec{\rho}} \, \psi^{\tilde{\mathbf{q}}+\vec{\rho}} \right) \right|_{\vec{\rho}=0}\;.
$$ (eq:pfrelations2)

At zeroth order in $\psi$, these reduce to

$$
\varpi_0 \simeq 1\;, \quad \varpi^a \simeq \frac{\log(\psi^a)}{2\pi i}\;, \quad
\frac{1}{2} \widetilde{\kappa}_{abc} \varpi^{ab} \simeq \frac{1}{2} \widetilde{\kappa}_{abc} \varpi^a \varpi^b - \frac{1}{24} \tilde{c}_a\;,
$$

$$
\frac{1}{3!} \widetilde{\kappa}_{abc} \varpi^{abc} \simeq \frac{1}{3!} \widetilde{\kappa}_{abc} \varpi^a \varpi^b \varpi^c - \frac{1}{24} \tilde{c}_a \varpi^a + \frac{\zeta(3)}{(2\pi i)^3} \chi(\widetilde{X})\;,
$$

in terms of the triple intersection numbers $\widetilde{\kappa}_{abc}$ of the mirror
$\widetilde{X}$, the second Chern class integrals
$\tilde{c}_a = \int_{\widetilde{X}} c_2(\widetilde{X}) \wedge \tilde{\beta}_a$,
and the Euler characteristic $\chi(\widetilde{X})$.


## Integral symplectic basis

Mirror symmetry implies that the LCS monodromies in Type IIB on $X$ equal the
large-volume monodromies of Type IIA on $\widetilde{X}$, which are determined by the
intersection form of $\widetilde{X}$.
Using these monodromies to fix all integration constants and adopting a suitable
normalization, one obtains the periods in an integral symplectic basis:

$$
\Pi(\psi) = \frac{1}{\varpi_0} \begin{pmatrix}
\frac{1}{3!} \widetilde{\kappa}_{abc} \varpi^{abc} + \frac{1}{12} \tilde{c}_a \varpi^a \\
-\frac{1}{2} \widetilde{\kappa}_{abc} \varpi^{bc} + \tilde{a}_{ab} \varpi^b \\
\varpi_0 \\
\varpi^a
\end{pmatrix}\;,
$$ (eq:period_symplectic)

where the matrix $\tilde{a}_{ab}$ is defined as

$$
\tilde{a}_{ab} \equiv \frac{1}{2} \begin{cases}
\widetilde{\kappa}_{aab} & a \geq b \\
\widetilde{\kappa}_{abb} & a < b
\end{cases}\;.
$$


## Gopakumar-Vafa invariants

The flat coordinates $z^a$ are related to the algebraic coordinates $\psi^a$ via

$$
z^a = \frac{\varpi^a}{\varpi_0} = \frac{\log(\psi^a)}{2\pi i} + \frac{1}{2\pi i} \frac{c^a(\psi)}{\varpi_0}\;,
$$ (eq:flatcoords)

with correction terms $c^a(\psi) = \sum_{\tilde{\mathbf{q}}} c^a_{\tilde{\mathbf{q}}} \psi^{\tilde{\mathbf{q}}}$ where $c^a_{\tilde{\mathbf{q}}} = \partial_{\rho_a} c_{\tilde{\mathbf{q}}+\vec{\rho}} |_{\vec{\rho}=0}$.
The LCS limit corresponds to $e^{2\pi i z^a} \ll 1$.

The non-perturbative part of the Type IIA prepotential can be written in terms of
genus-zero *Gopakumar-Vafa (GV) invariants* $\mathscr{N}_{\tilde{\mathbf{q}}} \in \mathbb{Z}$
[[hep-th/9809187](https://arxiv.org/abs/hep-th/9809187)],
[[hep-th/9812127](https://arxiv.org/abs/hep-th/9812127)]:

$$
\mathcal{F}_{\text{inst}}(z) = -\frac{1}{(2\pi i)^3} \sum_{\tilde{\mathbf{q}} \in \mathcal{M}(\widetilde{X})} \mathscr{N}_{\tilde{\mathbf{q}}} \, \text{Li}_3\!\left(e^{2\pi i \, \tilde{\mathbf{q}} \cdot \mathbf{z}}\right)\;,
$$ (eq:IIAprep)

where the sum runs over effective curve classes in the Mori cone $\mathcal{M}_{\widetilde{X}}$.
By comparing the instanton expansion {eq}`eq:IIAprep` with the period expressions, one
can extract the GV invariants order by order in $\psi$.
Specialized algorithms for this extraction in models with many moduli were implemented
in [[2303.00757](https://arxiv.org/abs/2303.00757)].


## Summary of LCS formulas

The prepotential $\mathcal{F}$ in the LCS patch decomposes as

$$
\mathcal{F}(z) = \mathcal{F}_{\text{poly}}(z) + \mathcal{F}_{\text{inst}}(z)\;,
$$ (eq:prepotential_decomp)

with the polynomial part

$$
\mathcal{F}_{\text{poly}}(z) = -\frac{1}{3!} \widetilde{\kappa}_{abc} z^a z^b z^c + \frac{1}{2} \tilde{a}_{ab} z^a z^b + \frac{1}{24} \tilde{c}_a z^a + \frac{\zeta(3) \chi(\widetilde{X})}{2(2\pi i)^3}\;,
$$ (eq:fpoly)

and the instanton part $\mathcal{F}_{\text{inst}}(z)$ given in {eq}`eq:IIAprep`, in terms
of the polylogarithm $\text{Li}_k(x) = \sum_{n=1}^\infty x^n / n^k$.

The polynomial terms are computed from the triple intersection numbers
$\widetilde{\kappa}_{abc}$ of the mirror threefold $\widetilde{X}$, together with

$$
\tilde{c}_a = \int_{\widetilde{X}} c_2(\widetilde{X}) \wedge \tilde{\beta}_a\;, \quad
\tilde{a}_{ab} \equiv \frac{1}{2} \begin{cases}
\widetilde{\kappa}_{aab} & a \geq b \\
\widetilde{\kappa}_{abb} & a < b
\end{cases}\;, \quad
\chi(\widetilde{X}) = \int_{\widetilde{X}} c_3(\widetilde{X})\;,
$$

for a basis $\{\tilde{\beta}_a\}_{a=1}^{h^{1,1}(\widetilde{X})}$ of
$H^2(\widetilde{X}, \mathbb{Z})$.
The instanton contribution $\mathcal{F}_{\text{inst}}(z)$ accounts for Type IIA worldsheet
instanton corrections, encoded in the genus-zero Gopakumar-Vafa invariants
$\mathscr{N}_{\tilde{\mathbf{q}}}$.

The period vector {eq}`eq:PeriodVecGen` can then be computed from $\mathcal{F}$ as
reviewed in {doc}`sugra`.


### Implementation in `jaxvacua`

The period computation infrastructure is spread across several modules:

```{eval-rst}
.. currentmodule:: jaxvacua.periods

.. autosummary::

    periods.period_vector_per
    periods.prepot_per
    periods.kahler_potential_per

```

```{eval-rst}
.. currentmodule:: jaxvacua.css

.. autosummary::

    css.F_LCS_poly
    css.F_inst
    css.F_LCS

```

The topological input data (intersection numbers $\widetilde{\kappa}_{abc}$,
second Chern class integrals $\tilde{c}_a$, GV invariants $\mathscr{N}_{\tilde{\mathbf{q}}}$)
is provided through the `jaxvacua.lcs` module, which constructs the data tree from
CYTools or from pre-computed model files.
