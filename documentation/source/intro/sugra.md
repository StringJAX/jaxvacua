# Type IIB Flux Compactifications


In this note, we review Type IIB flux compactifications on Calabi-Yau orientifolds.
The main purpose of this section is to concretise the type of equations that need to be solved for flux vacua.
These considerations are prerequisites for the subsequent implementation of sampling methods.


## 4D $\mathcal{N}=1$ Type IIB supergravity


To address questions related to moduli stabilisation, the object of interest is the $F$-term scalar potential $V_{F}$.
In $4$-dimensional Planck units $M_{P}=1$,
we define the $F$-term scalar potential $V_{F}$ as


```{math}
:label: eq:Vtot

V_{F}=\text{e}^{ K}\left ( K^{ A\overline{ B}}\, D_{ A} W\, D_{\overline{ B}}\overline{ W}-3| W|^{2}\right )\, .
```


Here, $\Phi_{ A}$ collectively denotes all complex scalars with $ K_{ A\overline{ B}}$ being the associated Kähler metric on field space. It is derived from a Kähler potential $K$ via


```{math}
:label: eq:intro-sugra-01

 K_{ A\overline{ B}}= \dfrac{\partial^{2} K}{\partial\Phi^{ A}\partial\overline{\Phi}^{ B}}\, .
```


Further, $W$ is the superpotential and $ K^{A\overline{ B}}$ is the inverse Kähler metric. The Kähler covariant derivative $D_{A}$ is defined as


```{math}
:label: eq:FTERMSDEFSUGRA

 
D_{A} W= \dfrac{\partial W}{\partial\Phi^{ A}}+\dfrac{\partial K}{\partial\Phi^{ A}}\,  W\equiv \partial_{ A} W+ K_{ A} W\, .
```


SUSY is preserved in the vacuum provided the following $F$-flatness conditions are satisfied


```{math}
:label: eq:FtermFlatnessSugra4D

D_{ A} W=0\; ,\quad  \forall\, \, \Phi_{ A}\, .
```


The scalar spectrum of the 4D $\mathcal{N} = 1$ EFT from chiral multiplets consists of the Kähler moduli $T_\alpha$, the complex structure moduli $z^i$ and the axio-dilaton ${\tau}$.
In what follows, we mainly focus on ($z^i, {\tau}$) defined as,


```{math}
:label: eq:N1coordsOrig

z^i = v^i + \text{i}\, u^i\; ,\quad  {\tau} \, =  c_0 + \text{i}\, s \, .
```


The $z^{i}$ parametrise the complex structure moduli space $\mathcal{M}_{\text{cs}}(X_{3})$,
while $\tau$ is the axio-dilaton.


The Kähler potential $ K$ in {eq}`eq:Vtot` needs to be computed order by order in the string-loop and $\alpha^{\prime}$-expansion.
Using appropriate chiral variables,
the classical K\"{a}hler potential $ K_{0}$ can be written as


```{math}
:label: eq:TreeLevKP

K= \mathbb{K}(z^i, \overline{z}^i) - \log\left(-\text{i}({\tau}-\overline{\tau})\right) -2\log \left ( \mathcal{V}\right )
```


Here, $\mathcal{V}$ is the Calabi-Yau volume in string units.  If
$\{\omega_\alpha\}$ is an integral divisor basis of $H^{1,1}(X_3,\mathbb{Z})$
and


```{math}
:label: eq:CYVolumeKahlerForm

J = t^\alpha \omega_\alpha
```


is the Kähler form, then


```{math}
:label: eq:CYAndDivisorVolumes

\mathcal{V}
= \frac{1}{3!}\int_{X_3} J\wedge J\wedge J
= \frac{1}{6}\kappa_{\alpha\beta\gamma}\,
   t^\alpha t^\beta t^\gamma\;,\qquad
\tau_\alpha
= \operatorname{Vol}(D_\alpha)
= \frac{1}{2}\int_{D_\alpha} J\wedge J
= \frac{1}{2}\int_{X_3}\omega_\alpha\wedge J\wedge J
= \frac{\partial\mathcal{V}}{\partial t^\alpha}\, .
```


The real parameters $t^\alpha$ are two-cycle volumes, while the real parts of
the Kähler moduli are the divisor, or four-cycle, volumes $\tau_\alpha$ up to
the convention-dependent placement of axions.


in terms of the complex structure Kähler potential


```{math}
:label: eq:Kgennn

\mathbb{K}(z^i, \overline{z}^i) =-\log( \mathcal{A}(z^{i},\overline{z}^{i}))\; ,\quad  \mathcal{A}(z^{i},\overline{z}^{i})=-\text{i}\int_{X}\Omega_3\wedge{\overline{\Omega}_3}\, .
```


It depends only on the $z^{i}$, $i=1,\ldots, h_{-}^{1,2}$, through the $3$-form $\Omega_{3}=\Omega_{3}(z^{i})$.
The moduli space $\mathcal{M}_{\text{cs}}(X_{3})$ is classically exact, i.e., it is not renormalised by any quantum corrections.


The $3$-form $\Omega_{3}$ in {eq}`eq:Kgennn` can be parametrised by the real, symplectic basis of $3$-forms $(\alpha_I, \beta^J)\in H_{-}^{3}(X_{3},\mathbb{Z})$, $I,J\in \{0, ..., h^{1,2}_-(X_3)\}$, together with a dual basis of $3$-cycles $(A_{I},B^{J})\in H^{-}_{3}(X_{3},\mathbb{Z})$ so that


```{math}
:label: eq:PeriodsInt

\Omega_{3}=  X^I \, \alpha_I - \,  F_{J} \,  \beta^J\; ,\quad  X^{I}= \int_{A_{I}}\, \Omega_{3}\; ,\quad  F_{J}= \int_{B^{J}}\, \Omega_{3}  \, .
```


Here, $(X^{I},F_{J})$ are the so-called *periods* of $\Omega_{3}$ which are usually arranged in the *period vector* $\Pi=(X^{I},F_{J})$.
The periods {eq}`eq:PeriodsInt` can be computed from solving Picard-Fuchs equations [hep-th/9308122](https://arxiv.org/abs/hep-th/9308122), [hep-th/9406055](https://arxiv.org/abs/hep-th/9406055) or using asymptotic Hodge theory [2105.02232](https://arxiv.org/abs/2105.02232).
For the concrete computational procedure used in JAXVacua see {doc}`periods`.
A parametrisation of the complex structure moduli space $\mathcal{M}_{\text{cs}}(X_{3})$ is conveniently obtained by choosing half of the periods, say $X^{I}$, as projective coordinates.
Specifically, we set


```{math}
:label: eq:intro-sugra-02

z^{i}=\dfrac{X^{i}}{X^{0}}\; ,\quad  i=1,\ldots ,h^{1,2}_{-}\, .
```


In terms of the period vector $\Pi$,
{eq}`eq:Kgennn` becomes


```{math}
:label: eq:DefACSM

 
\mathcal{A}(z^{i},\overline{z}^{i})=-\text{i}\, \Pi^{\dagger}\cdot\Sigma\cdot \Pi \; ,\quad  \Sigma=\left (\begin{array}{cc}
0 & 1 \\ 
-1 & 0
\end{array} \right )\, .
```


We can actually be more explicit since $\mathcal{M}_{\text{cs}}(X_{3})$ possesses a *special* Kähler structure.
This implies that one can compute the periods and Kähler potential $ \mathbb{K}$ from a *pre-potential* $F(z)$ through (setting $X^{0}=1$)


```{math}
:label: eq:PeriodVecGen

\Pi=\left (\begin{array}{c}
 2F-z^{i} \partial_{i}F\\ 
 \partial_{i}F\\ 
1 \\ 
z^{i}
\end{array} \right )\; ,\quad  \mathcal{A}=\text{i} \left [2(F-\overline{F})-(z-\overline{z})^{i}\,  \partial_{i}(F+\overline{F})\right ]\, .
```


### Implementation in `jaxvacua.css`


```{eval-rst}
.. currentmodule:: jaxvacua.css

.. autosummary::

    css.prepot
    css.kahler_potential
    css.kahler_metric

```


## Large complex structure limits


In explicit examples,
we need to compute the period vector $\Pi$.
An important class of models concerns large complex structure limits of $\mathcal{M}_{\text{cs}}(X_{3})$ for which the periods can be inferred from mirror symmetry.


Let us consider such a concrete setup to compute the pre-potential $F(Z)$.
Around so-called *Large Complex Structure* (LCS) points $\text{Im}(z^{i})\gg 1$ in $\mathcal{M}_{\text{cs}}(X_{3})$,
we can make use of mirror duality.
Specifically, the coordinates $z^{i}$ of $\mathcal{M}_{\text{cs}}(X_{3})$ are identified with the Kähler moduli $T_{IIA}^{i}$ in the large volume limit $ \text{Im}(T_{IIA}^{i})\gg 1$ of Type IIA string theory compactified on the mirror dual CY $\tilde{X}_{3}$ [hep-th/9111025](https://arxiv.org/abs/hep-th/9111025),[hep-th/9308122](https://arxiv.org/abs/hep-th/9308122), [hep-th/9406055](https://arxiv.org/abs/hep-th/9406055).
Thus, in the LCS limit,
mirror duality identifies the moduli space $\mathcal{M}_{\text{cs}}(X_{3})$ of Type IIB with the Kähler moduli space $\mathcal{M}_{\text{K}}(\tilde{X}_{3})$ of the mirror dual CY threefold $\tilde{X}_{3}$.
One finds that the pre-potential reads [hep-th/9111025](https://arxiv.org/abs/hep-th/9111025),[hep-th/9308122](https://arxiv.org/abs/hep-th/9308122), [hep-th/9406055](https://arxiv.org/abs/hep-th/9406055),


```{math}
:label: eq:prepotentialNew

F(z)= -\dfrac{1}{6}\kappa_{ijk} \,  z^i\,  z^j \,  z^k +  \frac{1}{2} \,{a_{ij} \,  z^i\,  z^j} +  \,{b_{i} \,   z^i} +  \frac{\text{i}}{2} \, \,{\tilde \xi} +    F_{inst}(z)\,.
```


Here, $\kappa_{ijk}$ are the triple intersection numbers on the mirror threefold $\tilde{X}_3$ which are defined as


```{math}
:label: eq:intro-sugra-03

\kappa_{ijk} &= \int_{\tilde{X_3}} \, J_i \wedge J_j \wedge J_k\; ,\quad  a_{ij} = \frac{1}{2}\int_{\tilde{X_3}} \, J_i \wedge J_j \wedge J_j\,\text{mod}\,\mathbb{Z}\; ,\quad \nonumber\\
b_j &= \frac{1}{4!}\int_{\tilde{X_3}} \,c_2(\tilde{X_3}) \wedge J_j\; ,\quad  \tilde{\xi} = \frac{\zeta(3)\, \chi(\tilde{X_3})}{(2\pi)^3}\,.
```


Here, $c_{2}(\tilde{X}_{3})$ denotes the second Chern class of $\tilde{X}_{3}$.
Further,
the $J_{i}\in H_{-}^{1,1}(\tilde{X}_{3},\mathbb{Z})$ are $(1,1)$-forms and $\chi(\tilde{X}_{3})$ is the Euler characteristic of $\tilde{X}_{3}$.
The validity of the ansatz {eq}`eq:prepotentialNew` for $F$ is limited to the region of convergence of the LCS expansion [hep-th/9406055](https://arxiv.org/abs/hep-th/9406055). 
The term $\sim \tilde{\xi}$ is a 1-loop correction to the Kähler metric of the hypermultiplets in Type IIA which is obtained from reducing the 8-derivative $R^{4}$ term in the 10D action as computed in [hep-th/9707013](https://arxiv.org/abs/hep-th/9707013).


Finally,
the string WS corrections on the mirror dual side give rise to [hep-th/9308122](https://arxiv.org/abs/hep-th/9308122), [hep-th/9406055](https://arxiv.org/abs/hep-th/9406055)


```{math}
:label: eq:InstCorrections

 
F_{\text{inst}}(z^{i})=-\dfrac{1}{(2\pi \text{i})^{3}}\sum_{\beta \in \mathcal{M}(\tilde{X}_{3})}\, n_{\beta}^{0}\, \text{Li}_{3}(q^{\beta})\; ,\quad  \text{Li}_{3}(x)=\sum_{m=1}^{\infty}\, \dfrac{x^{m}}{m^{3}}\; ,\quad  q^{\beta}= \text{e}^{2\pi i d_{i}z^{i}}
```


in terms of genus zero *Gopakumar-Vafa (GV) invariants* $n_{\beta}^{0}$ [hep-th/9809187](https://arxiv.org/abs/hep-th/9809187), [hep-th/9812127](https://arxiv.org/abs/hep-th/9812127) of effective curves $\beta$ in the *Mori cone* $\mathcal{M}(\tilde{X}_{3})$ of the mirror manifold $\tilde{X}_{3}$. Occasionally, it turns out to be more convenient to work with a different set of invariants obtained from a resummation of poly-logarithms.
These are the so-called (genus zero) *Gromov-Witten (GW) invariants* $N_{\beta}$ which are related to the GV invariants via


```{math}
:label: eq:intro-sugra-04

\sum_{\beta \in \mathcal{M}(\tilde{X}_{3})}\, n^{0}_{\beta}\, \text{Li}_{3}(q^{\beta})=\sum_{\beta \in \mathcal{M}(\tilde{X}_{3})}\, N_{\beta}\,  q^{\beta}\, .
```


These invariants can be computed systematically in many cases e.g. using [CYTools](https://cy.tools), see in particular [2303.00757](https://arxiv.org/abs/2303.00757).


Now, the first derivatives of the pre-potential $F$ are given by


```{math}
:label: eq:Prepder1

 \partial_{X^{0}}F=F_0 &= -\, \frac{1}{6}\, \kappa_{ijk}\, z^i \, z^j\, z^k + p_i \, z^i + \text{i} \, \tilde\xi +\left(2\, F_{inst} - z^i\, \partial_{i} F_{inst} \right), \\
 \partial_{X^{i}}F=F_i &= -\frac{1}{2}\, \kappa_{ijk} \, z^j\, z^k + p_{ij}\, z^j + p_i + \left(\partial_{i} F_{inst} \right)\, . \nonumber
```


In terms of the $z^{i}$, the period vector {eq}`eq:PeriodVecGen` becomes


```{math}
:label: eq:PVLCSExpansionExplicit

 
\Pi=\left (\begin{array}{c}
-\, \frac{1}{6}\, \kappa_{ijk}\, z^i \, z^j\, z^k + p_i \, z^i + \text{i} \, \tilde\xi+\sum_{\beta}\,  2N_{\beta}(1-\pi\text{i} \beta_{i}z^{i})\, \text{e}^{2\pi\text{i} \beta_{i}z^{i}}\\
\frac{1}{2}\, \kappa_{ijk} \, z^j\, z^k + p_{ij}\, z^j + p_i +\sum_{\beta}\, 2\pi \text{i} N_{\beta}\beta_{i} \text{e}^{2\pi\text{i} \beta_{i}z^{i}}  \\
1 \\ 
z^{i} 
\end{array}
\right )
```


and hence from {eq}`eq:DefACSM`


```{math}
:label: eq:FctAcs

\mathcal{A}(Z,\overline{Z}) &=   \frac{\text{i}}{3!} \kappa_{ijk} (z^{i} - \overline{z}^{i})(z^{j} - \overline{z}^{j})(z^{k} - \overline{z}^{k}) -2\tilde{\xi} \\
& \quad + \sum_{\beta} \, 2\text{i} N_{\beta}[1-\pi \text{i}  \beta_{i}(z^{i}-\overline{z}^{i})] \, \left[e^{2 \pi \text{i}  \beta_{i} z^{i}} + e^{-2 \pi \text{i}  \beta_{i} \overline{z}^{i}} \right] \vphantom{\frac{1}{3!}}  \ .
```


The terms in the first line of Eq.{eq}`eq:FctAcs` are invariant under shifts $z^{i} \rightarrow z^{i}+\lambda$, $\lambda\in \mathbb{R}$,
though this continuous shift symmetry is broken to a discrete one by the exponentially suppressed terms in the second line.
To understand this,
one recalls that under mirror duality the complex structure moduli $z^{i}$ are identified with the Kähler moduli $T_{IIA}^{i}=b_{IIA}^{i}+\text{i}\, t_{IIA}^{i}$ of Type IIA.
Here, the $b_{IIA}^{i}$ are axions arising from the reduction of the NSNS $2$-form field ${B}_{2}$.
The above shift symmetry is therefore related to the gauge symmetry ${B}_{2} \rightarrow {B}_{2}+\text{d} \Lambda$ in the $10$D Type IIA theory which is respected to all orders in perturbation theory.
The breaking to discrete shifts is induced by non-perturbative effects, namely WS instantons on $2$-cycles as explained in [Wen, Witten: 1985](https://inspirehep.net/literature/17562).


### Implementation in `jaxvacua.css`

```{eval-rst}
.. currentmodule:: jaxvacua.css

.. autosummary::

    css.F_LCS_poly
    css.F_inst
    css.F_LCS

```


## No-scale flux scalar potential

The three-form fluxes $F_3$ and $H_3$ generate a superpotential for the complex structure
moduli $z^i$ and the axio-dilaton $\tau$.
The **Gukov-Vafa-Witten (GVW) superpotential** is


```{math}
:label: eq:GVWsugra

W_{\rm flux} = \int_{X_3} G_3 \wedge \Omega_3 = \int_{X_3} (F_3 - \tau H_3) \wedge \Omega_3\;,
```


with $G_3 = F_3 - \tau H_3$ the complexified three-form flux.
In terms of the period vector and integer flux quanta $f = (f^I)$, $h = (h_J)$, this reads
$W_{\rm flux} = (f - \tau h)\cdot\Sigma\cdot\Pi$.

Due to the no-scale identity $K^{\alpha\bar\beta}\partial_\alpha K\partial_{\bar\beta}K = 3$
satisfied by the classical Kähler potential $K_K = -2\log\mathcal{V}$,
the Kähler moduli F-terms cancel the $-3|W|^2$ term in {eq}`eq:Vtot`.
The **no-scale flux potential** therefore takes the positive semi-definite form


```{math}
:label: eq:VfluxCS

V_{\rm flux} = e^K K^{a\bar{b}} D_a W_{\rm flux}\, D_{\bar{b}}\overline{W}_{\rm flux} \geq 0\;,
```


summing over complex structure and axio-dilaton indices $a, \bar b$ only.
This potential is minimised (to zero) at the ISD locus $D_a W_{\rm flux} = 0$,
which fixes $z^i$ and $\tau$ while leaving the Kähler moduli as flat directions.

For a detailed discussion of fluxes, tadpole cancellation, and the ISD condition
see {doc}`flux_compactifications`.
For Kähler moduli stabilisation see {doc}`moduli_stabilisation`.


### Implementation in `jaxvacua.flux_eft`

```{eval-rst}
.. currentmodule:: jaxvacua.flux_eft

.. autosummary::

    FluxEFT.map_to_fd_tau
    FluxEFT.superpotential
    FluxEFT.DW
    FluxEFT.scalar_potential

```


```{warning}
Note that python (and thus the code here) uses [zero-based indexing](https://en.wikipedia.org/wiki/Zero-based_numbering) while mathematical notation usually uses one-based indexing.
For consistency, the indexing in the notes here also starts at $0$.
```

