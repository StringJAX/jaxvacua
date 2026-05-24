# Type IIB Three-Form Fluxes

In this note we discuss three-form flux compactifications of Type IIB string theory on
Calabi-Yau orientifolds $X_3/\mathcal{O}$ (see {doc}`geometries` for the construction of
these geometries).
The fluxes generate a scalar potential that stabilises the complex structure moduli and
the axio-dilaton through the mechanism reviewed here.


## Three-form fluxes and the flux superpotential

Type IIB string theory contains two three-form field strengths:
the Ramond-Ramond (RR) three-form $F_3 = dC_2$ and the Neveu-Schwarz (NS) three-form
$H_3 = dB_2$.
These are quantized: their integrals over integral three-cycles $(A_I, B^J)$ of $X_3$
take integer values,

$$
\frac{1}{(2\pi)^2\alpha'}\int_{A_I} F_3 = f^I \in \mathbb{Z}\;, \quad
\frac{1}{(2\pi)^2\alpha'}\int_{B^J} H_3 = h_J \in \mathbb{Z}\;.
$$

The integer-valued vectors $f = (f^I)$ and $h = (h_J)$ define the **flux quanta**.
The complexified three-form

$$
G_3 = F_3 - \tau H_3
$$

combines the two fluxes using the axio-dilaton $\tau = C_0 + i/g_s$ (with $g_s$ the
string coupling).

The **Gukov-Vafa-Witten (GVW) superpotential** [[hep-th/9906070](https://arxiv.org/abs/hep-th/9906070)] reads

$$
W_{\rm flux} = \int_{X_3} G_3 \wedge \Omega_3 = \int_{X_3} (F_3 - \tau H_3) \wedge \Omega_3\;.
$$ (eq:GVW)

In terms of the period vector $\Pi = (X^I, F_J)$ introduced in {eq}`eq:PeriodsInt` and the
flux quanta, this becomes

$$
W_{\rm flux} = (f - \tau h) \cdot \Sigma \cdot \Pi\;,
$$

where $\Sigma$ is the symplectic pairing matrix from {eq}`eq:DefACSM`.


## Tadpole cancellation

The three-form fluxes source D3-brane charge.
The flux contribution to the D3-brane tadpole is

$$
N_{\rm flux} = \frac{1}{(2\pi\alpha')^2} \int_{X_3} H_3 \wedge F_3 = h \cdot f\;.
$$

Tadpole cancellation — required by the Gauss law constraint in the compact space —
constrains the total D3-brane charge:

$$
N_{\rm flux} + N_{\rm D3} = Q_O = \frac{\chi(X_3/\mathcal{O})}{24}\;,
$$ (eq:tadpole)

where $N_{\rm D3} \geq 0$ counts mobile D3-branes and $Q_O$ is the **orientifold charge**,
fixed by the geometry.
For the trilayer orientifold geometries considered in JAXVacua, one has
$Q_O = 2 + h^{1,1} + h^{2,1}$ (see {eq}`eq:qdef`).
A larger $Q_O$ admits more flux configurations and thus a richer landscape of vacua.


## Imaginary self-dual fluxes and SUSY vacua

SUSY-preserving flux backgrounds require the **imaginary self-dual (ISD)** condition

$$
G_3 = i \star_6 G_3\;.
$$

Under the Hodge decomposition into $(p,q)$-type forms, this restricts $G_3$ to

$$
G_3 \in H^{(2,1)}_{\rm prim} \oplus H^{(0,3)}\;.
$$

Supersymmetry additionally requires $G_3^{(0,3)} = 0$, i.e.\ $D_\tau W = 0$.
The unique SUSY flux configuration is therefore $G_3 \in H^{(2,1)}_{\rm prim}$.

The F-flatness conditions $D_i W_{\rm flux} = 0$ (complex structure) and
$D_\tau W_{\rm flux} = 0$ (axio-dilaton) can be collectively expressed as

$$
\Pi^\dagger \cdot \Sigma \cdot (f - \tau h) = 0\;,
$$ (eq:ISD_fterms)

together with the Hodge-type constraint on $G_3$.
These are the equations solved by the flux vacuum finder in JAXVacua.
Non-SUSY flux vacua — where the ISD condition is relaxed — are discussed in
{doc}`../applications/nonSUSY_vacua2023`.


## No-scale flux scalar potential

The flux-induced F-term potential takes the **no-scale** form

$$
V_{\rm flux} = e^K K^{a\bar{b}} D_a W_{\rm flux}\, D_{\bar{b}} \overline{W}_{\rm flux} \geq 0\;,
$$ (eq:Vflux)

where $a, \bar{b}$ run over complex structure moduli $z^i$ and the axio-dilaton $\tau$ only.
This is a consequence of the **no-scale identity**

$$
K^{\alpha\bar\beta}\, K_\alpha\, K_{\bar\beta} = 3
$$

satisfied by the classical Kähler potential $K_K = -2\log\mathcal{V}$:
the Kähler moduli F-term contributions exactly cancel the $-3|W|^2$ term in
{eq}`eq:Vtot`, yielding the positive semi-definite expression {eq}`eq:Vflux`.

The potential {eq}`eq:Vflux` is minimised (to zero) at the ISD locus $D_a W_{\rm flux} = 0$.
This fixes $z^i$ and $\tau$, while the Kähler moduli remain as flat directions.
Their stabilisation requires quantum corrections and is discussed in {doc}`moduli_stabilisation`.


## The flux landscape

The flux quanta $(f, h)$ are integer-valued vectors in a lattice of dimension $b_3 = 2(h^{2,1}+1)$.
Subject to the tadpole constraint {eq}`eq:tadpole`, the number of admissible flux vacua grows as

$$
\mathcal{N}_{\rm vac} \sim \frac{(2\pi Q_O)^{b_3/2}}{(b_3/2)!}\;,
$$

as estimated by Bousso and Polchinski [[hep-th/0004134](https://arxiv.org/abs/hep-th/0004134)]
and Ashok and Douglas [[hep-th/0307049](https://arxiv.org/abs/hep-th/0307049)].
For typical Calabi-Yau threefolds this yields an enormous number of vacua — the
**string landscape** — with estimates ranging up to $\mathcal{O}(10^{272{,}000})$ across
the full Kreuzer-Skarke database
[[2204.02317](https://arxiv.org/abs/2204.02317)].

The statistical distribution of physical observables (such as $|W_0|$) across this landscape
can be studied with the sampling tools in JAXVacua.
For applications see {doc}`../applications/W0_distribution2023`.
