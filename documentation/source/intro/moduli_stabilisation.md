# Moduli Stabilisation

## The moduli problem

A generic Calabi-Yau compactification gives rise to massless scalar fields (moduli)
in four dimensions:
the complex structure moduli $z^i$, the axio-dilaton $\tau$, and the Kähler moduli $T_\alpha$.
These parametrize flat directions of the classical scalar potential and are phenomenologically
unacceptable: massless scalars mediate long-range fifth forces, cause time-varying
fundamental constants, and produce overabundant non-relativistic matter in the early
universe (the *cosmological moduli problem*).

As described in {doc}`flux_compactifications`, three-form fluxes generate the no-scale
scalar potential

$$
V_{\rm flux} = e^K K^{a\bar b} D_a W_{\rm flux}\, D_{\bar b}\overline{W}_{\rm flux} \geq 0
$$

that stabilises $z^i$ and $\tau$ at the ISD locus.
However, the Kähler moduli $T_\alpha$ (which control the volumes of four-cycles
in $X_3$) remain flat directions at the classical level due to the no-scale structure.
Their stabilisation requires quantum corrections — perturbative or non-perturbative.


## KKLT scenario

The **KKLT scenario** [[hep-th/0301240](https://arxiv.org/abs/hep-th/0301240)] achieves
complete moduli stabilisation in two steps, followed by an uplifting procedure.

**Step 1: Flux stabilisation.**
Choose fluxes such that the no-scale minimum sits at

$$
W_0 = \langle W_{\rm flux} \rangle \ll 1\;.
$$

This fixes $z^i$ and $\tau$, leaving the Kähler moduli as flat directions.

**Step 2: Non-perturbative superpotential.**
Non-perturbative effects — from Euclidean D3-brane instantons wrapping four-cycles,
or from gaugino condensation on D7-branes — generate a correction

$$
W_{\rm np} = \sum_\alpha A_\alpha\, e^{-a_\alpha T_\alpha}\;,
$$

where $a_\alpha = 2\pi$ for a D3-instanton and $a_\alpha = 2\pi/N$ for $SU(N)$ gaugino
condensation.
The total superpotential is

$$
W = W_0 + W_{\rm np}\;.
$$

For a single Kähler modulus $T = \tau_K + i\theta$ with Kähler potential
$K = -3\log(T + \bar T)$, the scalar potential develops an **AdS minimum** at

$$
a\tau_K^* \approx -\frac{3}{2}\left(1 + \frac{W_0}{A e^{-a\tau_K^*}}\right)\;.
$$

The small $|W_0|$ ensures $\tau_K^*$ is in the perturbative regime.

**Step 3: Uplifting to de Sitter.**
The AdS minimum can be lifted to a metastable de Sitter vacuum by introducing a
positive energy contribution.
The canonical mechanism uses an **anti-D3 brane** placed at the tip of a
Klebanov-Strassler warped throat, contributing

$$
V_{\rm up} \sim \frac{\epsilon^4}{\mathcal{V}^2}
$$

where $\epsilon \ll 1$ is an exponential warping factor determined by the flux
quantization in the throat.


## Large Volume Scenario (LVS)

The **Large Volume Scenario**
[[hep-th/0502058](https://arxiv.org/abs/hep-th/0502058),
[hep-th/0505076](https://arxiv.org/abs/hep-th/0505076)]
stabilises Kähler moduli using $\alpha'$ corrections to the Kähler potential.

For a Calabi-Yau with at least two Kähler moduli — a *large* modulus $\tau_b$
controlling the overall volume $\mathcal{V} \sim \tau_b^{3/2}$ and a *small* modulus
$\tau_s$ — the $(\alpha')^3$-corrected Kähler potential reads

$$
K = -2\log\!\left(\mathcal{V} + \frac{\hat\xi}{2}\right)\;,
$$

where the correction coefficient is

$$
\hat\xi = -\frac{\chi(X_3)\,\zeta(3)}{2(2\pi)^3}
$$

and is proportional to the Euler characteristic $\chi(X_3)$.

Including the non-perturbative superpotential $W = W_0 + A_s\, e^{-a_s T_s}$, the
scalar potential takes the LVS form

$$
V_{\rm LVS} \simeq
  \lambda_s \frac{\sqrt{\tau_s}\,|W_0|^2\,e^{-2a_s\tau_s}}{\mathcal{V}}
- \mu_s \frac{|W_0|^2\,a_s\tau_s\,e^{-a_s\tau_s}}{\mathcal{V}^2}
+ \nu \frac{|W_0|^2\,\hat\xi}{\mathcal{V}^3}\;,
$$

where $\lambda_s, \mu_s, \nu$ are calculable numerical coefficients.
This potential has a **non-supersymmetric AdS minimum** at exponentially large volume

$$
\mathcal{V}_* \sim e^{a_s\tau_s^*}\;,
$$

where $\tau_s^*$ is determined by $\hat\xi$ and $A_s$.
The parametrically large volume suppresses higher-order corrections, making LVS a
controlled approximation.


## de Sitter vacua

Phenomenologically, a positive cosmological constant $\Lambda > 0$ is required to match
observations.
Constructing de Sitter vacua in string theory remains an active area of research.
The main uplifting mechanisms discussed in the literature include:

- **Anti-D3 branes** in warped throats (original KKLT proposal)
- **D-term contributions** from gauge fluxes on D7-branes
- **F-term uplifting** using matter fields (ISS-type hidden sector)
- **Perturbatively flat vacua (PFVs)** with specific $\alpha'$ and string-loop corrections (see {doc}`pfv`)

For a comprehensive and pedagogical treatment of these mechanisms, including recently
constructed explicit candidate de Sitter vacua, see the TASI lectures by McAllister and
Schachner [[arXiv:2512.17095](https://arxiv.org/abs/2512.17095)] and references therein.

JAXVacua currently focuses on finding **SUSY flux vacua** (ISD solutions minimising
$V_{\rm flux}$) and **non-SUSY vacua** (local minima of the full scalar potential).
For implementations see {doc}`../jaxvacua.flux_eft` and
{doc}`../applications/nonSUSY_vacua2023`.
