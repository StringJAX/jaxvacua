# Notebook Glossary

This glossary collects the short conventions used across the tutorial
notebooks.  It is a reminder page, not a replacement for the derivations in
the physics introduction or the worked notebooks.

(notebook-glossary-w0)=
## $W_0$

The value of the GVW superpotential after the complex-structure moduli and the
axio-dilaton have been fixed.  In the notebooks this usually means
$W_0 = W(z_\star, \tau_\star; F_3, H_3)$ at a converged flux vacuum.

(notebook-glossary-z-cf)=
## $z_{\rm cf}$

The local conifold modulus.  The conifold locus sits at $z_{\rm cf}=0$; in
coniLCS workflows this field is often parametrically heavy and can be solved
for analytically before working with the remaining bulk moduli.

(notebook-glossary-gs)=
## $g_s$

The string coupling, related to the axio-dilaton by
$\tau = C_0 + i/g_s$.  The weak-coupling regime is therefore
$g_s = 1 / {\rm Im}(\tau) \ll 1$.

(notebook-glossary-nflux)=
## $N_{\rm flux}$

The D3-brane charge induced by the three-form fluxes.  It is compared against
the tadpole budget, usually written $Q_{\rm O3}$ or $L_{\max}$ depending on
the notebook.

(notebook-glossary-isd)=
## ISD

Imaginary self-dual.  In the Type IIB flux sector this denotes the flux
condition $*_6 G_3 = i G_3$, and in these tutorials it is the geometric
condition behind the SUSY flux-vacuum equations for the complex-structure
moduli and axio-dilaton.

(notebook-glossary-aisd)=
## AISD

Anti-imaginary self-dual.  This is the complementary flux component satisfying
$*_6 G_3 = -i G_3$.  AISD components are useful diagnostics for departures from
the ISD locus and for non-SUSY deformations.

(notebook-glossary-pfv)=
## PFV

Perturbatively flat vacuum.  PFV workflows use special flux data for which the
perturbative part of the flux superpotential has a flat direction; instanton,
conifold, or racetrack effects then lift that direction.

(notebook-glossary-afv)=
## AFV

A development shorthand for a general flux-vacuum candidate outside the strict
PFV ansatz.  It is not a stable public API term; public notebooks should define
the local meaning explicitly whenever they use it.
