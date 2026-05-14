.. currentmodule:: jaxvacua


JAXVacua
===============


Package architecture
--------------------

The package is organised as a layered pipeline. ``lcs_tree`` is the
data interface; the linear chain ``periods → css → FluxEFT →
FluxVacuaFinder`` adds physics one layer at a time; orthogonal tools
(``sampling``, ``flux_bounding``, ``freezer``) plug into
``FluxVacuaFinder``; helper modules (``cytools_interface``,
``hypergeometric_models``, ``flux_utils``) feed the pipeline.

.. raw:: html
   :file: _static/figures/f2_architecture.html

Solid arrows are required code dependencies; dashed arrows indicate
"used by" tools that ``FluxVacuaFinder`` calls into rather than stages
it produces. The orange callout marks ``lcs_tree`` as the data hub —
every input on-ramp eventually populates it, and every model layer
reads from it. Concretely, ``lcs_tree`` carries the topological data
:math:`(\kappa_{ijk},\, c_2,\, \chi,\, n_\beta^0)`; the chain then
constructs the period vector :math:`\Pi(z) = (\mathcal{F}_I,\, X^I)`,
the Kähler potential :math:`K(z, \bar z)`, the GVW superpotential
:math:`W = \int_X G_3 \wedge \Omega`, and the F-terms
:math:`D_I W = \partial_I W + (\partial_I K)\, W`.


Subpackages
-------------


.. toctree::
    :maxdepth: 1

    jaxvacua.lcs
    jaxvacua.periods
    jaxvacua.css
    jaxvacua.flux_eft
    jaxvacua.flux_vacua_finder
    jaxvacua.flux_bounding
    jaxvacua.flux_utils
    jaxvacua.freezer
    jaxvacua.conifold
    jaxvacua.hypergeometric_models
    jaxvacua.sampling
    jaxvacua.util
    jaxvacua.cytools_interface
