# Non-supersymmetric flux vacua

<div style="text-align: justify">

In [2308.15525](https://arxiv.org/abs/2308.15525) we construct large ensembles of supersymmetry breaking solutions arising in the context of flux compactifications of type IIB string theory. This class of solutions was previously proposed in [[hep-th/0402135]](https://arxiv.org/abs/hep-th/0402135) (Saltman–Silverstein) for which we provide the first explicit examples in Calabi-Yau orientifold compactifications with discrete fluxes below their respective tadpole constraint. As a proof of concept, we study the degree-18 hypersurface in weighted projective space $\mathbb{CP}_{1,1,1,6,9}$. Furthermore, we look at 10 additional orientifolds with $h^{1,2} = 2, 3$. We find several flux vacua with hierarchical suppression of the vacuum energy with respect to the gravitino mass. These solutions provide a crucial stepping stone for the construction of explicit de Sitter vacua in string theory. Lastly, we also report the difference in the distribution of $W_0$ between supersymmetric and non-supersymmetric minima.

</div>

## Key results

- **First explicit examples**: We construct the first explicit non-supersymmetric flux vacua of the Saltman–Silverstein type [[hep-th/0402135]](https://arxiv.org/abs/hep-th/0402135) in Calabi-Yau orientifold compactifications with integer fluxes satisfying the tadpole constraint.
- **Geometries studied**: The degree-18 hypersurface $\mathbb{CP}_{1,1,1,6,9}$ (one complex structure modulus) and 10 additional orientifolds with $h^{1,2} = 2$ or $3$.
- **Hierarchical vacuum energy**: Multiple solutions are found where the vacuum energy is hierarchically suppressed relative to $m_{3/2}^2 M_{\mathrm{Pl}}^2$, a prerequisite for low-energy supersymmetry breaking scenarios and de Sitter uplifting.
- **$W_0$ distributions**: The $W_0$ distribution at non-supersymmetric minima is found to differ systematically from the Gaussian distribution of supersymmetric ISD vacua, providing a comparative benchmark for landscape statistics.

## Relevant JAXVacua modules and notebooks

The minimisation of the full scalar potential (beyond the ISD approximation) uses [`jaxvacua.css`](../jaxvacua.css) and [`jaxvacua.flux_eft`](../jaxvacua.flux_eft). The relevant tutorials are:

- **[Tutorial 5: Finding flux vacua](../notebooks/02_vacuum_finding/5_finding_flux_vacua)** — introduces scalar potential minimisation and the Newton refinement step used to locate exact vacua.
- **[Tutorial 6: Sampling module](../notebooks/02_vacuum_finding/6_sampling_module)** — batch flux sampling pipeline used for large-ensemble construction.

To cite our work, please use:

```
@article{Krippendorf:2023idy,
    author = "Krippendorf, Sven and Schachner, Andreas",
    title = "{New non-supersymmetric flux vacua in string theory}",
    eprint = "2308.15525",
    archivePrefix = "arXiv",
    primaryClass = "hep-th",
    reportNumber = "LMU-ASC 30/23",
    doi = "10.1007/JHEP12(2023)145",
    journal = "JHEP",
    volume = "12",
    pages = "145",
    year = "2023"
}
