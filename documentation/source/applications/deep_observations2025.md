# Deep observations of the Type IIB flux landscape

<div style="text-align: justify">

In [2501.03984](https://arxiv.org/abs/2501.03984), we present deep observations in targeted regions of the string landscape through a combination of analytic and dedicated numerical methods. Specifically, we devise an algorithm designed for the systematic construction of Type IIB flux vacua in finite regions of moduli space. Our algorithm is universally applicable across Calabi-Yau orientifold compactifications and can be used to enumerate flux vacua in a region given sufficient computational efforts. As a concrete example, we apply our methods to a two-modulus Calabi-Yau threefold, demonstrating that systematic enumeration is feasible and revealing intricate structures in vacuum distributions. Our results highlight local deviations from statistical expectations, providing insights into vacuum densities, superpotential distributions, and moduli mass hierarchies. This approach opens pathways for precise, data-driven mappings of the string landscape, complementing analytic studies and advancing the understanding of the distribution of flux vacua. This allows us to obtain different types of solutions with hierarchical suppressions, e.g. vacua with small values of the Gukov-Vafa-Witten superpotential $|W_0|$. We find an example with $|W_0| = 5.547 \times 10^{-5}$ at large complex structure, without light directions and the use of non-perturbative effects.

</div>

## Key results

- **Systematic flux enumeration**: A new algorithm that constructs all Type IIB flux vacua inside a bounded region of moduli space, applicable to any Calabi-Yau orientifold compactification. Bounding boxes are derived from the imaginary self-dual (ISD) condition and the tadpole constraint.
- **Two-modulus model**: The algorithm is demonstrated on a concrete two-modulus Calabi-Yau threefold, where exhaustive enumeration is shown to be computationally feasible.
- **Hierarchical suppression**: Vacua with highly suppressed superpotentials are found, including a solution with $|W_0| = 5.547 \times 10^{-5}$ at large complex structure — achieved without light directions or non-perturbative effects.
- **Landscape structure**: The resulting vacuum distributions reveal local deviations from statistical expectations, providing data-driven insights into vacuum densities, $W_0$ distributions, and moduli mass hierarchies.

## Relevant JAXVacua modules and notebooks

The core algorithm of this paper is implemented in the [`jaxvacua.flux_bounding`](../jaxvacua.flux_bounding) module. The relevant tutorials are:

- **[Tutorial 10: Flux bounding](../notebooks/03_flux_bounding/10_flux_bounding)** — introduces the bounding-box construction and the `bounded_fluxes` / `enumerate_fluxes` interface.
- **[Tutorial 10b: Stochastic flux search](../notebooks/03_flux_bounding/10b_stochastic_flux_search)** — Monte-Carlo sampling inside the bounding box.
- **[Tutorial 10c: Step-by-step bounded flux sampling](../notebooks/03_flux_bounding/10c_sample_bounded_fluxes_stepbystep)** — walks through each step of the algorithm in detail.

To cite our work, please use:

```
@article{Chauhan:2025rdj,
    author = "Chauhan, Aman and Cicoli, Michele and Krippendorf, Sven and Maharana, Anshuman and Piantadosi, Pellegrino and Schachner, Andreas",
    title = "{Deep observations of the Type IIB flux landscape}",
    eprint = "2501.03984",
    archivePrefix = "arXiv",
    primaryClass = "hep-th",
    month = "1",
    year = "2025"
}
```


