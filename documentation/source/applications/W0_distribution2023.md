# Distribution of $W_0$

<div style="text-align: justify">

In [2307.15749](https://arxiv.org/abs/2307.15749), we explore the distribution of vacuum expectation values of the superpotential $W_0$ in explicit Type IIB flux compactifications. We show that the distribution can be approximated universally across geometries by a two-dimensional Gaussian with a model-dependent standard deviation. We identify this behaviour in 20 Calabi-Yau orientifold compactifications with between two and five complex structure moduli by constructing a total of $\mathcal{O}(10^7)$ flux vacua. We observe a characteristic scaling behaviour of the width $\sigma$ of our distributions with respect to the D3-charge contributions $N_{\mathrm{flux}}$ from fluxes which can be approximated by $\sigma \sim \sqrt{N_{\mathrm{flux}}}$. This $W_0$ distribution implies that locating small values of $|W_0|$ as a preferred regime associated with classes of string theory solutions typically featuring hierarchies, simplifies to the basic statement of finding small Euclidean norms of normally distributed values. We also identify small modifications to this Gaussian behaviour in our samples, which might be seen as indications for the breakdown of the continuous flux approximation commonly used in the context of statistical analyses of the flux landscape.

</div>

## Key results

- **Universal Gaussian distribution**: The distribution of the Gukov-Vafa-Witten superpotential $W_0$ is well approximated by a two-dimensional Gaussian, $W_0 \sim \mathcal{N}(0, \sigma^2)$, across all 20 geometries studied.
- **Scaling with tadpole**: The width of the distribution scales as $\sigma \sim \sqrt{N_{\mathrm{flux}}}$, where $N_{\mathrm{flux}}$ is the D3-charge contribution from fluxes.
- **Scale of the study**: $\mathcal{O}(10^7)$ supersymmetric flux vacua constructed across 20 Calabi-Yau orientifolds with $h^{1,2} \in \{2, 3, 4, 5\}$ complex structure moduli.
- **Implication for small $|W_0|$**: Finding vacua with exponentially small $|W_0|$ reduces to sampling the tails of a normal distribution — with no special fine-tuning mechanism required beyond flux discreteness.
- **Non-Gaussian corrections**: Small deviations from the Gaussian approximation are observed, consistent with the breakdown of the continuous flux approximation at low $N_{\mathrm{flux}}$.

## Relevant JAXVacua modules and notebooks

The large-scale sampling of ISD flux vacua underlying this paper uses the [`jaxvacua.sampling`](../jaxvacua.sampling) module. The relevant tutorials are:

- **[NB06: ISD sampling principle](../notebooks/02_vacuum_finding/06_ISD_sampling_principle.ipynb)** — derives the ISD completion step for a single vacuum.
- **[NB07: ISD sampling](../notebooks/02_vacuum_finding/07_ISD_sampling.ipynb)** — large-scale SUSY and non-SUSY sampling workflows.
- **[NB15: Landscape statistics](../notebooks/04_analysis_and_pipelines/15_landscape_statistics.ipynb)** — post-processing and distribution diagnostics for vacuum ensembles.

To cite our work, please use:

```
@article{Ebelt:2023clh,
    author = "Ebelt, Julian and Krippendorf, Sven and Schachner, Andreas",
    title = "{W0\_sample = np.random.normal(0,1)?}",
    eprint = "2307.15749",
    archivePrefix = "arXiv",
    primaryClass = "hep-th",
    doi = "10.1016/j.physletb.2024.138786",
    journal = "Phys. Lett. B",
    volume = "855",
    pages = "138786",
    year = "2024"
}
```

