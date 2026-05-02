# ==============================================================================
# critical_points.py
#
# Tools for finding critical points of the scalar potential V in Type IIB
# flux compactifications, including non-SUSY minima, saddle points, and
# (meta)stable de Sitter vacua.
#
# The main class `CriticalPointFinder` generates flux candidates via ISD
# sampling (modes F, H, ISD+, ISD-), then solves ∂V/∂φ = 0 using either
# a JIT-compiled Newton solver or optax gradient-based optimisers.
#
# Usage:
#   from jaxvacua.critical_points import CriticalPointFinder
#   finder = CriticalPointFinder(model, sampler, Nmax=200)
#   results = finder.sample_critical_points(n_target=100, isd_mode="ISD-")
# ==============================================================================

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from functools import partial
from typing import List, Tuple, Any, Callable
import time

try:
    import optax
    _HAS_OPTAX = True
except ImportError:
    _HAS_OPTAX = False


class CriticalPointFinder:
    r"""
    **Description:**
    Finds critical points of the scalar potential :math:`V` in Type IIB flux
    compactifications, including both SUSY (:math:`D_I W = 0`) and non-SUSY
    (:math:`\partial_I V = 0,\; D_I W \neq 0`) vacua.

    .. admonition:: Background
        :class: dropdown

        The no-scale scalar potential is

        .. math::
            V_{\rm ns} = \mathrm{e}^K\, D_{\bar I}\bar W\, K^{I\bar J}\, D_J W
            \geq 0

        Its critical points satisfy :math:`\partial_\alpha V_{\rm ns} = 0` in
        real coordinates :math:`\phi^\alpha = (\mathrm{Re}\,z^i,
        \mathrm{Im}\,z^i, c_0, s)`.  SUSY vacua (:math:`D_I W = 0`) are
        always critical points of :math:`V_{\rm ns}`, but there exist additional
        non-SUSY critical points where :math:`D_I W \neq 0`.

        The full :math:`\mathcal{N}=1` SUGRA potential
        :math:`V = \mathrm{e}^K(|DW|^2 - 3|W|^2)` can also be used
        (``noscale=False``), but the no-scale version produces a much
        higher fraction of minima (~85% vs ~50%).

    Args:
        model: A :class:`FluxEFT` or :class:`FluxVacuaFinder` instance.
        sampler: A :class:`data_sampler` for moduli/tau sampling.
        Nmax (int): Maximum D3-tadpole charge.
        noscale (bool): If ``True``, use no-scale potential. Defaults to ``True``.
        flux_prior (str): Prior for ISD input vectors:
            ``"gaussian"`` — isotropic :math:`\mathcal{N}(0, \sigma^2 I)`,
            best for F, ISD± modes.
            ``"M_weighted"`` — :math:`\mathcal{N}(0, \sigma^2 M)` weighted
            by the ISD matrix eigensystem, best for H mode (3× more valid
            candidates). Falls back to isotropic for non-H modes.
            ``"uniform"`` — uniform random integers (original behaviour).
            Defaults to ``"gaussian"``.
        flux_prior_sigma (float): Scale factor for Gaussian prior.
            If ``None``, uses mode-dependent defaults tuned empirically.
        moduli_max (float): Upper bound on :math:`\max(|z_i|, |\tau|)` for
            physical solutions. Candidates exceeding this are rejected as
            runaways. Also used for early termination in Newton iterations.
            Defaults to ``100.0``. Set to ``None`` to disable.

    See also: :func:`jaxvacua.flux_bounding.bounded_fluxes.sample_bounded_fluxes`
    """

    def __init__(
        self,
        model: Any,
        sampler: Any,
        Nmax: int = 10,
        noscale: bool = True,
        flux_prior: str = "gaussian",
        flux_prior_sigma: float = None,
        moduli_max: float = 100.0,
        map_to_fd: bool | None = None,
    ):
        self.model = model
        self.sampler = sampler
        self.Nmax = int(Nmax)
        self.noscale = noscale
        self.n_fluxes = model.n_fluxes          # 2*(h12+1)
        self.dimension_H3 = model.dimension_H3    # h12+1
        self.flux_prior = flux_prior
        self.flux_prior_sigma = flux_prior_sigma
        self.moduli_max = moduli_max
        self._map_to_fd = getattr(model, '_map_to_fd', False) if map_to_fd is None else map_to_fd

        # Pre-computed M eigensystem (for M_weighted prior)
        self._M_eigvecs = None
        self._M_scales = None
        self._M_cond = None
        self._tr_Minv_median = None
        self._s_min = float(getattr(sampler, 's_lower', np.sqrt(3) / 2))

        # Calibrated σ values per mode (populated by calibrate_priors
        # or _estimate_sigmas; empty = use hardcoded defaults)
        self._calibrated_sigmas: dict = {}

        # Pre-compute M eigensystem (fast, ~0.5s, needed for M_weighted)
        if flux_prior == "M_weighted":
            self._precompute_M_eigensystem()

    # ------------------------------------------------------------------
    #  Fundamental domain mapping & dedup key
    # ------------------------------------------------------------------

    def _map_result_to_fd(self, flux, moduli, tau):
        r"""Map a single (flux, moduli, tau) to the fundamental domain.

        If ``_map_to_fd`` is ``False``, returns the inputs unchanged.
        """
        if not self._map_to_fd:
            return flux, moduli, tau
        moduli_fd, tau_fd, fluxes_fd = self.model.map_to_FD(
            jnp.asarray(moduli), complex(tau), jnp.asarray(flux),
        )
        return np.asarray(fluxes_fd, dtype=np.int32), np.asarray(moduli_fd), complex(tau_fd)

    @staticmethod
    def _dedup_key(flux, moduli, tau, n_digits=6):
        r"""Build a hashable dedup key from (flux, moduli, tau).

        Includes moduli and tau (rounded to ``n_digits``) because a single
        flux vector can admit multiple distinct critical points.
        """
        f_part = np.round(np.asarray(flux).real).astype(np.int32).tobytes()
        z_re = np.around(np.asarray(moduli).real, n_digits)
        z_im = np.around(np.asarray(moduli).imag, n_digits)
        t_part = (round(complex(tau).real, n_digits), round(complex(tau).imag, n_digits))
        return (f_part, z_re.tobytes(), z_im.tobytes(), t_part)

    # ------------------------------------------------------------------
    #  Physicality check (general h12)
    # ------------------------------------------------------------------

    def _is_physical(self, x: np.ndarray) -> bool:
        r"""
        **Description:**
        Check whether a solution point is in the physical region of moduli
        space.  Checks are applied in order from cheapest to most expensive:

        0. **Runaway check**: :math:`\max(|z_i|, |\tau|) \leq` ``moduli_max``.
        1. **Dilaton floor**: :math:`\mathrm{Im}(\tau) > s_{\min}`.
        2. **Hyperplane check** (if ``lcs_tree.hyperplanes`` is available):
           :math:`\mathrm{Im}(z)` must dot positively with all hyperplane
           normals, i.e. ``hyperplanes @ Im(z) > 0`` component-wise.
        3. **Kähler metric check** (fallback): all eigenvalues of the
           Kähler metric must be positive.
        4. **Basic check** (last resort): :math:`\mathrm{Im}(z_i) > 0`
           for all moduli and :math:`\mathrm{Im}(\tau) > 0`.
        """
        x_jax = jnp.array(x)
        moduli, _, tau, _ = self.model._convert_real_to_complex(x_jax)
        im_z = jnp.imag(moduli)
        s = float(jnp.imag(tau))

        # Check 0: runaway bound (cheapest — single comparison)
        if self.moduli_max is not None:
            max_abs = float(jnp.max(jnp.abs(jnp.append(moduli, tau))))
            if max_abs > self.moduli_max:
                return False

        # Check 1: Im(tau) must be positive (use SL(2,Z) floor as minimum)
        s_lo = getattr(self.sampler, 's_lower', 0.0)
        if s <= s_lo:
            return False

        # Check 2: hyperplanes (fast, preferred)
        hp = getattr(self.model.lcs_tree, 'hyperplanes', None)
        if hp is not None:
            dots = jnp.array(hp) @ im_z
            return bool(jnp.all(dots > 0))

        # Check 3: Kähler metric positive-definiteness (slower)
        try:
            moduli_c = jnp.conj(moduli)
            tau_c = jnp.conj(tau)
            K = self.model.kahler_metric(moduli, moduli_c, tau, tau_c)
            eigs = jnp.linalg.eigvalsh(K)
            return bool(jnp.all(eigs > 0))
        except Exception:
            pass

        # Check 4: basic Im(z_i) > 0
        return bool(jnp.all(im_z > 0))

    # ------------------------------------------------------------------
    #  Matrix pre-computation
    # ------------------------------------------------------------------

    def _precompute_M_eigensystem(self, n_sample: int = 50):
        r"""
        **Description:**
        Pre-compute the ISD matrix eigensystem at representative moduli
        points.  Selects the best-conditioned :math:`M` to define the
        Gaussian covariance for M-weighted sampling.

        Also stores summary eigenvalue data used by :meth:`_estimate_sigmas`.
        """
        mod = np.array(self.sampler.get_complex_moduli(n_sample))
        mod_jax = jnp.array(mod, dtype=complex)

        best_cond = np.inf
        best_M = None
        all_tr_Minv = []
        for i in range(n_sample):
            M = np.array(self.model.ISD_matrix(
                mod_jax[i], jnp.conj(mod_jax[i])))
            eigs = np.abs(np.linalg.eigvalsh(M))
            cond = eigs.max() / max(eigs.min(), 1e-30)
            all_tr_Minv.append(np.sum(1.0 / np.maximum(eigs, 1e-30)))
            if cond < best_cond:
                best_cond = cond
                best_M = M

        eigvals_M, eigvecs_M = np.linalg.eigh(best_M)
        eigvals_M = np.abs(eigvals_M)

        s_min = float(getattr(self.sampler, 's_lower', np.sqrt(3) / 2))
        sigma_sq = float(self.Nmax) / (self.n_fluxes * s_min)
        self._M_scales = np.sqrt(sigma_sq * eigvals_M)
        self._M_eigvecs = eigvecs_M

        # Store summary stats for analytical σ estimation
        self._s_min = s_min
        self._tr_Minv_median = float(np.median(all_tr_Minv))
        self._M_cond = best_cond

    # ------------------------------------------------------------------
    #  Analytical σ estimation
    # ------------------------------------------------------------------

    def _estimate_sigmas(self):
        r"""
        **Description:**
        Estimate optimal σ for each ISD mode analytically from the
        matrix eigenvalue structure and tadpole constraint.

        For **H mode** (input = h, tadpole ~ :math:`s\,h^T M^{-1} h`):

        .. math::
            \sigma^2 = \frac{N_{\max}}{s_{\min}\,\mathrm{tr}(M^{-1})}

        For **F mode** (input = f, completion h ~ M^{-1} f):
        The completed h has magnitude ~ σ/|τ|, and the tadpole goes as
        σ²·tr(M^{-1})/|τ|⁴.  We use a heuristic scale.

        For **ISD±** modes, the scale depends on the gauge kinetic matrix
        eigenvalues (not easily accessible at construction time), so we
        use the M-based estimate with empirical correction factors.
        """
        Nmax = float(self.Nmax)
        s_min = self._s_min
        tr_Minv = self._tr_Minv_median
        d = float(self.n_fluxes)

        # H mode: E[s·h^T M^{-1} h] = s·σ²·tr(M^{-1}) for h ~ N(0,σ²I)
        # Want ≈ Nmax → σ² = Nmax / (s_min · tr(M^{-1}))
        sigma_H = np.sqrt(Nmax / (s_min * tr_Minv))

        # F mode: heuristic — the M^{-1} weighting in the completion
        # amplifies small eigenvalue directions. Empirically σ_F ≈ σ_H
        # works well (the completion acts as a natural regulariser).
        sigma_F = sigma_H

        # ISD- mode: completion uses N^{-1}, which amplifies inputs.
        # Empirical correction: σ_ISD- ≈ 2 × σ_H (allows larger inputs)
        sigma_ISDm = 2.0 * sigma_H

        # ISD+ mode: completion uses N directly, which amplifies outputs.
        # Need smaller inputs: σ_ISD+ ≈ 0.5 × σ_H
        sigma_ISDp = 0.5 * sigma_H

        self._calibrated_sigmas = {
            "F": float(sigma_F),
            "H": float(sigma_H),
            "ISD-": float(sigma_ISDm),
            "ISD+": float(sigma_ISDp),
        }

    # ------------------------------------------------------------------
    #  Empirical calibration
    # ------------------------------------------------------------------

    def calibrate_priors(
        self,
        modes: List[str] = None,
        n_test: int = 200,
        target_acceptance: float = 0.8,
        verbose: bool = True,
    ) -> dict:
        r"""
        **Description:**
        Empirically calibrate the Gaussian σ for each ISD mode by
        binary-searching for the σ that gives the target acceptance rate.

        This takes ~1-5 seconds per mode and produces model-adaptive
        parameters that can be cached via :meth:`save_calibration`.

        Args:
            modes (List[str]): Modes to calibrate. Defaults to all four.
            n_test (int): Candidates per trial. Defaults to 200.
            target_acceptance (float): Target fraction of valid candidates.
                Defaults to 0.8.
            verbose (bool): Print calibration results.

        Returns:
            dict: Calibrated σ values per mode.
        """
        if modes is None:
            modes = ["F", "H", "ISD+", "ISD-"]

        mod_pts = jnp.array(
            self.sampler.get_complex_moduli(n_test), dtype=complex)
        tau_pts = jnp.array(
            self.sampler.get_complex_tau(n_test), dtype=complex)

        def _acceptance(sigma, mode):
            """Measure acceptance rate at given σ."""
            rng = np.random.default_rng(42)
            inp_len = (self.n_fluxes if mode in ("F", "H")
                       else 2 * self.dimension_H3)
            inputs = np.round(rng.normal(0, sigma, (n_test, inp_len)))
            valid = 0
            for i in range(n_test):
                inp = jnp.array(inputs[i], dtype=float)
                if jnp.all(inp == 0):
                    continue
                try:
                    fl = self.sampler.ISD_sampling(
                        mod_pts[i], jnp.conj(mod_pts[i]),
                        tau_pts[i], jnp.conj(tau_pts[i]),
                        inp, mode=mode)
                except Exception:
                    continue
                if fl is None or not jnp.all(jnp.isfinite(fl)):
                    continue
                fl_int = np.round(np.array(jnp.array(fl).real)).astype(float)
                tad = abs(float(jnp.real(
                    self.model.tadpole(jnp.array(fl_int)))))
                if 0 < tad <= self.Nmax:
                    valid += 1
            return valid / n_test

        if verbose:
            print("[calibrate_priors] Calibrating σ for each ISD mode...")

        for mode in modes:
            # Binary search: find σ where acceptance ≈ target
            lo, hi = 0.1, 20.0
            best_sigma = self._calibrated_sigmas.get(mode, 1.0)
            best_diff = abs(_acceptance(best_sigma, mode) - target_acceptance)

            for _ in range(12):  # ~12 iterations for good precision
                mid = (lo + hi) / 2
                acc = _acceptance(mid, mode)
                diff = abs(acc - target_acceptance)
                if diff < best_diff:
                    best_diff = diff
                    best_sigma = mid
                if acc > target_acceptance:
                    lo = mid  # σ too small → increase
                else:
                    hi = mid  # σ too large → decrease

            self._calibrated_sigmas[mode] = best_sigma
            if verbose:
                final_acc = _acceptance(best_sigma, mode)
                print(f"  {mode:>5}: σ = {best_sigma:.3f} "
                      f"(acceptance = {100*final_acc:.0f}%)")

        # Diagnostic: run a quick solve to measure runaway fraction.
        # Always runs when moduli_max is set (results stored in return dict).
        runaway_info = None
        if self.moduli_max is not None:
            if verbose:
                print(f"\n[calibrate_priors] Runaway diagnostic "
                      f"(moduli_max={self.moduli_max}, 50 candidates, "
                      f"mode=ISD-, solver=scipy):")

            diag_finder = CriticalPointFinder(
                self.model, self.sampler, Nmax=self.Nmax,
                noscale=self.noscale, flux_prior=self.flux_prior,
                flux_prior_sigma=self.flux_prior_sigma,
                moduli_max=None)  # no filter for diagnostic
            diag_finder._calibrated_sigmas = dict(self._calibrated_sigmas)

            diag_results = diag_finder.sample_critical_points(
                n_target=500, n_batch=50, max_batches=1,
                isd_mode="ISD-", solver="scipy", verbose=False)

            if diag_results:
                n_phys = sum(1 for r in diag_results
                             if self._is_physical(
                                 np.asarray(self.model._convert_complex_to_real(
                                     jnp.array(r['moduli']),
                                     jnp.conj(jnp.array(r['moduli'])),
                                     r['tau'], np.conj(r['tau'])))))
                n_run = len(diag_results) - n_phys
                runaway_info = {
                    'n_converged': len(diag_results),
                    'n_physical': n_phys,
                    'n_runaways': n_run,
                    'runaway_fraction': n_run / len(diag_results),
                }
                if verbose:
                    print(f"  {len(diag_results)} converged, "
                          f"{n_phys} physical, "
                          f"{n_run} runaways "
                          f"({100*n_run/len(diag_results):.0f}%)")
                    if n_run > 0.8 * len(diag_results):
                        print(f"  Warning: >80% runaways. Consider increasing "
                              f"moduli_max or using solver='hybrid'.")
            elif verbose:
                print("  No solutions found in diagnostic run.")

        result = dict(self._calibrated_sigmas)
        if runaway_info is not None:
            result['_runaway_diagnostic'] = runaway_info
        return result

    # ------------------------------------------------------------------
    #  Save / load calibration
    # ------------------------------------------------------------------

    def _get_model_dir(self) -> str:
        r"""
        **Description:**
        Derive the model data directory from the model's lcs_tree metadata.
        Returns the directory containing the model's ``.p`` file, e.g.
        ``jaxvacua/models/KS/h12_2/``.
        """
        import os
        try:
            home_dir = os.path.dirname(os.path.realpath(__file__))
            model_type = getattr(self.model, 'model_type', 'KS')
            h12 = self.model.h12
            return os.path.join(home_dir, "models", model_type, f"h12_{h12}")
        except Exception:
            return None

    def save_calibration(self, path: str = None) -> str:
        r"""
        **Description:**
        Save calibrated prior parameters to a JSON file for reuse.

        If ``path`` is ``None``, saves in the model's data directory
        (e.g. ``jaxvacua/models/KS/h12_2/critical_points_prior_Nmax200.json``).
        Falls back to the current directory if the model directory
        cannot be determined.

        Returns:
            str: Path to the saved file.
        """
        import os, json

        if path is None:
            model_dir = self._get_model_dir()
            if model_dir is not None and os.path.isdir(model_dir):
                path = os.path.join(
                    model_dir,
                    f"critical_points_prior_Nmax{self.Nmax}.json")
            else:
                path = f"critical_points_prior_Nmax{self.Nmax}.json"

        data = {
            'Nmax': self.Nmax,
            'n_fluxes': self.n_fluxes,
            'flux_prior': self.flux_prior,
            'sigmas': self._calibrated_sigmas,
            'M_cond': float(self._M_cond) if self._M_cond else None,
            'tr_Minv_median': self._tr_Minv_median,
            's_min': self._s_min,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        return path

    def load_calibration(self, path: str) -> dict:
        r"""
        **Description:**
        Load calibrated prior parameters from a JSON file.

        Returns:
            dict: Loaded σ values per mode.
        """
        import json
        with open(path) as f:
            data = json.load(f)
        self._calibrated_sigmas = data.get('sigmas', {})
        return dict(self._calibrated_sigmas)

    # ==========================================================================
    #  Flux candidate generation
    # ==========================================================================

    def _generate_flux_candidates(
        self,
        n_candidates: int,
        moduli_pts: jnp.ndarray,
        tau_pts: jnp.ndarray,
        isd_mode: str = "ISD-",
        rng: np.random.Generator = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        **Description:**
        Generate integer flux candidates via ISD sampling in the specified mode.

        Args:
            n_candidates (int): Number of (moduli, flux) pairs to attempt.
            moduli_pts (Array): Starting moduli, shape ``(N, h12)``.
            tau_pts (Array): Starting axio-dilatons, shape ``(N,)``.
            isd_mode (str): One of ``"F"``, ``"H"``, ``"ISD+"``, ``"ISD-"``.
            rng (np.random.Generator): Random number generator. If ``None``,
                a deterministic ``np.random.default_rng(0)`` is used so that
                tests and downstream consumers see reproducible flux draws;
                pass an explicit ``rng`` to vary the seed.

        Returns:
            Tuple of (x0_array, flux_array, indices) for valid candidates.
        """
        if rng is None:
            rng = np.random.default_rng(0)

        n_fl = self.n_fluxes
        dim_H3 = self.dimension_H3
        N = min(n_candidates, len(moduli_pts))

        # Determine input vector length
        if isd_mode in ("F", "H"):
            inp_len = n_fl
        elif isd_mode in ("ISD+", "ISD-"):
            inp_len = 2 * dim_H3
        else:
            raise ValueError(f"Unknown isd_mode: {isd_mode}. "
                             f"Must be one of 'F', 'H', 'ISD+', 'ISD-'.")

        # Generate input vectors using the configured prior.
        # σ is determined by: (1) explicit flux_prior_sigma override,
        # (2) calibrated values (if calibrate_priors() was called),
        # (3) empirically tuned defaults.
        if self.flux_prior in ("gaussian", "M_weighted"):
            if self.flux_prior_sigma is not None:
                sigma = self.flux_prior_sigma
            elif isd_mode in self._calibrated_sigmas:
                sigma = self._calibrated_sigmas[isd_mode]
            else:
                # Empirically tuned defaults (h12=2, KS model benchmark)
                sigma = {
                    "F": 1.0,     # isotropic σ=1 gives ~87% valid
                    "H": 1.0,     # isotropic σ=1 gives ~28% valid
                    "ISD+": 0.5,  # σ=0.5 gives ~53% valid
                    "ISD-": 2.0,  # σ=2 gives ~93% valid
                }.get(isd_mode, 1.0)

            if (self.flux_prior == "M_weighted" and isd_mode == "H"
                    and self._M_eigvecs is not None):
                # M-weighted: h ~ N(0, scale² · σ²_M · M) via eigensystem.
                # _M_scales already contain sqrt(Nmax/(d·s_min) · λ_M_i).
                # scale_factor rescales the overall magnitude.
                scale_factor = self.flux_prior_sigma or 0.3
                z = rng.standard_normal((N, inp_len))
                inputs = np.round(
                    (z * self._M_scales[None, :] * scale_factor)
                    @ self._M_eigvecs.T)
            else:
                # Isotropic Gaussian N(0, σ²I)
                inputs = np.round(rng.normal(0, sigma, (N, inp_len)))

        elif self.flux_prior == "uniform":
            inputs = rng.integers(-3, 4, (N, inp_len)).astype(float)
        else:
            raise ValueError(f"Unknown flux_prior: {self.flux_prior}. "
                             f"Must be 'gaussian', 'M_weighted', or 'uniform'.")

        x0_list = []
        flux_list = []

        for i in range(N):
            inp = jnp.array(inputs[i], dtype=float)
            if jnp.all(inp == 0):
                continue
            try:
                fl = self.sampler.ISD_sampling(
                    moduli_pts[i], jnp.conj(moduli_pts[i]),
                    tau_pts[i], jnp.conj(tau_pts[i]),
                    inp, mode=isd_mode,
                )
            except Exception:
                continue

            if fl is None or not jnp.all(jnp.isfinite(fl)):
                continue

            # Round to integers
            fl_int = np.round(np.array(jnp.array(fl).real, copy=True)).astype(float)
            tad = abs(float(jnp.real(self.model.tadpole(jnp.array(fl_int)))))
            if tad <= 0 or tad > self.Nmax:
                continue

            # Starting point in real coordinates: (Re z1, Im z1, ..., Re tau, Im tau)
            x0 = np.asarray(self.model._convert_complex_to_real(
                moduli_pts[i], jnp.conj(moduli_pts[i]),
                tau_pts[i], jnp.conj(tau_pts[i])))

            x0_list.append(x0)
            flux_list.append(fl_int)

        if not x0_list:
            n_fields = 2 * (self.model.h12 + 1)
            return (np.zeros((0, n_fields)),
                    np.zeros((0, 2 * n_fl)),
                    np.array([], dtype=int))

        return np.array(x0_list), np.array(flux_list), np.arange(len(x0_list))

    # ==========================================================================
    #  Newton solver for ∂V/∂φ = 0
    # ==========================================================================

    def _solve_dV_newton_single(
        self,
        x0: np.ndarray,
        flux: np.ndarray,
        step_size: float = 1.0,
        tol: float = 1e-10,
        max_iters: int = 300,
    ) -> Tuple[np.ndarray, float, bool]:
        r"""
        **Description:**
        Solve :math:`\partial V / \partial \phi = 0` for a single
        (starting point, flux) pair using Newton's method.

        Uses ``model.dV_x`` (gradient) and ``model.ddV_x`` (Hessian)
        which are JIT-compiled.

        Returns:
            Tuple of (x_solution, residual, converged).
        """
        x_jax = jnp.array(x0)
        fl_jax = jnp.array(flux)
        ns = self.noscale
        _mmax = self.moduli_max

        for _ in range(max_iters):
            dV = self.model.dV_x(x_jax, fl_jax, noscale=ns)
            res = float(jnp.sum(jnp.abs(dV)))
            if res < tol:
                return np.asarray(x_jax), res, True
            if not jnp.all(jnp.isfinite(dV)):
                return np.asarray(x_jax), float('inf'), False

            ddV = self.model.ddV_x(x_jax, fl_jax, noscale=ns)
            try:
                delta = -jnp.linalg.solve(ddV, dV)
            except Exception:
                return np.asarray(x_jax), res, False

            if not jnp.all(jnp.isfinite(delta)):
                return np.asarray(x_jax), res, False

            x_jax = x_jax + step_size * delta

            # Early escape: abort if moduli have run away
            if _mmax is not None and float(jnp.max(jnp.abs(x_jax))) > _mmax:
                return np.asarray(x_jax), float('inf'), False

        dV = self.model.dV_x(x_jax, fl_jax, noscale=ns)
        res = float(jnp.sum(jnp.abs(dV)))
        return np.asarray(x_jax), res, res < tol

    def _solve_dV_optax_single(
        self,
        x0: np.ndarray,
        flux: np.ndarray,
        optimiser=None,
        n_steps: int = 1000,
        tol: float = 1e-10,
    ) -> Tuple[np.ndarray, float, bool]:
        r"""
        **Description:**
        Minimise :math:`|\nabla V|^2` using an optax optimiser for a single
        (starting point, flux) pair.

        Args:
            optimiser: An ``optax.GradientTransformation``. Defaults to
                ``optax.adam(1e-3)`` with exponential decay.
        """
        if not _HAS_OPTAX:
            raise ImportError("optax is required for optax-based solvers. "
                              "Install with: pip install optax")

        fl_jax = jnp.array(flux)
        ns = self.noscale

        @jax.jit
        def loss_fn(x):
            dV = self.model.dV_x(x, fl_jax, noscale=ns)
            return jnp.sum(dV ** 2)

        grad_fn = jax.jit(jax.grad(loss_fn))

        if optimiser is None:
            schedule = optax.exponential_decay(
                init_value=1e-3, transition_steps=200, decay_rate=0.5)
            optimiser = optax.adam(learning_rate=schedule)

        x_cur = jnp.array(x0)
        opt_state = optimiser.init(x_cur)

        for step in range(n_steps):
            g = grad_fn(x_cur)
            if not jnp.all(jnp.isfinite(g)):
                break
            updates, opt_state = optimiser.update(g, opt_state, x_cur)
            x_cur = optax.apply_updates(x_cur, updates)

            if step % 50 == 49:
                loss = float(loss_fn(x_cur))
                if loss < tol ** 2:
                    break

        # Final residual
        dV_final = self.model.dV_x(x_cur, fl_jax, noscale=ns)
        res = float(jnp.sum(jnp.abs(dV_final)))
        return np.asarray(x_cur), res, res < tol

    # ==========================================================================
    #  Vectorised optax solver
    # ==========================================================================

    def _build_optax_kernel(
        self,
        optimiser,
        n_steps: int = 2000,
        objective: str = "dV2",
    ) -> Callable:
        r"""
        **Description:**
        Build a JIT-compiled, vmapped optax optimisation kernel.

        The returned function has signature
        ``(x0_batch, flux_batch) -> (x_final_batch, residual_batch)``
        and is fully vectorised over the batch dimension.

        Args:
            optimiser: An ``optax.GradientTransformation``.
            n_steps (int): Number of optimisation steps per candidate.
            objective (str): Loss function variant:
                ``"dV2"`` for :math:`|\nabla V|^2`,
                ``"log_dV2"`` for :math:`\log(|\nabla V|^2 + \epsilon)`,
                ``"V"`` for :math:`V` directly (finds minima, not saddle points).

        Returns:
            A callable ``(x0_batch, flux_batch) -> (x_finals, residuals)``.
        """
        if not _HAS_OPTAX:
            raise ImportError("optax is required. Install with: pip install optax")

        model = self.model
        ns = self.noscale

        # --- Build objective function with flux as an explicit argument ---
        if objective == "dV2":
            @jax.jit
            def loss_fn(x, fl):
                dV = model.dV_x(x, fl, noscale=ns)
                return jnp.sum(dV ** 2)
        elif objective == "log_dV2":
            @jax.jit
            def loss_fn(x, fl):
                dV = model.dV_x(x, fl, noscale=ns)
                return jnp.log(jnp.sum(dV ** 2) + 1e-30)
        elif objective == "V":
            @jax.jit
            def loss_fn(x, fl):
                return model.V_x(x, fl, noscale=ns)
        else:
            raise ValueError(f"Unknown objective: {objective}")

        grad_fn = jax.grad(loss_fn)

        def solve_single(x0, fl):
            """Optimise a single (x0, flux) pair using lax.scan."""
            opt_state = optimiser.init(x0)

            def step_fn(carry, _):
                x, opt_st = carry
                g = grad_fn(x, fl)
                # Replace NaN gradients with zero (halts progress, avoids divergence)
                g = jnp.where(jnp.isfinite(g), g, 0.0)
                updates, new_opt_st = optimiser.update(g, opt_st, x)
                new_x = optax.apply_updates(x, updates)
                # Protect against NaN in x
                new_x = jnp.where(jnp.isfinite(new_x), new_x, x)
                return (new_x, new_opt_st), None

            (x_final, _), _ = lax.scan(step_fn, (x0, opt_state), None,
                                        length=n_steps)

            # Compute final residual: sum |dV|
            dV_final = model.dV_x(x_final, fl, noscale=ns)
            residual = jnp.sum(jnp.abs(dV_final))
            return x_final, residual

        # Vmap over batch dimension and JIT-compile once
        solve_batch = jax.jit(vmap(solve_single))
        return solve_batch

    def _solve_dV_optax_batch(
        self,
        x0_batch: np.ndarray,
        flux_batch: np.ndarray,
        optimiser=None,
        n_steps: int = 2000,
        objective: str = "dV2",
        tol: float = 1e-10,
        sub_batch_size: int = 64,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        **Description:**
        Vectorised optax solver for :math:`\partial V / \partial \phi = 0`.

        Uses ``lax.scan`` for the step loop (JIT-traceable) and ``vmap`` over
        the batch dimension, compiled into a single XLA kernel.

        Args:
            x0_batch (Array): Starting points, shape ``(N, 2*(h12+1))``.
            flux_batch (Array): Integer fluxes, shape ``(N, 2*n_fluxes)``.
            optimiser: An ``optax.GradientTransformation``.
                Defaults to ``optax.adam`` with exponential decay.
            n_steps (int): Optimisation steps per candidate. Defaults to 2000.
            objective (str): ``"dV2"``, ``"log_dV2"``, or ``"V"``.
            tol (float): Convergence tolerance on :math:`\sum|\partial V|`.
            sub_batch_size (int): Process in sub-batches to limit memory.

        Returns:
            Tuple of ``(x_finals, residuals, converged)`` arrays.
        """
        if not _HAS_OPTAX:
            raise ImportError("optax is required. Install with: pip install optax")

        if optimiser is None:
            schedule = optax.cosine_decay_schedule(
                init_value=0.5, decay_steps=n_steps)
            optimiser = optax.chain(
                optax.clip_by_global_norm(10.0),
                optax.adam(learning_rate=schedule))

        kernel = self._build_optax_kernel(optimiser, n_steps, objective)

        N = len(x0_batch)
        all_x = []
        all_res = []

        # Process in sub-batches to limit memory usage
        for i in range(0, N, sub_batch_size):
            j = min(i + sub_batch_size, N)
            x_sub = jnp.array(x0_batch[i:j])
            fl_sub = jnp.array(flux_batch[i:j])
            x_out, res_out = kernel(x_sub, fl_sub)
            all_x.append(np.asarray(x_out))
            all_res.append(np.asarray(res_out))

        x_finals = np.concatenate(all_x, axis=0)
        residuals = np.concatenate(all_res, axis=0)
        converged = residuals < tol

        return x_finals, residuals, converged

    # ==========================================================================
    #  Vectorised Newton solver
    # ==========================================================================

    def _build_newton_kernel(
        self,
        n_iters: int = 50,
        step_size: float = 1.0,
    ) -> Callable:
        r"""
        **Description:**
        Build a JIT-compiled, vmapped Newton solver kernel.

        Uses ``lax.scan`` for a fixed number of Newton iterations with
        NaN-guarded updates. The returned function has signature
        ``(x0_batch, flux_batch) -> (x_final_batch, residual_batch)``.

        Args:
            n_iters (int): Fixed number of Newton iterations. Defaults to 50.
            step_size (float): Newton step size. Defaults to 1.0.

        Returns:
            A callable ``(x0_batch, flux_batch) -> (x_finals, residuals)``.
        """
        model = self.model
        ns = self.noscale
        alpha = step_size

        def solve_single(x0, fl):
            def newton_step(x, _):
                dV = model.dV_x(x, fl, noscale=ns)
                ddV = model.ddV_x(x, fl, noscale=ns)
                delta = -jnp.linalg.solve(ddV, dV)
                # Guard against NaN/Inf: keep old x if step is invalid
                new_x = x + alpha * delta
                valid = jnp.all(jnp.isfinite(new_x))
                new_x = jnp.where(valid, new_x, x)
                return new_x, None

            x_final, _ = lax.scan(newton_step, x0, None, length=n_iters)
            dV_final = model.dV_x(x_final, fl, noscale=ns)
            residual = jnp.sum(jnp.abs(dV_final))
            return x_final, residual

        return jax.jit(vmap(solve_single))

    def _solve_dV_newton_batch(
        self,
        x0_batch: np.ndarray,
        flux_batch: np.ndarray,
        n_iters: int = 50,
        step_size: float = 1.0,
        tol: float = 1e-10,
        sub_batch_size: int = 64,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        **Description:**
        Vectorised Newton solver for :math:`\partial V / \partial \phi = 0`.

        Uses ``lax.scan`` (fixed iteration count) + ``vmap`` over the batch,
        compiled into a single XLA kernel.

        Args:
            x0_batch (Array): Starting points, shape ``(N, 2*(h12+1))``.
            flux_batch (Array): Integer fluxes, shape ``(N, 2*n_fluxes)``.
            n_iters (int): Fixed Newton iterations. Defaults to 50.
            step_size (float): Newton step size. Defaults to 1.0.
            tol (float): Convergence tolerance on :math:`\sum|\partial V|`.
            sub_batch_size (int): Process in sub-batches to limit memory.

        Returns:
            Tuple of ``(x_finals, residuals, converged)`` arrays.
        """
        kernel = self._build_newton_kernel(n_iters, step_size)

        N = len(x0_batch)
        all_x = []
        all_res = []

        for i in range(0, N, sub_batch_size):
            j = min(i + sub_batch_size, N)
            x_sub = jnp.array(x0_batch[i:j])
            fl_sub = jnp.array(flux_batch[i:j])
            x_out, res_out = kernel(x_sub, fl_sub)
            all_x.append(np.asarray(x_out))
            all_res.append(np.asarray(res_out))

        x_finals = np.concatenate(all_x, axis=0)
        residuals = np.concatenate(all_res, axis=0)
        converged = residuals < tol

        return x_finals, residuals, converged

    # ==========================================================================
    #  Classification
    # ==========================================================================

    def _classify_solution(
        self,
        x_sol: np.ndarray,
        flux: np.ndarray,
    ) -> dict:
        r"""
        **Description:**
        Classify a critical point: compute V, |DW|, Hessian eigenvalues,
        and determine whether it is SUSY, a minimum, etc.

        Returns:
            dict with keys: V, |DW|, eigenvalues, is_susy, is_minimum, Nflux.
        """
        x_jax = jnp.array(x_sol)
        fl_jax = jnp.array(flux)
        ns = self.noscale

        z, z_c, tau, tau_c = self.model._convert_real_to_complex(x_jax)

        V = float(self.model.scalar_potential(
            z, z_c, tau, tau_c, fl_jax, noscale=ns).real)
        DW = self.model.DW(z, z_c, tau, tau_c, fl_jax)
        dw_norm = float(jnp.sum(jnp.abs(DW)))

        ddV = self.model.ddV_x(x_jax, fl_jax, noscale=ns)
        eigs = np.sort(np.asarray(jnp.linalg.eigvalsh(ddV).real))

        tad = abs(float(jnp.real(self.model.tadpole(fl_jax))))

        return {
            'V': V,
            '|DW|': dw_norm,
            'eigenvalues': eigs,
            'is_susy': dw_norm < 1e-6,
            'is_minimum': bool(np.all(eigs > -1e-6)),
            'Nflux': tad,
        }

    def _build_classify_kernel(self) -> Callable:
        r"""
        **Description:**
        Build a JIT-compiled, vmapped classification kernel.

        Returns a callable ``(x_batch, flux_batch) -> (V, dw_norm, eigs, tad)``
        where each output is an array over the batch dimension.
        """
        model = self.model
        ns = self.noscale

        def classify_single(x, fl):
            z, zb, tau, taub = model._convert_real_to_complex(x)

            V = model.scalar_potential(z, zb, tau, taub, fl, noscale=ns).real
            DW = model.DW(z, zb, tau, taub, fl)
            dw_norm = jnp.sum(jnp.abs(DW))

            ddV = model.ddV_x(x, fl, noscale=ns)
            eigs = jnp.sort(jnp.linalg.eigvalsh(ddV).real)

            tad = jnp.abs(jnp.real(model.tadpole(fl)))

            return V, dw_norm, eigs, tad

        return jax.jit(vmap(classify_single))

    def _classify_batch(
        self,
        x_batch: np.ndarray,
        flux_batch: np.ndarray,
        sub_batch_size: int = 64,
    ) -> List[dict]:
        r"""
        **Description:**
        Vectorised classification of critical points. Computes V, |DW|,
        Hessian eigenvalues, and tadpole for a batch of solutions.

        Args:
            x_batch (Array): Solution points, shape ``(N, 2*(h12+1))``.
            flux_batch (Array): Flux vectors, shape ``(N, 2*n_fluxes)``.
            sub_batch_size (int): Sub-batch size for memory control.

        Returns:
            List of dicts with keys: V, |DW|, eigenvalues, is_susy,
            is_minimum, Nflux.
        """
        kernel = self._build_classify_kernel()
        N = len(x_batch)

        all_V, all_dw, all_eigs, all_tad = [], [], [], []
        for i in range(0, N, sub_batch_size):
            j = min(i + sub_batch_size, N)
            V, dw, eigs, tad = kernel(
                jnp.array(x_batch[i:j]), jnp.array(flux_batch[i:j]))
            all_V.append(np.asarray(V))
            all_dw.append(np.asarray(dw))
            all_eigs.append(np.asarray(eigs))
            all_tad.append(np.asarray(tad))

        V_arr = np.concatenate(all_V)
        dw_arr = np.concatenate(all_dw)
        eigs_arr = np.concatenate(all_eigs, axis=0)
        tad_arr = np.concatenate(all_tad)

        results = []
        for k in range(N):
            results.append({
                'V': float(V_arr[k]),
                '|DW|': float(dw_arr[k]),
                'eigenvalues': eigs_arr[k],
                'is_susy': bool(dw_arr[k] < 1e-6),
                'is_minimum': bool(np.all(eigs_arr[k] > -1e-6)),
                'Nflux': float(tad_arr[k]),
            })
        return results

    # ==========================================================================
    #  Main entry point
    # ==========================================================================

    def sample_critical_points(
        self,
        n_target: int = 100,
        n_batch: int = 10_000,
        max_batches: int = 20,
        isd_mode: str = "ISD-",
        solver: str = "newton",
        step_size: float = 1.0,
        newton_tol: float = 1e-10,
        newton_max_iters: int = 300,
        optax_steps: int = 1000,
        optax_objective: str = "dV2",
        optimiser=None,
        classify: bool = True,
        deduplicate: bool = True,
        verbose: bool = True,
    ) -> List[dict]:
        r"""
        **Description:**
        Find critical points of the scalar potential :math:`V` by
        generating ISD flux candidates and solving
        :math:`\partial V / \partial \phi = 0`.

        Args:
            n_target (int): Target number of critical points.
            n_batch (int): Flux candidates per batch.
            max_batches (int): Maximum number of batches.
            isd_mode (str): ISD sampling mode (``"F"``, ``"H"``, ``"ISD+"``,
                ``"ISD-"``). Defaults to ``"ISD-"`` (best non-SUSY yield).
            solver (str): Solver backend: ``"newton"`` (JAX Newton),
                ``"adam"`` / ``"lbfgs"`` (optax, per-candidate loop),
                ``"adam_v"`` (vectorised optax Adam via ``lax.scan`` + ``vmap``),
                ``"hybrid"`` (vectorised Adam warm-start + Newton refinement),
                ``"scipy"`` (scipy.optimize.root).
            step_size (float): Newton step size. Defaults to ``1.0``.
            newton_tol (float): Convergence tolerance. Defaults to ``1e-10``.
            newton_max_iters (int): Max Newton iterations. Defaults to ``300``.
            optax_steps (int): Max steps for optax solvers. Defaults to ``1000``.
            optax_objective (str): Loss function for optax solvers:
                ``"dV2"`` (:math:`|\nabla V|^2`),
                ``"log_dV2"`` (:math:`\log(|\nabla V|^2 + \epsilon)`),
                ``"V"`` (:math:`V` directly). Defaults to ``"dV2"``.
            optimiser: Custom ``optax.GradientTransformation``. Overrides
                ``solver`` when provided.
            classify (bool): Compute Hessian eigenvalues for min/saddle
                classification. Defaults to ``True``.
            deduplicate (bool): Remove duplicate solutions based on
                ``(flux, moduli, tau)``.  If ``map_to_fd`` is set, vacua are
                mapped to the fundamental domain before comparison.
                Defaults to ``True``.
            verbose (bool): Print progress. Defaults to ``True``.

        Returns:
            List[dict]: Each entry has keys ``flux``, ``moduli``, ``tau``,
            ``residual``, and (if ``classify=True``) ``V``, ``|DW|``,
            ``eigenvalues``, ``is_susy``, ``is_minimum``, ``Nflux``.
        """
        t0 = time.perf_counter()

        def _elapsed():
            s = time.perf_counter() - t0
            if s < 120: return f"{s:.1f}s"
            elif s < 3600: return f"{s/60:.1f}m"
            else:
                h = int(s // 3600); m = int((s % 3600) // 60)
                return f"{h}h {m}m"

        if verbose:
            ns_label = "no-scale" if self.noscale else "full SUGRA"
            print(f"[critical_points] Searching for critical points of V ({ns_label})")
            print(f"  Nmax={self.Nmax}, mode={isd_mode}, solver={solver}"
                  + (f", moduli_max={self.moduli_max}"
                     if self.moduli_max is not None else ""))

        results = []
        seen = set()
        n_tried = 0
        n_valid = 0
        n_runaway_total = 0
        rng = np.random.default_rng()

        # Build optax optimiser if needed
        optax_opt = optimiser
        if solver == "adam" and optax_opt is None and _HAS_OPTAX:
            schedule = optax.exponential_decay(
                init_value=1e-3, transition_steps=200, decay_rate=0.5)
            optax_opt = optax.adam(learning_rate=schedule)
        elif solver == "lbfgs" and optax_opt is None and _HAS_OPTAX:
            optax_opt = optax.lbfgs()

        for batch_idx in range(max_batches):
            if len(results) >= n_target:
                break

            # Sample moduli/tau starting points
            mod_pts = jnp.array(
                self.sampler.get_complex_moduli(n_batch), dtype=complex)
            tau_pts = jnp.array(
                self.sampler.get_complex_tau(n_batch), dtype=complex)

            # Generate flux candidates
            x0_arr, flux_arr, _ = self._generate_flux_candidates(
                n_batch, mod_pts, tau_pts, isd_mode=isd_mode, rng=rng)

            n_tried += n_batch
            n_valid += len(x0_arr)

            if len(x0_arr) == 0:
                if verbose:
                    print(f"  Batch {batch_idx+1}: 0 valid flux candidates  [{_elapsed()}]")
                continue

            # Solve ∂V/∂φ = 0
            batch_results = []
            n_runaway_batch = 0

            if solver in ("adam_v", "hybrid"):
                # --- Vectorised optax path ---
                x_finals, residuals, conv_arr = self._solve_dV_optax_batch(
                    x0_arr, flux_arr,
                    optimiser=optax_opt, n_steps=optax_steps,
                    objective=optax_objective, tol=newton_tol)

                if solver == "hybrid":
                    # Filter runaways before Newton (avoids wasting iterations)
                    if self.moduli_max is not None:
                        max_abs = np.max(np.abs(x_finals), axis=1)
                        runaway = max_abs > self.moduli_max
                        n_runaway_batch = int(runaway.sum())
                        conv_arr[runaway] = False
                        residuals[runaway] = np.inf
                    else:
                        n_runaway_batch = 0

                    # Phase 2: Newton refinement on near-converged candidates.
                    # Sequential with early exit is faster on CPU than vmapped
                    # fixed-iteration Newton (median convergence: 14 iters).
                    near_mask = (residuals < 1.0) & (~conv_arr)
                    for j in np.where(near_mask)[0]:
                        x_warm, res_w, conv_w = self._solve_dV_newton_single(
                            x_finals[j], flux_arr[j],
                            step_size=step_size, tol=newton_tol,
                            max_iters=newton_max_iters)
                        if conv_w:
                            x_finals[j] = x_warm
                            residuals[j] = res_w
                            conv_arr[j] = True

                # Filter converged, physical, unique solutions
                conv_idx = np.where(conv_arr)[0]
                for j in conv_idx:
                    if not self._is_physical(x_finals[j]):
                        continue
                    z_sol, _, tau_sol, _ = self.model._convert_real_to_complex(
                        jnp.array(x_finals[j]))
                    fv, mv, tv = self._map_result_to_fd(
                        flux_arr[j], np.asarray(z_sol), complex(tau_sol))
                    if deduplicate:
                        key = self._dedup_key(fv, mv, tv)
                        if key in seen:
                            continue
                        seen.add(key)
                    entry = {'flux': fv, 'moduli': mv, 'tau': tv,
                             'residual': float(residuals[j])}
                    if classify:
                        entry.update(self._classify_solution(
                            x_finals[j], flux_arr[j]))
                    batch_results.append(entry)
            else:
                # --- Per-candidate loop path ---
                for j in range(len(x0_arr)):
                    if solver == "newton":
                        x_sol, res, conv = self._solve_dV_newton_single(
                            x0_arr[j], flux_arr[j],
                            step_size=step_size, tol=newton_tol,
                            max_iters=newton_max_iters)
                    elif solver in ("adam", "lbfgs") or optimiser is not None:
                        x_sol, res, conv = self._solve_dV_optax_single(
                            x0_arr[j], flux_arr[j],
                            optimiser=optax_opt, n_steps=optax_steps,
                            tol=newton_tol)
                    elif solver == "scipy":
                        from scipy.optimize import root as scipy_root
                        fl_np = flux_arr[j]
                        def f(x): return np.asarray(
                            self.model.dV_x(jnp.array(x), jnp.array(fl_np),
                                            noscale=self.noscale))
                        def jac(x): return np.asarray(
                            self.model.ddV_x(jnp.array(x), jnp.array(fl_np),
                                             noscale=self.noscale))
                        res_sp = scipy_root(f, x0=x0_arr[j], method='hybr',
                                            jac=jac)
                        x_sol = res_sp.x
                        res = max(abs(res_sp.fun)) if res_sp.success else float('inf')
                        conv = res_sp.success and res < newton_tol
                    else:
                        raise ValueError(f"Unknown solver: {solver}")

                    if not conv:
                        continue

                    # Check physicality (hyperplanes / Kähler metric / Im>0)
                    if not self._is_physical(x_sol):
                        continue

                    # Build result and dedup
                    z_sol, _, tau_sol, _ = self.model._convert_real_to_complex(
                        jnp.array(x_sol))
                    fv, mv, tv = self._map_result_to_fd(
                        flux_arr[j], np.asarray(z_sol), complex(tau_sol))
                    if deduplicate:
                        key = self._dedup_key(fv, mv, tv)
                        if key in seen:
                            continue
                        seen.add(key)

                    entry = {
                        'flux': fv,
                        'moduli': mv,
                        'tau': tv,
                        'residual': res,
                    }

                    if classify:
                        info = self._classify_solution(x_sol, flux_arr[j])
                        entry.update(info)

                    batch_results.append(entry)

            results.extend(batch_results)

            if verbose:
                n_susy = sum(1 for r in batch_results if r.get('is_susy', False))
                n_ns = len(batch_results) - n_susy
                n_min = sum(1 for r in batch_results if r.get('is_minimum', False))
                run_str = (f", {n_runaway_batch} runaways"
                           if n_runaway_batch > 0 else "")
                print(
                    f"  Batch {batch_idx+1}: {len(x0_arr)} candidates → "
                    f"{len(batch_results)} critical pts "
                    f"({n_susy} SUSY, {n_ns} non-SUSY, {n_min} minima{run_str}) "
                    f"| total: {len(results)}  [{_elapsed()}]")

            n_runaway_total += n_runaway_batch

        if verbose:
            n_susy_tot = sum(1 for r in results if r.get('is_susy', False))
            n_ns_tot = len(results) - n_susy_tot
            n_min_tot = sum(1 for r in results if r.get('is_minimum', False))
            n_sad_tot = len(results) - n_min_tot
            run_str = (f", {n_runaway_total} runaways filtered"
                       if n_runaway_total > 0 else "")
            print(
                f"\n[critical_points] Done: {len(results)} critical points "
                f"({n_susy_tot} SUSY, {n_ns_tot} non-SUSY, "
                f"{n_min_tot} minima, {n_sad_tot} saddle{run_str}) "
                f"from {n_valid} valid candidates ({n_tried} tried)  [{_elapsed()}]")

        return results
