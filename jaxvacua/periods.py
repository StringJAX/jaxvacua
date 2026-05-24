# Copyright 2022-2026 Andreas Schachner
#
# This file is part of JAXVacua.
#
# JAXVacua is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# JAXVacua is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with JAXVacua. If not, see <https://www.gnu.org/licenses/>.

"""Period and prepotential machinery for Calabi-Yau threefolds.

Purpose
-------
Define the ``periods`` class, the low-level object that evaluates period
vectors, prepotentials and derived special-geometry data in supported regions
of complex-structure moduli space.

Main public API
---------------
- ``periods``: loads or constructs model data from ``lcs_tree``, dictionaries,
  saved model files, CYTools input or custom period/prepotential callables.
- LCS, coniLCS and one-modulus hypergeometric period/prepotential
  implementations, including polynomial and instanton contributions.
- Period vectors, derivatives, gauge-kinetic matrices and projective/affine
  coordinate conversions consumed by ``css`` and ``FluxEFT``.

Design notes
------------
This is the numerical core below the geometry classes.  It is JAX-pytree
compatible and keeps model data in forms suitable for JIT-compiled downstream
calculations.
"""


# Important standard libraries
import os, sys, warnings
import numpy as np
import itertools
from functools import partial

# Important JAX libraries
import jax
from jax import jit, vmap, config
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import zeta
from jax.numpy import pi as Pi
from jax.tree_util import register_pytree_node


#Polylog imports
from jaxpolylog import jax_polylog_vmap

# JAXVacua custom imports
from .util import *
from .lcs import lcs_tree



class periods:
    """
    **Description:** 
    Period class for Calabi-Yau threefold compactifications. Provides functions to compute periods, prepotential, Kähler potential, gauge kinetic matrix, and derived objects at various moduli space limits (LCS, coniLCS, etc.).
    """

    def __init__(
                self,
                
                # Model and limit data
                h12: int | None = None,
                model_ID: int | None = None,
                model_type: str = "KS",
                limit: str = "LCS",
                save_file: bool = False,
                
                # CYTools interface
                use_cytools: bool = False,
                mirror_cy: object | None = None,
                
                # Model data input
                lcs_tree_input: object | None = None,
                model_file: str = "",
                model_data: dict | None = None,
                
                # Basis transformation 
                basis_change: Array | None = None,
                
                # Instaton data input
                maximum_degree: int = 0,
                grading_vector: Array | None = None,
                use_gvs: bool = True,
                prange: int = 5,
                
                # Conifold data for coniLCS limits
                ncf: int | None = None,
                conifold_curve: object | None = None,
                conifold_basis: bool | None = True,
                
                # Custom period input
                period_input: Callable | None = None,
                prepotential_input: Callable | None = None,
                
                # Extra inputs 
                **kwargs
                ) -> None:
        r"""

        **Description:**

        Period class.

        This class defines functions of the periods of Calabi-Yau threefolds :math:`X`. They are obtained 
        from integrating the holomorphic 3-form :math:`\Omega` over a symplectic basis of :math:`H_3(X,\mathbb{Z})`.

        Args:
            h12 (int): The number of moduli for the compactified geometry.
            model_type (string): The type of manifold considered for the compactification. Currently, ``"KS"`` and ``"CICY"`` are available.
            model_ID (int): ID specifying a certain model.
            limit (string): String identifying the type of periods to be considered. Currently, only ``"LCS"`` is available.
            model_data (dictionary): Contains model data like triple intersection numbers etc.
            maximum_degree (int): Maximum degree used for the instanton sum.
            use_cytools (boolean): Whether or not to use CYTools to compute topological data of Calabi-Yau.
            mirror_cy (cytools.CalabiYau): Mirror Calabi-Yau threefold.
            basis_change (Array): Basis transformation to be applied to topological data of Calabi-Yau.
            grading_vector (Array): Grading vector to be used for the GV computation.
            lcs_tree_input (lcs_tree | None, optional): Pre-built lcs_tree object. If provided, skips model construction. Defaults to ``None``.
            ncf (int | None, optional): Number of conifolds for the coniLCS limit. Defaults to ``None``.
            conifold_curve (Array | None, optional): Conifold curve charges. Defaults to ``None``.
            prange (int, optional): Truncation order for the polylogarithm series. Defaults to ``500``.
            use_gvs (bool, optional): Whether to use GV invariants instead of GW invariants for the instanton sum. Defaults to ``False``.
            period_input (function): Input for periods.
            prepotential_input (function): Input for prepotential.
            save_file (bool, optional): Save files for new models. Defaults to ``False``.
            **kwargs: Extra inputs.

        Raises:
            ValueError: If model inputs are inconsistent or required data is missing.

        Attributes:
            h12 (int): :math:`h^{1,2}(X,\mathbb{Z})` of the CY :math:`X`.
            model_type (str): Model type. Defaults to ``None``. Currently, ``"KS"`` and ``"CICY"`` are available.
            limit (str): Moduli space limit. Defaults to ``None``. Currently, only ``"LCS"`` is available.
            model_ID (int): Internal model ID. Defaults to ``None``.
            dimension_H3 (int): Dimension of :math:`H^{3}(X,\mathbb{Z})`.
            _dimension_H3_tot (int): Dimension of :math:`H^{3}(X,\mathbb{Z})` plus dimension of :math:`H_{3}(X,\mathbb{Z})`.
            period_input (object): Input period function.
            prepotential_input (object): Input prepotential function.
            

        """
        
        # Testing inputs...
        ## Check correct model types
        model_types=[None,"KS","CICY","hypergeometric"]
        if model_type not in model_types:
            raise ValueError(f"Model type must be one of {model_types}!")
            
        
        # Initialisation of class
        ## Setting general attributes from input
        self.model_type = model_type
        self.limit = limit
        self.model_ID = model_ID
        self.maximum_degree = maximum_degree
        self.prange = prange

        # Recognised moduli-space limits.
        _LCS_FAMILY_LIMITS = ("LCS", "coniLCS", "coniLCS_series", "coniLCS_bulk")
        _NON_LCS_LIMITS    = ("Kpoint", "Cpoint")
        _RECOGNISED_LIMITS = _LCS_FAMILY_LIMITS + _NON_LCS_LIMITS

        if limit is not None and limit not in _RECOGNISED_LIMITS:
            raise ValueError(
                f"Unknown limit {limit!r}; expected one of "
                f"{_RECOGNISED_LIMITS} or None."
            )

        # Hypergeometric model auto-detection: if the user passes a non-LCS
        # limit + a known label, resolve the closed-form prepotential from
        # the registry and tag the model_type. Registry lookup logic lives in
        # HypergeometricModels.resolve_prepotential to keep this constructor
        # simple; periods owns the h12 invariant locally.
        if limit in _NON_LCS_LIMITS:
            if h12 != 1:
                raise ValueError(
                    f"limit={limit!r} is only defined for one-modulus (h12=1) "
                    f"hypergeometric models; got h12={h12}."
                )
            from .hypergeometric_models import HypergeometricModels
            self.model_type = "hypergeometric"
            if prepotential_input is None:
                prepotential_input = HypergeometricModels.resolve_prepotential(
                    model_ID=model_ID, limit=limit
                )
        
        # Set whether to use GV invariants for the instanton sum
        self.use_gvs = use_gvs
        
        # If no instanton corrections are requested, disable GV invariant usage
        if self.maximum_degree==0:
            self.use_gvs=False
            self.include_mirror_wsi = False
        else:
            self.include_mirror_wsi = True
            
        # coniLCS limits need a conifold basis choice; default to the aligned
        # (conifold-at-index-0) basis, but honour an explicit conifold_basis=False
        # so a general (non-aligned) basis can be requested.
        if limit in ["coniLCS_series","coniLCS_bulk","coniLCS"] and conifold_basis is None:
            conifold_basis = True
        
        # Set lcs_tree from input or construct it from provided data or CYTools interface. The lcs_tree object holds all the relevant data of the CY model and is used as input for the period calculations. If no lcs_tree_input is provided, we construct the lcs_tree object from the provided data or using the CYTools interface. If an lcs_tree_input is provided, we use it directly.
        if prepotential_input is not None or period_input is not None:
            if lcs_tree_input is not None:
                self._lcs_tree = lcs_tree_input.__copy__()
                self.limit = lcs_tree_input.limit
            else:
                self._lcs_tree = lcs_tree(intnums=jnp.array([[[0]]]),c2=jnp.array([0]))
            #    self.lcs_tree = {}
            
        else:
            if lcs_tree_input is None:
                
                if use_cytools or mirror_cy is not None:
                    
                    if mirror_cy is None:
                        raise ValueError(f"Need to provide `mirror_cy` if `use_cytools=True`!")
                    
                    time_out = kwargs.get('time_out', 120)
                    compute_gws = kwargs.get('compute_gws', False)
                    
                    if self.maximum_degree>0:
                        if self.use_gvs==False:
                            compute_gws = True
                    
                    out = lcs_tree.from_cytools(cy = mirror_cy,
                                                maximum_degree=maximum_degree,
                                                basis_change=basis_change,
                                                ncf=ncf,
                                                conifold_curve=conifold_curve,
                                                conifold_basis=conifold_basis,
                                                grading_vector=grading_vector,
                                                limit=limit,
                                                save_file=save_file,
                                                model_ID=model_ID,
                                                time_out=time_out,
                                                compute_gws=compute_gws
                                                )
                    
                    self._lcs_tree = out
                else:
                    if model_file == "":
                        if model_data is None:
                            if model_ID is None:
                                raise ValueError(f"Need to provide `lcs_tree_input`, `model_file`, `model_data` or `model_ID` if not using CYTools interface!")

                            # Some global variables
                            home_dir=os.path.dirname(os.path.realpath(__file__))
                            files_dir=home_dir+"/models"

                            file = files_dir+f"/h12_{h12}/model_{model_ID}.p"
                            
                            if not os.path.isfile(file):
                                raise ValueError(f"Could not find file for path {file}!")
                            
                            model_data = load_zipped_pickle(file)
                            

                        if type(model_data) != dict:
                            raise ValueError(f"`model_data` needs to be of type `dict`, but got {type(model_data)}!")
                        
                        model_data["maximum_degree"]=maximum_degree
                        model_data["limit"]=limit
                        model_data["conifold_basis"]=conifold_basis
                        
                        self._lcs_tree = lcs_tree.from_dict(model_data)
                    else:
                        if not os.path.isfile(model_file):
                            raise ValueError(f"Could not find file for path {model_file}!")
                        
                        self._lcs_tree = lcs_tree.from_file(model_file,
                                                            maximum_degree=maximum_degree,
                                                            limit=limit,
                                                            conifold_basis=conifold_basis)
                        
                
                        
            else:
                self._lcs_tree = lcs_tree_input.__copy__()
                self.limit = lcs_tree_input.limit

        # Set h12 from lcs_tree
        if h12 is None:
            self.h12 = self._lcs_tree.h12
        else:
            self.h12 = h12

        # Define the symplectic pairing matrix for the periods. This is a constant matrix that encodes the symplectic structure of the period vector. It is defined as
        self.sigma = self.sigma()
        
        # Set the dimension of the 3rd cohomology group H^3(X) to h12 + 1
        self.dimension_H3 = self.h12 + 1

        # Set the total dimension of H^3 (both A and B cycles) to 2*(h12 + 1)
        self._dimension_H3_tot = 2*(self.h12 + 1)
        
        # Validate the specified moduli space limit
        self.period_input = None
        self._period_input_used = False
        self.prepotential_input = None
        self._prepotential_input_used = False
        
        
        
        LCS_FAMILY_LIMITS = ("LCS", "coniLCS", "coniLCS_series", "coniLCS_bulk")
        NON_LCS_LIMITS    = ("Kpoint", "Cpoint")

        # Custom-input dispatch: K/Cpoint always use it; LCS-family fall back to
        # it only when no lcs_tree was supplied.
        if (limit in NON_LCS_LIMITS) or (limit not in LCS_FAMILY_LIMITS) or (self._lcs_tree is None):
            self._setup_custom_input(period_input, prepotential_input)
            
        # Number of terms in the conifold expansion to be included. Only relevant for the "coniLCS_series" limit.
        # Setting default to 2 in order to get linear expansion at the level of the superpotential.
        self.nmax = kwargs.get("nmax", 2)
                

        # DONE
        # -----------------------------------------------------------------------------------

    
    @property
    def lcs_tree(self):
        r"""
        Description:
        The underlying lcs_tree object holding the CY model data.
        
        Args:
            None
            
        Returns:
            lcs_tree: The lcs_tree object holding the CY model data.
        
        """
        return self._lcs_tree

    @lcs_tree.setter
    def lcs_tree(self, value):
        r"""
        **Description:**
        Sets the underlying lcs_tree object holding the CY model data.
        
        Args:
            value (lcs_tree): The lcs_tree object to set.
        
        """
        
        self._lcs_tree = value.__copy__()
        
        
    def _setup_custom_input(
                            self,
                            period_input: Callable | None,
                            prepotential_input: Callable | None,
                            ) -> None:
        r"""
        **Description:**
        Validate and register a user-supplied period or prepotential function.
        Called during initialisation when ``limit`` is not one of the built-in
        moduli-space limits.

        Checks that the callable returns the expected shape and is consistent
        with complex conjugation, then sets ``self.period_input``,
        ``self.prepotential_input``, and the corresponding ``_used`` flags.

        Args:
            period_input (callable | None): Period function ``Pi(z, conj=False)`` returning an array of shape ``(2*(h12+1),)``.
            prepotential_input (callable | None): Prepotential function ``F(z, conj=False)`` returning a complex scalar.

        Raises:
            ValueError: If neither input is provided, if the output shape is wrong, or if the function is inconsistent with complex conjugation.
        """
        
        # If custom moduli space limit is used, require either period or prepotential input
        if period_input is None and prepotential_input is None:
            raise ValueError("Need to provide input for periods or prepotential.\
                    Currently, only LCS is implemented as moduli space limit!")
        else:
            warnings.warn("Implementation for general input periods not tested!")
        
        # Warn if both prepotential and periods are provided as inputs
        if period_input is not None and prepotential_input is not None:
            warnings.warn("If both periods and prepotential are provided,\
                please make sure input is consistent. \
                Only the provided period vector \
                will be used for the computation!")

        # Validate the provided period input function
        if period_input is not None:
        
            # Generate random test input for validation
            z = np.random.uniform(-1,1,self.h12+1)+1j*np.random.uniform(-1,1,self.h12+1)
            
            # Compute output of period function and check that it has the expected shape of a period vector with dimension 2*(h12+1)
            try:
                period_output = period_input(z)
            except:
                raise ValueError("Failed to evaluate `period_input`. Please check input function!")
        
            # Check that output shape matches expected period vector dimension
            if period_output.shape != ((self.h12+1)*2,):
                raise ValueError(f"Wrong output shape for period.\
                    Output shape is {period_output.shape}, but should be {((self.h12+1)*2,)}")
                
            # Compute complex conjugate of output and check consistency with conjugate input
            try:
                period_output_conj = period_input(jnp.conj(z),conj=True)
            except:
                raise ValueError("Failed to evaluate `period_input` with conjugate input. Please check input function!")
        
            # Check that output shape matches expected period vector dimension
            if period_output_conj.shape != ((self.h12+1)*2,):
                raise ValueError(f"Wrong output shape for period.\
                    Output shape is {period_output_conj.shape}, but should be {((self.h12+1)*2,)}")
                
            # Check that the output of the period function is consistent with complex conjugation
            if not jnp.allclose(jnp.conj(period_output), period_output_conj, atol=1e-12, rtol=1e-12):
                raise ValueError("Output of `period_input` is not consistent with complex conjugation! Please check input function!")
            
            self.period_input = period_input
            self._period_input_used = True
            self.prepotential_input = prepotential_input
            self._prepotential_input_used = False
        else:
            
            
            # Validate the provided prepotential input function
            if prepotential_input is not None:
            
                # Generate random test input for validation
                z = np.random.uniform(-1,1,self.h12+1)+1j*np.random.uniform(-1,1,self.h12+1)
                z = jnp.array(z)

                try:
                    prepotential_output = prepotential_input(z)
                except:
                    raise ValueError("Failed to evaluate `prepotential_input`. Please check input function!")
                    
                # Check that output is a scalar complex number
                if prepotential_output.shape!=():
                    
                    raise ValueError(f"Wrong output shape for prepotential.\
                        Output shape is {prepotential_output.shape}, but should be {()}")
                    
                try:
                    prepotential_output_conj = prepotential_input(jnp.conj(z),conj=True)
                except:
                    raise ValueError("Failed to evaluate `prepotential_input` with conjugate input. Please check input function!")
                    
                # Check that output is a scalar complex number
                if prepotential_output_conj.shape!=():
                    
                    raise ValueError(f"Wrong output shape for prepotential on conjugate input.\
                        Output shape is {prepotential_output_conj.shape}, but should be {()}")
                    
                if not jnp.allclose(jnp.conj(prepotential_output_conj), prepotential_output, atol=1e-12, rtol=1e-12):
                    raise ValueError("Output of `prepotential_input` is not consistent with complex conjugation! Please check input function!")
                    
                self.prepotential_input = prepotential_input
                self._prepotential_input_used = True

    def __repr__(self) -> str:
        r"""
        **Description:**
        String representation of the periods object.

        Returns:
            str: A string summarizing h12 of the model.
        """
        return f"Period calculations for h12={self.h12}."
    
    def sigma(self) -> Array:
        r"""
        **Description:**
        Returns the symplectic matrix
        
        .. math::
            \Sigma=\left (\begin{array}{cc}0 & 1\\ -1&0\end{array}\right )\, .
                
        Args:
            None
        
        Returns:
            Array: Symplectic pairing matrix.
        
        """

        Block1=-jnp.identity(2*(self.h12+1),dtype=jnp.int32)[:self.h12+1]
        Block2=jnp.identity(2*(self.h12+1),dtype=jnp.int32)[self.h12+1:]

        return jnp.concatenate((Block2,Block1))

    def compute_a_shift_monodromy(self, shift: Array) -> Array:
        r"""
        **Description:**
        Symplectic monodromy matrix induced by a shift of the prepotential
        :math:`a`-matrix, :math:`a \to a + S`.

        .. admonition:: Details
            :class: dropdown

            The LCS prepotential carries a quadratic term
            :math:`\tfrac{1}{2}\, a_{ij}\, X^i X^j`, so under :math:`a \to a + S`
            it gains :math:`\tfrac{1}{2}\, S_{ij}\, X^i X^j`.  The dual periods
            :math:`\mathcal{F}_I = \partial_I F` then transform as

            .. math::
                \mathcal{F}_i \to \mathcal{F}_i + S_{ij}\, X^j
                \quad (i,j=1,\ldots,h^{1,2})\,, \qquad
                \mathcal{F}_0 \to \mathcal{F}_0\,,

            while every :math:`X^I` is unchanged (the :math:`\mathcal{F}_0` term
            is invariant because the degree-two piece cancels by Euler's
            theorem).  On the period vector :math:`\Pi = (\mathcal{F}_I,\, X^I)`
            (dual periods first, see :func:`period_vector_per`) this is the
            unipotent transformation

            .. math::
                M(S) = \begin{pmatrix} \mathbb{1} & \widehat{S} \\
                                       0 & \mathbb{1} \end{pmatrix}\,, \qquad
                \widehat{S} = \begin{pmatrix} 0 & 0 \\ 0 & S \end{pmatrix}\,,

            where :math:`\widehat{S}` embeds the :math:`h^{1,2}\times h^{1,2}`
            shift :math:`S` with a zero :math:`X^0` row and column.  :math:`M(S)`
            is symplectic (:math:`M^T \Sigma M = \Sigma`, with :math:`\Sigma`
            from :func:`sigma`) **iff** :math:`S` is symmetric — which every
            :math:`a`-matrix from :func:`jaxvacua.conifold.compute_a_matrix`
            satisfies (:math:`a_{ij} = \kappa_{ijj}/2 = a_{ji}`) — and integer
            iff :math:`S` is integer.

            Periods, fluxes and the full flux vector all transform by the
            **same** :math:`M(S)` (:math:`\Pi \to M\Pi`,
            :math:`\text{flux} \to M\,\text{flux}`), leaving the Kähler
            potential :math:`e^{-K} = i\,\Pi^\dagger \Sigma \Pi` and the
            superpotential :math:`W = (f - \tau h)\cdot \Sigma \cdot \Pi`
            invariant.  This makes :math:`M(S)` the dictionary between two
            :math:`a`-conventions (equivalently two integral flux bases), e.g.
            the ``conifold_basis=True`` and ``conifold_basis=False`` descriptions
            of the same geometry.

        Args:
            shift (Array): Symmetric :math:`(h^{1,2}, h^{1,2})` shift matrix
                :math:`S = a' - a`.

        Returns:
            Array: The :math:`(2(h^{1,2}+1), 2(h^{1,2}+1))` symplectic monodromy
            matrix :math:`M(S)`.

        See also: :func:`sigma`, :func:`period_vector_per`,
            :func:`jaxvacua.conifold.compute_a_matrix`
        """
        S = np.asarray(shift)
        if S.ndim != 2 or S.shape[0] != S.shape[1]:
            raise ValueError(f"a-shift S must be a square matrix; got shape {S.shape}.")
        if S.shape[0] != self.h12:
            raise ValueError(
                f"a-shift S must be ({self.h12}, {self.h12}) to match h12={self.h12}; "
                f"got shape {S.shape}."
            )
        if not np.allclose(S, S.T):
            raise ValueError("a-shift S must be symmetric for M(S) to be symplectic.")

        h = self.h12
        S_jnp = jnp.asarray(S)
        S_hat = jnp.zeros((h + 1, h + 1), dtype=S_jnp.dtype).at[1:, 1:].set(S_jnp)
        identity = jnp.eye(h + 1, dtype=S_jnp.dtype)
        zero = jnp.zeros((h + 1, h + 1), dtype=S_jnp.dtype)
        return jnp.block([[identity, S_hat], [zero, identity]])



    ###################################################################################################################################
    #################################### PREPOTENTIAL ETC. AS FUNCTIONS OF PERIODS FOR LCS ############################################
    ###################################################################################################################################

    @partial(jit, static_argnums = (2,))
    def F_LCS_poly_per(self, XPer: Array, conj: bool = False) -> complex:
        r"""
        **Description:**
        Computes the polynomial part :math:`F_{\mathrm{poly}}` of the LCS prepotential :math:`F_{\mathrm{LCS}}`
        in terms of the periods. 

        .. admonition:: Details
            :class: dropdown

            The polynomial part :math:`F_{\mathrm{poly}}` of the LCS prepotential :math:`F_{\mathrm{LCS}}`
            can be expressed in terms of the periods :math:`X^I=(X^0,X^i)` as
            
            .. math::
                F_{\mathrm{poly}}(X) = -\frac{1}{6X^0}\widetilde{\kappa}_{ijk}X^iX^jX^k+\frac{1}{2}a_{ij}X^iX^j
                    +b_{i}X^i\, X^0 + \dfrac{\text{i}}{2}\tilde{\xi}\, (X^0)^2\, .

            Here, :math:`\widetilde{\kappa}_{ijk}` are the triple intersection numbers of
            the mirror dual Calabi-Yau threefold :math:`\widetilde{X}`.
            Here, we defined

            .. math::
                a_{ij} = \dfrac{1}{2}\begin{cases}
                                        \widetilde{\kappa}_{iij} & i\geq j\\[0.3em]
                                        \widetilde{\kappa}_{ijj} & i<j
                                    \end{cases} \, , \quad 
                b_i = \dfrac{1}{24} \int_{\tilde{D}^i}\, c_2(\widetilde{X})\, , \quad  
                \tilde{\xi}=\frac{\zeta(3)\, \chi(\widetilde{X})}{(2\pi)^3}\, .
                
            The :math:`a_{ij}` are rational numbers, while the :math:`b_i` are integers.
            The :math:`\tilde{D}^i` are the divisors dual to the basis of :math:`H^{1,1}(\widetilde{X},\mathbb{Z})`.
            Finally, :math:`\chi(\widetilde{X})` is the Euler characteristic of :math:`\widetilde{X}`.
            The last term in :math:`F_{\mathrm{poly}}` is the so-called :math:`(\alpha')^3` 1-loop correction
            on the mirror dual side.

        Args:
            XPer (Array): Values of periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.

        Returns:
            complex: Value of the polynomial contribution :math:`F_{\mathrm{poly}}` to 
                the LCS prepotential :math:`F_{\mathrm{LCS}}`.

        See also: :func:`F_LCS_per`

        """
        
        cubic = -jnp.dot(self.lcs_tree.intnums_coo_sym[:,-1],XPer[self.lcs_tree.intnums_coo_sym[:,0]+1]*XPer[self.lcs_tree.intnums_coo_sym[:,1]+1]*XPer[self.lcs_tree.intnums_coo_sym[:,2]+1])/6./XPer[0]
        #quadratic = jnp.einsum('ij,i,j',self.lcs_tree.a_matrix,XPer[1:],XPer[1:])/(2.)
        quadratic = jnp.matmul(jnp.matmul(self.lcs_tree.a_matrix,XPer[1:]),XPer[1:])/(2.)
        # Alternative sparse version, which does not seem to be faster....
        #quadratic = jnp.dot(self.a_matrix_sparse_values,XPer[self.a_matrix_sparse[:,0]+1]*XPer[self.a_matrix_sparse[:,1]+1])/(2.)
        #linear = jnp.einsum('i,i',self.lcs_tree.b_vector,XPer[1:])*XPer[0]
        linear = jnp.matmul(self.lcs_tree.b_vector,XPer[1:])*XPer[0]
        val =  cubic + quadratic + linear
        
        if not conj:
            return val + self.lcs_tree.K0/2.*XPer[0]**(2)
        else:
            return val - self.lcs_tree.K0/2.*XPer[0]**(2)

    @partial(jit, static_argnums = (2,))
    def F_inst_per(self,XPer: Array, conj: bool = False) -> complex:
        r"""
        **Description:**
        Computes the instanton part :math:`F_{\mathrm{inst}}` of the LCS prepotential :math:`F_{\mathrm{LCS}}`
        in terms of the periods.

        .. admonition:: Details
            :class: dropdown

            The instanton part :math:`F_{\mathrm{inst}}` of the LCS prepotential :math:`F_{\mathrm{LCS}}`
            can be expressed in terms of the periods :math:`X^I=(X^0,X^i)` as

            .. math::
                F_{\mathrm{inst}}(X) = -\frac{(X^0)^2}{(2\pi\mathrm{i})^3}\, \sum_{q\in\mathcal{M}(\widetilde{X})}\, 
                n_q^{0}\, \text{Li}_3\left (\text{e}^{2\pi \text{i} q_i X^i / X^0}\right )\; , \quad 
                \text{Li}_3\left (x\right )=\sum_{m=1}^{\infty}\, \dfrac{x^{m}}{m^{3}}\, .
        
            Here the sum is performed over all effective curve classes :math:`q\in\mathcal{M}(\widetilde{X})`
            in the Mori cone :math:`\mathcal{M}(\widetilde{X})` of the mirror dual manifold :math:`\widetilde{X}`.
            Here, the :math:`n_q^{0}` are the genus-0 Gopakumar-Vafa (GV) invariants which can be computed
            systematically using methods described in `hep-th/9308122 <https://arxiv.org/pdf/hep-th/9308122.pdf>`_.
        
            The infinite sum appearing in the poly-logarithm :math:`\text{Li}_3` can be rewritten to arrive at

            .. math::
                \sum_{q\in\mathcal{M}(\widetilde{X})}\, n_q^{0}\, \text{Li}_3\left (\text{e}^{2\pi \text{i} q_i X^i / X^0}\right )
                = \sum_{q\in\mathcal{M}(\widetilde{X})}\, N_q\, \text{e}^{2\pi \text{i} q_i X^i / X^0}

            in terms of genus-0 Gromov-Witten (GW) invariants :math:`N_q`. We typically work with the latter to simplify the calculation.
            The relation between the two types of invariants is more explicitly given by
            
            .. math::
                N_q = \sum_{d|q}\, \dfrac{1}{d^3}\, n_{q/d}^{0}\, .
                
            Here, the sum runs over all divisors :math:`d` of the curve class :math:`q`.
            The :math:`N_q` are typically rational numbers, while the :math:`n_q^{0}` are integers. 
            The curve classes :math:`q` are specified in a basis of the Mori cone :math:`\mathcal{M}(\widetilde{X})`
            of the mirror dual Calabi-Yau threefold :math:`\widetilde{X}`.
            The Mori cone is dual to the Kähler cone of :math:`\widetilde{X}`.
            The curve classes :math:`q` can be expressed in terms of the generators of the Mori cone.
            The curve classes are also referred to as curve charges in the following.
            The curve charges are stored in the attribute ``GW_charges`` (``GV_charges``) for the GW (GV) invariants.
            The corresponding invariants are stored in the attributes ``GW_inv`` (``GV_inv``).
            The curve charges and invariants are limited to a certain maximum degree :math:`d=\text{max_deg}` in the attributes
            ``GW_charges_lim``, ``GV_charges_lim``, ``GW_inv_lim``, ``GV_inv_lim``.
            The maximum degree can be specified when initialising the class.
            The maximum degree is defined with respect to a grading vector which can be specified when initialising
            the class. If no grading vector is provided, the default grading vector :math:`(1,1,\ldots,1)` is used.
            The maximum available degree for the instanton data is stored in the attribute ``max_available_deg``.
            If the specified maximum degree exceeds the maximum available degree, a warning is raised and
            the maximum available degree is used instead.
            
        .. admonition:: Note
            :class: dropdown
            
            The sum over curve classes is truncated at a certain maximum degree :math:`d=\text{max_deg}` in our implementation.
            This is justified since the instanton contributions are exponentially suppressed at large complex structure.
            The maximum degree can be specified when initialising the class.
        

        Args:
            XPer (Array): Values of periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.

        Returns:
            complex: Value of the instanton part :math:`F_{\mathrm{inst}}` of the LCS prepotential :math:`F_{\mathrm{LCS}}`.

        See also: :func:`F_LCS_per`

        """
        
        if self.limit in ["LCS","coniLCS_series","coniLCS_bulk"]:
            approx="inf"
            #approx="patch"
        elif self.limit == "coniLCS":
            approx="patch"
        else:
            approx="inf"
            
        coeff = 2.*Pi*1j
        if conj:
            coeff = - coeff
        
        if self.use_gvs:
            t = jnp.matmul(self.lcs_tree.gv_charges,XPer[1:])
            invs = self.lcs_tree.gv_invariants
        else:
            t = jnp.matmul(self.lcs_tree.gw_charges,XPer[1:])
            invs = self.lcs_tree.gw_invariants
            
        t = t/XPer[0]
        z = jnp.exp(coeff*t)
        
        if self.use_gvs:
            z = jax_polylog_vmap(z,3,self.prange,approx=approx)
            
        sum_wsi = jnp.matmul(invs,z)
        
        return -sum_wsi/(coeff)**(3)*XPer[0]**(2)


    

    @partial(jit, static_argnums = (2,))
    def F_LCS_per(self, XPer: Array, conj: bool = False) -> complex:
        r"""

        **Description:**
        Calculates the value of the LCS prepotential :math:`F_{\text{LCS}}` in terms of the periods :math:`X^I`.

        .. admonition:: Details
            :class: dropdown

            At Large Complex Structure (LCS), the prepotential can be expressed as

            .. math::
                F_{\mathrm{LCS}}(X) = F_{\mathrm{poly}}(X) + F_{\mathrm{inst}}(X)
            
            where

            .. math::
                F_{\mathrm{poly}}(X) = -\frac{1}{6X^0}\widetilde{\kappa}_{ijk}X^iX^jX^k+\frac{1}{2}a_{ij}X^iX^j
                    +b_{i}X^i\, X^0 + \dfrac{\text{i}}{2}\tilde{\xi}\, (X^0)^2\, ,

            and

            .. math::
                F_{\mathrm{inst}}(X) = -\frac{(X^0)^2}{(2\pi\mathrm{i})^3}\, \sum_{q\in\mathcal{M}(\widetilde{X})}\, 
                n_q^{0}\, \text{Li}_3\left (\text{e}^{2\pi \text{i} q_i X^i / X^0}\right )\; , \quad 
                \text{Li}_3\left (x\right )=\sum_{m=1}^{\infty}\, \dfrac{x^{m}}{m^{3}}\, .

            The former is computed via :func:`F_LCS_poly_per`, while the latter in :func:`F_inst_per`.
            See the respective function for more details.
            
        Args:
            XPer (Array): Values of periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.

        Returns:
            complex: Value of the LCS prepotential :math:`F_{\text{LCS}}`.

        See also: :func:`F_LCS_poly_per`

        See also: :func:`F_inst_per`

        """
        if self.include_mirror_wsi:
            return self.F_LCS_poly_per(XPer,conj=conj)+self.F_inst_per(XPer,conj=conj)
        else:
            return self.F_LCS_poly_per(XPer,conj=conj)
        
    
    
    
    ###################################################################################################################################
    ####################################### PREPOTENTIAL ETC. AS FUNCTIONS OF PERIODS #################################################
    ###################################################################################################################################


    @partial(jit, static_argnums = (2,))
    def prepot_per(self, XPer: Array, conj: bool = False) -> complex:
        r"""

        **Description:**
        Computes the prepotential :math:`F` in terms of the periods :math:`X^I`.

        .. admonition:: Details
            :class: dropdown

            At Large Complex Structure (LCS), the prepotential can be computed from mirror symmetry as
            
            .. math::
                F_{\mathrm{LCS}}(X) = F_{\mathrm{poly}}(X) + F_{\mathrm{inst}}(X)
            
            where

            .. math::
                F_{\mathrm{poly}}(X) = -\frac{1}{6X^0}\widetilde{\kappa}_{ijk}X^iX^jX^k+\frac{1}{2}a_{ij}X^iX^j
                    +b_{i}X^i\, X^0 + \dfrac{\text{i}}{2}\tilde{\xi}\, (X^0)^2\, ,

            and

            .. math::
                F_{\mathrm{inst}}(X) = -\frac{(X^0)^2}{(2\pi\mathrm{i})^3}\, \sum_{q\in\mathcal{M}(\widetilde{X})}\, 
                n_q^{0}\, \text{Li}_3\left (\text{e}^{2\pi \text{i} q_i X^i / X^0}\right )\; , \quad 
                \text{Li}_3\left (x\right )=\sum_{m=1}^{\infty}\, \dfrac{x^{m}}{m^{3}}\, .

            The former is computed via :func:`F_LCS_poly_per`, while the latter in :func:`F_inst_per`.
            This limit in moduli space is implemented in :func:`F_LCS_per`.

        .. warning::
            The moduli space limit around which the prepotential is computed is set by the 
            global parameter ``self.periods.limit``. Currently, only the limits ``"LCS"``, ``"coniLCS_series"``, and ``"coniLCS_bulk"``
            are supported.

        .. note::
            This function is used to compute the gauge kinetic matrix :math:`\mathcal{N}_{IJ}`
            of second derivatives of the prepotential which is used to check the ISD-condition for 
            flux vacua.

        Args:
            XPer (Array): Values of periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.

        Returns:
            complex: Value of the prepotential :math:`F`.

        See also: :func:`F_LCS_per`

        See also: :func:`prepot_grad_grad_per`

        See also: :func:`gauge_kinetic_matrix`

        """
        
        if self.limit in ["LCS","coniLCS"]:
            return self.F_LCS_per(XPer,conj=conj)
        elif self.limit == "coniLCS_series":
            return self.F_coniLCS_series_per(XPer,conj=conj)
        elif self.limit == "coniLCS_bulk":
            return self.F_coniLCS_bulk_per(XPer,conj=conj)
        elif self._prepotential_input_used:
            return self.prepotential_input(XPer,conj=conj)
        else:
            raise ValueError("Prepotential undefined! Please specify the limit in moduli space or provide a prepotential or period vector input!")
    
    
    @partial(jit, static_argnums = (2,))
    def prepot_grad_per(self, XPer: Array, conj: bool = False) -> Array:
        r"""

        **Description:**
        Computes the holomorphic derivatives :math:`F_{I}=\partial_{X^I}F` of the prepotential :math:`F`
        with respect to the periods :math:`X^I`.

        Args:
            XPer (Array): Values of periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.

        Returns:
            Array: Holomorphic derivatives :math:`F_{I}=\partial_{X^I}F` of the prepotential :math:`F`.

        """

        #Derivative of prepotential:
        return jax.grad(self.prepot_per,holomorphic=True)(XPer,conj=conj)

    @partial(jit, static_argnums = (2,))
    def prepot_grad_grad_per(self, XPer: Array, conj: bool = False) -> Array:
        r"""

        **Description:**
        Computes the second holomorphic derivatives :math:`F_{IJ}=\partial_{X^I}\partial_{X^J}F` of the prepotential
        :math:`F` with respect to the periods :math:`X^I`.

        .. note::

            This matrix is used among others to compute the gauge kinetic matrix entering the ISD condition
            for SUSY flux vacua, see :func:`gauge_kinetic_matrix_prepotential`.

        Args:
            XPer (Array): Values of periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.

        Returns:
            ddF (jax array of arrays): :math:`(h^{1,2}+1)\times(h^{1,2}+1)`-matrix with entries corresponding to 2nd derivatives of :math:`F`.

        See also: :func:`gauge_kinetic_matrix`.

        """

        #Second derivative of prepotential:
        return jax.jacrev(self.prepot_grad_per,holomorphic=True)(XPer,conj=conj)

    @partial(jit, static_argnums = (2,))
    def period_vector_per(self, XPer: Array, conj: bool = False) -> Array:
        r"""

        **Description:**
        Computes the period vector :math:`\Pi` in terms of the periods :math:`X^I`.

        .. admonition:: Details
            :class: dropdown

            We introduce a symplectic basis of :math:`\{\Sigma_{I},\Sigma^I\} \subset H_3(X,\mathbb{Z})` together with the corresponding 
            Poincaré dual forms :math:`\{\alpha^I,\beta_I\}`. We then define the *periods* by integrating the holomorphic :math:`3`-form 
            :math:`\Omega` over these cycles,
            
            .. math::
                X^I=\int_{\Sigma_{I}}\Omega=\int_{X} \Omega\wedge \alpha^I\, ,\quad \mathcal{F}_I=\int_{\Sigma^I}\Omega=\int_{X} \Omega \wedge \beta_I \, .
            
            We collect the periods :math:`X^I,\mathcal{F}_I` in the period vector :math:`\Pi`, that is,

            .. math::
                \Pi=\left (\begin{array}{c} \mathcal{F}_I\\ X^I \end{array}\right )\, I=0,1,\ldots,h^{1,2}\, .


            The periods :math:`X^I` serve as  homogeneous  complex coordinates on a local patch of the complex structure moduli 
            space of :math:`X`. Away from the locus :math:`X^0=0`, we introduce projective coordinates 

            .. math::
                Z^i =\dfrac{X^i}{X^0}\, , i=1,\ldots,h^{2,1}(X)\, , 

            and normalise :math:`\Omega` such that :math:`X^0=1`. The dual periods :math:`\mathcal{F}_I=\mathcal{F}_I(Z)` are then 
            determined by a prepotential :math:`F(Z)` through
            
            .. math::
                \mathcal{F}_i(Z)=\partial_{Z^i} F \, ,\quad \mathcal{F}_0 =2F-Z^i\partial_{Z^i}F\, .


        Args:
            XPer (Array): Values of periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.

        Returns:
            Array: Period vector :math:`\Pi`.

        """
        
        if self._period_input_used:
            return self.period_input(XPer,conj=conj)
        elif self._prepotential_input_used or self.limit in ["LCS","coniLCS","coniLCS_bulk"]:
            return jnp.concatenate((self.prepot_grad_per(XPer,conj=conj), XPer))
        else:
            raise ValueError("Period vector undefined! If no input was provided, use one of the pre-implemented methods!")

    @partial(jit, static_argnums = ())
    def A_per(self, XPer: Array, cXPer: Array) -> complex:
        r"""

        **Description:**
        Computes the value of the mirror CY volume :math:`\tilde{\mathcal{V}}` as a function of the periods :math:`X^I`.

        .. admonition:: Details
            :class: dropdown

            The mirror CY volume :math:`\tilde{\mathcal{V}}` is computed from the period vector :math:`\Pi` as
            
            .. math::
                \tilde{\mathcal{V}} = -\text{i}\, \Pi^\dagger\cdot \Sigma\cdot\Pi\, .

        Args:
            XPer (Array): Values of periods.
            cXPer (Array): Complex conjugate values of periods.

        Returns:
            Array: Value of the mirror CY volume in terms of the period :math:`X^I`.

        """

        return -1.j*jnp.matmul(self.period_vector_per(cXPer,conj=True), jnp.matmul(self.sigma, self.period_vector_per(XPer)))
    
    @partial(jit, static_argnums = ())
    def kahler_potential_per(self, XPer: Array, cXPer: Array) -> complex:
        r"""

        **Description:**
        Computes the value of the Kähler potential :math:`K` as a function of the periods :math:`X^I`.

        .. admonition:: Details
            :class: dropdown

            The Kähler potential on complex structure moduli space is defined as

            .. math::
                K_{cs} = -\text{ln}(\tilde{\mathcal{V}})      

            in terms of the mirror CY volume :math:`\tilde{\mathcal{V}}`
            
            .. math::
                \tilde{\mathcal{V}} = -\text{i}\, \Pi^\dagger\cdot \Sigma\cdot\Pi\, .


        Args:
            XPer (Array): Values of periods.
            cXPer (Array): Complex conjugate values of periods.

        Returns:
            Array: Value of the Kähler potential :math:`K`.

        """

        return -jnp.log(self.A_per(XPer,cXPer))



    
    @partial(jit, static_argnums = (2,))
    def grad_period_vector_per(self, XPer: Array, conj: bool = False) -> Array:
        r"""

        **Description:**
        Computes derivatives :math:`\partial_{X^I}\Pi` of the period vector :math:`\Pi` with respect to the
        periods :mathh`X^I`.

        Args:
            XPer (Array): Values of periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.

        Returns:
            Array: Derivatives :math:`\partial_{X^I}\Pi` of the period vector :math:`\Pi`.

        """
        
        return jax.jacrev(self.period_vector_per,argnums=0,holomorphic=True)(XPer,conj=conj)
    
    
    
    @partial(jit, static_argnums = (3,))
    def grad_kahler_potential_per(self, XPer: Array, cXPer: Array, conj: bool = False) -> Array:
        r"""

        **Description:**
        Computes derivatives :math:`\partial_{X^I}K` of the Kähler potential :math:`K` with respect to the
        periods :mathh`X^I`.

        .. warning::
            If we set ``conj=True``, we compute the anti-holomorphic derivative 
            :math:`\partial_{\overline{X}^I}K` instead!

        Args:
            XPer (Array): Values of periods.
            cXPer (Array): Complex conjugate values of periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.

        Returns:
            Array: First derivative :math:`\partial_{X^I}K` of the Kähler potential :math:`K`.

        """

        if not conj:
            return jax.grad(self.kahler_potential_per,holomorphic=True,argnums=0)(XPer,cXPer)
        else:
            return jax.grad(self.kahler_potential_per,holomorphic=True,argnums=1)(XPer,cXPer)

    
    @partial(jit, static_argnums = (3,))
    def D_period_vector_per(self, XPer: Array, cXPer: Array, conj: bool = False) -> Array:
        r"""

        **Description:**
        Computes the Kähler convariant derivative :math:`D_I\Pi`. of the period vector :math:`\Pi`.

        Args:
            XPer (Array): Values of periods.
            cXPer (Array): Complex conjugate values of periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
            
        Returns:
            Array: Values of :math:`D_I\Pi`.

        """
        
        dK = self.grad_kahler_potential_per(XPer,cXPer,conj=conj)

        if not conj:
            dPi = self.grad_period_vector_per(XPer,conj=conj)
            return dPi+jnp.outer(self.period_vector_per(XPer,conj=conj),dK)
        else:
            dPi = self.grad_period_vector_per(cXPer,conj=conj)
            return dPi+jnp.outer(self.period_vector_per(cXPer,conj=conj),dK)

    
    
    @partial(jit, static_argnums = (3,))
    def P_per(self, XPer: Array, cXPer: Array, conj: bool = False) -> Array:
        r"""

        **Description:**
        Computes the matrix :math:`P_{IJ}`.

        .. admonition:: Details
            :class: dropdown
            
            Computes the matrix :math:`P_{IJ}`. in terms of the periods :math:`X^I` and :math:`\mathcal{F}_I` as

            .. math::
                P_{IJ} = (\mathcal{F}_{I} ,\, D_{\bar{\imath}}\overline{\mathcal{F}}_I )_{J}\, ,

            see Eq. (3.5) in `2310.06040 <https://arxiv.org/abs/2310.06040>`_.

        Args:
            XPer (Array): Values of periods.
            cXPer (Array): Complex conjugate values of periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
            
        Returns:
            Array: Values of :math:`P_{IJ}`.

        """
        

        # see https://arxiv.org/abs/2310.06040
        if not conj:
            
            PPi = self.period_vector_per(XPer,conj=False)[:self.h12+1]
            DPi = self.D_period_vector_per(XPer,cXPer,conj=True).T[:,:self.h12+1]
        else:

            PPi = self.period_vector_per(cXPer,conj=True)[:self.h12+1]
            DPi = self.D_period_vector_per(XPer,cXPer,conj=False).T[:,:self.h12+1]
        
        return jnp.append(jnp.array([PPi]),DPi[1:],axis=0)
    
    @partial(jit, static_argnums = (3,))
    def Q_per(self, XPer: Array, cXPer: Array, conj: bool = False) -> Array:
        r"""

        **Description:**
        Computes the matrix :math:`Q^{I}\,_{J}`.

        .. admonition:: Details
            :class: dropdown

            Computes the matrix :math:`Q^{I}\,_{J}` in terms of the periods :math:`X^I` and :math:`\mathcal{F}_I` as

            .. math::
                Q^{I}\,_{J} = (X^{I},\, D_{\bar{\jmath}}\overline{X}^I)_{J}\, ,

            see Eq. (3.5) in `2310.06040 <https://arxiv.org/abs/2310.06040>`_.

        Args:
            XPer (Array): Values of periods.
            cXPer (Array): Complex conjugate values of periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
            
        Returns:
            Array: Values of :math:`Q^{I}\,_{J}`.

        """

        # see https://arxiv.org/abs/2310.06040
        # Note 1st and 2nd term are opposite for conjugation!!!
        if not conj:
            
            PPi = self.period_vector_per(XPer,conj=False)[self.h12+1:]
            DPi = self.D_period_vector_per(XPer,cXPer,conj=True).T[:,self.h12+1:]
        else:

            PPi = self.period_vector_per(cXPer,conj=True)[self.h12+1:]
            DPi = self.D_period_vector_per(XPer,cXPer,conj=False).T[:,self.h12+1:]
            
        return jnp.append(jnp.array([PPi]),DPi[1:],axis=0)
    
    @partial(jit, static_argnums = (3,))
    def Q_inv_per(self, XPer: Array, cXPer: Array, conj: bool = False) -> Array:
        r"""

        **Description:**
        Computes the inverse of the matrix :math:`Q^{I}\,_{J}`.

        .. admonition:: Details
            :class: dropdown

            Computes the inverse of the matrix :math:`Q^{I}\,_{J}` defined in terms of the periods :math:`X^I` and :math:`\mathcal{F}_I` as

            .. math::
                Q^{I}\,_{J} = (X^{I},\, D_{\bar{\jmath}}\overline{X}^I)_{J}\, ,

            see Eq. (3.5) in `2310.06040 <https://arxiv.org/abs/2310.06040>`_.

        Args:
            XPer (Array): Values of periods.
            cXPer (Array): Complex conjugate values of periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
            
        Returns:
            Array: Values of :math:`(Q^{-1})^{I}\,_{J}`.

        """

        # see https://arxiv.org/abs/2310.06040
        return jnp.linalg.inv(self.Q_per(XPer,cXPer,conj=conj))

    @partial(jit, static_argnums = (3,))
    def PQ_per(self, XPer: Array, cXPer: Array, conj: bool = False) -> Tuple[Array,Array]:
        r"""

        **Description:**
        Computes the matrices :math:`P_{IJ}` and :math:`Q^{I}\,_{J}`.

        .. admonition:: Details
            :class: dropdown

            Computes the two matrices :math:`P_{IJ}` and :math:`Q^{I}\,_{J}` in terms of the periods :math:`X^I` and :math:`\mathcal{F}_I` as

            .. math::
                P_{IJ} = (\mathcal{F}_{I} ,\, D_{\bar{\imath}}\overline{\mathcal{F}}_I )_{J}
                \, ,\quad Q^{I}\,_{J} = (X^{I},\, D_{\bar{\jmath}}\overline{X}^I)_{J}\, ,

            see Eq. (3.5) in `2310.06040 <https://arxiv.org/abs/2310.06040>`_.

        Args:
            XPer (Array): Values of periods.
            cXPer (Array): Complex conjugate values of periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
            
        Returns:
            Array: Values of :math:`P_{IJ}`.
            Array: Values of :math:`Q^{I}\,_{J}`.

        """
        

        # see https://arxiv.org/abs/2310.06040
        # Note 1st and 2nd term are opposite for conjugation!!!
        if not conj:
            
            PPi = self.period_vector_per(XPer,conj=False)
            DPi = self.D_period_vector_per(XPer,cXPer,conj=True).T
        else:

            PPi = self.period_vector_per(cXPer,conj=True)
            DPi = self.D_period_vector_per(XPer,cXPer,conj=False).T

        
        P = jnp.append(jnp.array([PPi[:self.h12+1]]),DPi[:,:self.h12+1][1:],axis=0)
        Q = jnp.append(jnp.array([PPi[self.h12+1:]]),DPi[:,self.h12+1:][1:],axis=0)

        return P,Q
    
    

    ##########################################################################################
    ############################## GAUGE KINETIC MATRIX ######################################
    ##########################################################################################
    
    @partial(jit, static_argnums = (3,))
    def gauge_kinetic_matrix_prepotential(self, XPer: Array, cXPer: Array, conj: bool = False) -> Array:
        r"""

        **Description:**
        Computes value of the gauge kinetic matrix :math:`\mathcal{N}` from prepotential.

        .. admonition:: Details
            :class: dropdown

            In the presence of a prepotential :math:`F`, the gauge kinetic matrix :math:`\mathcal{N}` can be computed as follows
            (see e.g. Eq. (2.12) in `2110.05511 <https://inspirehep.net/files/14d463478651c33420dee54ab42be7c6>`_)

            .. math::
                \mathcal{N}_{IJ}= \overline{F}_{IJ} + +2\text{i}\, \dfrac{\text{Im}(F_{I L})X^{L} \, 
                                    \text{Im}(F_{J K})X^{K}}{X^{M}\text{Im}(F_{MN})X^{N}}

            where :math:`F_{IJ}=\partial_{X^{I}}\partial_{X^{J}}F` is the second derivative of the prepotential :math:`F`
            with respect to the periods :math:`X^I`.


        Args:
            XPer (Array): Values of periods.
            cXPer (Array): Complex conjugate values of periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.

        Returns:
            Array: Value of the gauge kinetic matrix :math:`\mathcal{N}` from prepotential.

        """

        # Compute matrix of second holomorphic derivatives of the prepotential
        FIJ=self.prepot_grad_grad_per(XPer)

        # Compute the complex conjugate matrix of second anti-holomorphic derivatives of the conjugate prepotential
        cFIJ=self.prepot_grad_grad_per(cXPer,conj=True)

        # Split into conjugated and not conjugated function
        if not conj:

            TFI=jnp.dot(-1j*(FIJ-cFIJ)/2.,XPer)

            return cFIJ+2.*1j*jnp.dot(jnp.array([TFI]).T,jnp.array([TFI]))/jnp.dot(XPer,TFI)
        else:
            
            TFI=jnp.dot(-1j*(FIJ-cFIJ)/2.,cXPer)

            return FIJ-2.*1j*jnp.dot(jnp.array([TFI]).T,jnp.array([TFI]))/jnp.dot(cXPer,TFI)


    @partial(jit, static_argnums = (3,))
    def gauge_kinetic_matrix_periods(self, XPer: Array, cXPer: Array, conj: bool = False) -> Array:
        r"""

        **Description:**
        Computes value of the gauge kinetic matrix :math:`\mathcal{N}` from periods.

        .. admonition:: Details
            :class: dropdown

            The gauge kinetic matrix :math:`\mathcal{N}` is defined in terms of the periods :math:`X^I` and :math:`\mathcal{F}_I` as

            .. math::
                \mathcal{N}_{IJ} = (\mathcal{F}_{I} ,\, D_{\bar{\imath}}\overline{\mathcal{F}}_I )\, \cdot\, (X^{J},\, D_{\bar{\jmath}}\overline{X}^J)^{-1}
                                 = P_{IK} \, (Q^{-1})^{K}\,_{J}

            see e.g. Sect. 2 above Eq. (2.12) in `2110.05511 <https://inspirehep.net/files/14d463478651c33420dee54ab42be7c6>`_.
            Here we defined the two matrices :math:`P_{IJ}` and :math:`Q^{I}\,_{J}` as

            .. math::
                P_{IJ} = (\mathcal{F}_{I} ,\, D_{\bar{\imath}}\overline{\mathcal{F}}_I )_{J}
                \, ,\quad Q^{I}\,_{J} = (X^{I},\, D_{\bar{\jmath}}\overline{X}^I)_{J}\, ,

            see Eq. (3.5) in `2310.06040 <https://arxiv.org/abs/2310.06040>`_.
            

        Args:
            XPer (Array): Values of periods.
            cXPer (Array): Complex conjugate values of periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
            
        Returns:
            Array: Value of the gauge kinetic matrix :math:`\mathcal{N}` from periods.

        """

        P,Q = self.PQ_per(XPer,cXPer,conj=conj)

        #Qinv = jnp.linalg.inv(Q)
        
        #return jnp.matmul(Qinv,P)
        
        return jnp.linalg.solve(Q, P)
    
    
    
    
    @partial(jit, static_argnums = (3,))
    def gauge_kinetic_matrix(self, XPer: Array, cXPer: Array, conj: bool = False) -> Array:
        r"""

        **Description:**
        Computes the value of the gauge kinetic matrix :math:`\mathcal{N}`.

        .. admonition:: Details
            :class: dropdown

            The gauge kinetic matrix :math:`\mathcal{N}` is defined in terms of the periods :math:`X^I` and :math:`\mathcal{F}_I` as

            .. math::
                \mathcal{N}_{IJ} = (\mathcal{F}_{I} ,\, D_{\bar{\imath}}\overline{\mathcal{F}}_I )\, \cdot\, (X^{J},\, D_{\bar{\jmath}}\overline{X}^J)^{-1}
                                 = P_{IK} \, (Q^{-1})^{K}\,_{J}

            see e.g. Sect. 2 above Eq. (2.12) in `2110.05511 <https://inspirehep.net/files/14d463478651c33420dee54ab42be7c6>`_.
            Here we defined the two matrices :math:`P_{IJ}` and :math:`Q^{I}\,_{J}` as

            .. math::
                P_{IJ} = (\mathcal{F}_{I} ,\, D_{\bar{\imath}}\overline{\mathcal{F}}_I )_{J}
                \, ,\quad Q^{I}\,_{J} = (X^{I},\, D_{\bar{\jmath}}\overline{X}^I)_{J}\, .

            In the presence of a prepotential :math:`F`, this object can be computed as follows
            (see e.g. Eq. (2.12) in `2110.05511 <https://inspirehep.net/files/14d463478651c33420dee54ab42be7c6>`_)

            .. math::
                \mathcal{N}_{IJ}= \overline{F}_{IJ} + +2\text{i}\, \dfrac{\text{Im}(F_{I L})X^{L} \, 
                                    \text{Im}(F_{J K})X^{K}}{X^{M}\text{Im}(F_{MN})X^{N}}

            where :math:`F_{IJ}=\partial_{X^{I}}\partial_{X^{J}}F` is the second derivative of the prepotential :math:`F`
            with respect to the periods :math:`X^I`.

            Both formulas for the gauge kinetic matrix :math:`\mathcal{N}` have been implemented. The mode of computation
            depends on the provided input:

            * at LCS, we use the formula from the periods,
            * with input periods, we use the formula from the periods, and
            * otherwise, with input prepotential, we use the formula for the prepotential.


        Args:
            XPer (Array): Values of periods.
            cXPer (Array): Complex conjugate values of periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.

        Returns:
            Array: Value of the gauge kinetic matrix :math:`\mathcal{N}`.

        """

        if self.limit in ["LCS","coniLCS","coniLCS_series","coniLCS_bulk"] or self.period_input is not None:
            return self.gauge_kinetic_matrix_periods(XPer,cXPer,conj=conj)
        elif self.prepotential_input is not None:
            return self.gauge_kinetic_matrix_prepotential(XPer,cXPer,conj=conj)
        else:
            raise ValueError("Could not compute gauge kinetic matrix! Please check input!")

    N = gauge_kinetic_matrix

    @partial(jit, static_argnums = (3,))
    def dN(self, XPer: Array, cXPer: Array, conj: bool = False) -> Array:
        r"""

        **Description:**
        Returns the holomorphic derivative :math:`\partial_{X^I}\mathcal{N}` of the 
        gauge kinetic matrix :math:`\mathcal{N}` with respect to the periods :math:`X^I`.


        Args:
            XPer (Array): JAX array of shape (:math:`h^{1,2}+1`, ) containing the value of the periods.
            cXPer (Array): JAX array of shape (:math:`h^{1,2}+1`, ) containing the complex conjugate value of the periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.

        Returns:
            Array: Holomorphic derivative :math:`\partial_{X^I}\mathcal{N}` of the 
                gauge kinetic matrix :math:`\mathcal{N}`.

        """
        
        return jax.jacrev(self.gauge_kinetic_matrix,holomorphic=True,argnums=0)(XPer,cXPer,conj=conj)

    @partial(jit, static_argnums = (3,))
    def dN_c(self, XPer: Array, cXPer: Array, conj: bool = False) -> Array:
        r"""

        **Description:**
        Returns the anti-holomorphic derivative :math:`\partial_{\overline{X}^I}\mathcal{N}` of the 
        gauge kinetic matrix :math:`\mathcal{N}` with respect to the 
        complex conjugate periods :math:`\overline{X}^I`.

        Args:
            XPer (Array): JAX array of shape (:math:`h^{1,2}+1`, ) containing the value of the periods.
            cXPer (Array): JAX array of shape (:math:`h^{1,2}+1`, ) containing the complex conjugate value of the periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
            
        Returns:
            Array: Anti-holomorphic derivative :math:`\partial_{\overline{X}^I}\mathcal{N}` of the 
                gauge kinetic matrix :math:`\mathcal{N}`.

        """
        
        return jax.jacrev(self.gauge_kinetic_matrix,holomorphic=True,argnums=1)(XPer,cXPer,conj=conj)

    ##########################################################################################
    ###################################### ISD MATRIX ########################################
    ##########################################################################################

    @partial(jit, static_argnums = ())
    def ISD_matrix(self, XPer: Array, cXPer: Array) -> Array:
        r"""

        **Description:**
        Computes the value of the ISD-matrix :math:`\mathcal{M}`.


        .. admonition:: Details
            :class: dropdown

            Computes a matrix representation :math:`\mathcal{M}` for the Hodge-:math:`\star` operator for Calabi-Yau threefolds.
            It can be expressed in terms of the real and imaginary parts of the gauge kinetic matrix
            :math:`\mathcal{N} = \mathcal{R}+\mathrm{i}\mathcal{I}` as

            .. math::
                \mathcal{M} = \left (\begin{array}{cc} -\mathcal{I}^{-1} & \mathcal{I}^{-1}\mathcal{R} \\ 
                                \mathcal{R}\mathcal{I}^{-1} & -\mathcal{I}-\mathcal{R}\mathcal{I}^{-1}\mathcal{R} \end{array}\right )

            see Eq. (2.15) in `2110.05511 <https://inspirehep.net/files/14d463478651c33420dee54ab42be7c6>`_ 
            or alternatively Eq. (3.7) in `2310.06040 <https://arxiv.org/abs/2310.06040>`_.

            This so-called ISD-matrix :math:`\mathcal{M}` possesses the following properties

            .. math::
                \mathcal{M} = \mathcal{M}^T\; ,\quad \mathcal{M}^T\Sigma\mathcal{M} = \Sigma
                \; ,\quad \mathcal{M}^{-1}=\Sigma^T\mathcal{M}\Sigma\, .

            Moreover, it has positive eigenvalues which can be useful to bound the number of vacua
            satisfying the ISD condition :math:`DW=0`, see for example `2310.06040 <https://arxiv.org/abs/2310.06040>`_
            and `2501.03984 <https://arxiv.org/abs/2501.03984>`_.

        .. warning::

            Our convention differs slightly from that of Eq. (3.7) in `2310.06040 <https://arxiv.org/abs/2310.06040>`_
            due to the ordering of periods in the period vector. Specifically, we have

            .. math::
                \mathcal{M}_{\mathrm{here}} = \mathcal{M}_{\mathrm{there}}^{-1}\, .

        Args:
            XPer (Array): Values of periods.
            cXPer (Array): Complex conjugate values of periods.

        Returns:
            Array: Value of the ISD-matrix :math:`\mathcal{M}`.

        Aliases:
            :func:`M`

        """

        # Compute value of gauge kinetic matrix N
        NMatVal=self.gauge_kinetic_matrix(XPer,cXPer)

        # Compute complex conjugate value of gauge kinetic matrix N
        cNMatVal=self.gauge_kinetic_matrix(XPer,cXPer,conj=True)

        # Define real and imaginary part (R and I) of gauge kinetic matrix
        RMat=(NMatVal+cNMatVal)/2.
        IMat=-1j*(NMatVal-cNMatVal)/2.

        # Compute inverse of imaginary part I of gauge kinetic matrix N
        IIMat=jnp.linalg.inv(IMat)

        # Compute matrix product of I^(-1)@R
        IIRMat=jnp.matmul(IIMat,RMat)

        # Construct blocks for ISD matrix M
        Block1=jnp.concatenate((IMat+jnp.matmul(RMat,IIRMat),jnp.matmul(RMat,IIMat)),axis=1)
        Block2=jnp.concatenate((IIRMat,IIMat),axis=1)

        # Combine blocks into ISD matrix
        return (-1.)*jnp.concatenate((Block1,Block2))

    M = ISD_matrix

    @partial(jit, static_argnums = ())
    def dM(self, XPer: Array, cXPer: Array) -> Array:
        r"""

        **Description:**
        Returns the holomorphic derivative :math:`\partial_{X^I}\mathcal{M}` of the 
        ISD-matrix :math:`\mathcal{M}` with respect to the periods :math:`X^I`.

        Args:
            XPer (Array): Values of periods.
            cXPer (Array): Complex conjugate values of periods.
            
        Returns:
            Array: Holomorphic derivative :math:`\partial_{X^I}\mathcal{M}` of the 
                ISD-matrix :math:`\mathcal{M}`.

        """

        return jax.jacrev(self.M,holomorphic=True,argnums=0)(XPer,cXPer)

    @partial(jit, static_argnums = ())
    def dM_c(self, XPer: Array, cXPer: Array) -> Array:
        r"""

        **Description:**
        Returns the anti-holomorphic derivative :math:`\partial_{\overline{X}^I}\mathcal{M}` of the 
        ISD-matrix :math:`\mathcal{M}` with respect to
        the complex conjugate periods :math:`\overline{X}^I`.

        Args:
            XPer (Array): Values of periods.
            cXPer (Array): Complex conjugate values of periods.
            
        Returns:
            Array: Anti-holomorphic derivative :math:`\partial_{\overline{X}^I}\mathcal{M}` of the 
                ISD-matrix :math:`\mathcal{M}`.

        """

        return jax.jacrev(self.M,holomorphic=True,argnums=1)(XPer,cXPer)

    
"""
from .conifold_utils import F_coniLCS_series_per,F_coniLCS_exp_per, F_inst_per_coni,F_coniLCS_poly_split_per,dF_coniLCS_poly_per,ddF_coniLCS_poly_per,dddF_coniLCS_poly_per, ddddF_coniLCS_poly_per,F_coniLCS_bulk_per,F_coni_per,delete_coni_index

periods.F_coniLCS_series_per = F_coniLCS_series_per
periods.F_coniLCS_exp_per = F_coniLCS_exp_per
periods.F_coni_per = F_coni_per 
periods.F_inst_per_coni = F_inst_per_coni 
periods.F_coniLCS_poly_split_per = F_coniLCS_poly_split_per 
periods.dF_coniLCS_poly_per = dF_coniLCS_poly_per 
periods.ddF_coniLCS_poly_per = ddF_coniLCS_poly_per 
periods.dddF_coniLCS_poly_per = dddF_coniLCS_poly_per 
periods.ddddF_coniLCS_poly_per = ddddF_coniLCS_poly_per 
periods.F_coniLCS_bulk_per = F_coniLCS_bulk_per 
periods.delete_coni_index = delete_coni_index
"""

# Conifold methods are attached via the ``_ConifoldGated`` descriptor so they
# are only surfaced when ``self.limit ∈ {coniLCS, coniLCS_series, coniLCS_bulk}``.
# The list of method names lives in :mod:`jaxvacua.conifold` (single source of
# truth — append there to add a new attached method, no edit needed here).
from jaxvacua import conifold as _cf

for _name in _cf._PERIODS_METHODS:
    setattr(periods, _name, _cf._ConifoldGated(getattr(_cf, _name)))

unflatten_func = lambda aux_data, children: unflatten_func_class(aux_data, children, periods)

register_pytree_node(periods, flatten_func, unflatten_func)
