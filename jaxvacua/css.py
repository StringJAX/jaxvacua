# ==============================================================================
# This code is written by Andreas Schachner. Without the author's permission, 
# this code must not be shared with anyone else or used for any other projects 
# than those involving the author directly.
#
# If any questions arise, please feel free to reach out to me (Andreas) either at
# andreas.schachner@gmx.net or at as3475@cornell.edu or at a.schachner@lmu.de.
# ==============================================================================
#
# ------------------------------------------------------------------------------
# This file holds functions to construct complex structure moduli sector of
# Type IIB compactifications.
# ------------------------------------------------------------------------------


# Important standard libraries
import os, sys, warnings
import numpy as np
from functools import partial

# Important JAX libraries
import jax
from jax import jit, vmap, config
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import zeta
from jax.numpy import pi as Pi
from jax.tree_util import register_pytree_node


# Enable 64 bit precision
#Polylog imports
from jaxpolylog import jax_polylog_vmap

# JAXVacua custom imports
from .util import *
from .utils_jaxvacua import flatten_func, unflatten_func_class
from .periods import periods




class css:


    def __init__(self, 
                 h12: int | None = None,
                 model_ID: int | str | None = None, 
                 model_type: str = "KS",
                 limit: str = "LCS",
                 maximum_degree: int = 0,
                 mirror_cy: Optional["cytools.CalabiYau"] = None,
                 model_data: dict | None = None,
                 lcs_tree_input: object | None = None,
                 model_file: str = "",
                 use_cytools: bool = False,
                 basis_change: Array | None = None,
                 ncf: int | None = None,
                 conifold_curve: Array | None = None,
                 grading_vector: Array | None = None, 
                 period_input: Callable | None = None, 
                 prepotential_input: Callable | None = None,
                 gauge_choice: complex = 1.+0.*1j, 
                 prange: int = 500, 
                 use_gvs: bool = False,
                 save_file: bool = False, 
                 **kwargs) -> None:
        r"""
        **Description:**
        This class defines the complex structure sector in Type IIB orientifold compactifications.
        It contains a class object of type :class:`jaxvacua.periods.periods` to compute periods, the prepotential,
        the gauge kinetic matrix etc. The moduli space limit (LCS, coniLCS, etc.) is selected via the ``limit`` parameter.
        
        
        Args:
            h12 (int): The number of moduli for the compactified geometry.
            model_type (string): The type of manifold considered for the compactification. Currently, ``"KS"`` and ``"CICY"`` are available.
            model_ID (int): ID specifying a certain model.
            limit (string): String identifying the type of periods to be considered. Currently, only ``"LCS"`` is available.
            model_data (dictionary): Contains model data like triple intersection numbers etc.
            instanton_data (list): List of GV and GW invariants.
            maximum_degree (int): Maximum degree used for the instanton sum.
            use_cytools (boolean): Whether or not to use CYTools to compute topological data of Calabi-Yau.
            mirror_cy (cytools.CalabiYau): Mirror Calabi-Yau threefold.
            basis_change (Array): Basis transformation to be applied to topological data of Calabi-Yau.
            grading_vector (Array): Grading vector to be used for the GV computation.
            period_input (function): Input for periods.
            prepotential_input (function): Input for prepotential.
            save_file (bool, optional): Save files for new models. Defaults to ``False``.
            **kwargs: Extra inputs.

        Attributes:
            dimension_H3 (int): Dimension of :math:`H^{3}(X,\mathbb{Z})`.
            _dimension_H3_tot (int): Dimension of :math:`H^{3}(X,\mathbb{Z})` plus dimension of :math:`H_{3}(X,\mathbb{Z})`.
            gauge_choice (complex): Choice of gauge for projective coordinates.
            gauge_choice_conj (complex): Choice of gauge for conjugate projective coordinates.
            h12 (int): Hodge number h12 of the Calabi-Yau setting the number of complex structure moduli.
            
        
        """
        
        # Initialise periods class object, which contains all information about the periods, the prepotential, the gauge kinetic matrix etc. for the complex structure sector.
        self.periods = periods(h12=h12,
                               model_ID = model_ID,
                               model_type = model_type,
                               limit = limit,
                               maximum_degree = maximum_degree,
                               mirror_cy = mirror_cy,
                               model_data = model_data,
                               lcs_tree_input = lcs_tree_input,
                               model_file = model_file,
                               use_cytools = use_cytools,
                               basis_change = basis_change,
                               conifold_curve=conifold_curve,
                               ncf=ncf,
                               grading_vector = grading_vector, 
                               period_input = period_input, 
                               prepotential_input = prepotential_input,
                               prange = prange,
                               use_gvs = use_gvs,
                               save_file=save_file, 
                               **kwargs)

        
        # Set class attributes
        #self._lcs_tree = self.periods.lcs_tree  # Removed: duplicate pytree leaf (lcs_tree already in periods)
        
        # Set gauge choice for projective coordinates
        self.gauge_choice = gauge_choice
        self.gauge_choice_conj = jnp.conj(gauge_choice)
        
        # Set h12 and dimension of H3, which are needed for the gauge fixing functions etc.
        self.h12 = self.periods.h12
        self.dimension_H3 = self.periods.dimension_H3
        self._dimension_H3_tot = self.periods._dimension_H3_tot
        
    @property
    def lcs_tree(self):
        r"""
        Description:
        The :class:`~jaxvacua.lcs.lcs_tree` object containing the topological data
        (intersection numbers, second Chern class, GV invariants, etc.) for the
        underlying Calabi-Yau geometry.

        Returns:
            lcs_tree: The topological data tree.
        """
        return self.periods._lcs_tree

    @lcs_tree.setter
    def lcs_tree(self, value):
        r"""
        Description:
        Sets the :class:`~jaxvacua.lcs.lcs_tree` object. A copy of the input is stored
        to avoid unintended mutation of the original object.

        Args:
            value (lcs_tree): The new topological data tree.
        """
        cvalue = value.__copy__()
        self.periods._lcs_tree = cvalue
        
        
    def __repr__(self) -> str:
        r"""
        **Description:**
        Class object description.
        
        Returns:
            str: Class object description.
        """
        return f"Complex structure sector for h12={self.h12} complex structure moduli in {self.periods.limit}-limit."
    
    ###################################################################################################################################
    ####################################### Prepotential LCS #############################################
    ###################################################################################################################################
    
    @partial(jit, static_argnums = (2,))
    def F_LCS_poly(self, 
                moduli: Array, 
                conj: bool = False
                ) -> complex:
        r"""
        
        **Description:**
        Computes the polynomial contribution :math:`F_{\mathrm{poly}}` to the LCS prepotential 
        :math:`F_{\mathrm{LCS}}` in terms of the complex structure moduli :math:`z^i`.

        .. admonition:: Details
            :class: dropdown

            At Large Complex Structure (LCS), the polynomial part :math:`F_{\mathrm{poly}}`
            of the prepotential :math:`F_{\mathrm{LCS}}` is given by

            .. math::
                F_{\text{poly}}(Z)=\dfrac{1}{6}\widetilde{\kappa}_{ijk}\, z^i z^j z^k
                    +\dfrac{1}{2} a_{ij}z^i z^j+b_i\, z^i
                    +\dfrac{\text{i}}{2}\tilde{\xi}\, .

            Here, :math:`\widetilde{\kappa}_{ijk}` are the triple intersection numbers of
            the mirror dual Calabi-Yau threefold :math:`\widetilde{X}`.
            Further, we defined

            .. math::
                a_{ij} = \dfrac{1}{2}\begin{cases}
                                        \widetilde{\kappa}_{iij} & i\geq j\\[0.3em]
                                        \widetilde{\kappa}_{ijj} & i<j
                                    \end{cases} \, , \quad 
                b_i = \dfrac{1}{24} \int_{\tilde{D}^i}\, c_2(\widetilde{X})\, , \quad  
                \tilde{\xi}=\frac{\zeta(3)\, \chi(\widetilde{X})}{(2\pi)^3}\, .
        
        Args:
            moduli (Array): Complex structure moduli values.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            complex: Value of the polynomial contribution :math:`F_{\mathrm{poly}}` to 
                the LCS prepotential :math:`F_{\mathrm{LCS}}`.
            
        See also: :func:`F_LCS`
        
        """

        val = -jnp.dot(self.lcs_tree.intnums_coo_sym[:,-1],moduli[self.lcs_tree.intnums_coo_sym[:,0]]*moduli[self.lcs_tree.intnums_coo_sym[:,1]]*moduli[self.lcs_tree.intnums_coo_sym[:,2]])/6. 
        #val += jnp.einsum('ij,i,j',self.lcs_tree.a_matrix,moduli,moduli)/(2.) 
        val += jnp.matmul(jnp.matmul(self.lcs_tree.a_matrix,moduli),moduli)/(2.) 
        # Alternative sparse version, which does not seem to be faster....
        #val += jnp.dot(self.periods.a_matrix_sparse_values,moduli[self.periods.a_matrix_sparse[:,0]]*moduli[self.periods.a_matrix_sparse[:,1]])/(2.) 
        #val += jnp.einsum('i,i',self.lcs_tree.b_vector,moduli)
        val += jnp.dot(self.lcs_tree.b_vector,moduli)
        
        # Notice that we use K_0 = i * \tilde{\xi} in the code
        if not conj:
            return  val + self.lcs_tree.K0/2.
        else:
            return  val - self.lcs_tree.K0/2.
        

    @partial(jit, static_argnums = (2,))
    def F_inst(self,
            moduli: Array, 
            conj: bool = False
            ) -> complex:
        r"""
        
        **Description:**
        Returns the instanton part :math:`F_{\mathrm{inst}}` of the LCS prepotential :math:`F_{\mathrm{LCS}}`
        in terms of the complex structure moduli :math:`z^i`.
        
        .. admonition:: Details
            :class: dropdown
        
            The sum over world-sheet instantons in the Type IIA language is given by
        
            .. math::
                F_{\text{inst}}(z^1,\ldots , z^{h^{1,2}})=-\frac{1}{(2\pi\mathrm{i})^3}\, \sum_{q\in\mathcal{M}(\widetilde{X})}\, 
                n_q^{0}\, \text{Li}_3\left (\text{e}^{2\pi \text{i} q_i z^i}\right )\; , \quad 
                \text{Li}_3\left (\right )=\sum_{m=1}^{\infty}\, \dfrac{x^{m}}{m^{3}}
        
            where the sum is performed over all effective curve classes :math:`q\in\mathcal{M}(\widetilde{X})`
            in the Mori cone :math:`\mathcal{M}(\widetilde{X})` of the mirror dual manifold :math:`\widetilde{X}`.
            Here, the :math:`n_q^{0}` are the genus-0 Gopakumar-Vafa (GV) invariants which can be computed
            systematically using methods described in `hep-th/9308122 <https://arxiv.org/pdf/hep-th/9308122.pdf>`_.
        
            The infinite sum appearing in the poly-logarithm :math:`\text{Li}_3` can be rewritten to arrive at
        
            .. math::
                \sum_{q\in\mathcal{M}(\widetilde{X})}\, n_q^{0}\, \text{Li}_3\left (\text{e}^{2\pi \text{i} q_i z^i}\right ) = \sum_{q\in\mathcal{M}(\widetilde{X})}\, N_q\, \text{e}^{2\pi \text{i} q_i z^i}
        
            in terms of genus-0 Gromov-Witten (GW) invariants :math:`N_q`. We typically work with the latter to simplify the calculation.
        
        
        
        
        Args:
            moduli (Array): Complex structure moduli values.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            complex: Value of the instanton part :math:`F_{\mathrm{inst}}` of the LCS prepotential 
                :math:`F_{\mathrm{LCS}}`.
            
        See also: :func:`F_LCS`
    
        See also: :func:`F_LCS_poly`
        
        """
        if self.periods.limit in ["LCS","coniLCS_series","coniLCS_bulk"]:
            approx="inf"
        elif self.periods.limit == "coniLCS":
            approx="patch"
        else:
            approx="inf"
            
        coeff = 2.*Pi*1j
        if conj:
            coeff = - coeff
            
        
        if self.periods.use_gvs:
            t = jnp.einsum("ki,i",self.lcs_tree.gv_charges,moduli)
            invs = self.lcs_tree.gv_invariants
            
        else:
            t = jnp.einsum("ki,i",self.lcs_tree.gw_charges,moduli)
            invs = self.lcs_tree.gw_invariants
            
        z = jnp.exp(coeff*t)
        
        if self.periods.use_gvs:
            z = jax_polylog_vmap(z,3,self.periods.prange,approx=approx)
            
        sum_wsi = jnp.sum(invs*z)
        
        return -sum_wsi/(coeff)**(3)
    
    
    @partial(jit, static_argnums = (2,))
    def F_LCS(self, 
            moduli: Array, 
            conj: bool = False
            ) -> complex:
        r"""
        
        **Description:**
        Calculates the value of the LCS prepotential in terms of the complex structure moduli :math:`z^{i}`.
        
        
        .. admonition:: Details
            :class: dropdown
        
            At LCS, we can write the prepotential as
        
            .. math::
                F_{\text{LCS}}(z^1,\ldots , z^{h^{1,2}})=F_{\text{poly}}(z^1,\ldots , z^{h^{1,2}}) + F_{\text{inst}}(z^1,\ldots , z^{h^{1,2}})
        
            see e.g. Eq. (2.13) in `1312.0014 <https://arxiv.org/pdf/1312.0014.pdf>`_ for an equivalent formula. Here, the polynomial piece is given by
        
            .. math::
                F_{\text{poly}}(z^1,\ldots , z^{h^{1,2}})=\dfrac{1}{6}\kappa_{ijk}\, z^i z^j z^k+\dfrac{1}{2} a_{ij}z^i z^j+b_i\, z^i+\dfrac{\text{i}}{2}\tilde{\xi}\, ,
        
            Here, :math:`\widetilde{\kappa}_{ijk}` are the triple intersection numbers of
            the mirror dual Calabi-Yau threefold :math:`\widetilde{X}`.
            Further, we defined

            .. math::
                a_{ij} = \dfrac{1}{2}\begin{cases}
                                        \widetilde{\kappa}_{iij} & i\geq j\\[0.3em]
                                        \widetilde{\kappa}_{ijj} & i<j
                                    \end{cases} \, , \quad 
                b_i = \dfrac{1}{24} \int_{\tilde{D}^i}\, c_2(\widetilde{X})\, , \quad  
                \tilde{\xi}=\frac{\zeta(3)\, \chi(\widetilde{X})}{(2\pi)^3}\, .

            The instanton contributions read

            .. math::
                F_{\mathrm{inst}}(z) = -\frac{1}{(2\pi\mathrm{i})^3}\, \sum_{q\in\mathcal{M}(\widetilde{X})}\, 
                n_q^{0}\, \text{Li}_3\left (\text{e}^{2\pi \text{i} q_i z^i}\right )\; , \quad 
                \text{Li}_3\left (x\right )=\sum_{m=1}^{\infty}\, \dfrac{x^{m}}{m^{3}}\, .
        
            in terms of genus-0 Gopakumar-Vafa (GV) invariants :math:`n_q^0` and the 3rd polylogarithm :math:`\text{Li}_3(x)`. 
            Alternatively, the instanton contributions can be expressed as
            
            .. math::
                F_{\text{inst}}(z^1,\ldots , z^{h^{1,2}})=\sum_{q\in\mathcal{M}(\widetilde{X})}\, N_q\, \text{e}^{2\pi \text{i} q_i z^i}
        
            in terms of genus-0 Gromov-Witten (GW) invariants :math:`N_q`. 
            Both pieces are computed in separate functions, see :func:`F_LCS_poly` and :func:`F_inst` for details.

        
        Args:
            moduli (Array): Complex structure moduli values.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            complex: Value of the LCS prepotential :math:`F_{\text{LCS}}`.
            
        See also: :func:`F_LCS_poly`, :func:`F_inst`

        """

        if self.periods.include_mirror_wsi>0:
            return self.F_LCS_poly(moduli,conj=conj)+self.F_inst(moduli,conj=conj)
        else:
            return self.F_LCS_poly(moduli,conj=conj)
    
    ###################################################################################################################################
    ####################################### GAUGE FIXING FUCTIONS FROM PERIOD CALCULATION #############################################
    ###################################################################################################################################
        
    @partial(jit, static_argnums = (2,))
    def moduli_to_periods(
                        self, 
                        moduli: Array, 
                        conj: bool = False
                        ) -> Array:
        r"""
        **Description:**
        Transforms complex structure moduli to periods for the global choice of gauge.
        
        Args:
            moduli (Array): Values of the complex structure moduli.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Value of the periods.

        """

        val = jnp.concatenate((jnp.ones(1),moduli))

        if conj:
            return val*self.gauge_choice_conj
        else:
            return val*self.gauge_choice


    @partial(jit, static_argnums = ())
    def periods_to_moduli(
                        self, 
                        XPer: Array
                        ) -> Array:
        r"""

        **Description:**
        Transforms periods to complex structure moduli.
        
        Args:
            XPer (Array): Values of the periods.
        
        Returns:
            Array: Value of the complex structure moduli.

        """

        return XPer[1:]/XPer[0]

        
    
    @partial(jit, static_argnums = (3,))
    def gauge_kinetic_matrix(
                            self, 
                            moduli: Array, 
                            moduli_c: Array, 
                            conj: bool = False
                            ) -> Array:
        r"""

        **Description:**
        Computes the value of the gauge kinetic matrix :math:`\mathcal{N}`.

        .. note::

            This function descends from :func:`jaxvacua.periods.periods.gauge_kinetic_matrix`
            upon gauge fixing, i.e., making a choice of  homogeneous complex coordinates 
            on the complex structure moduli space of :math:`X`.
        
        Args:
            moduli (Array): Values of the complex structure moduli.
            moduli_c (Array): Complex conjugate values of the complex structure moduli.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Value of the gauge kinetic matrix :math:`\mathcal{N}`.

        Aliases:
            :func:`N`

        See also: :func:`jaxvacua.periods.periods.gauge_kinetic_matrix`

        """

        return self.periods.gauge_kinetic_matrix(self.moduli_to_periods(moduli),self.moduli_to_periods(moduli_c,conj=True),conj=conj)
    
    N = gauge_kinetic_matrix
    
    @partial(jit, static_argnums = ())
    def ISD_matrix(
                self, 
                moduli: Array, 
                moduli_c: Array
                ) -> Array:
        r"""

        **Description:**
        Computes the value of the ISD-matrix :math:`\mathcal{M}`.

        .. note::

            This function descends from :func:`jaxvacua.periods.periods.ISD_matrix`
            upon gauge fixing, i.e., making a choice of  homogeneous complex coordinates 
            on the complex structure moduli space of :math:`X`.
        
        Args:
            moduli (Array): Values of the complex structure moduli.
            moduli_c (Array): Complex conjugate values of the complex structure moduli.
        
        Returns:
            Array: Value of the ISD-matrix :math:`\mathcal{M}`.

        Aliases:
            :func:`M`

        See also: :func:`jaxvacua.periods.periods.ISD_matrix`

        """
        return self.periods.ISD_matrix(self.moduli_to_periods(moduli),self.moduli_to_periods(moduli_c,conj=True))
    
    M = ISD_matrix

    @partial(jit, static_argnums = (3,))
    def dN_X(
            self, 
            moduli: Array, 
            moduli_c: Array, 
            conj: bool = False
            ) -> Array:
        r"""

        **Description:**
        Returns the holomorphic derivative :math:`\partial_{X^I}\mathcal{N}` of the 
        gauge kinetic matrix :math:`\mathcal{N}` with respect to the periods :math:`X^I`.

        .. note::

            This function descends from :func:`jaxvacua.periods.periods.dN`
            upon gauge fixing, i.e., making a choice of  homogeneous complex coordinates 
            on the complex structure moduli space of :math:`X`.
        
        Args:
            moduli (Array): Values of the complex structure moduli.
            moduli_c (Array): Complex conjugate values of the complex structure moduli.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Holomorphic derivative :math:`\partial_{X^I}\mathcal{N}` of the 
                gauge kinetic matrix :math:`\mathcal{N}`.

        """

        return self.periods.dN(self.moduli_to_periods(moduli),self.moduli_to_periods(moduli_c,conj=True),conj=conj)
    
    @partial(jit, static_argnums = (3,))
    def dN(
            self, 
            moduli: Array, 
            moduli_c: Array, 
            conj: bool = False
            ) -> Array:
        r"""

        **Description:**
        Returns the holomorphic derivative :math:`\partial_{z^i}\mathcal{N}` of the 
        gauge kinetic matrix :math:`\mathcal{N}` with respect to the 
        complex structure moduli :math:`z^i`.
        
        
        Args:
            moduli (Array): Values of the complex structure moduli.
            moduli_c (Array): Complex conjugate values of the complex structure moduli.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Holomorphic derivative :math:`\partial_{z^i}\mathcal{N}` of the 
                gauge kinetic matrix :math:`\mathcal{N}`.

        """

        return self.dN_X(moduli,moduli_c,conj=conj)[:,:,1:]


    @partial(jit, static_argnums = (3,))
    def dN_cX(
            self, 
            moduli: Array, 
            moduli_c: Array, 
            conj: bool = False
            ) -> Array:
        r"""

        **Description:**
        Returns the anti-holomorphic derivative :math:`\partial_{\overline{X}^I}\mathcal{N}` of the 
        gauge kinetic matrix :math:`\mathcal{N}` with respect to the 
        complex conjugate periods :math:`\overline{X}^I`.

        .. note::

            This function descends from :func:`jaxvacua.periods.periods.dN_c`
            upon gauge fixing, i.e., making a choice of  homogeneous complex coordinates 
            on the complex structure moduli space of :math:`X`.
        
        Args:
            moduli (Array): Values of the complex structure moduli.
            moduli_c (Array): Complex conjugate values of the complex structure moduli.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Anti-holomorphic derivative :math:`\partial_{\overline{X}^I}\mathcal{N}` of the 
                gauge kinetic matrix :math:`\mathcal{N}`.

        """

        return self.periods.dN_c(self.moduli_to_periods(moduli),self.moduli_to_periods(moduli_c,conj=True),conj=conj)
    
    @partial(jit, static_argnums = (3,))
    def dN_c(
            self, 
            moduli: Array, 
            moduli_c: Array, 
            conj: bool = False
            ) -> Array:
        r"""

        **Description:**
        Returns the anti-holomorphic derivative :math:`\partial_{\overline{z}^i}\mathcal{N}` of the 
        gauge kinetic matrix :math:`\mathcal{N}` with respect to the complex conjugate 
        complex structure moduli :math:`\overline{z}^i`.
        
        Args:
            moduli (Array): Values of the complex structure moduli.
            moduli_c (Array): Complex conjugate values of the complex structure moduli.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Anti-holomorphic derivative :math:`\partial_{\overline{z}^i}\mathcal{N}` of the 
                gauge kinetic matrix :math:`\mathcal{N}`.

        """

        return self.dN_cX(moduli,moduli_c,conj=conj)[:,:,1:]


    @partial(jit, static_argnums = ())
    def dM_X(
            self, 
            moduli: Array, 
            moduli_c: Array
            ) -> Array:
        r"""

        **Description:**
        Returns the holomorphic derivative :math:`\partial_{X^I}\mathcal{M}` of the 
        ISD-matrix :math:`\mathcal{M}` with respect to the periods :math:`X^I`.

        .. note::

            This function descends from :func:`jaxvacua.periods.periods.dM`
            upon gauge fixing, i.e., making a choice of  homogeneous complex coordinates 
            on the complex structure moduli space of :math:`X`.
        
        Args:
            moduli (Array): Values of the complex structure moduli.
            moduli_c (Array): Complex conjugate values of the complex structure moduli.

        Returns:
            Array: Holomorphic derivative :math:`\partial_{X^I}\mathcal{M}` of the 
                ISD-matrix :math:`\mathcal{M}`.

        """

        return self.periods.dM(self.moduli_to_periods(moduli),self.moduli_to_periods(moduli_c,conj=True))

    @partial(jit, static_argnums = ())
    def dM_cX(
            self, 
            moduli: Array, 
            moduli_c: Array
            ) -> Array:
        r"""

        **Description:**
        Returns the anti-holomorphic derivative :math:`\partial_{\overline{X}^I}\mathcal{M}` of the 
        ISD-matrix :math:`\mathcal{M}` with respect to
        the complex conjugate periods :math:`\overline{X}^I`.

        .. note::

            This function descends from :func:`jaxvacua.periods.periods.dM_c`
            upon gauge fixing, i.e., making a choice of  homogeneous complex coordinates 
            on the complex structure moduli space of :math:`X`.
        
        Args:
            moduli (Array): Values of the complex structure moduli.
            moduli_c (Array): Complex conjugate values of the complex structure moduli.

        Returns:
            Array: Anti-holomorphic derivative :math:`\partial_{\overline{X}^I}\mathcal{M}` of the 
                ISD-matrix :math:`\mathcal{M}`.

        """
        
        return self.periods.dM_c(self.moduli_to_periods(moduli),self.moduli_to_periods(moduli_c,conj=True))
    

    @partial(jit, static_argnums = ())
    def dM(
            self, 
            moduli: Array, 
            moduli_c: Array
            ) -> Array:
        r"""

        **Description:**
        Returns the holomorphic derivative :math:`\partial_{z^i}\mathcal{M}` of the 
        ISD-matrix :math:`\mathcal{M}` with respect to
        the complex structure moduli :math:`z^i`.
        
        Args:
            moduli (Array): Values of the complex structure moduli.
            moduli_c (Array): Complex conjugate values of the complex structure moduli.

        Returns:
            Array: Holomorphic derivative :math:`\partial_{z^i}\mathcal{M}` of the 
                ISD-matrix :math:`\mathcal{M}`.
        """
        
        return self.dM_X(moduli,moduli_c)[:,:,1:]

    @partial(jit, static_argnums = ())
    def dM_c(
            self, 
            moduli: Array, 
            moduli_c: Array
            ) -> Array:
        r"""

        **Description:**
        Returns the anti-holomorphic derivative :math:`\partial_{\overline{z}^i}\mathcal{M}` of the 
        ISD-matrix :math:`\mathcal{M}` with respect to
        the complex conjugate complex structure moduli :math:`\overline{z}^i`.
        
        Args:
            moduli (Array): Values of the complex structure moduli.
            moduli_c (Array): Complex conjugate values of the complex structure moduli.

        Returns:
            Array: Anti-holomorphic derivative :math:`\partial_{\overline{z}^i}\mathcal{M}` of the 
                ISD-matrix :math:`\mathcal{M}`.

        """
        
        return self.dM_cX(moduli,moduli_c)[:,:,1:]

    ###################################################################################################################################
    ########################################### PREPOTENTIAL, KÄHLER POTENTIAL ETC. ###################################################
    ###################################################################################################################################
    
    @partial(jit, static_argnums = (2,))
    def prepot(
            self, 
            moduli: Array, 
            conj: bool = False
            ) -> complex:
        r"""
        
        **Description:**
        Computes the pre-potential for given values of the moduli.
        
        .. note::

            We return the value of the pre-potential in terms projective coordinates
        
            .. math::
                z^{i}=\frac{X^i}{X^0}\, .
        
            Per default, we work in the gauge choice :math:`X^0=1`, but other gauge choices can be provided
            as inputs. 

        .. note::

            We provide the option to compute the pre-potential and some additional functions
            in terms of the periods directly, see in particular :func:`jaxvacua.periods.periods.F_LCS_per`, 
            :func:`jaxvacua.periods.periods.prepot_per`
            and :func:`jaxvacua.periods.periods.period_vector_per`.
        
        .. warning::
            The moduli space limit around which the pre-potential is computed is set by the global parameter ``self.periods.limit``. 
            Currently, only ``self.periods.limit="LCS"`` is supported.
        
        Args:
            moduli (Array): Complex structure moduli values.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            complex: Value of the prepotential :math:`F(z^i)`.
            
        Errors:
            ValueError: If the moduli space limit is not identified.

        Aliases:
            :func:`F`
            
        See also: :func:`prepot_per`
        
        """
        
        if self.periods.prepotential_input is None and self.gauge_choice==1.:
            if self.periods.limit in ["LCS","coniLCS"]:
                return self.F_LCS(moduli,conj=conj)
            elif self.periods.limit == "coniLCS_series":
                return self.F_coniLCS_series(moduli,conj=conj)
            elif self.periods.limit == "coniLCS_bulk":
                return self.F_coniLCS_bulk(moduli,conj=conj)
            else:
                raise ValueError("Could not identify mode for computing the prepotential in complex structure class!")
        else:
                
            mod = self.moduli_to_periods(moduli,conj=conj)

            return self.periods.prepot_per(mod,conj=conj)
    
    F = prepot
        
    @partial(jit, static_argnums = (2,))
    def dF(self, 
        moduli: Array, 
        conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Computes the holomorphic derivative :math:`\partial_{z^i} F` of the prepotential :math:`F`
        for given values of the moduli.
        
        Args:
            moduli (Array): Complex structure moduli values.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Value of the holomorphic derivatives :math:`\partial_{z^i}F` of the prepotential :math:`F(z^i)`.

        See also: :func:`prepot`

        """
        
        return jax.grad(self.prepot,holomorphic=True)(moduli,conj=conj)
    
    @partial(jit, static_argnums = (2,))
    def period_vector(
                    self, 
                    moduli: Array, 
                    conj: bool = False
                    ) -> Array:
        r"""
        
        **Description:**
        Returns the period vector :math:`\Pi` at a given point in moduli space.
        
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
                z^i =\dfrac{X^i}{X^0}\, , i=1,\ldots,h^{2,1}(X)\, , 

            and normalise :math:`\Omega` such that :math:`X^0=1`. The dual periods :math:`\mathcal{F}_I=\mathcal{F}_I(z^i)` are then 
            determined by a prepotential :math:`F(z^i)` through
            
            .. math::
                \mathcal{F}_i(z^i)=\partial_{z^i} F \, ,\quad \mathcal{F}_0 =2F-z^i\partial_{z^i}F\, .
            
            The period vector can then be computed from the derivatives of the prepotential :math:`F(z^i)` as follows
            
            .. math::
                \Pi(z^1,\ldots , z^{h^{1,2}})=\left (\begin{array}{c} 2F-F_i z^i\\ F_i\\ 1\\ z^i \end{array}\right )\, i=1,\ldots,h^{1,2}\, .
            
            Here, we make use of standard identities for computing periods in terms of the projective
            coordinates directly.

            
        .. note::
            To compute the period vector in terms of the periods directly, please use
            :func:`jaxvacua.periods.periods.period_vector_per`.
        
        Args:
            moduli (Array): Complex structure moduli values.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Value of the period vector :math:`\Pi`.

        See also: :func:`prepot`
    
        See also: :func:`dF`
    
        See also: :func:`jaxvacua.periods.periods.period_vector_per`
        
        """
        
        if self.periods.period_input is None:

            # Compute the value of the gradient of the prepotential
            dF = self.dF(moduli,conj=conj)

            # Get lower half of periods
            X = self.moduli_to_periods(moduli,conj=conj)

            return jnp.concatenate((jnp.concatenate((jnp.array([2*self.prepot(moduli,conj=conj) - jnp.dot(moduli,dF)]), dF)), X))
        else:
                
            X = self.moduli_to_periods(moduli,conj=conj)

            return self.periods.period_vector_per(X,conj=conj)
        
    @partial(jit, static_argnums = ())
    def mirror_volume(
                    self, 
                    moduli: Array, 
                    moduli_c: Array
                    ) -> complex:
        r"""
        
        **Description:**
        Returns the value of the mirror dual Calabi-Yau volume.
        
        .. admonition:: Details
            :class: dropdown
        
            Let :math:`X` be a Calabi-Yau threefold and :math:`\Omega_3` its holomorphic :math:`(3,0)`-form. Then this function returns the value of the following integral:
        
            .. math::
                    \mathcal{A}(z^i,\overline{z}^i)=-\text{i}\, \int_{X}\, \Omega_3\wedge \overline{\Omega}_3 = -\text{i}\, \Pi^\dagger(\overline{z}^i)\cdot \Sigma\cdot \Pi(z^i)
        
            This object appears as the argument inside the Kähler potential, see :func:`kahler_potential`. Given that it is dual to the volume of the mirror Calabi-Yau threefold :math:`\widetilde{X}`, it is manifestly positive, provided the Kähler cone conditions of :math:`\widetilde{X}` are satisfied.
        
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
        
        Returns:
            complex: Mirror dual Calabi-Yau volume.

        Aliases:
            :func:`A`, :func:`mirror_volume`, :func:`V_tilde`

        See also: :func:`period_vector`
    
        See also: :func:`kahler_potential`
        
        """
        
        return -1.j*jnp.matmul(self.period_vector(moduli_c,conj=True), jnp.matmul(self.periods.sigma, self.period_vector(moduli)))
        

    A = mirror_volume
    V_tilde = mirror_volume  

    # Kahler potential in terms of the axio dilation and the period vectors
    @partial(jit, static_argnums = ())
    def kahler_potential(
                        self, 
                        moduli: Array, 
                        moduli_c: Array, 
                        tau: complex, 
                        tau_c: complex
                        ) -> complex:
        r"""
        
        **Description:**
        Returns the value of the Kähler potential.
        
        
        .. admonition:: Details
            :class: dropdown
        
            Let :math:`X` be a Calabi-Yau threefold and :math:`\Omega_3` its holomorphic :math:`(3,0)`-form. Then this function returns the value of the following integral:
        
            .. math::
                K(z^i,\overline{z}^i,\tau,\overline{\tau})=-\log\bigl [-\text{i}(\tau-\overline{\tau})\bigr ]-\log\bigl [\mathcal{A}(z^i,\overline{z}^i)\bigr ]
        
            where :math:`\mathcal{A}(z^i,\overline{z}^i)` is the mirror dual CY volume defined as
        
            .. math::
                \mathcal{A}(z^i,\overline{z}^i)=-\text{i}\, \int_{X}\, \Omega_3\wedge \overline{\Omega}_3 = -\text{i}\, \Pi^\dagger(\overline{z}^i)\cdot \Sigma\cdot \Pi(z^i)\, .
        
            The period vector can be computed via :func:`period_vector` or :func:`jaxvacua.periods.periods.period_vector_per`.

        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            complex: Value of the Kähler potential.

        Aliases:
            :func:`K`, :func:`kahler_potential`
        
        See also: :func:`A`
        """
        
        return -jnp.log(-1.j*(tau-tau_c))-jnp.log(self.A(moduli, moduli_c))

    K = kahler_potential
    
    
    @partial(jit, static_argnums = ())
    def dK_z(self, 
            moduli: Array, 
            moduli_c: Array, 
            tau: complex, 
            tau_c: complex
            ) -> Array:
        r"""
        
        **Description:**
        Returns the holomorphic derivative :math:`\partial_{z^i}K` of the Kähler potential :math:`K`
        with respect to the complex structure moduli :math:`z^i`.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            Array: Holomorphic derivative :math:`\partial_{z^i}K` of the Kähler potential :math:`K`.
            
        See also: :func:`kahler_potential`
        
        """
        
        # Gradient of the kahler potential w.r.t moduli and axio dilaton
        return jax.grad(self.kahler_potential,argnums=0,holomorphic=True)(moduli,moduli_c,tau,tau_c)

    
    @partial(jit, static_argnums = ())
    def dK_cz(self, 
            moduli: Array, 
            moduli_c: Array, 
            tau: complex, 
            tau_c: complex
            ) -> Array:
        r"""
        
        **Description:**
        Returns the anti-holomorphic derivative :math:`\partial_{\overline{z}^i}K` of the Kähler potential :math:`K` 
        with respect to the conjugate complex structure moduli :math:`\overline{z}^i`.
        
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            Array: Anti-holomorphic derivative :math:`\partial_{\overline{z}^i}K` of the Kähler potential :math:`K`.
            
        See also: :func:`kahler_potential`
        
        """
        # Gradient of the kahler potential w.r.t moduli and axio dilaton
        return jax.grad(self.kahler_potential,argnums=1,holomorphic=True)(moduli, moduli_c, tau, tau_c)

    

    @partial(jit, static_argnums = ())
    def dK_tau(self, 
            moduli: Array, 
            moduli_c: Array, 
            tau: complex, 
            tau_c: complex
            ) -> complex:
        r"""
        
        **Description:**
        Returns the holomorphic derivative :math:`\partial_{\tau}K` of the Kähler potential :math:`K`
        with respect to the axio-dilaton :math:`\tau`.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            complex: Holomorphic derivative :math:`\partial_{\tau}K` of the Kähler potential :math:`K`.
            
        See also: :func:`kahler_potential`
        
        """
        
        return jax.grad(self.kahler_potential,argnums=2,holomorphic=True)(moduli,moduli_c,tau,tau_c)
    
    
    @partial(jit, static_argnums = ())
    def dK_ctau(self, 
            moduli: Array, 
            moduli_c: Array, 
            tau: complex, 
            tau_c: complex
            ) -> complex:
        r"""
        
        **Description:**
        Returns the anti-holomorphic derivative :math:`\partial_{\overline{\tau}}K` of the Kähler potential :math:`K`
        with respect to the conjugate axio-dilaton :math:`\overline{\tau}`.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            complex: Anti-holomorphic derivative :math:`\partial_{\overline{\tau}}K` of the Kähler potential :math:`K`.
            
        See also: :func:`kahler_potential`
        
        """
        
        #return jax.grad(self.kahler_potential_conj,argnums=3,holomorphic=True)(moduli, moduli_c, tau, tau_c)
        return jax.grad(self.kahler_potential,argnums=3,holomorphic=True)(moduli, moduli_c, tau, tau_c)


    @partial(jit, static_argnums = ())
    def dK(self, 
            moduli: Array, 
            moduli_c: Array, 
            tau: complex, 
            tau_c: complex
            ) -> Array:
        r"""
        
        **Description:**
        Returns the holomorphic derivative :math:`\partial_I K` of the Kähler potential :math:`K` 
        with respect to the complex structure moduli :math:`z^{i}` and the axio-dilaton :math:`\tau`.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            Array: Values of :math:`\partial_I K`. 


        See also: :func:`kahler_potential`
        
        """
        
        dKz = self.dK_z(moduli, moduli_c, tau, tau_c)
        dKtau = self.dK_tau(moduli, moduli_c, tau, tau_c)
        
        return jnp.append(dKz,dKtau)
    
    
    
    @partial(jit, static_argnums = ())
    def dK_c(self, 
            moduli: Array, 
            moduli_c: Array, 
            tau: complex, 
            tau_c: complex
            ) -> Array:
        r"""
        
        **Description:**
        Returns the holomorphic derivative :math:`\partial_{\overline{I}} K` of the Kähler potential :math:`K` 
        with respect to the complex conjugate complex structure moduli :math:`\overline{z}^{i}` 
        and the axio-dilaton :math:`\overline{\tau}`.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            Array: Values of :math:`\partial_{\overline{I}} K`. 


        See also: :func:`kahler_potential`
        
        """
        
        dKcz = self.dK_cz(moduli, moduli_c, tau, tau_c)
        dKctau = self.dK_ctau(moduli, moduli_c, tau, tau_c)
        
        return jnp.append(dKcz,dKctau)


    @partial(jit, static_argnums = ())
    def ddK_z_cz(self, 
                moduli: Array, 
                moduli_c: Array, 
                tau: complex, 
                tau_c: complex
                ) -> Array:
        r"""
        
        **Description:**
        Returns the second derivatives :math:`K_{i\bar{\jmath}}=\partial_{z^i}\partial_{\overline{z}^j}K`
        of the Kähler potential :math:`K` with respect to the complex structure moduli :math:`z^i` and their conjugate.
        
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            Array: Second derivatives :math:`K_{i\bar{\jmath}}=\partial_{z^i}\partial_{\overline{z}^j}K`
                of the Kähler potential :math:`K`.
            
        See also: :func:`dK_z`
        
        """
    
        return jax.jacrev(self.dK_z,argnums =1, holomorphic = True)(moduli, moduli_c, tau, tau_c) #K_Z\bar{Z}

    @partial(jit, static_argnums = ())
    def ddK_cz_z(self, 
                moduli: Array, 
                moduli_c: Array, 
                tau: complex, 
                tau_c: complex
                ) -> Array:
        r"""
        
        **Description:**
        Returns the second derivatives :math:`K_{\bar{\jmath}i}=\partial_{\overline{z}^j}\partial_{z^i}K`
        of the Kähler potential :math:`K` with respect to the complex structure moduli :math:`z^i` and their conjugate.
        
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            Array: Second derivatives :math:`K_{\bar{\jmath}i}=\partial_{\overline{z}^j}\partial_{z^i}K`
                of the Kähler potential :math:`K`.
            
        See also: :func:`dK_z`
        
        """
    
        return jax.jacrev(self.dK_cz,argnums =0, holomorphic = True)(moduli, moduli_c, tau, tau_c) #K_Z\bar{Z}


    @partial(jit, static_argnums = ())
    def ddK_z_tau(self, 
                moduli: Array, 
                moduli_c: Array, 
                tau: complex, 
                tau_c: complex
                ) -> Array:
        r"""
        
        **Description:**
        Returns the second derivatives :math:`K_{i\tau}=\partial_{z^i}\partial_{\tau}K`
        of the Kähler potential :math:`K` with respect to the complex structure moduli :math:`z^i` and 
        the axio-dilaton :math:`\tau`.
        
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            Array: Second derivatives :math:`K_{i\tau}=\partial_{z^i}\partial_{\tau}K`
                of the Kähler potential :math:`K`.
        
        """
        
        return jax.jacrev(self.dK_z,argnums =2, holomorphic = True)(moduli, moduli_c, tau, tau_c)

    @partial(jit, static_argnums = ())
    def ddK_cz_ctau(self, 
                    moduli: Array, 
                    moduli_c: Array, 
                    tau: complex, 
                    tau_c: complex
                    ) -> Array:
        r"""
        
        **Description:**
        Returns the second derivatives :math:`K_{\bar{\imath}\overline{\tau}}=\partial_{\overline{z}^i}\partial_{\overline{\tau}}K`
        of the Kähler potential :math:`K` with respect to the complex conjugate complex structure moduli :math:`\overline{z}^i` and 
        the axio-dilaton :math:`\overline{\tau}`.
        
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            Array: Second derivatives :math:`K_{\bar{\imath}\overline{\tau}}=\partial_{\overline{z}^i}\partial_{\overline{\tau}}K`
                of the Kähler potential :math:`K`.
        
        """
        
        return jax.jacrev(self.dK_cz,argnums =3, holomorphic = True)(moduli, moduli_c, tau, tau_c)
    

    @partial(jit, static_argnums = ())
    def ddK_z_ctau(self, 
                moduli: Array, 
                moduli_c: Array, 
                tau: complex, 
                tau_c: complex
                ) -> Array:
        r"""
        
        **Description:**
        Returns the second derivatives :math:`K_{i\overline{\tau}}=\partial_{z^i}\partial_{\overline{\tau}}K`
        of the Kähler potential :math:`K` with respect to the complex structure moduli :math:`z^i` and 
        the conjugate axio-dilaton :math:`\overline{\tau}`.
        
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            Array: Second derivatives :math:`K_{i\overline{\tau}}=\partial_{z^i}\partial_{\overline{\tau}}K`
                of the Kähler potential :math:`K`.
        
        """
        
        return jax.jacrev(self.dK_z,argnums =3, holomorphic = True)(moduli, moduli_c, tau, tau_c)

    @partial(jit, static_argnums = ())
    def ddK_cz_tau(self, 
                moduli: Array, 
                moduli_c: Array, 
                tau: complex, 
                tau_c: complex
                ) -> Array:
        r"""
        
        **Description:**
        Returns the second derivatives :math:`K_{\tau\overline{j}}=\partial_{\tau}\partial_{\overline{z}^{j}}K` 
        of the Kähler potential :math:`K` with respect to the conjugate complex structure moduli :math:`\overline{z}^i`
        and the axio-dilaton :math:`\tau`.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            Array: Second derivatives :math:`K_{\tau\overline{j}}=\partial_{\tau}\partial_{\overline{z}^{j}}K` 
                of the Kähler potential :math:`K`.
        
        """
        
        return jax.jacrev(self.dK_cz,argnums =2, holomorphic = True)(moduli, moduli_c, tau, tau_c)

    @partial(jit, static_argnums = ())
    def ddK_tau_ctau(self, 
                    moduli: Array, 
                    moduli_c: Array, 
                    tau: complex, 
                    tau_c: complex
                    ) -> complex:
        r"""
        
        **Description:**
        Returns the second derivatives :math:`\partial_{\tau}\partial_{\overline{\tau}}K` of the
        Kähler potential :math:`K` with respect to the axio-dilaton :math:`\tau` and its conjugate.
        
        .. note::
        
            The returned scalar is given by
        
            .. math::
                K_{\tau\overline{\tau}}(z^i,\overline{z}^i,\tau,\overline{\tau})=\partial_{\tau}\partial_{\overline{\tau}}K\, .

            For the standard Kähler potential :math:`K \supset - \log (-\text{i}(\tau-\overline{\tau}))`, we find

            .. math::
                K_{\tau\overline{\tau}} = -\frac{1}{(\tau-\overline{\tau})^2}= \frac{1}{4\mathrm{Im}(\tau)^2}\, .
        
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            complex: Second derivatives :math:`\partial_{\tau}\partial_{\overline{\tau}}K`
                of the Kähler potential :math:`K`.
        
        """
        
        return jax.jacrev(self.dK_tau,argnums =3, holomorphic = True)(moduli, moduli_c, tau, tau_c)
        
    @partial(jit, static_argnums = ())
    def ddK_z_z(self, 
                moduli: Array, 
                moduli_c: Array, 
                tau: complex, 
                tau_c: complex
                ) -> Array:
        r"""
        
        **Description:**
        Returns the second holomorphic derivatives of the Kähler potential with respect to the complex structure moduli :math:`z^i`.
        
        .. admonition:: Details
            :class: dropdown
        
            The returned array has the components
        
            .. math::
                K_{ij}(z^i,\overline{z}^i,\tau,\overline{\tau})=\partial_{z^i}\partial_{z^{j}}K(z^i,\overline{z}^i,\tau,\overline{\tau})\, .
        
        .. note::
            These holomorphic derivatives are used e.g. in :func:`DDW` to compute the second Kähler covariant derivative of the superpotential.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            Array: :math:`h^{1,2}\times h^{1,2}` matrix of the second holomorphic derivatives of the Kähler potential with respect to :math:`z^i`.
        
        """
        
        return jax.jacrev(self.dK_z,argnums =0, holomorphic = True)(moduli, moduli_c, tau, tau_c)

    @partial(jit, static_argnums = ())
    def ddK_cz_cz(self, 
                moduli: Array, 
                moduli_c: Array, 
                tau: complex, 
                tau_c: complex
                ) -> Array:
        r"""
        
        **Description:**
        Returns the second anti-holomorphic derivatives of the Kähler potential with respect to the complex structure moduli :math:`z^i`.
        
        .. admonition:: Details
            :class: dropdown
        
            The returned array has the components
        
            .. math::
                K_{ij}(z^i,\overline{z}^i,\tau,\overline{\tau})=\partial_{z^i}\partial_{z^{j}}K(z^i,\overline{z}^i,\tau,\overline{\tau})\, .
        
        .. note::
            These holomorphic derivatives are used e.g. in :func:`DDW` to compute the second Kähler covariant derivative of the superpotential.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            Array: :math:`h^{1,2}\times h^{1,2}` matrix of the second holomorphic derivatives of the Kähler potential with respect to :math:`z^i`.
        
        """
        
        return jax.jacrev(self.dK_cz,argnums = 1, holomorphic = True)(moduli, moduli_c, tau, tau_c)
        
    @partial(jit, static_argnums = ())
    def ddK_z_tau(self, 
                moduli: Array, 
                moduli_c: Array, 
                tau: complex, 
                tau_c: complex
                ) -> Array:
        r"""
        
        **Description:**
        Returns the second holomorphic derivatives of the Kähler potential with respect to the complex structure moduli :math:`z^i` and the axio-dilaton :math:`\tau`.
        
        .. admonition:: Details
            :class: dropdown
        
            The returned array has the components
        
            .. math::
                K_{i\tau}(z^i,\overline{z}^i,\tau,\overline{\tau})=\partial_{z^i}\partial_{\tau}K(z^i,\overline{z}^i,\tau,\overline{\tau})\, .
        
        .. note::
            These holomorphic derivatives are used e.g. in :func:`DDW` to compute the second Kähler covariant derivative of the superpotential.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            Array: :math:`h^{1,2}\times 1` matrix of the second holomorphic derivatives of the Kähler potential with respect to :math:`z^i` and :math:`\tau`.
        
        """
        
        return jax.jacrev(self.dK_z,argnums =2, holomorphic = True)(moduli, moduli_c, tau, tau_c)
        
    @partial(jit, static_argnums = ())
    def ddK_tau_tau(self, 
                    moduli: Array, 
                    moduli_c: Array, 
                    tau: complex, 
                    tau_c: complex
                    ) -> complex:
        r"""
        
        **Description:**
        Returns the second holomorphic derivatives of the Kähler potential with respect to the axio-dilaton :math:`\tau`.
        
        .. admonition:: Details
            :class: dropdown
        
            The returned array has the components
        
            .. math::
                K_{\tau\tau}(z^i,\overline{z}^i,\tau,\overline{\tau})=\partial_{\tau}\partial_{\tau}K(z^i,\overline{z}^i,\tau,\overline{\tau})\, .
        
        .. note::
            These holomorphic derivatives are used e.g. in :func:`DDW` to compute the second Kähler covariant derivative of the superpotential.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            complex: Second holomorphic derivative of the Kähler potential with respect to :math:`\tau`.
        
        """
        
        return jax.jacrev(self.dK_tau,argnums =2, holomorphic = True)(moduli, moduli_c, tau, tau_c)

    @partial(jit, static_argnums = ())
    def ddK_ctau_ctau(self, 
                    moduli: Array, 
                    moduli_c: Array, 
                    tau: complex, 
                    tau_c: complex
                    ) -> complex:
        r"""
        
        **Description:**
        Returns the second holomorphic derivatives of the Kähler potential with respect to the axio-dilaton :math:`\tau`.
        
        .. admonition:: Details
            :class: dropdown
        
            The returned array has the components
        
            .. math::
                K_{\tau\tau}(z^i,\overline{z}^i,\tau,\overline{\tau})=\partial_{\tau}\partial_{\tau}K(z^i,\overline{z}^i,\tau,\overline{\tau})\, .
        
        .. note::
            These holomorphic derivatives are used e.g. in :func:`DDW` to compute the second Kähler covariant derivative of the superpotential.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            complex: Second holomorphic derivative of the Kähler potential with respect to :math:`\tau`.
        
        """
        
        return jax.jacrev(self.dK_ctau,argnums =3, holomorphic = True)(moduli, moduli_c, tau, tau_c)
    
    @partial(jit, static_argnums = ())
    def _kahler_metric_general(self, 
                            moduli: Array, 
                            moduli_c: Array, 
                            tau: complex, 
                            tau_c: complex
                            ) -> Array:
        r"""
        
        **Description:**
        Returns the Kähler metric by combining the matrices of second derivatives.
        
        .. admonition:: Details
            :class: dropdown
        
            We assemble the various components of second derivatives obtained from :func:`ddK_z_cz`, :func:`ddK_z_ctau`, :func:`ddK_cz_tau` and :func:`ddK_tau_ctau` in such a way that
        
            .. math::
                K_{A\overline{B}}=\left (\begin{array}{cc}
                K_{i\overline{j}} & K_{i\overline{\tau}} \
                K_{\tau\overline{j}} & K_{\tau\overline{\tau}}
                \end{array} \right )\, .
        
        
        .. note::
            This function is general by which we mean it does not make any prior assumption of the Kähler potential. If we simply consider the tree level scalar potential for the complex structure moduli and the axio dilaton :math:`K=-\log\left (-\text{i}\int\Omega_{3}\wedge\overline{\Omega}_{3}\right ) - \log (-\text{i}(\tau-\overline{\tau}))`, one can in principle make use of the block diagonal structure of the Kähler metric, see :func:`kahler_metric_block_diagonal`.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            Array: Kähler metric :math:`K_{\overline{I}J}`.

        See also: :func:`ddK_z_cz`
    
        See also: :func:`ddK_z_ctau`
    
        See also: :func:`ddK_cz_tau`
    
        See also: :func:`ddK_tau_ctau`
        
        """

        val1 = self.ddK_z_cz(moduli, moduli_c, tau, tau_c) #Component with K_i\bar{j}
        val2 = self.ddK_z_ctau(moduli, moduli_c, tau, tau_c) #Component with K_i\bar{tau}
        val3 = self.ddK_cz_tau(moduli, moduli_c, tau, tau_c) #Component with K_tau\bar{j}
        val4 = self.ddK_tau_ctau(moduli, moduli_c, tau, tau_c) #Component with K_tau\bar{tau}

        # Combine blocks into array of the form [K_i\bar{j}, K_i\bar{tau}]
        a = jnp.hstack((jnp.asarray(val1), jnp.asarray(val2).reshape(self.h12, 1)))

        # Combine blocks into array of the form [K_tau\bar{j}, K_tau\bar{tau}]
        b = jnp.hstack((jnp.asarray(val3), jnp.asarray(val4)))

        # Stack the two blocks and return result
        return jnp.vstack((a, b))
    
    @partial(jit, static_argnums = ())
    def _kahler_metric_block_diagonal(self, 
                                    moduli: Array, 
                                    moduli_c: Array, 
                                    tau: complex, 
                                    tau_c: complex
                                    ) -> Array:
        r"""
        
        **Description:**
        Returns the Kähler metric by combining the matrices of second derivatives assuming a block-diagonal
        structure for the Kähler metric.
        
        .. admonition:: Details
            :class: dropdown
        
            We assemble the various components of second derivatives obtained from :func:`ddK_z_cz`, :func:`ddK_z_ctau`, :func:`ddK_cz_tau` and :func:`ddK_tau_ctau` in such a way that
        
            .. math::
                K_{A\overline{B}}=\left (\begin{array}{cc}
                K_{i\overline{j}} & 0 \\[0.3em]
                0 & K_{\tau\overline{\tau}}
                \end{array} \right )\, .
        
            Here, we assume that there is a block-diagonal structure in the Kahler potential :math:`K` given by
        
            .. math::
                K=-\log\left (-\text{i}\int\Omega_{3}\wedge\overline{\Omega}_{3}\right ) - \log (-\text{i}(\tau-\overline{\tau}))
        
            In this case, the mixed second derivatives like :math:`K_{i\overline{\tau}}=K_{\tau\overline{i}}=0` vanish.
        
        .. warning::
            This function relies on a block-diagonal structure in the Kahler potential :math:`K`. 
            This may not necessarily true if further corrections to the tree level Kähler potential are taken into account.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            Array: Kähler metric :math:`K_{\overline{I}J}` assuming :math:`K_{i\overline{\tau}}=K_{\tau\overline{i}}=0`.

        See also: :func:`ddK_z_cz`
    
        See also: :func:`ddK_tau_ctau`
        
        """

        val1 = self.ddK_z_cz(moduli, moduli_c, tau, tau_c) #Component with K_i\bar{j}
        val2 = jnp.zeros((self.h12,1),dtype=jnp.complex_)  #Component with K_i\bar{tau}=0
        val3 = jnp.zeros((self.h12,),dtype=jnp.complex_)   #Component with K_tau\bar{j}=0
        val4 = self.ddK_tau_ctau(moduli, moduli_c, tau, tau_c) #Component with K_tau\bar{tau}

        # Combine blocks into array of the form [K_i\bar{j}, 0]
        a = jnp.hstack((jnp.asarray(val1), jnp.asarray(val2)))

        # Combine blocks into array of the form [0, K_tau\bar{tau}]
        b = jnp.hstack((jnp.asarray(val3), jnp.asarray(val4)))

        # Stack the two blocks and return result
        return jnp.vstack((a, b))
        
    @partial(jit, static_argnums = (5,))
    def kahler_metric(self, 
                    moduli: Array, 
                    moduli_c: Array, 
                    tau: complex, 
                    tau_c: complex, 
                    mode: str = "block diagonal"
                    ) -> Array:
        r"""
        
        **Description:**
        Computes the Kähler metric :math:`K_{\overline{I}J}`.
        
        .. admonition:: Details
            :class: dropdown
        
            We assemble the various components of second derivatives obtained from :func:`ddK_z_cz`, 
            :func:`ddK_z_ctau`, :func:`ddK_cz_tau` and :func:`ddK_tau_ctau` in such a way that
        
            .. math::
                K_{A\overline{B}}=\left (\begin{array}{cc}
                K_{i\overline{j}} & K_{i\overline{\tau}} \\[0.3em]
                K_{\tau\overline{j}} & K_{\tau\overline{\tau}}
                \end{array} \right )
        
            Depending on the use case, we assume a block diagonal structure of the Kähler metric 
            in which case the mixed second derivatives vanish 
            :math:`K_{i\overline{\tau}}=K_{\tau\overline{j}}=0`.

        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
            mode (string, optional): Mode to compute the Kahler metric. Defaults to ``"block diagonal"`` 
                meaning that there are no mixing terms between complex structure moduli and 
                the axio-dilaton in the Kähler potential. If set to ``None`` instead, the full set of 2nd 
                derivatives of the Kähler potential is computed.
        
        Returns:
            Array: Kähler metric :math:`K_{\overline{I}J}`.

        Raises:
            ValueError: Wrong mode for computation. Only ``mode=None`` and ``mode="block diagonal"``
                are supported.
        
        """
    
        if mode not in ["block diagonal",None]:
            raise ValueError("CANNOT IDENTIFY MODE FOR COMPUTING THE KAHLER METRIC!")
             
        if mode is None:
            return self._kahler_metric_general( moduli, moduli_c, tau, tau_c)
        elif mode=="block diagonal":
            return self._kahler_metric_block_diagonal( moduli, moduli_c, tau, tau_c)


    @partial(jit, static_argnums = (5,))
    def inverse_kahler_metric(self, 
                            moduli: Array, 
                            moduli_c: Array, 
                            tau: complex, 
                            tau_c: complex, 
                            mode: str = "block diagonal"
                            ) -> Array:
        r"""
        
        **Description:**
        Returns the inverse Kähler metric :math:`K^{\overline{I}J}`.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
            mode (str, optional): Description
        
        Returns:
            Array: Inverse Kähler metric :math:`K^{\overline{I}J}`.
            

        See also: :func:`kahler_metric`
        
        """
    
        return jnp.linalg.inv(self.kahler_metric(moduli,moduli_c,tau,tau_c,mode=mode))
        
    @partial(jit, static_argnums = (5,))
    def inverse_kahler_metric_grad(self, 
                                moduli: Array, 
                                moduli_c: Array, 
                                tau: complex, 
                                tau_c: complex, 
                                mode: str = "block diagonal"
                                ) -> Tuple[Array,Array,complex,complex]:
        r"""
        
        **Description:**
        Returns the gradient of the inverse Kähler metric.
        
        .. note::
            This function is currently not being used, but might turn out to be useful 
            when taking derivatives of the F-terms :math:`F^i = K^{i\overline{j}}D_{\overline{j}}\overline{W}`.
            The output corresponds to the tuple

            .. math::
                (\partial_{z^i}K^{\bar{J}L},\partial_{\overline{z}^i}K^{\bar{J}L},\partial_{\tau}K^{\bar{J}L},\partial_{\overline{\tau}}K^{\bar{J}L})\, .
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
            mode (str, optional): Description
        
        Returns:
            Array: Gradient of the inverse Kähler metric :math:`\partial_{I}K^{\overline{J}L},\partial_{\overline{I}}K^{\overline{J}L}`.

        See also: :func:`kahler_metric`

        See also: :func:`inverse_kahler_metric`
        
        """
        
        return jax.jacrev(self.inverse_kahler_metric,holomorphic=True)(moduli,moduli_c,tau,tau_c,mode=mode)


    ###################################################################################################################
    ############################  Kähler connection and curvature  #####################################################
    ###################################################################################################################

    @partial(jit, static_argnums = ())
    def dddK(
            self,
            moduli: Array,
            moduli_c: Array,
            tau: complex,
            tau_c: complex
            ) -> Array:
        r"""

        **Description:**
        Returns the third holomorphic-mixed Kähler derivative
        :math:`\partial_{A} K_{C\bar{F}}` with respect to the combined
        field-space index :math:`A = (z^i, \tau)`.

        .. admonition:: Details
            :class: dropdown

            Computes the holomorphic derivative of the Kähler metric
            :math:`K_{C\bar{F}} = \partial_C\partial_{\bar{F}} K` with
            respect to each holomorphic direction :math:`A`:

            .. math::
                (\texttt{dddK})_{C\bar{F}A} = \partial_A K_{C\bar{F}}
                = \partial_A \partial_C \partial_{\bar{F}} K\,.

            These third derivatives appear in the Christoffel symbols of the
            Kähler connection via

            .. math::
                \Gamma^E_{AC} = K^{E\bar{F}}\,\partial_A K_{C\bar{F}} \,.

        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.

        Returns:
            Array: Third Kähler derivative :math:`\partial_A K_{C\bar{F}}`
                with shape ``(n, n, n)`` and index ordering ``[C, F̄, A]``
                where ``n = h^{1,2} + 1``.

        See also: :func:`kahler_metric`, :func:`christoffel_symbols`

        """
        km_z = jax.jacrev(self.kahler_metric, argnums=0, holomorphic=True)(
            moduli, moduli_c, tau, tau_c)
        km_tau = jax.jacrev(self.kahler_metric, argnums=2, holomorphic=True)(
            moduli, moduli_c, tau, tau_c)
        return jnp.concatenate([km_z, km_tau[:, :, None]], axis=-1)

    @partial(jit, static_argnums = ())
    def dddK_c(
            self,
            moduli: Array,
            moduli_c: Array,
            tau: complex,
            tau_c: complex
            ) -> Array:
        r"""

        **Description:**
        Returns the third anti-holomorphic-mixed Kähler derivative
        :math:`\partial_{\bar{B}} K_{C\bar{F}}` with respect to the combined
        anti-holomorphic field-space index :math:`\bar{B} = (\bar{z}^i, \bar{\tau})`.

        .. admonition:: Details
            :class: dropdown

            Computes the anti-holomorphic derivative of the Kähler metric
            :math:`K_{C\bar{F}}`:

            .. math::
                (\texttt{dddK\_c})_{C\bar{F}\bar{B}} = \partial_{\bar{B}} K_{C\bar{F}}
                = \partial_{\bar{B}} \partial_C \partial_{\bar{F}} K\,.

            These appear in the anti-holomorphic derivative of the inverse
            Kähler metric and in the Riemann curvature tensor.

        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.

        Returns:
            Array: Third Kähler derivative :math:`\partial_{\bar{B}} K_{C\bar{F}}`
                with shape ``(n, n, n)`` and index ordering ``[C, F̄, B̄]``
                where ``n = h^{1,2} + 1``.

        See also: :func:`kahler_metric`, :func:`riemann_tensor`

        """
        km_zc = jax.jacrev(self.kahler_metric, argnums=1, holomorphic=True)(
            moduli, moduli_c, tau, tau_c)
        km_tc = jax.jacrev(self.kahler_metric, argnums=3, holomorphic=True)(
            moduli, moduli_c, tau, tau_c)
        return jnp.concatenate([km_zc, km_tc[:, :, None]], axis=-1)

    @partial(jit, static_argnums = ())
    def christoffel_symbols(
            self,
            moduli: Array,
            moduli_c: Array,
            tau: complex,
            tau_c: complex
            ) -> Array:
        r"""

        **Description:**
        Returns the Christoffel symbols :math:`\Gamma^E_{AC}` of the
        Levi-Civita connection on the Kähler moduli space.

        .. admonition:: Details
            :class: dropdown

            The Christoffel connection on a Kähler manifold is given by

            .. math::
                \Gamma^E_{AC} = K^{E\bar{F}}\,\partial_A K_{C\bar{F}}

            where :math:`K_{C\bar{F}}` is the Kähler metric and :math:`K^{E\bar{F}}`
            its inverse. The indices :math:`A, C, E` run over the combined field space
            :math:`(z^1,\ldots,z^{h^{1,2}},\tau)`.

            For a Kähler manifold the connection is torsion-free,
            :math:`\Gamma^E_{AC} = \Gamma^E_{CA}`, and the only non-vanishing
            components have purely holomorphic lower indices.

        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.

        Returns:
            Array: Christoffel symbols :math:`\Gamma^E_{AC}` with shape
                ``(n, n, n)`` and index ordering ``[E, A, C]``.

        See also: :func:`dddK`, :func:`inverse_kahler_metric`, :func:`riemann_tensor`

        """
        K3h = self.dddK(moduli, moduli_c, tau, tau_c)
        IKM = self.inverse_kahler_metric(moduli, moduli_c, tau, tau_c)
        return jnp.einsum('ef,cfa->eac', IKM, K3h)

    @partial(jit, static_argnums = ())
    def riemann_tensor(
            self,
            moduli: Array,
            moduli_c: Array,
            tau: complex,
            tau_c: complex
            ) -> Array:
        r"""

        **Description:**
        Returns the Riemann curvature tensor :math:`R_{i\bar{\jmath}k\bar{l}}` of the
        Kähler moduli space.

        .. admonition:: Details
            :class: dropdown

            The Riemann tensor of a Kähler manifold can be expressed in terms
            of the Kähler metric and its derivatives as

            .. math::
                R_{i\bar{\jmath}k\bar{l}}
                = \partial_k\partial_{\bar{l}} K_{i\bar{\jmath}}
                - K^{m\bar{n}}\,
                  (\partial_k K_{i\bar{n}})\,
                  (\partial_{\bar{l}} K_{m\bar{\jmath}})\,.

            The first term is the fourth mixed Kähler derivative
            :math:`K_{i\bar{\jmath}k\bar{l}} = \partial_k\partial_{\bar{l}} K_{i\bar{\jmath}}`,
            while the second subtracts the Christoffel-symbol contraction.

            The tensor satisfies the Kähler symmetry

            .. math::
                R_{i\bar{\jmath}k\bar{l}} = R_{k\bar{\jmath}i\bar{l}}\,.

            It enters the Hessian of the :math:`\mathcal{N}=1` supergravity scalar
            potential at generic points in moduli space (see e.g.
            `hep-th/0411183 <https://arxiv.org/abs/hep-th/0411183>`_, Eq. 2.3).

        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.

        Returns:
            Array: Riemann tensor :math:`R_{i\bar{\jmath}k\bar{l}}` with shape
                ``(n, n, n, n)`` and index ordering ``[i, j̄, k, l̄]``.

        See also: :func:`christoffel_symbols`, :func:`dddK`, :func:`dddK_c`

        """
        K3h = self.dddK(moduli, moduli_c, tau, tau_c)
        K3a = self.dddK_c(moduli, moduli_c, tau, tau_c)
        IKM = self.inverse_kahler_metric(moduli, moduli_c, tau, tau_c)

        # K4[C, F̄, A, B̄] = ∂_A ∂_{B̄} K_{CF̄}
        K3h_zc = jax.jacrev(self.dddK, argnums=1, holomorphic=True)(
            moduli, moduli_c, tau, tau_c)
        K3h_tc = jax.jacrev(self.dddK, argnums=3, holomorphic=True)(
            moduli, moduli_c, tau, tau_c)
        K4 = jnp.concatenate([K3h_zc, K3h_tc[:, :, :, None]], axis=-1)

        return K4 - jnp.einsum('mn,ink,mjl->ijkl', IKM, K3h, K3a)

    @partial(jit, static_argnums = ())
    def dIKM_c(
            self,
            moduli: Array,
            moduli_c: Array,
            tau: complex,
            tau_c: complex
            ) -> Array:
        r"""

        **Description:**
        Returns the anti-holomorphic derivative of the inverse Kähler metric
        :math:`\partial_{\bar{B}} K^{I\bar{J}}`.

        .. admonition:: Details
            :class: dropdown

            From the identity :math:`K_{I\bar{F}}\,K^{I\bar{J}} = \delta_{\bar{F}}^{\bar{J}}`
            one obtains

            .. math::
                \partial_{\bar{B}} K^{I\bar{J}}
                = - K^{I\bar{F}}\,(\partial_{\bar{B}} K_{C\bar{F}})\,K^{C\bar{J}}\,.

            This quantity appears in the anti-holomorphic variation of the
            :math:`F`-term scalar :math:`S = D_{\bar{I}}\bar{W}\,K^{I\bar{J}}\,D_J W`.

        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.

        Returns:
            Array: :math:`\partial_{\bar{B}} K^{I\bar{J}}` with shape ``(n, n, n)``
                and index ordering ``[I, J̄, B̄]``.

        See also: :func:`dddK_c`, :func:`inverse_kahler_metric`

        """
        K3a = self.dddK_c(moduli, moduli_c, tau, tau_c)
        IKM = self.inverse_kahler_metric(moduli, moduli_c, tau, tau_c)
        return -jnp.einsum('if,cfb,cj->ijb', IKM, K3a, IKM)

    @partial(jit, static_argnums = ())
    def ddIKM(
            self,
            moduli: Array,
            moduli_c: Array,
            tau: complex,
            tau_c: complex
            ) -> Array:
        r"""

        **Description:**
        Returns the mixed second derivative of the inverse Kähler metric
        :math:`\partial_A\partial_{\bar{B}} K^{I\bar{J}}`.

        .. admonition:: Details
            :class: dropdown

            Differentiating
            :math:`\partial_{\bar{B}} K^{I\bar{J}} = -K^{I\bar{F}}(\partial_{\bar{B}}K_{C\bar{F}})K^{C\bar{J}}`
            with respect to :math:`z^A` yields three terms via the product rule:

            .. math::
                \partial_A\partial_{\bar{B}} K^{I\bar{J}}
                = \underbrace{-\Gamma^I_{AQ}\,(\partial_{\bar{B}} K^{Q\bar{J}})}_{\text{A1}}
                \;\underbrace{-\;K^{I\bar{F}}\,R_{C\bar{F}A\bar{B}}\,K^{C\bar{J}}}_{\text{A2}_R}
                \;\underbrace{-\;K^{I\bar{F}}\,(K^{M\bar{N}} K_{3h,C\bar{N}A}\,K_{3a,M\bar{F}\bar{B}})\,K^{C\bar{J}}}_{\text{A2}_\Gamma}
                \;\underbrace{+\;K^{I\bar{F}}\,K_{3a,C\bar{F}\bar{B}}\,\Gamma^C_{AD}\,K^{D\bar{J}}}_{\text{A3}}

            where :math:`K_{3h}` and :math:`K_{3a}` are the holomorphic and anti-holomorphic
            third Kähler derivatives (:func:`dddK`, :func:`dddK_c`),
            :math:`\Gamma^E_{AC}` are the Christoffel symbols (:func:`christoffel_symbols`),
            and :math:`R_{i\bar{\jmath}k\bar{l}}` is the Riemann tensor (:func:`riemann_tensor`).

            The Riemann tensor enters through the holomorphic derivative of the
            anti-holomorphic Kähler-metric derivative
            :math:`\partial_A(\partial_{\bar{B}} K_{C\bar{F}}) = K_{C\bar{F}A\bar{B}}`,
            which decomposes as :math:`R + \Gamma`-contraction via the metric formula
            for the Riemann tensor.

        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.

        Returns:
            Array: :math:`\partial_A\partial_{\bar{B}} K^{I\bar{J}}` with shape
                ``(n, n, n, n)`` and index ordering ``[I, J̄, B̄, A]``.

        See also: :func:`riemann_tensor`, :func:`christoffel_symbols`, :func:`dIKM_c`

        """
        K3h = self.dddK(moduli, moduli_c, tau, tau_c)
        K3a = self.dddK_c(moduli, moduli_c, tau, tau_c)
        IKM = self.inverse_kahler_metric(moduli, moduli_c, tau, tau_c)
        Gamma = self.christoffel_symbols(moduli, moduli_c, tau, tau_c)
        R = self.riemann_tensor(moduli, moduli_c, tau, tau_c)
        dIKM_bar = self.dIKM_c(moduli, moduli_c, tau, tau_c)

        # A1: -Γ^I_{AQ} (∂_{B̄} K^{QJ̄})
        A1 = -jnp.einsum('iaq,qjb->ijba', Gamma, dIKM_bar)

        # A2_R: -K^{IF̄} R_{CF̄AB̄} K^{CJ̄}  (Riemann contribution)
        A2_R = -jnp.einsum('if,cfab,cj->ijba', IKM, R, IKM)

        # A2_Gamma: -K^{IF̄} (K^{MN̄} K3h_{CN̄A} K3a_{MF̄B̄}) K^{CJ̄}
        K4_Gamma = jnp.einsum('mn,cna,mfb->cfab', IKM, K3h, K3a)
        A2_Gamma = -jnp.einsum('if,cfab,cj->ijba', IKM, K4_Gamma, IKM)

        # A3: +K^{IF̄} K3a_{CF̄B̄} Γ^C_{AD} K^{DJ̄}
        temp = jnp.einsum('if,cfb->icb', IKM, K3a)
        A3 = jnp.einsum('icb,cad,dj->ijba', temp, Gamma, IKM)

        return A1 + A2_R + A2_Gamma + A3

    @partial(jit, static_argnums = ())
    def dGamma(
            self,
            moduli: Array,
            moduli_c: Array,
            tau: complex,
            tau_c: complex
            ) -> Array:
        r"""

        **Description:**
        Returns the holomorphic derivative of the Christoffel symbols
        :math:`\partial_B \Gamma^E_{AC}`.

        .. admonition:: Details
            :class: dropdown

            Differentiates :math:`\Gamma^E_{AC} = K^{E\bar{F}}\,\partial_A K_{C\bar{F}}`
            with respect to the holomorphic direction :math:`z^B` (or :math:`\tau`).

            This quantity appears in the holomorphic second derivative of the inverse
            Kähler metric (and thus in the holomorphic Hessian :math:`\partial_A\partial_B V`)
            via

            .. math::
                \partial_A\partial_B K^{I\bar{J}}
                = -(\partial_B\Gamma^I_{AC})\,K^{C\bar{J}}
                + \Gamma^I_{AC}\,\Gamma^C_{BD}\,K^{D\bar{J}}\,.

        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.

        Returns:
            Array: :math:`\partial_B \Gamma^E_{AC}` with shape ``(n, n, n, n)``
                and index ordering ``[E, A, C, B]``.

        See also: :func:`christoffel_symbols`

        """
        dG_z = jax.jacrev(self.christoffel_symbols, argnums=0, holomorphic=True)(
            moduli, moduli_c, tau, tau_c)
        dG_tau = jax.jacrev(self.christoffel_symbols, argnums=2, holomorphic=True)(
            moduli, moduli_c, tau, tau_c)
        return jnp.concatenate([dG_z, dG_tau[:, :, :, None]], axis=-1)


    ###################################################################################################################################
    ########################################### MONODROMY MATRICES ##################################################################
    ###################################################################################################################################

    def monodromy_matrix_single(self, b: int) -> np.ndarray:
        r"""
        **Description:**
        Computes the monodromy matrix :math:`T_b` for the shift :math:`z^b \to z^b + 1`.

        The period vector transforms as :math:`\Pi \to T_b \cdot \Pi`.

        At LCS, uses the analytical formula from the intersection numbers,
        a-matrix, and b-vector. For other limits, falls back to a numerical
        computation via the period vector.

        .. note::

            The monodromy matrices receive no instanton corrections: the instanton
            sum :math:`F_{\mathrm{inst}}` involves :math:`e^{2\pi i q_a z^a}` which
            is invariant under :math:`z^a \to z^a + 1`.

        Args:
            b (int): Index of the modulus to shift (0-based: ``b = 0, ..., h12-1``).

        Returns:
            np.ndarray: Integer monodromy matrix of shape ``(2*h12+2, 2*h12+2)``.

        See also: :func:`monodromy_matrix`, :func:`verify_monodromy`
        """
        if self.periods.limit in ["LCS", "coniLCS", "coniLCS_bulk", "coniLCS_series"]:
            return self._monodromy_matrix_LCS(b)
        else:
            return self._monodromy_matrix_numerical(b)

    def monodromy_matrix(self, n) -> np.ndarray:
        r"""
        **Description:**
        Computes the monodromy matrix :math:`T(\vec{n})` for a general integer shift
        :math:`z^a \to z^a + n^a`.

        This is the product :math:`T(\vec{n}) = \prod_b T_b^{n_b}`. The LCS monodromy
        matrices commute (maximally unipotent monodromy), so the order does not matter.

        Args:
            n (array-like): Integer shift vector of shape ``(h12,)``.

        Returns:
            np.ndarray: Integer monodromy matrix of shape ``(2*h12+2, 2*h12+2)``.

        See also: :func:`monodromy_matrix_single`
        """
        n_arr = np.asarray(n, dtype=int)
        h = self.h12
        assert n_arr.shape == (h,), f"Shift vector must have shape ({h},), got {n_arr.shape}"

        dim = 2 * h + 2
        T = np.eye(dim, dtype=int)

        for b in range(h):
            if n_arr[b] == 0:
                continue
            T_b = self.monodromy_matrix_single(b)
            T = T @ np.linalg.matrix_power(T_b, int(n_arr[b]))

        return T

    def _monodromy_matrix_LCS(self, b: int) -> np.ndarray:
        r"""
        **Description:**
        Analytical monodromy matrix :math:`T_b` from LCS topological data.

        Uses the formula:

        * :math:`T[\mathcal{F}_a, z^c] = -\kappa_{abc}`
        * :math:`T[\mathcal{F}_a, 1] = -\kappa_{abb}/2 + a_{ab}`
        * :math:`T[\mathcal{F}_0, \mathcal{F}_b] = -1`
        * :math:`T[\mathcal{F}_0, z^c] = \kappa_{bbc}/2 + a_{bc}`
        * :math:`T[\mathcal{F}_0, 1] = \kappa_{bbb}/6 + 2 b_b`

        Args:
            b (int): Direction index (0-based).

        Returns:
            np.ndarray: Integer monodromy matrix.
        """
        h = self.h12
        n = 2 * h + 2
        kappa = np.array(self.lcs_tree.intnums)
        a_mat = np.array(self.lcs_tree.a_matrix)
        b_vec = np.array(self.lcs_tree.b_vector)

        T = np.eye(n, dtype=float)

        # Index map: F_0=0, F_a=1..h, X^0=h+1, z^a=h+2..2h+1

        # z block: z^b -> z^b + 1
        T[h + 2 + b, h + 1] = 1.0

        # F_a block
        for aa in range(h):
            for c in range(h):
                T[1 + aa, h + 2 + c] += -kappa[aa, b, c]
            T[1 + aa, h + 1] += -0.5 * kappa[aa, b, b] + a_mat[aa, b]

        # F_0 row
        T[0, 1 + b] += -1.0
        for c in range(h):
            T[0, h + 2 + c] += 0.5 * kappa[b, b, c] + a_mat[b, c]
        T[0, h + 1] += kappa[b, b, b] / 6.0 + 2.0 * b_vec[b]

        return np.round(T).astype(int)

    def _monodromy_matrix_numerical(self, b: int, n_samples: int = 15) -> np.ndarray:
        r"""
        **Description:**
        Numerical monodromy matrix :math:`T_b` computed by solving the linear system
        :math:`T_b \cdot \Pi(z) = \Pi(z + e_b)` at multiple random points.

        Args:
            b (int): Direction index (0-based).
            n_samples (int): Number of sample points for the linear solve.

        Returns:
            np.ndarray: Integer monodromy matrix.
        """
        h = self.h12
        rng = np.random.default_rng(42)

        A_rows = []
        B_rows = []

        for _ in range(n_samples):
            z = jnp.array(rng.uniform(-0.3, 0.3, h) + 1j * rng.uniform(2, 5, h))
            Pi_z = np.array(self.period_vector(z))
            z_shifted = z.at[b].add(1.0)
            Pi_shifted = np.array(self.period_vector(z_shifted))
            A_rows.append(Pi_z)
            B_rows.append(Pi_shifted)

        A = np.array(A_rows)
        B = np.array(B_rows)

        T_T, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        T = T_T.T

        T_int = np.round(T.real).astype(int)

        int_err = np.max(np.abs(T - T_int))
        if int_err > 1e-6:
            import warnings
            warnings.warn(
                f"Monodromy matrix T_{b} deviates from integer by {int_err:.2e}. "
                f"This may indicate a problem with the period vector computation."
            )

        return T_int

    def conifold_monodromy_matrix(self, conifold_curve=None, conifold_index=None):
        r"""
        **Description:**
        Picard-Lefschetz monodromy matrix around a conifold singularity
        where a 3-cycle :math:`\gamma = \sum_a c_a A^a` shrinks to zero.

        The monodromy acts on the period vector
        :math:`\Pi = (F_0, F_a, X^0, z^a)` as

        .. math::
            F_a \;\to\; F_a + c_a \sum_b c_b\, z^b\,,

        with all other periods invariant.  This follows from the
        Picard-Lefschetz formula
        :math:`\delta \to \delta + (\delta \cdot \gamma)\,\gamma`
        applied to the symplectic basis of :math:`H_3(X,\mathbb{Z})`.

        Args:
            conifold_curve (array-like, optional): Charge vector
                :math:`c = (c_1, \dots, c_{h^{2,1}})` of the vanishing
                cycle.  The vanishing period is :math:`w = c^T z`.
                If ``None``, uses ``self.lcs_tree.conifold.conifold_curve``
                when available.
            conifold_index (int, optional): Index (0-based among
                :math:`z^1 \dots z^h`) of the conifold modulus,
                equivalent to ``conifold_curve = e_{index}``.

        Returns:
            np.ndarray: Integer matrix of shape ``(2*h12+2, 2*h12+2)``.

        Raises:
            ValueError: If both ``conifold_curve`` and ``conifold_index``
                are provided, or if neither is given and
                ``lcs_tree.conifold.conifold_curve`` is unavailable.
        """
        h = self.h12

        if conifold_curve is not None and conifold_index is not None:
            raise ValueError(
                "Specify at most one of conifold_curve or conifold_index."
            )

        if conifold_curve is None and conifold_index is None:
            cf = getattr(self.lcs_tree, 'conifold', None)
            if cf is not None and cf.conifold_curve is not None:
                conifold_curve = np.asarray(cf.conifold_curve)
            else:
                raise ValueError(
                    "No conifold_curve or conifold_index given and "
                    "lcs_tree.conifold.conifold_curve is not available."
                )

        if conifold_index is not None:
            c = np.zeros(h, dtype=int)
            c[conifold_index] = 1
        else:
            c = np.asarray(conifold_curve, dtype=int).ravel()
            if c.shape[0] != h:
                raise ValueError(
                    f"conifold_curve has length {c.shape[0]}, expected h12={h}."
                )

        n = 2 * h + 2
        T = np.eye(n, dtype=int)
        # F_a rows (1..h), z^b columns (h+2..2h+1): += c_a c_b
        T[1:h + 1, h + 2:n] += np.outer(c, c)
        return T

    def verify_monodromy(self, b: int, z=None, tol: float = 1e-8) -> dict:
        r"""
        **Description:**
        Numerically verify the monodromy matrix by checking
        :math:`T_b \cdot \Pi(z) = \Pi(z + e_b)`.

        Args:
            b (int): Direction index for the shift.
            z (Array, optional): Test point. If ``None``, a random point is generated.
            tol (float): Tolerance for the check.

        Returns:
            dict: Dictionary with keys ``'T_b'``, ``'max_error'``, ``'passed'``.
        """
        h = self.h12
        if z is None:
            rng = np.random.default_rng(123)
            z = jnp.array(rng.uniform(-0.3, 0.3, h) + 1j * rng.uniform(2, 5, h))

        T_b = self.monodromy_matrix_single(b)

        Pi = np.array(self.period_vector(z))
        z_shifted = z.at[b].add(1.0)
        Pi_shifted = np.array(self.period_vector(z_shifted))

        T_Pi = T_b @ Pi
        error = np.max(np.abs(T_Pi - Pi_shifted))

        return {
            'T_b': T_b,
            'max_error': float(error),
            'passed': error < tol,
        }



"""
from .conifold_utils import F_coniLCS_bulk,F_coniLCS_series, dK_cf_bulk, F_coniLCS_exp, dF_coniLCS_exp
css.F_coniLCS_bulk = F_coniLCS_bulk 
css.F_coniLCS_series = F_coniLCS_series 
css.dK_cf_bulk = dK_cf_bulk
css.F_coniLCS_exp = F_coniLCS_exp
css.dF_coniLCS_exp = dF_coniLCS_exp
"""


from . import conifold_utils as _cu

_CONIFOLD_CSS_METHODS = (
    "F_coniLCS_bulk", "F_coniLCS_series",
    "dK_cf_bulk", "F_coniLCS_exp", "dF_coniLCS_exp",
)
for _name in _CONIFOLD_CSS_METHODS:
    setattr(css, _name, _cu._ConifoldGated(getattr(_cu, _name)))


unflatten_func = lambda aux_data, children: unflatten_func_class(aux_data, children, css)

register_pytree_node(css, flatten_func, unflatten_func)

       

