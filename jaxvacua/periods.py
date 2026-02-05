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
# This file holds functions to compute periods and derived objects in CY threefold 
# compactifications.
# ------------------------------------------------------------------------------


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
#from jax.typing import ArrayLike
from numpy.typing import ArrayLike
from jax.scipy.special import zeta
from jax.numpy import pi as Pi

os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Enable 64 bit precision
config.update("jax_enable_x64", True)

# JAXVacua custom imports
from .util import *
from .periods_LCS import periods_LCS
from .periods_coniLCS import periods_coniLCS
from .periods_coniLCSbulk import periods_coniLCSbulk



class periods(periods_LCS,periods_coniLCS,periods_coniLCSbulk):
    
    def __init__(
        self,
        h12: int | None = None,
        model_ID: int | None = None,
        model_type: str = "KS",
        moduli_space_limit: str = "LCS",
        maximum_degree: int = 0,
        mirror_cy: object | None = None,
        model_data: dict | None = None,
        instanton_data: list | None = None,
        use_cytools: bool = False,
        basis_transformation: ArrayLike | None = None,
        grading_vector: ArrayLike | None = None,
        ncf: int | None = None,
        conifold_curve: object | None = None,
        period_input: Callable | None = None,
        prepotential_input: Callable | None = None,
        prange: int = 500,
        use_gvs: bool = False,
        save_file: bool = False,
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
            moduli_space_limit (string): String identifying the type of periods to be considered. Currently, only ``"LCS"`` is available.
            model_data (dictionary): Contains model data like triple intersection numbers etc.
            instanton_data (list): List of GV and GW invariants.
            maximum_degree (int): Maximum degree used for the instanton sum.
            use_cytools (boolean): Whether or not to use CYTools to compute topological data of Calabi-Yau.
            mirror_cy (cytools.CalabiYau): Mirror Calabi-Yau threefold.
            basis_transformation (ArrayLike): Basis transformation to be applied to topological data of Calabi-Yau.
            grading_vector (ArrayLike): Grading vector to be used for the GV computation.
            period_input (function): Input for periods.
            prepotential_input (function): Input for prepotential.
            save_file (bool, optional): Save files for new models. Defaults to ``False``.
            **kwargs: Extra inputs.

        Attributes:
            h12 (int): :math:`h^{1,2}(X,\mathbb{Z})` of the CY :math:`X`.
            model_type (str): Model type. Defaults to ``None``. Currently, ``"KS"`` and ``"CICY"`` are available.
            moduli_space_limit (str): Moduli space limit. Defaults to ``None``. Currently, only ``"LCS"`` is available.
            model_ID (int): Internal model ID. Defaults to ``None``.
            dimension_H3 (int): Dimension of :math:`H^{3}(X,\mathbb{Z})`.
            _dimension_H3_tot (int): Dimension of :math:`H^{3}(X,\mathbb{Z})` plus dimension of :math:`H_{3}(X,\mathbb{Z})`.
            period_input (object): Input period function.
            prepotential_input (object): Input prepotential function.
            

        """
        
        # -----------------------------------------------------------------------------------
        # Testing inputs...
        ## Check correct model types
        model_types=[None,"KS","CICY"]
        if model_type not in model_types:
            raise ValueError(f"Model type must be one of {model_types}!")
            
        
        # DONE
        # -----------------------------------------------------------------------------------


        # -----------------------------------------------------------------------------------
        # Initialisation of class
        ## Setting general attributes from input
        self.model_type = model_type
        self.moduli_space_limit = moduli_space_limit
        self.model_ID = model_ID
        self.period_input = period_input
        self.prepotential_input = prepotential_input

        if h12 is not None:
            self.h12 = h12
        else:
            if use_cytools:
                self.h12 = mirror_cy.h11()
            else:
                if model_type=="KS":
                    raise ValueError("For `model_type=KS`, the number of moduli should be provided explicitly!")
            
        
        ## Defining class based on model_type
        if self.moduli_space_limit!="LCS":
            file_KS = None
            file_KS_inst = None
        else:
            if use_cytools:
                file_KS = None
                file_KS_inst = None
            else:
                if self.model_ID is None:
                    if model_data is None:
                        raise ValueError(f"Need to provide model_ID if grabbing model_data and instanton_data from files!")

            
        ## Grabbing model data     
        if self.moduli_space_limit=="LCS":

            periods_LCS.__init__(self, h12=h12, model_ID = model_ID,model_type = model_type, maximum_degree = maximum_degree, mirror_cy = mirror_cy,
                                model_data = model_data,instanton_data = instanton_data,use_cytools = use_cytools,
                                basis_transformation = basis_transformation,grading_vector = grading_vector, prange = prange, use_gvs = use_gvs,save_file=save_file,**kwargs)

            class_attr_rm = [periods_coniLCS,periods_coniLCSbulk]
        elif self.moduli_space_limit=="coniLCS":

            periods_coniLCS.__init__(self, h12=h12, model_ID = model_ID,model_type = model_type, maximum_degree = maximum_degree, mirror_cy = mirror_cy,
                                model_data = model_data,instanton_data = instanton_data,use_cytools = use_cytools, ncf=ncf,conifold_curve=conifold_curve,
                                basis_transformation = basis_transformation,grading_vector = grading_vector, prange = prange, use_gvs = use_gvs,save_file=save_file,**kwargs)

            class_attr_rm = [periods_LCS,periods_coniLCSbulk]
        elif self.moduli_space_limit=="coniLCSbulk":

            periods_coniLCSbulk.__init__(self, h12=h12, model_ID = model_ID,model_type = model_type, maximum_degree = maximum_degree, mirror_cy = mirror_cy,
                                model_data = model_data,instanton_data = instanton_data,use_cytools = use_cytools, ncf=ncf,conifold_curve=conifold_curve,
                                basis_transformation = basis_transformation,grading_vector = grading_vector, prange = prange, use_gvs = use_gvs,save_file=save_file,**kwargs)
            
            # Need to keep LCS attributes!
            class_attr_rm = [periods_coniLCS]

        else:
            
            #raise NotImplementedError("Implementation for other moduli space limits not yet implemented!")
            warnings.warn("Implementation for other moduli space limits not yet tested!")

        
        """
        for class_attr in class_attr_rm:
            periods_attributes = [x for x in dir(class_attr) if "__" not in x]
            for attr in periods_attributes:
                delattr(class_attr, str(attr))
        """
        
        self.dimension_H3 = self.h12 + 1 #Dimension of the 3rd cohomology group

        self._dimension_H3_tot = 2*(self.h12 + 1)

        try:
            if self.maximum_degree>0:
                if self.use_gvs==False:
                    if len(self.GW_inv_lim)==1 and self.GW_inv_lim[0]==0:
                        print("GW invariants not available. Use GVs instead for computations!")
                        self.use_gvs = True
        except:
            self.Q = None
            self.name = ""
            pass

        if 'amatrix' in kwargs:
            self.a_matrix = kwargs["amatrix"]

        ## Check moduli space limit
        moduli_space_limits = ["LCS","coniLCS","coniLCSbulk"]
        if moduli_space_limit not in moduli_space_limits:
            ### Test input for 
            if period_input is None and prepotential_input is None:
                raise ValueError("Need to provide input for periods or prepotential.\
                                    Currently, only LCS is implemented as moduli space limit!")
            else:
                #raise NotImplementedError("Implementation for general input periods not completed!")
                warnings.warn("Implementation for general input periods not tested!")
                
            ### Raise warning if both prepotential and periods are being provided as inputs...
            if period_input is not None and prepotential_input is not None:
                warnings.warn("If both periods and prepotential are provided,\
                                please make sure input is consistent. \
                                Only the provided period vector \
                                will be used for the computation!")

            if period_input is not None:
            
                z = np.random.uniform(-1,1,self.h12+1)+1j*np.random.uniform(-1,1,self.h12+1)
                
                try:
                    period_output = period_input(z)
                except:
                    raise ValueError("Failed to evaluate `period_input`. Please check input function!")
                
                if period_output.shape != ((self.h12+1)*2,):
                    raise ValueError(f"Wrong output shape for period.\
                            Output shape is {period_output.shape}, but should be {((self.h12+1)*2,)}")

            if prepotential_input is not None:
            
                z = np.random.uniform(-1,1,self.h12+1)+1j*np.random.uniform(-1,1,self.h12+1)
                z = jnp.array(z)

                try:
                    prepotential_output = prepotential_input(z)
                except:
                    raise ValueError("Failed to evaluate `period_input`. Please check input function!")
                    

                if prepotential_output.shape!=():
                    
                    raise ValueError(f"Wrong output type for prepotential.\
                            Output type is {type(prepotential_output)}, but should be {np.complex128}")

        # DONE
        # -----------------------------------------------------------------------------------

    def __repr__(self) -> str:
    
        return f"Period calculations for h12={self.h12}."

    # Symplectic Matrix
    @partial(jit, static_argnums = (0,))
    def sigma(self) -> Array:
        r"""
        
        **Description:**
        Returns the symplectic matrix
        
        .. math::
            \Sigma=\left (\begin{array}{cc}0 & 1\\ -1&0\end{array}\right )\, .
                
        Args:
            None
        
        Returns:
            ArrayLike: Symplectic pairing matrix.
        
        """

        Block1=-jnp.identity(2*(self.h12+1),dtype=jnp.int32)[:self.h12+1]
        Block2=jnp.identity(2*(self.h12+1),dtype=jnp.int32)[self.h12+1:]

        return jnp.concatenate((Block2,Block1))
    
    
    ###################################################################################################################################
    ####################################### PREPOTENTIAL ETC. AS FUNCTIONS OF PERIODS #################################################
    ###################################################################################################################################


    @partial(jit, static_argnums = (0,2,))
    def prepot_per(self, XPer: ArrayLike, conj: bool = False) -> complex:
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
            global parameter ``self.moduli_space_limit``. Currently, only the limits ``"LCS"``, ``"coniLCS"``, and ``"coniLCSbulk"``
            are supported.

        .. note::
            This function is used to compute the gauge kinetic matrix :math:`\mathcal{N}_{IJ}`
            of second derivatives of the prepotential which is used to check the ISD-condition for 
            flux vacua.

        Args:
            XPer (ArrayLike): Values of periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.

        Returns:
            complex: Value of the prepotential :math:`F`.

        See also: :func:`F_LCS_per`

        See also: :func:`prepot_grad_grad_per`

        See also: :func:`gauge_kinetic_matrix`

        """
        
        if self.moduli_space_limit == "LCS":
            return self.F_LCS_per(XPer,conj=conj)
        elif self.moduli_space_limit == "coniLCS":
            return self.F_coniLCS_per(XPer,conj=conj)
        elif self.moduli_space_limit == "coniLCSbulk":
            return self.F_coniLCSbulk_per(XPer,conj=conj)
        elif self.prepotential_input is not None:
            return self.prepotential_input(XPer,conj=conj)
        else:
            raise ValueError("Prepotential undefined! If no input was provided, use one of the pre-implemented methods!")
    
    
    @partial(jit, static_argnums = (0,2,))
    def prepot_grad_per(self, XPer: ArrayLike, conj: bool = False) -> Array:
        r"""

        **Description:**
        Computes the holomorphic derivatives :math:`F_{I}=\partial_{X^I}F` of the prepotential :math:`F`
        with respect to the periods :math:`X^I`.

        Args:
            XPer (ArrayLike): Values of periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.

        Returns:
            ArrayLike: Holomorphic derivatives :math:`F_{I}=\partial_{X^I}F` of the prepotential :math:`F`.

        """

        #Derivative of prepotential:
        return jax.grad(self.prepot_per,holomorphic=True)(XPer,conj=conj)

    @partial(jit, static_argnums = (0,2,))
    def prepot_grad_grad_per(self, XPer: ArrayLike, conj: bool = False) -> Array:
        r"""

        **Description:**
        Computes the second holomorphic derivatives :math:`F_{IJ}=\partial_{X^I}\partial_{X^J}F` of the prepotential
        :math:`F` with respect to the periods :math:`X^I`.

        .. note::

            This matrix is used among others to compute the gauge kinetic matrix entering the ISD condition
            for SUSY flux vacua, see :func:`gauge_kinetic_matrix_prepotential`.

        Args:
            XPer (ArrayLike): Values of periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.

        Returns:
            ddF (jax array of arrays): :math:`(h^{1,2}+1)\times(h^{1,2}+1)`-matrix with entries corresponding to 2nd derivatives of :math:`F`.

        See also: :func:`gauge_kinetic_matrix`.

        """

        #Second derivative of prepotential:
        return jax.jacrev(self.prepot_grad_per,holomorphic=True)(XPer,conj=conj)

    @partial(jit, static_argnums = (0,2,))
    def period_vector_per(self, XPer: ArrayLike, conj: bool = False) -> Array:
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
            XPer (ArrayLike): Values of periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.

        Returns:
            ArrayLike: Period vector :math:`\Pi`.

        """
        
        if self.period_input is not None:
            return self.period_input(XPer,conj=conj)
        elif self.prepotential_input is not None or self.moduli_space_limit in ["LCS","coniLCS","coniLCSbulk"]:
            return jnp.concatenate((self.prepot_grad_per(XPer,conj=conj), XPer))
        else:
            raise ValueError("Period vector undefined! If no input was provided, use one of the pre-implemented methods!")

    @partial(jit, static_argnums = (0,))
    def A_per(self, XPer: ArrayLike, cXPer: ArrayLike) -> complex:
        r"""

        **Description:**
        Computes the value of the mirror CY volume :math:`\tilde{\mathcal{V}}` as a function of the periods :math:`X^I`.

        .. admonition:: Details
            :class: dropdown

            The mirror CY volume :math:`\tilde{\mathcal{V}}` is computed from the period vector :math:`\Pi` as
            
            .. math::
                \tilde{\mathcal{V}} = -\text{i}\, \Pi^\dagger\cdot \Sigma\cdot\Pi\, .

        Args:
            XPer (ArrayLike): Values of periods.
            cXPer (ArrayLike): Complex conjugate values of periods.

        Returns:
            ArrayLike: Value of the mirror CY volume in terms of the period :math:`X^I`.

        """

        return -1.j*jnp.matmul(self.period_vector_per(cXPer,conj=True), jnp.matmul(self.sigma(), self.period_vector_per(XPer)))
    
    @partial(jit, static_argnums = (0,))
    def kahler_potential_per(self, XPer: ArrayLike, cXPer: ArrayLike) -> complex:
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
            XPer (ArrayLike): Values of periods.
            cXPer (ArrayLike): Complex conjugate values of periods.

        Returns:
            ArrayLike: Value of the Kähler potential :math:`K`.

        """

        return -jnp.log(self.A_per(XPer,cXPer))



    
    @partial(jit, static_argnums = (0,2,))
    def grad_period_vector_per(self, XPer: ArrayLike, conj: bool = False) -> Array:
        r"""

        **Description:**
        Computes derivatives :math:`\partial_{X^I}\Pi` of the period vector :math:`\Pi` with respect to the
        periods :mathh`X^I`.

        Args:
            XPer (ArrayLike): Values of periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.

        Returns:
            ArrayLike: Derivatives :math:`\partial_{X^I}\Pi` of the period vector :math:`\Pi`.

        """
        
        return jax.jacrev(self.period_vector_per,argnums=0,holomorphic=True)(XPer,conj=conj)
    
    
    
    @partial(jit, static_argnums = (0,3,))
    def grad_kahler_potential_per(self, XPer: ArrayLike, cXPer: ArrayLike, conj: bool = False) -> Array:
        r"""

        **Description:**
        Computes derivatives :math:`\partial_{X^I}K` of the Kähler potential :math:`K` with respect to the
        periods :mathh`X^I`.

        .. warning::
            If we set ``conj=True``, we compute the anti-holomorphic derivative 
            :math:`\partial_{\overline{X}^I}K` instead!

        Args:
            XPer (ArrayLike): Values of periods.
            cXPer (ArrayLike): Complex conjugate values of periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.

        Returns:
            ArrayLike: First derivative :math:`\partial_{X^I}K` of the Kähler potential :math:`K`.

        """

        if not conj:
            return jax.grad(self.kahler_potential_per,holomorphic=True,argnums=0)(XPer,cXPer)
        else:
            return jax.grad(self.kahler_potential_per,holomorphic=True,argnums=1)(XPer,cXPer)

    
    @partial(jit, static_argnums = (0,3,))
    def D_period_vector_per(self, XPer: ArrayLike, cXPer: ArrayLike, conj: bool = False) -> Array:
        r"""

        **Description:**
        Computes the Kähler convariant derivative :math:`D_I\Pi`. of the period vector :math:`\Pi`.

        Args:
            XPer (ArrayLike): Values of periods.
            cXPer (ArrayLike): Complex conjugate values of periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
            
        Returns:
            ArrayLike: Values of :math:`D_I\Pi`.

        """
        
        dK = self.grad_kahler_potential_per(XPer,cXPer,conj=conj)

        if not conj:
            dPi = self.grad_period_vector_per(XPer,conj=conj)
            return dPi+jnp.outer(self.period_vector_per(XPer,conj=conj),dK)
        else:
            dPi = self.grad_period_vector_per(cXPer,conj=conj)
            return dPi+jnp.outer(self.period_vector_per(cXPer,conj=conj),dK)

    
    
    @partial(jit, static_argnums = (0,3,))
    def P_per(self, XPer: ArrayLike, cXPer: ArrayLike, conj: bool = False) -> Array:
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
            XPer (ArrayLike): Values of periods.
            cXPer (ArrayLike): Complex conjugate values of periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
            
        Returns:
            ArrayLike: Values of :math:`P_{IJ}`.

        """
        

        # see https://arxiv.org/abs/2310.06040
        if not conj:
            
            PPi = self.period_vector_per(XPer,conj=False)[:self.h12+1]
            DPi = self.D_period_vector_per(XPer,cXPer,conj=True).T[:,:self.h12+1]
        else:

            PPi = self.period_vector_per(cXPer,conj=True)[:self.h12+1]
            DPi = self.D_period_vector_per(XPer,cXPer,conj=False).T[:,:self.h12+1]
        
        return jnp.append(jnp.array([PPi]),DPi[1:],axis=0)
    
    @partial(jit, static_argnums = (0,3,))
    def Q_per(self, XPer: ArrayLike, cXPer: ArrayLike, conj: bool = False) -> Array:
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
            XPer (ArrayLike): Values of periods.
            cXPer (ArrayLike): Complex conjugate values of periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
            
        Returns:
            ArrayLike: Values of :math:`Q^{I}\,_{J}`.

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
    
    @partial(jit, static_argnums = (0,3,))
    def Q_inv_per(self, XPer: ArrayLike, cXPer: ArrayLike, conj: bool = False) -> Array:
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
            XPer (ArrayLike): Values of periods.
            cXPer (ArrayLike): Complex conjugate values of periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
            
        Returns:
            ArrayLike: Values of :math:`(Q^{-1})^{I}\,_{J}`.

        """

        # see https://arxiv.org/abs/2310.06040
        return jnp.linalg.inv(self.Q_per(XPer,cXPer,conj=conj))

    @partial(jit, static_argnums = (0,3,))
    def PQ_per(self, XPer: ArrayLike, cXPer: ArrayLike, conj: bool = False) -> Tuple[Array,Array]:
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
            XPer (ArrayLike): Values of periods.
            cXPer (ArrayLike): Complex conjugate values of periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
            
        Returns:
            ArrayLike: Values of :math:`P_{IJ}`.
            ArrayLike: Values of :math:`Q^{I}\,_{J}`.

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
    
    @partial(jit, static_argnums = (0,3,))
    def gauge_kinetic_matrix_prepotential(self, XPer: ArrayLike, cXPer: ArrayLike, conj: bool = False) -> Array:
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
            XPer (ArrayLike): Values of periods.
            cXPer (ArrayLike): Complex conjugate values of periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.

        Returns:
            ArrayLike: Value of the gauge kinetic matrix :math:`\mathcal{N}` from prepotential.

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


    @partial(jit, static_argnums = (0,3,))
    def gauge_kinetic_matrix_periods(self, XPer: ArrayLike, cXPer: ArrayLike, conj: bool = False) -> Array:
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
            XPer (ArrayLike): Values of periods.
            cXPer (ArrayLike): Complex conjugate values of periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
            
        Returns:
            ArrayLike: Value of the gauge kinetic matrix :math:`\mathcal{N}` from periods.

        """

        P,Q = self.PQ_per(XPer,cXPer,conj=conj)

        Qinv = jnp.linalg.inv(Q)
        
        return jnp.matmul(Qinv,P)
    
    
    
    
    @partial(jit, static_argnums = (0,3,))
    def gauge_kinetic_matrix(self, XPer: ArrayLike, cXPer: ArrayLike, conj: bool = False) -> Array:
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
            XPer (ArrayLike): Values of periods.
            cXPer (ArrayLike): Complex conjugate values of periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.

        Returns:
            ArrayLike: Value of the gauge kinetic matrix :math:`\mathcal{N}`.

        """

        if self.moduli_space_limit in ["LCS","coniLCS","coniLCSbulk"] or self.period_input is not None:
            return self.gauge_kinetic_matrix_periods(XPer,cXPer,conj=conj)
        elif self.prepotential_input is not None:
            return self.gauge_kinetic_matrix_prepotential(XPer,cXPer,conj=conj)
        else:
            raise ValueError("Could not compute gauge kinetic matrix! Please check input!")

    N = gauge_kinetic_matrix

    @partial(jit, static_argnums = (0,3,))
    def dN(self, XPer: ArrayLike, cXPer: ArrayLike, conj: bool = False) -> Array:
        r"""

        **Description:**
        Returns the holomorphic derivative :math:`\partial_{X^I}\mathcal{N}` of the 
        gauge kinetic matrix :math:`\mathcal{N}` with respect to the periods :math:`X^I`.


        Args:
            XPer (ArrayLike): JAX array of shape (:math:`h^{1,2}+1`, ) containing the value of the periods.
            cXPer (ArrayLike): JAX array of shape (:math:`h^{1,2}+1`, ) containing the complex conjugate value of the periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.

        Returns:
            ArrayLike: Holomorphic derivative :math:`\partial_{X^I}\mathcal{N}` of the 
                gauge kinetic matrix :math:`\mathcal{N}`.

        """
        
        return jax.jacrev(self.gauge_kinetic_matrix,holomorphic=True,argnums=0)(XPer,cXPer,conj=conj)

    @partial(jit, static_argnums = (0,3,))
    def dN_c(self, XPer: ArrayLike, cXPer: ArrayLike, conj: bool = False) -> Array:
        r"""

        **Description:**
        Returns the anti-holomorphic derivative :math:`\partial_{\overline{X}^I}\mathcal{N}` of the 
        gauge kinetic matrix :math:`\mathcal{N}` with respect to the 
        complex conjugate periods :math:`\overline{X}^I`.

        Args:
            XPer (ArrayLike): JAX array of shape (:math:`h^{1,2}+1`, ) containing the value of the periods.
            cXPer (ArrayLike): JAX array of shape (:math:`h^{1,2}+1`, ) containing the complex conjugate value of the periods.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
            
        Returns:
            ArrayLike: Anti-holomorphic derivative :math:`\partial_{\overline{X}^I}\mathcal{N}` of the 
                gauge kinetic matrix :math:`\mathcal{N}`.

        """
        
        return jax.jacrev(self.gauge_kinetic_matrix,holomorphic=True,argnums=1)(XPer,cXPer,conj=conj)

    ##########################################################################################
    ###################################### ISD MATRIX ########################################
    ##########################################################################################

    @partial(jit, static_argnums = (0,))
    def ISD_matrix(self, XPer: ArrayLike, cXPer: ArrayLike) -> Array:
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
            XPer (ArrayLike): Values of periods.
            cXPer (ArrayLike): Complex conjugate values of periods.

        Returns:
            ArrayLike: Value of the ISD-matrix :math:`\mathcal{M}`.

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

    @partial(jit, static_argnums = (0,))
    def dM(self, XPer: ArrayLike, cXPer: ArrayLike) -> Array:
        r"""

        **Description:**
        Returns the holomorphic derivative :math:`\partial_{X^I}\mathcal{M}` of the 
        ISD-matrix :math:`\mathcal{M}` with respect to the periods :math:`X^I`.

        Args:
            XPer (ArrayLike): Values of periods.
            cXPer (ArrayLike): Complex conjugate values of periods.
            
        Returns:
            ArrayLike: Holomorphic derivative :math:`\partial_{X^I}\mathcal{M}` of the 
                ISD-matrix :math:`\mathcal{M}`.

        """

        return jax.jacrev(self.M,holomorphic=True,argnums=0)(XPer,cXPer)

    @partial(jit, static_argnums = (0,))
    def dM_c(self, XPer: ArrayLike, cXPer: ArrayLike) -> Array:
        r"""

        **Description:**
        Returns the anti-holomorphic derivative :math:`\partial_{\overline{X}^I}\mathcal{M}` of the 
        ISD-matrix :math:`\mathcal{M}` with respect to
        the complex conjugate periods :math:`\overline{X}^I`.

        Args:
            XPer (ArrayLike): Values of periods.
            cXPer (ArrayLike): Complex conjugate values of periods.
            
        Returns:
            ArrayLike: Anti-holomorphic derivative :math:`\partial_{\overline{X}^I}\mathcal{M}` of the 
                ISD-matrix :math:`\mathcal{M}`.

        """

        return jax.jacrev(self.M,holomorphic=True,argnums=1)(XPer,cXPer)

    
        


