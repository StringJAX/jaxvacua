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
#from jax.typing import ArrayLike
from numpy.typing import ArrayLike
from jax.scipy.special import zeta
from jax.numpy import pi as Pi

# Set CPU environment
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Enable 64 bit precision
config.update("jax_enable_x64", True)


# JAXVacua custom imports
from .util import *
from .periods import periods
from .css_LCS import css_LCS
from .css_coniLCS import css_coniLCS
from .css_coniLCSbulk import css_coniLCSbulk

class css(css_LCS,css_coniLCS,css_coniLCSbulk):
    
    
    def __init__(self, h12=None,model_ID = None,model_type = "KS",moduli_space_limit = "LCS",
                 maximum_degree = 0,mirror_cy = None,model_data = None,instanton_data = None,
                 use_cytools = False,basis_transformation = None,ncf=None,conifold_curve=None,
                 grading_vector = None, period_input = None, prepotential_input = None,
                 gauge_choice = 1.+0.*1j, prange = 500, use_gvs = False,save_file=False, **kwargs):
        r"""
        **Description:**
        This class defines class for the complex structure sector in Type IIB orientifold compactifications. 
        It inherits from the classes :class:`css_LCS`, :class:`css_coniLCS` and :class:`css_coniLCSbulk`
        depending on the moduli space limit considered. It also contains a class object of type
        :class:`jaxvacua.periods.periods` to compute periods, the prepotential, the gauge kinetic matrix etc.
        
        
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
            dimension_H3 (int): Dimension of :math:`H^{3}(X,\mathbb{Z})`.
            _dimension_H3_tot (int): Dimension of :math:`H^{3}(X,\mathbb{Z})` plus dimension of :math:`H_{3}(X,\mathbb{Z})`.
            gauge_choice (complex): Choice of gauge for projective coordinates.
            gauge_choice_conj (complex): Choice of gauge for conjugate projective coordinates.
            h12 (int): Hodge number h12 of the Calabi-Yau setting the number of complex structure moduli.
            
        
        """
        
        # -----------------------------------------------------------------------------------
        # Initialise model
        
        

        self.periods = periods(h12=h12,model_ID = model_ID,model_type =model_type,moduli_space_limit = moduli_space_limit,
                    maximum_degree = maximum_degree,mirror_cy = mirror_cy,model_data = model_data,instanton_data = instanton_data,
                    use_cytools = use_cytools,basis_transformation = basis_transformation,conifold_curve=conifold_curve,ncf=ncf,grading_vector = grading_vector, 
                    period_input = period_input, prepotential_input = prepotential_input,prange = prange, use_gvs = use_gvs,save_file=save_file, **kwargs)

        if moduli_space_limit == "LCS":

            css_LCS.__init__(self, prange = prange, use_gvs = use_gvs, **kwargs)

            class_attr_rm = [css_coniLCS,css_coniLCSbulk]

        elif moduli_space_limit == "coniLCS":

            css_coniLCS.__init__(self, prange = prange, use_gvs = use_gvs, **kwargs)

            class_attr_rm = [css_LCS,css_coniLCSbulk]

        elif moduli_space_limit == "coniLCSbulk":

            css_coniLCSbulk.__init__(self, prange = prange, use_gvs = use_gvs, **kwargs)

            # Need to keep LCS attributes!
            class_attr_rm = [css_coniLCS]

        #else:    

            #raise NotImplementedError(f"Requested moduli space limit {moduli_space_limit} not implemented!")
            #warnings.warn(f"Requested moduli space limit {moduli_space_limit} not implemented!")

        """
        for class_attr in class_attr_rm:
            csm_attributes = [x for x in dir(class_attr) if "__" not in x]
            for attr in csm_attributes:
                delattr(class_attr, str(attr))
        """

        self.gauge_choice = gauge_choice
        self.gauge_choice_conj = jnp.conj(gauge_choice)
        self.h12 = self.periods.h12
        self.dimension_H3 = self.periods.dimension_H3
        self._dimension_H3_tot = self.periods._dimension_H3_tot

        # DONE
        # -----------------------------------------------------------------------------------
        
        
    def __repr__(self) -> str:
        r"""
        **Description:**
        Class object description.
        
        Returns:
            str: Class object description.
        """
        return f"Complex structure sector for h12={self.h12} complex structure moduli in {self.periods.moduli_space_limit}-limit."
    
    ###################################################################################################################################
    ####################################### GAUGE FIXING FUCTIONS FROM PERIOD CALCULATION #############################################
    ###################################################################################################################################
        
    @partial(jit, static_argnums = (0,2,))
    def moduli_to_periods(
                        self, 
                        moduli: ArrayLike, 
                        conj: bool = False
                        ) -> Array:
        r"""
        **Description:**
        Transforms complex structure moduli to periods for the global choice of gauge.
        
        Args:
            moduli (ArrayLike): Values of the complex structure moduli.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Value of the periods.

        """

        val = jnp.concat((jnp.ones(1),moduli))

        if conj:
            return val*self.gauge_choice_conj
        else:
            return val*self.gauge_choice


    @partial(jit, static_argnums = (0,))
    def periods_to_moduli(
                        self, 
                        XPer: ArrayLike
                        ) -> Array:
        r"""

        **Description:**
        Transforms periods to complex structure moduli.
        
        Args:
            XPer (ArrayLike): Values of the periods.
        
        Returns:
            ArrayLike: Value of the complex structure moduli.

        """

        return Xper[1:]/X[0]

        
    
    @partial(jit, static_argnums = (0,3,))
    def gauge_kinetic_matrix(
                            self, 
                            moduli: ArrayLike, 
                            moduli_c: ArrayLike, 
                            conj: bool = False
                            ) -> Array:
        r"""

        **Description:**
        Computes the value of the gauge kinetic matrix :math:`\mathcal{N}`.

        .. note::

            This function descends from :func:`jaxvacua.periods.periods.gauge_kinetic_matrix`
            upong gauge fixing, i.e., making a choice of  homogeneous complex coordinates 
            on the complex structure moduli space of :math:`X`.
        
        Args:
            moduli (ArrayLike): Values of the complex structure moduli.
            moduli_c (ArrayLike): Complex conjugate values of the complex structure moduli.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Value of the gauge kinetic matrix :math:`\mathcal{N}`.

        Aliases:
            :func:`N`

        See also: :func:`jaxvacua.periods.periods.gauge_kinetic_matrix`

        """

        return self.periods.gauge_kinetic_matrix(self.moduli_to_periods(moduli),self.moduli_to_periods(moduli_c,conj=True),conj=conj)
    
    N = gauge_kinetic_matrix
    
    @partial(jit, static_argnums = (0,))
    def ISD_matrix(
                self, 
                moduli: ArrayLike, 
                moduli_c: ArrayLike
                ) -> Array:
        r"""

        **Description:**
        Computes the value of the ISD-matrix :math:`\mathcal{M}`.

        .. note::

            This function descends from :func:`jaxvacua.periods.periods.ISD_matrix`
            upong gauge fixing, i.e., making a choice of  homogeneous complex coordinates 
            on the complex structure moduli space of :math:`X`.
        
        Args:
            moduli (ArrayLike): Values of the complex structure moduli.
            moduli_c (ArrayLike): Complex conjugate values of the complex structure moduli.
        
        Returns:
            ArrayLike: Value of the ISD-matrix :math:`\mathcal{M}`.

        Aliases:
            :func:`M`

        See also: :func:`jaxvacua.periods.periods.ISD_matrix`

        """
        return self.periods.ISD_matrix(self.moduli_to_periods(moduli),self.moduli_to_periods(moduli_c,conj=True))
    
    M = ISD_matrix

    @partial(jit, static_argnums = (0,3,))
    def dN_X(
            self, 
            moduli: ArrayLike, 
            moduli_c: ArrayLike, 
            conj: bool = False
            ) -> Array:
        r"""

        **Description:**
        Returns the holomorphic derivative :math:`\partial_{X^I}\mathcal{N}` of the 
        gauge kinetic matrix :math:`\mathcal{N}` with respect to the periods :math:`X^I`.

        .. note::

            This function descends from :func:`jaxvacua.periods.periods.dN`
            upong gauge fixing, i.e., making a choice of  homogeneous complex coordinates 
            on the complex structure moduli space of :math:`X`.
        
        Args:
            moduli (ArrayLike): Values of the complex structure moduli.
            moduli_c (ArrayLike): Complex conjugate values of the complex structure moduli.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Holomorphic derivative :math:`\partial_{X^I}\mathcal{N}` of the 
                gauge kinetic matrix :math:`\mathcal{N}`.

        """

        return self.periods.dN(self.moduli_to_periods(moduli),self.moduli_to_periods(moduli_c,conj=True),conj=conj)
    
    @partial(jit, static_argnums = (0,3,))
    def dN(
            self, 
            moduli: ArrayLike, 
            moduli_c: ArrayLike, 
            conj: bool = False
            ) -> Array:
        r"""

        **Description:**
        Returns the holomorphic derivative :math:`\partial_{z^i}\mathcal{N}` of the 
        gauge kinetic matrix :math:`\mathcal{N}` with respect to the 
        complex structure moduli :math:`z^i`.
        
        
        Args:
            moduli (ArrayLike): Values of the complex structure moduli.
            moduli_c (ArrayLike): Complex conjugate values of the complex structure moduli.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Holomorphic derivative :math:`\partial_{z^i}\mathcal{N}` of the 
                gauge kinetic matrix :math:`\mathcal{N}`.

        """

        return self.dN_X(moduli,moduli_c,conj=conj)[:,:,1:]


    @partial(jit, static_argnums = (0,3,))
    def dN_cX(
            self, 
            moduli: ArrayLike, 
            moduli_c: ArrayLike, 
            conj: bool = False
            ) -> Array:
        r"""

        **Description:**
        Returns the anti-holomorphic derivative :math:`\partial_{\overline{X}^I}\mathcal{N}` of the 
        gauge kinetic matrix :math:`\mathcal{N}` with respect to the 
        complex conjugate periods :math:`\overline{X}^I`.

        .. note::

            This function descends from :func:`jaxvacua.periods.periods.dN_c`
            upong gauge fixing, i.e., making a choice of  homogeneous complex coordinates 
            on the complex structure moduli space of :math:`X`.
        
        Args:
            moduli (ArrayLike): Values of the complex structure moduli.
            moduli_c (ArrayLike): Complex conjugate values of the complex structure moduli.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Anti-holomorphic derivative :math:`\partial_{\overline{X}^I}\mathcal{N}` of the 
                gauge kinetic matrix :math:`\mathcal{N}`.

        """

        return self.periods.dN_c(self.moduli_to_periods(moduli),self.moduli_to_periods(moduli_c,conj=True),conj=conj)
    
    @partial(jit, static_argnums = (0,3,))
    def dN_c(
            self, 
            moduli: ArrayLike, 
            moduli_c: ArrayLike, 
            conj: bool = False
            ) -> Array:
        r"""

        **Description:**
        Returns the anti-holomorphic derivative :math:`\partial_{\overline{z}^i}\mathcal{N}` of the 
        gauge kinetic matrix :math:`\mathcal{N}` with respect to the complex conjugate 
        complex structure moduli :math:`\overline{z}^i`.
        
        Args:
            moduli (ArrayLike): Values of the complex structure moduli.
            moduli_c (ArrayLike): Complex conjugate values of the complex structure moduli.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Anti-holomorphic derivative :math:`\partial_{\overline{z}^i}\mathcal{N}` of the 
                gauge kinetic matrix :math:`\mathcal{N}`.

        """

        return self.dN_cX(moduli,moduli_c,conj=conj)[:,:,1:]


    @partial(jit, static_argnums = (0,))
    def dM_X(
            self, 
            moduli: ArrayLike, 
            moduli_c: ArrayLike
            ) -> Array:
        r"""

        **Description:**
        Returns the holomorphic derivative :math:`\partial_{X^I}\mathcal{M}` of the 
        ISD-matrix :math:`\mathcal{M}` with respect to the periods :math:`X^I`.

        .. note::

            This function descends from :func:`jaxvacua.periods.periods.dM`
            upong gauge fixing, i.e., making a choice of  homogeneous complex coordinates 
            on the complex structure moduli space of :math:`X`.
        
        Args:
            moduli (ArrayLike): Values of the complex structure moduli.
            moduli_c (ArrayLike): Complex conjugate values of the complex structure moduli.

        Returns:
            ArrayLike: Holomorphic derivative :math:`\partial_{X^I}\mathcal{M}` of the 
                ISD-matrix :math:`\mathcal{M}`.

        """

        return self.periods.dM(self.moduli_to_periods(moduli),self.moduli_to_periods(moduli_c,conj=True))

    @partial(jit, static_argnums = (0,))
    def dM_cX(
            self, 
            moduli: ArrayLike, 
            moduli_c: ArrayLike
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
            moduli (ArrayLike): Values of the complex structure moduli.
            moduli_c (ArrayLike): Complex conjugate values of the complex structure moduli.

        Returns:
            ArrayLike: Anti-holomorphic derivative :math:`\partial_{\overline{X}^I}\mathcal{M}` of the 
                ISD-matrix :math:`\mathcal{M}`.

        """
        
        return self.periods.dM_c(self.moduli_to_periods(moduli),self.moduli_to_periods(moduli_c,conj=True))
    

    @partial(jit, static_argnums = (0,))
    def dM(
            self, 
            moduli: ArrayLike, 
            moduli_c: ArrayLike
            ) -> Array:
        r"""

        **Description:**
        Returns the holomorphic derivative :math:`\partial_{z^i}\mathcal{M}` of the 
        ISD-matrix :math:`\mathcal{M}` with respect to
        the complex structure moduli :math:`z^i`.
        
        Args:
            moduli (ArrayLike): Values of the complex structure moduli.
            moduli_c (ArrayLike): Complex conjugate values of the complex structure moduli.

        Returns:
            ArrayLike: Holomorphic derivative :math:`\partial_{z^i}\mathcal{M}` of the 
                ISD-matrix :math:`\mathcal{M}`.
        """
        
        return self.dM_X(moduli,moduli_c)[:,:,1:]

    @partial(jit, static_argnums = (0,))
    def dM_c(
            self, 
            moduli: ArrayLike, 
            moduli_c: ArrayLike
            ) -> Array:
        r"""

        **Description:**
        Returns the anti-holomorphic derivative :math:`\partial_{\overline{z}^i}\mathcal{M}` of the 
        ISD-matrix :math:`\mathcal{M}` with respect to
        the complex conjugate complex structure moduli :math:`\overline{z}^i`.
        
        Args:
            moduli (ArrayLike): Values of the complex structure moduli.
            moduli_c (ArrayLike): Complex conjugate values of the complex structure moduli.

        Returns:
            ArrayLike: Anti-holomorphic derivative :math:`\partial_{\overline{z}^i}\mathcal{M}` of the 
                ISD-matrix :math:`\mathcal{M}`.

        """
        
        return self.dM_cX(moduli,moduli_c)[:,:,1:]

    ###################################################################################################################################
    ########################################### PREPOTENTIAL, KÄHLER POTENTIAL ETC. ###################################################
    ###################################################################################################################################
    
    @partial(jit, static_argnums = (0,2,))
    def prepot(
            self, 
            moduli: ArrayLike, 
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
            The moduli space limit around which the pre-potential is computed is set by the global parameter ``self.periods.moduli_space_limit``. 
            Currently, only ``self.periods.moduli_space_limit="LCS"`` is supported.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
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
            if self.periods.moduli_space_limit == "LCS":
                return self.F_LCS(moduli,conj=conj)
            elif self.periods.moduli_space_limit == "coniLCS":
                return self.F_coniLCS(moduli,conj=conj)
            elif self.periods.moduli_space_limit == "coniLCSbulk":
                return self.F_coniLCSbulk(moduli,conj=conj)
            else:
                return ValueError("Could not identify mode for computing the prepotential in complex structure class!")
        else:
                
            mod = self.moduli_to_periods(moduli,conj=conj)

            return self.periods.prepot_per(mod,conj=conj)
    
    F = prepot
        
    @partial(jit, static_argnums = (0,2,))
    def dF(
                        self, 
                        moduli: ArrayLike, 
                        conj: bool = False
                        ) -> Array:
        r"""
        
        **Description:**
        Computes the holomorphic derivative :math:`\partial_{z^i} F` of the prepotential :math:`F`
        for given values of the moduli.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Value of the holomorphic derivatives :math:`\partial_{z^i}F` of the prepotential :math:`F(z^i)`.

        See also: :func:`prepot`

        """
        
        return jax.grad(self.prepot,holomorphic=True)(moduli,conj=conj)
    
    @partial(jit, static_argnums = (0,2,))
    def period_vector(
                    self, 
                    moduli: ArrayLike, 
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
            moduli (ArrayLike): Complex structure moduli values.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Value of the period vector :math:`\Pi`.

        See also: :func:`prepot`
    
        See also: :func:`dF`
    
        See also: :func:`jaxvacua.periods.periods.period_vector_per`
        
        """
        
        if self.periods.period_input is None:

            # Compute the value of the gradient of the prepotential
            dF = self.dF(moduli,conj=conj)

            # Get lower half of periods
            X = self.moduli_to_periods(moduli,conj=conj)

            return jnp.concat((jnp.concat((jnp.array([2*self.prepot(moduli,conj=conj) - jnp.einsum('i,i',moduli,dF)]), dF)), X))
        else:
                
            X = self.moduli_to_periods(moduli,conj=conj)

            return self.periods.period_vector_per(X,conj=conj)
        
    @partial(jit, static_argnums = (0,))
    def mirror_volume(
                    self, 
                    moduli: ArrayLike, 
                    moduli_c: ArrayLike
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
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
        
        Returns:
            complex: Mirror dual Calabi-Yau volume.

        Aliases:
            :func:`A`, :func:`mirror_volume`, :func:`V_tilde`

        See also: :func:`period_vector`
    
        See also: :func:`kahler_potential`
        
        """
        
        return -1.j*jnp.matmul(self.period_vector(moduli_c,conj=True), jnp.matmul(self.periods.sigma(), self.period_vector(moduli)))
        

    A = mirror_volume
    V_tilde = mirror_volume  

    # Kahler potential in terms of the axio dilation and the period vectors
    @partial(jit, static_argnums = (0,))
    def kahler_potential(
                        self, 
                        moduli: ArrayLike, 
                        moduli_c: ArrayLike, 
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
        
            The period vector :func:`jaxvacua.periods.periods.period_vector_per` and :func:`period_vector`

        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
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
    
    
    @partial(jit, static_argnums = (0,))
    def dK_z(self, 
            moduli: ArrayLike, 
            moduli_c: ArrayLike, 
            tau: complex, 
            tau_c: complex
            ) -> Array:
        r"""
        
        **Description:**
        Returns the holomorphic derivative :math:`\partial_{z^i}K` of the Kähler potential :math:`K`
        with respect to the complex structure moduli :math:`z^i`.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            ArrayLike: Holomorphic derivative :math:`\partial_{z^i}K` of the Kähler potential :math:`K`.
            
        See also: :func:`kahler_potential`
        
        """
        
        # Gradient of the kahler potential w.r.t moduli and axio dilaton
        return jax.grad(self.kahler_potential,argnums=0,holomorphic=True)(moduli,moduli_c,tau,tau_c)

    
    @partial(jit, static_argnums = (0,))
    def dK_cz(self, 
            moduli: ArrayLike, 
            moduli_c: ArrayLike, 
            tau: complex, 
            tau_c: complex
            ) -> Array:
        r"""
        
        **Description:**
        Returns the anti-holomorphic derivative :math:`\partial_{\overline{z}^i}K` of the Kähler potential :math:`K` 
        with respect to the conjugate complex structure moduli :math:`\overline{z}^i`.
        
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            ArrayLike: Anti-holomorphic derivative :math:`\partial_{\overline{z}^i}K` of the Kähler potential :math:`K`.
            
        See also: :func:`kahler_potential`
        
        """
        # Gradient of the kahler potential w.r.t moduli and axio dilaton
        return jax.grad(self.kahler_potential,argnums=1,holomorphic=True)(moduli, moduli_c, tau, tau_c)

    

    @partial(jit, static_argnums = (0,))
    def dK_tau(self, 
            moduli: ArrayLike, 
            moduli_c: ArrayLike, 
            tau: complex, 
            tau_c: complex
            ) -> complex:
        r"""
        
        **Description:**
        Returns the holomorphic derivative :math:`\partial_{\tau}K` of the Kähler potential :math:`K`
        with respect to the axio-dilaton :math:`\tau`.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            complex: Holomorphic derivative :math:`\partial_{\tau}K` of the Kähler potential :math:`K`.
            
        See also: :func:`kahler_potential`
        
        """
        
        return jax.grad(self.kahler_potential,argnums=2,holomorphic=True)(moduli,moduli_c,tau,tau_c)
    
    
    @partial(jit, static_argnums = (0,))
    def dK_ctau(self, 
            moduli: ArrayLike, 
            moduli_c: ArrayLike, 
            tau: complex, 
            tau_c: complex
            ) -> complex:
        r"""
        
        **Description:**
        Returns the anti-holomorphic derivative :math:`\partial_{\overline{\tau}}K` of the Kähler potential :math:`K`
        with respect to the conjugate axio-dilaton :math:`\overline{\tau}`.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            complex: Anti-holomorphic derivative :math:`\partial_{\overline{\tau}}K` of the Kähler potential :math:`K`.
            
        See also: :func:`kahler_potential`
        
        """
        
        #return jax.grad(self.kahler_potential_conj,argnums=3,holomorphic=True)(moduli, moduli_c, tau, tau_c)
        return jax.grad(self.kahler_potential,argnums=3,holomorphic=True)(moduli, moduli_c, tau, tau_c)


    @partial(jit, static_argnums = (0,))
    def dK(self, 
            moduli: ArrayLike, 
            moduli_c: ArrayLike, 
            tau: complex, 
            tau_c: complex
            ) -> Array:
        r"""
        
        **Description:**
        Returns the holomorphic derivative :math:`\partial_I K` of the Kähler potential :math:`K` 
        with respect to the complex structure moduli :math:`z^{i}` and the axio-dilaton :math:`\tau`.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            ArrayLike: Values of :math:`\partial_I K`. 


        See also: :func:`kahler_potential`
        
        """
        
        dKz = self.dK_z(moduli, moduli_c, tau, tau_c)
        dKtau = self.dK_tau(moduli, moduli_c, tau, tau_c)
        
        return jnp.append(dKz,dKtau)
    
    
    
    @partial(jit, static_argnums = (0,))
    def dK_c(self, 
            moduli: ArrayLike, 
            moduli_c: ArrayLike, 
            tau: complex, 
            tau_c: complex
            ) -> Array:
        r"""
        
        **Description:**
        Returns the holomorphic derivative :math:`\partial_{\overline{I}} K` of the Kähler potential :math:`K` 
        with respect to the complex conjugate complex structure moduli :math:`\overline{z}^{i}` 
        and the axio-dilaton :math:`\overline{\tau}`.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            ArrayLike: Values of :math:`\partial_{\overline{I}} K`. 


        See also: :func:`kahler_potential`
        
        """
        
        dKcz = self.dK_cz(moduli, moduli_c, tau, tau_c)
        dKctau = self.dK_ctau(moduli, moduli_c, tau, tau_c)
        
        return jnp.append(dKcz,dKctau)


    @partial(jit, static_argnums = (0,))
    def ddK_z_cz(self, 
                moduli: ArrayLike, 
                moduli_c: ArrayLike, 
                tau: complex, 
                tau_c: complex
                ) -> Array:
        r"""
        
        **Description:**
        Returns the second derivatives :math:`K_{i\bar{\jmath}}=\partial_{z^i}\partial_{\overline{z}^j}K`
        of the Kähler potential :math:`K` with respect to the complex structure moduli :math:`z^i` and their conjugate.
        
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            ArrayLike: Second derivatives :math:`K_{i\bar{\jmath}}=\partial_{z^i}\partial_{\overline{z}^j}K`
                of the Kähler potential :math:`K`.
            
        See also: :func:`dK_z`
        
        """
    
        return jax.jacrev(self.dK_z,argnums =1, holomorphic = True)(moduli, moduli_c, tau, tau_c) #K_Z\bar{Z}

    @partial(jit, static_argnums = (0,))
    def ddK_cz_z(self, 
                moduli: ArrayLike, 
                moduli_c: ArrayLike, 
                tau: complex, 
                tau_c: complex
                ) -> Array:
        r"""
        
        **Description:**
        Returns the second derivatives :math:`K_{\bar{\jmath}i}=\partial_{\overline{z}^j}\partial_{z^i}K`
        of the Kähler potential :math:`K` with respect to the complex structure moduli :math:`z^i` and their conjugate.
        
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            ArrayLike: Second derivatives :math:`K_{\bar{\jmath}i}=\partial_{\overline{z}^j}\partial_{z^i}K`
                of the Kähler potential :math:`K`.
            
        See also: :func:`dK_z`
        
        """
    
        return jax.jacrev(self.dK_cz,argnums =0, holomorphic = True)(moduli, moduli_c, tau, tau_c) #K_Z\bar{Z}


    @partial(jit, static_argnums = (0,))
    def ddK_z_tau(self, 
                moduli: ArrayLike, 
                moduli_c: ArrayLike, 
                tau: complex, 
                tau_c: complex
                ) -> Array:
        r"""
        
        **Description:**
        Returns the second derivatives :math:`K_{i\tau}=\partial_{z^i}\partial_{\tau}K`
        of the Kähler potential :math:`K` with respect to the complex structure moduli :math:`z^i` and 
        the axio-dilaton :math:`\tau`.
        
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            ArrayLike: Second derivatives :math:`K_{i\tau}=\partial_{z^i}\partial_{\tau}K`
                of the Kähler potential :math:`K`.
        
        """
        
        return jax.jacrev(self.dK_z,argnums =2, holomorphic = True)(moduli, moduli_c, tau, tau_c)

    @partial(jit, static_argnums = (0,))
    def ddK_cz_ctau(self, 
                    moduli: ArrayLike, 
                    moduli_c: ArrayLike, 
                    tau: complex, 
                    tau_c: complex
                    ) -> Array:
        r"""
        
        **Description:**
        Returns the second derivatives :math:`K_{\bar{\imath}\overline{\tau}}=\partial_{\overline{z}^i}\partial_{\overline{\tau}}K`
        of the Kähler potential :math:`K` with respect to the complex conjugate complex structure moduli :math:`\overline{z}^i` and 
        the axio-dilaton :math:`\overline{\tau}`.
        
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            ArrayLike: Second derivatives :math:`K_{\bar{\imath}\overline{\tau}}=\partial_{\overline{z}^i}\partial_{\overline{\tau}}K`
                of the Kähler potential :math:`K`.
        
        """
        
        return jax.jacrev(self.dK_cz,argnums =3, holomorphic = True)(moduli, moduli_c, tau, tau_c)
    

    @partial(jit, static_argnums = (0,))
    def ddK_z_ctau(self, 
                moduli: ArrayLike, 
                moduli_c: ArrayLike, 
                tau: complex, 
                tau_c: complex
                ) -> Array:
        r"""
        
        **Description:**
        Returns the second derivatives :math:`K_{i\overline{\tau}}=\partial_{z^i}\partial_{\overline{\tau}}K`
        of the Kähler potential :math:`K` with respect to the complex structure moduli :math:`z^i` and 
        the conjugate axio-dilaton :math:`\overline{\tau}`.
        
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            ArrayLike: Second derivatives :math:`K_{i\overline{\tau}}=\partial_{z^i}\partial_{\overline{\tau}}K`
                of the Kähler potential :math:`K`.
        
        """
        
        return jax.jacrev(self.dK_z,argnums =3, holomorphic = True)(moduli, moduli_c, tau, tau_c)

    @partial(jit, static_argnums = (0,))
    def ddK_cz_tau(self, 
                moduli: ArrayLike, 
                moduli_c: ArrayLike, 
                tau: complex, 
                tau_c: complex
                ) -> Array:
        r"""
        
        **Description:**
        Returns the second derivatives :math:`K_{\tau\overline{j}}=\partial_{\tau}\partial_{\overline{z}^{j}}K` 
        of the Kähler potential :math:`K` with respect to the conjugate complex structure moduli :math:`\overline{z}^i`
        and the axio-dilaton :math:`\tau`.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            ArrayLike: Second derivatives :math:`K_{\tau\overline{j}}=\partial_{\tau}\partial_{\overline{z}^{j}}K` 
                of the Kähler potential :math:`K`.
        
        """
        
        return jax.jacrev(self.dK_cz,argnums =2, holomorphic = True)(moduli, moduli_c, tau, tau_c)

    @partial(jit, static_argnums = (0,))
    def ddK_tau_ctau(self, 
                    moduli: ArrayLike, 
                    moduli_c: ArrayLike, 
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
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            complex: Second derivatives :math:`\partial_{\tau}\partial_{\overline{\tau}}K`
                of the Kähler potential :math:`K`.
        
        """
        
        return jax.jacrev(self.dK_tau,argnums =3, holomorphic = True)(moduli, moduli_c, tau, tau_c)
        
    @partial(jit, static_argnums = (0,))
    def ddK_z_z(self, 
                moduli: ArrayLike, 
                moduli_c: ArrayLike, 
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
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            ArrayLike: :math:`h^{1,2}\times h^{1,2}` matrix of the second holomorphic derivatives of the Kähler potential with respect to :math:`z^i`.
        
        """
        
        return jax.jacrev(self.dK_z,argnums =0, holomorphic = True)(moduli, moduli_c, tau, tau_c)

    @partial(jit, static_argnums = (0,))
    def ddK_cz_cz(self, 
                moduli: ArrayLike, 
                moduli_c: ArrayLike, 
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
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            ArrayLike: :math:`h^{1,2}\times h^{1,2}` matrix of the second holomorphic derivatives of the Kähler potential with respect to :math:`z^i`.
        
        """
        
        return jax.jacrev(self.dK_cz,argnums = 1, holomorphic = True)(moduli, moduli_c, tau, tau_c)
        
    @partial(jit, static_argnums = (0,))
    def ddK_z_tau(self, 
                moduli: ArrayLike, 
                moduli_c: ArrayLike, 
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
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            ArrayLike: :math:`h^{1,2}\times 1` matrix of the second holomorphic derivatives of the Kähler potential with respect to :math:`z^i` and :math:`\tau`.
        
        """
        
        return jax.jacrev(self.dK_z,argnums =2, holomorphic = True)(moduli, moduli_c, tau, tau_c)
        
    @partial(jit, static_argnums = (0,))
    def ddK_tau_tau(self, 
                    moduli: ArrayLike, 
                    moduli_c: ArrayLike, 
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
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            complex: Second holomorphic derivative of the Kähler potential with respect to :math:`\tau`.
        
        """
        
        return jax.jacrev(self.dK_tau,argnums =2, holomorphic = True)(moduli, moduli_c, tau, tau_c)

    @partial(jit, static_argnums = (0,))
    def ddK_ctau_ctau(self, 
                    moduli: ArrayLike, 
                    moduli_c: ArrayLike, 
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
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            complex: Second holomorphic derivative of the Kähler potential with respect to :math:`\tau`.
        
        """
        
        return jax.jacrev(self.dK_ctau,argnums =3, holomorphic = True)(moduli, moduli_c, tau, tau_c)
    
    @partial(jit, static_argnums = (0,))
    def _kahler_metric_general(self, 
                            moduli: ArrayLike, 
                            moduli_c: ArrayLike, 
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
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            ArrayLike: Kähler metric :math:`K_{\overline{I}J}`.

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
    
    @partial(jit, static_argnums = (0,))
    def _kahler_metric_block_diagonal(self, 
                                    moduli: ArrayLike, 
                                    moduli_c: ArrayLike, 
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
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
        
        Returns:
            ArrayLike: Kähler metric :math:`K_{\overline{I}J}` assuming :math:`K_{i\overline{\tau}}=K_{\tau\overline{i}}=0`.

        See also: :func:`ddK_z_cz`
    
        See also: :func:`ddK_tau_ctau`
        
        """

        val1 = self.ddK_z_cz(moduli, moduli_c, tau, tau_c) #Component with K_i\bar{j}
        val2 = np.zeros((self.h12,1),dtype=np.complex128)  #Component with K_i\bar{tau}=0
        val3 = np.zeros((1,self.h12),dtype=np.complex128)[0] #Component with K_tau\bar{j}=0
        val4 = self.ddK_tau_ctau(moduli, moduli_c, tau, tau_c) #Component with K_tau\bar{tau}

        # Combine blocks into array of the form [K_i\bar{j}, 0]
        a = jnp.hstack((jnp.asarray(val1), jnp.asarray(val2)))

        # Combine blocks into array of the form [0, K_tau\bar{tau}]
        b = jnp.hstack((jnp.asarray(val3), jnp.asarray(val4)))

        # Stack the two blocks and return result
        return jnp.vstack((a, b))
        
    @partial(jit, static_argnums = (0,5,))
    def kahler_metric(self, 
                    moduli: ArrayLike, 
                    moduli_c: ArrayLike, 
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
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
            mode (string, optional): Mode to compute the Kahler metric. Defaults to ``"block diagonal"`` 
                meaning that there are no mixing terms between complex structure moduli and 
                the axio-dilaton in the Kähler potential. If set to ``None`` instead, the full set of 2nd 
                derivatives of the Kähler potential is computed.
        
        Returns:
            ArrayLike: Kähler metric :math:`K_{\overline{I}J}`.

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


    @partial(jit, static_argnums = (0,5,))
    def inverse_kahler_metric(self, 
                            moduli: ArrayLike, 
                            moduli_c: ArrayLike, 
                            tau: complex, 
                            tau_c: complex, 
                            mode: str = "block diagonal"
                            ) -> Array:
        r"""
        
        **Description:**
        Returns the inverse Kähler metric :math:`K^{\overline{I}J}`.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
            mode (str, optional): Description
        
        Returns:
            ArrayLike: Inverse Kähler metric :math:`K^{\overline{I}J}`.
            

        See also: :func:`kahler_metric`
        
        """
    
        return jnp.linalg.inv(self.kahler_metric(moduli,moduli_c,tau,tau_c,mode=mode))
        
    @partial(jit, static_argnums = (0,5,))
    def inverse_kahler_metric_grad(self, 
                                moduli: ArrayLike, 
                                moduli_c: ArrayLike, 
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
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value of the axio-dilaton.
            mode (str, optional): Description
        
        Returns:
            ArrayLike: Gradient of the inverse Kähler metric :math:`\partial_{I}K^{\overline{J}L},\partial_{\overline{I}}K^{\overline{J}L}`.

        See also: :func:`kahler_metric`

        See also: :func:`inverse_kahler_metric`
        
        """
        
        return jax.jacrev(self.inverse_kahler_metric,holomorphic=True)(moduli,moduli_c,tau,tau_c,mode=mode)
    
    


    

       

