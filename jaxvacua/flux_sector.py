# ==============================================================================
# This code is written by Andreas Schachner. Without the author's permission, this 
# code must not be shared with anyone else or used for any other projects than 
# those involving the author directly.
#
# If any questions arise, please feel free to reach out to me (Andreas) either at
# andreas.schachner@gmx.net or at as3475@cornell.edu or at a.schachner@lmu.de.
# ==============================================================================
#
# ------------------------------------------------------------------------------
# This file holds the class to construct flux vacua.
# ------------------------------------------------------------------------------


# Important standard libraries
import os, sys, warnings, time
import numpy as np
from typing import Optional, Tuple, Any, Callable
from functools import partial


# Important JAX libraries
import jax
from jax import jit, vmap, config
import jax.numpy as jnp
from jax import Array
#from jax.typing import ArrayLike
from numpy.typing import ArrayLike

# Enable 64 bit precision
config.update("jax_enable_x64", True)

# JAXVacua custom imports
from .util import *
from .css import css

class flux_sector(css):
    r"""
    **Description:**
    A class representing the flux sector in a 4D EFT obtained from Type IIB
    compactification on orientifolds of CY threefolds with 3-form flux backgrounds.
    """
    
    def __init__(
        self,
        h12: Optional[int] = None,
        model_ID: Optional[int] = None,
        model_type: str = "KS",
        moduli_space_limit: str = "LCS",
        maximum_degree: int = 0,
        mirror_cy: Optional[Any] = None,
        model_data: Optional[dict] = None,
        instanton_data: Optional[list] = None,
        use_cytools: bool = False,
        basis_transformation: Optional[ArrayLike] = None,
        ncf: Optional[int] = None,
        conifold_curve: Optional[Any] = None,
        grading_vector: Optional[ArrayLike] = None,
        period_input: Optional[Callable] = None,
        prepotential_input: Optional[Callable] = None,
        Q: Optional[int] = None,
        gauge_choice: complex = 1.0 + 0.0j,
        prange: int = 500,
        use_gvs: bool = False,
        save_file: bool = False,
        **kwargs: Any,
    ) -> None:
        r"""

        **Description:**
        A class representing the flux sector in a 4D EFT obtained from Type IIB 
        compactification on CY threefolds with 3-form flux backgrounds.
        
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
            gauge_choice (complex, optional): Gauge choice parameter.
            save_file (bool, optional): Save files for new models. Defaults to ``False``.
            **kwargs: Additional keyword arguments.
            
        Attributes:
            D3_tadpole (int): Maximum allowed D3-tadpole.
            n_fluxes (int): Number of fluxes.


        """

        # Initialise flux sector as subclass of complex structure sector!
        super(flux_sector, self).__init__(
            h12=h12,
            model_ID=model_ID,
            model_type=model_type,
            moduli_space_limit=moduli_space_limit,
            maximum_degree=maximum_degree,
            mirror_cy=mirror_cy,
            model_data=model_data,
            instanton_data=instanton_data,
            use_cytools=use_cytools,
            basis_transformation=basis_transformation,
            ncf=ncf,
            conifold_curve=conifold_curve,
            grading_vector=grading_vector,
            period_input=period_input,
            prepotential_input=prepotential_input,
            gauge_choice=gauge_choice,
            prange=prange,
            use_gvs=use_gvs,
            save_file=save_file,
            **kwargs
        )
        
        if Q is None:
            self.D3_tadpole = self.periods.Q
        else:
            self.D3_tadpole = Q

        self.n_fluxes = self._dimension_H3_tot

    def __repr__(self) -> str:
        r"""
        **Description:**
        Returns a string representation of the flux sector class.
        
        Returns:
            str: Description of the class.
        """
        if self.periods.name!="":
            return f"Flux sector for {self.periods.name} with h12={self.h12} complex structure moduli in the {self.periods.moduli_space_limit} limit."
        else:
            return f"Flux sector with h12={self.h12} complex structure moduli in the {self.periods.moduli_space_limit} limit."
            
    def map_to_FD_tau(
        self, 
        tau: complex, 
        fluxes: ArrayLike,
        return_SL2Z_matrix: bool = False, 
        verbose: bool = False
        ):
        r"""
        
        **Description:**
        Map of the axio-dilaton value and the flux vector to the fundamental domain (FD) of :math:`\text{SL}(2,\mathbb{Z})`.
        
        .. admonition:: Details
            :class: dropdown
        
            The axio-dilaton transforms as
        
            .. math::
                \tau \rightarrow \dfrac{a\tau+b}{c\tau+d}\; ,\quad ad-bc=1\; ,\quad \Sigma = \left (\begin{array}{cc}a & b \ c & d \end{array} \right )\in \text{SL}(2,\mathbb{Z})\, .
        
            The flux vector of RR-flux :math:`f` and NSNS-flux :math:`h` transform as
        
            .. math::
                \left (\begin{array}{c}f \ h \end{array} \right )\rightarrow\left (\begin{array}{c}a\, f+b\, h \\ c\, f+d\, h\end{array} \right )\, .
        
            Given a value :math:`\tau_0` for the axio-dilaton, we apply translations and S-duality transformations
        
            .. math::
                 \tau_0 \rightarrow \tau_0+b\; ,\quad \tau_0 \rightarrow -\dfrac{1}{\tau_0}\, ,
        
            until we find a value of :math:`\tau` in the fundamental domain :math:`\mathcal{F}_{\text{SL}(2,\mathbb{Z})}` of :math:`\text{SL}(2,\mathbb{Z})` given by
        
            .. math::
                \mathcal{F}_{\text{SL}(2,\mathbb{Z})} = \lbrace \tau\in\mathbb{C}:\, |\text{Re}(\tau)|\leq 0.5\, ,\, |\tau|\leq 1\rbrace \, .
        
        
        .. note::
            Per default, this function only returns the value of the axio-dilaton and the fluxes. It is however also possible to return the :math:`\text{SL}(2,\mathbb{Z})` transformation taking the input value :math:`\tau` of the axio-dilaton to the fundamental domain.
        
        Args:
            tau (complex): Axio-dilaton value.
            fluxes (ArrayLike): The flux vector used in the mapping.
            return_SL2Z_matrix (bool, optional): If ``True``, returns the :math:`\text{SL}(2,\mathbb{Z})`-matrix :math:`\Sigma` for the transformation. Defaults to ``False``.
            verbose (boolean, optional): If ``True``, enables verbose mode. Defaults to ``False``.
        
        Returns:
            complex: The axio-dilaton value in the fundamental domain.
            ArrayLike: The corresponding flux vector.
            ArrayLike, optional: The corresponding :math:`\text{SL}(2,\mathbb{Z})` transformation matrix.
        
        """

        # Initialise the SL(2,Z) matrix
        if return_SL2Z_matrix:
            sig=np.array([[1,0],[0,1]])
            
        tau1=jnp.real(tau)
        tau2=jnp.imag(tau)

        if tau2<0:
            # Negative string coupling
            if return_SL2Z_matrix:
                return None,None,None
            else:
                return None,None
        
        if jnp.abs(tau1)<=0.5 and jnp.abs(tau)>=1.:
            # Already in fundamental domain -> return values
            if return_SL2Z_matrix==True:
                return tau,fluxes,sig
            else:
                return tau,fluxes

        FFlux = fluxes[:self.n_fluxes]
        HFlux = fluxes[self.n_fluxes:]
        
        count=0
    
        end_loop=0
        
        while end_loop<1:
        
            if tau1>10**15 or tau2>10**15:
                warnings.warn("Needed to stop map to FD domain!")
                if return_SL2Z_matrix:
                    return None,None,None
                else:
                    return None,None
        
            if count>10**(4):
                warnings.warn("Needed to stop map to FD domain!")
                if return_SL2Z_matrix:
                    return None,None,None
                else:
                    return None,None

            temp1=int(jnp.floor(tau1))
            
            
            if tau1<0.:
                p_list=[jnp.fmod(tau1,1.)+1.,tau2]
            else:
                p_list=[jnp.fmod(tau1,1.),tau2]

            FFlux=FFlux-temp1*HFlux

            if return_SL2Z_matrix:
                #b=-temp1
                #a=1
                #c=0
                #d=1

                sig = np.matmul(np.array([[1,-temp1],[0,1]]),sig)

                if verbose:
                    print("Sigma 1: ",sig)

            if jnp.abs(p_list[0])>0.5:

                temp2=jnp.sign(p_list[0])
                p_list[0]=p_list[0]-temp2

                FFlux=FFlux-temp2*HFlux

                if return_SL2Z_matrix:
                    #b=-temp2
                    #a=1
                    #c=0
                    #d=1

                    sig=np.matmul(np.array([[1,-temp2],[0,1]]),sig)
                    if verbose:
                        print("Sigma 2: ",sig)

            if jnp.sqrt(p_list[0]**2+p_list[1]**2)<1.:

                FFlux_old=FFlux.copy()
                FFlux=HFlux.copy()
                HFlux=-FFlux_old.copy()

                norm_p=p_list[0]**2+p_list[1]**2
                p_list[0]=-p_list[0]/norm_p
                p_list[1]=p_list[1]/norm_p

                if return_SL2Z_matrix:
                    #a=0
                    #b=1
                    #c=-1
                    #d=0
                    sig=np.matmul(np.array([[0,1],[-1,0]]),sig)

                if verbose:
                    print("Sigma 3: ",sig)

            tau1=p_list[0]
            tau2=p_list[1]
            count=count+1

            if jnp.abs(tau1)<=0.5 and jnp.sqrt(tau1**2+tau2**2)>=1.:
                end_loop=1


        fluxes = np.hstack((FFlux, HFlux))
        tau = tau1+1j*tau2
        
        if return_SL2Z_matrix:
            return tau,fluxes,sig
        else:
            return tau,fluxes


    @partial(jit, static_argnums = (0,))
    def _convert_real_to_complex(
        self, x: ArrayLike
        ) -> float:
        r"""
        
        **Description:**
        Converts the real field components to complex values of the complex structure moduli :math:`z^{i}`
        and the axio-dilaton :math:`\tau`.

        .. note::
            The input shape should be

            .. math:
                \bigl (\mathrm{Re}(z^{1}),\mathrm{Im}(z^{1}),\ldots , \mathrm{Re}(z^{h^{1,2}}),
                \mathrm{Im}(z^{h^{1,2}}), \mathrm{Re}(\tau),\mathrm{Im}(\tau) \bigl )\, .
        
        
        Args:
            x (ArrayLike): Array of shape (:math:`2(h^{1,2}+1)`, ) containing real and imaginary parts of
                the complex structure moduli :math:`z^i` and axio-dilaton :math:`\tau`.
        
        Returns:
            ArrayLike: Complex structure moduli values.
            ArrayLike: Complex conjugate values for complex structure moduli.
            complex: Axio-dilaton value.
            complex: Complex conjugate value for axio-dilaton.
        
        """
        moduli = x[0:-2:2]+1.j*x[1:-2:2]
        moduli_c = x[0:-2:2]-1.j*x[1:-2:2]
        tau = x[-2]+1.j*x[-1]
        tau_c = x[-2]-1.j*x[-1]

        return moduli,moduli_c,tau,tau_c

    @partial(jit, static_argnums = (0,))
    def _convert_complex_to_real(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex
        ) -> float:
        r"""
        
        **Description:**
        Converts the complex values of the complex structure moduli :math:`z^{i}`
        and the axio-dilaton :math:`\tau` to real components.

        .. note::
            The output shape is 

            .. math:
                \bigl (\mathrm{Re}(z^{1}),\mathrm{Im}(z^{1}),\ldots , \mathrm{Re}(z^{h^{1,2}}),
                \mathrm{Im}(z^{h^{1,2}}), \mathrm{Re}(\tau),\mathrm{Im}(\tau) \bigl )\, .
        
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): Axio-dilaton value.
            tau_c (complex): Complex conjugate value for axio-dilaton.
        
        Returns:
            ArrayLike: Array of shape (:math:`2(h^{1,2}+1)`, ) containing real and imaginary parts of
                the complex structure moduli :math:`z^i` and axio-dilaton :math:`\tau`.
            
        
        """
        real_z = (moduli+moduli_c)/2.
        imag_z = -1j*(moduli-moduli_c)/2.
        real_tau = (tau+tau_c)/2.
        imag_tau = -1j*(tau-tau_c)/2.

        x = jnp.append(jnp.array([real_z]),jnp.array([imag_z]),axis=0).T.flatten()

        return jnp.append(x,jnp.array([real_tau,imag_tau])).astype(jnp.float64)

    @partial(jit, static_argnums = (0,))
    def _convert_complex_to_real_nondif(
        self, moduli: ArrayLike, tau: complex
        ) -> float:
        r"""
        
        **Description:**
        Converts the complex values of the complex structure moduli :math:`z^{i}`
        and the axio-dilaton :math:`\tau` to real components.

        .. note::
            The output shape is 

            .. math:
                \bigl (\mathrm{Re}(z^{1}),\mathrm{Im}(z^{1}),\ldots , \mathrm{Re}(z^{h^{1,2}}),
                \mathrm{Im}(z^{h^{1,2}}), \mathrm{Re}(\tau),\mathrm{Im}(\tau) \bigl )\, .
        
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            tau (complex): Axio-dilaton value.
        
        Returns:
            ArrayLike: Array of shape (:math:`2(h^{1,2}+1)`, ) containing real and imaginary parts of
                the complex structure moduli :math:`z^i` and axio-dilaton :math:`\tau`.
            
        
        """
        moduli_c = jnp.conj(moduli)
        tau_c = jnp.conj(tau)
        real_z = (moduli+moduli_c)/2.
        imag_z = -1j*(moduli-moduli_c)/2.
        real_tau = (tau+tau_c)/2.
        imag_tau = -1j*(tau-tau_c)/2.

        x = jnp.append(jnp.array([real_z]),jnp.array([imag_z]),axis=0).T.flatten()

        return jnp.append(x,jnp.array([real_tau,imag_tau])).astype(jnp.float64)


    @partial(jit, static_argnums = (0,))
    def tadpole(
        self, fluxes: ArrayLike
        ) -> float:
        r"""
        
        **Description:**
        Calculates the D3-charge for given fluxes.
        
        .. admonition:: Details
            :class: dropdown
        
            The D3-charge induced by fluxes is defined as

            .. math::
                    N_{\mathrm{flux}} = \vec{f}\Sigma \vec{h}\, ,

            where :math`\vec{f}, \vec{h}` are the NSNS- and RR-fluxes respectively.
            The symplectic pairing :math:`\Sigma` is implemented in the periods class :func:`jaxvacua.periods.periods`.

        Args:
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
        
        Returns:
            int: D3-charge induced by 3-form fluxes.
        
        """
        
        FF = fluxes[:self.n_fluxes]
        HF = fluxes[self.n_fluxes:]
        
        return jnp.matmul(FF, jnp.matmul(self.periods.sigma(), HF))
            

    ###################################################################################################################
    ############################################ GVW SUPERPOTENTIAL ###################################################
    ###################################################################################################################

    
    @partial(jit, static_argnums = (0,4,5,))
    def superpotential(
        self, moduli: ArrayLike, tau: complex, fluxes: ArrayLike, conj: bool = False, normalise: bool = False
        ) -> complex:
        r"""
        
        **Description:**
        Calculates the value of the superpotential for given flux, moduli and axio-dilaton.
        
        
        .. admonition:: Details
            :class: dropdown
        
            For a Calabi-Yau manifold :math:`X`, the Gukov-Vafa-Witten superpotential `hep-th/9906070 <https://inspirehep.net/literature/501505>`_ is defined in terms of the complex 3-form flux  :math:`G_3=F_3-\tau H_3` and the holomorphic 3-form  :math:`\Omega_3` as
        
            .. math::
                    W(Z,\tau) = \int_{X} \, G_3\wedge \Omega_3 = (\vec{f}-\tau\vec{h})^T\cdot\Sigma\cdot\Pi(Z)
        
        
            following the conventions of eq. (4) in `1912.10047 <https://inspirehep.net/literature/1772253>`_.
            The period vector :math:`\Pi` and the symplectic pairing :math:`\Sigma` are defined in the periods class :func:`jaxvacua.periods.periods`.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            tau (complex): Axio-dilaton value.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
            normalise (bool, optional): If ``True``, rescales superpotential by :math:`\sqrt{2/\pi}`. Defaults to ``False``.
        
        Returns:
            complex: Value of the superpotential.
        
        """

        FFlux = fluxes[:self.n_fluxes]
        HFlux = fluxes[self.n_fluxes:]

        W0 = jnp.matmul((FFlux-tau*HFlux),jnp.matmul(self.periods.sigma(), self.period_vector(moduli,conj=conj)))

        if normalise:
            W0 = jnp.sqrt(2./jnp.pi)*W0

        return W0

        
    
    W = superpotential
    
    @partial(jit, static_argnums = (0,4,5,))
    def superpotential_gauge_invariant(
        self, moduli: ArrayLike, tau: complex, fluxes: ArrayLike, conj: bool = False, normalise: bool = True
        ) -> complex:
        r"""
        
        **Description:**
        Calculates the value of the gauge invariant version of the flux superpotential.
        
        
        .. admonition:: Details
            :class: dropdown
        
            Following the conventions of `1912.10047 <https://inspirehep.net/literature/1772253>`_,
            we define the gauge invariant version of the superpotential as
        
            .. math::
                    \tilde{W} = \sqrt{\dfrac{2}{\pi}}\, \mathrm{e}^{K/2}\, W

            Its absolute value is proportional to the gravitino mass.
            The normalisation is taken from `1908.04788 <https://inspirehep.net/literature/1749542>`_.

        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            tau (complex): Axio-dilaton value.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
            normalise (bool, optional): If ``True``, rescales superpotential by :math:`\sqrt{2/\pi}`. Defaults to ``True``.
        
        Returns:
            complex: Value of the gauge invariant superpotential.
            

        See also: :func:`superpotential`
    
        See also: :func:`kahler_potential`
        """
            
        KP = self.kahler_potential(moduli,jnp.conj(moduli),tau,jnp.conj(tau))

        W = self.superpotential(moduli,tau,fluxes,conj=conj,normalise=normalise)
        
        return jnp.exp(KP.real/2.)*W
    
    W0 = superpotential_gauge_invariant
        
    @partial(jit, static_argnums = (0,4,))
    def dW_z(
        self, moduli: ArrayLike, tau: complex, fluxes: ArrayLike, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Calculates the holomorphic derivative :math:`W_i=\partial_{z^i}W` 
        of the superpotential :math:`W` with respect to the complex structure moduli :math:`z^{i}`.
                
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            tau (complex): Value of the axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Value of the holomorphic derivative of the superpotential with respect to  
                the complex structure moduli :math:`z^{i}`.
        
        """
        
        # Gradient of superpotential w.r.t the moduli and the axio dilaton
        return jax.grad(self.superpotential,argnums=0,holomorphic=True)(moduli,tau,fluxes,conj=conj)

    
    @partial(jit, static_argnums = (0,4,))
    def dW_tau(
        self, moduli: ArrayLike, tau: complex, fluxes: ArrayLike, conj: bool = False
        ) -> complex:
        r"""
        
        **Description:**
        Calculates the holomorphic derivative :math:`W_\tau=\partial_{\tau}W` 
        of the superpotential :math:`W` with respect to the axio-dilaton :math:`\tau`.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            tau (complex): Value of the axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            complex: Value of the holomorphic gradient of the superpotential with respect to 
                the axio-dilaton :math:`\tau`.
        
        """
        
        return jax.grad(self.superpotential,argnums=1,holomorphic=True)(moduli,tau,fluxes,conj=conj)


    @partial(jit, static_argnums = (0,4,))
    def dW(
        self, moduli: ArrayLike, tau: complex, fluxes: ArrayLike, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Calculates the holomorphic derivative :math:`W_I` of the superpotential :math:`W`
        with respect to the complex structure moduli :math:`z^{i}` and the axio-dilaton :math:`\tau`.
                
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            tau (complex): Value of the axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Value of the holomorphic derivative of the superpotential with respect to 
                the complex structure moduli :math:`z^{i}` and the axio-dilaton :math:`\tau`.
        
        """
        
        dWz = self.dW_z(moduli,tau,fluxes,conj=conj)
        dWtau = self.dW_tau(moduli,tau,fluxes,conj=conj)

        return jnp.append(dWz,dWtau)


    @partial(jit, static_argnums = (0,4,))
    def ddW_z_z(
        self, moduli: ArrayLike, tau: complex, fluxes: ArrayLike, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Calculates the second holomorphic derivatives :math:`W_{i j}=\partial_{z^i}\partial_{z^j}W`
        of the superpotential :math:`W` with respect to the complex structure moduli :math:`z^{i}`.
                
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            tau (complex): Value of the axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            complex: Value of :math:`W_{i j}=\partial_{z^i}\partial_{z^j}W`.
        
        """
        
        return jax.jacrev(self.dW_z,argnums=0,holomorphic=True)(moduli,tau,fluxes,conj=conj)

        
        
    
    @partial(jit, static_argnums = (0,4,))
    def ddW_z_tau(
        self, moduli: ArrayLike, tau: complex, fluxes: ArrayLike, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Calculates the second mixed holomorphic derivatives :math:`W_{i \tau}=\partial_{z^i}\partial_{\tau}W`
        of the superpotential :math:`W` with respect to the axio-dilaton :math:`\tau` and 
        the complex structure moduli :math:`z^{i}`.
                
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            tau (complex): Value of the axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Value of :math:`W_{i \tau}=\partial_{z^i}\partial_{\tau}W`.
        
        """
        return jax.jacrev(self.dW_z,argnums=1,holomorphic=True)(moduli,tau,fluxes,conj=conj)

    @partial(jit, static_argnums = (0,4,))
    def ddW_tau_tau(
        self, moduli: ArrayLike, tau: complex, fluxes: ArrayLike, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Calculates the second mixed holomorphic derivatives :math:`W_{\tau \tau}=\partial_{\tau}\partial_{\tau}W`
        of the superpotential :math:`W` with respect to the axio-dilaton :math:`\tau`.
                
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            tau (complex): Value of the axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Value of :math:`W_{\tau \tau}=\partial_{\tau}\partial_{\tau}W`.
        
        """
        return jax.jacrev(self.dW_tau,argnums=1,holomorphic=True)(moduli,tau,fluxes,conj=conj)


    @partial(jit, static_argnums = (0,4,))
    def ddW(
        self, moduli: ArrayLike, tau: complex, fluxes: ArrayLike, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Calculates the second mixed holomorphic derivatives :math:`W_{IJ}=\partial_{I}\partial_{J}W`
        of the superpotential :math:`W` with respect to the axio-dilaton :math:`\tau` and 
        the complex structure moduli :math:`z^{i}`.
                
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            tau (complex): Value of the axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Value of :math:`W_{IJ}=\partial_{I}\partial_{J}W`.
        
        """

        ddW_z_z = self.ddW_z_z(moduli, tau, fluxes, conj=conj)
        ddW_z_tau = self.ddW_z_tau(moduli, tau, fluxes, conj=conj)
        ddW_tau_tau = self.ddW_tau_tau(moduli, tau, fluxes, conj=conj)
        
        block1 = jnp.append(ddW_z_z,ddW_z_tau.reshape(1,-1),axis=0)
        block2 = jnp.append(ddW_z_tau,ddW_tau_tau)
        
        return jnp.append(block1,block2.reshape(-1,1),axis=1)
        
    ###################################################################################################################
    ######################################### F-TERM CONDITIONS ETC.  #################################################
    ###################################################################################################################
    
    @partial(jit, static_argnums = (0,6,))
    def DW_z(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Calculates the Kähler covariant derivative of the superpotential :math:`W` 
        with respect to the complex structure moduli :math:`z^{i}`.
        
        .. admonition:: Details
            :class: dropdown
        
            The :math:`F`-term conditions for the complex structure moduli :math:`z^{i}` are given by
            
            .. math::
                D_i W = \partial_{z^i}W+(\partial_{z^i}K)\, W = W_i +K_i\, W \, .

            They appear prominently in the :math:`F`-term scalar potential :func:`scalar_potential`.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): Axio-dilaton value.
            tau_c (complex): Complex conjugate value for axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Values of the :math:`F`-term conditions for the complex structure moduli :math:`z^{i}`.

        See also: :func:`superpotential`
    
        See also: :func:`dW_z`
    
        See also: :func:`dK_z`

        See also: :func:`dK_cz`
        
        """
        
        # F-Terms as the Kähler covariant derivative of W w.r.t cs moduli and axio dilaton
        if not conj:
            return self.dW_z(moduli,tau,fluxes)+self.dK_z(moduli,moduli_c,tau,tau_c)*self.superpotential(moduli,tau,fluxes)
        else:
            return self.dW_z(moduli_c,tau_c,fluxes,conj=True)+self.dK_cz(moduli,moduli_c,tau,tau_c)*self.superpotential(moduli_c, tau_c,fluxes,conj=True)
        
    
    
    @partial(jit, static_argnums = (0,6,))
    def DW_tau(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, conj: bool = False
        ) -> complex:
        r"""
        
        **Description:**
        Calculates the Kähler covariant derivative of the superpotential with respect to the axio-dilaton :math:`\tau`.
        
        .. admonition:: Details
            :class: dropdown
        
            The :math:`F`-term conditions for the axio-dilaton :math:`\tau` are given by

            .. math::
                D_{\tau} W=\partial_{\tau}W+(\partial_{\tau}K)\, W = W_\tau+K_\tau\, W

            They appear prominently in the :math:`F`-term scalar potential :func:`scalar_potential`.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): Axio-dilaton value.
            tau_c (complex): Complex conjugate value for axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            complex: Values of the :math:`F`-term conditions for the axio-dilaton :math:`\tau`.
            

        See also: :func:`superpotential`
    
        See also: :func:`dW_tau`
    
        See also: :func:`dK_tau`

        See also: :func:`dK_ctau`
        
        """
        if not conj:
            return self.dW_tau(moduli,tau,fluxes)+self.dK_tau(moduli,moduli_c,tau,tau_c)*self.superpotential(moduli,tau,fluxes)
        else:
            return self.dW_tau(moduli_c,tau_c,fluxes,conj=True)+self.dK_ctau(moduli,moduli_c,tau,tau_c)*self.superpotential(moduli_c,tau_c,fluxes,conj=True)
    
    
    @partial(jit, static_argnums = (0,6,))
    def DW(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the holomorphic Kähler covariant derivatives of the superpotential with respect to
        the complex structure moduli :math:`z^{i}` and the axio-dilaton :math:`\tau`.
        
        .. admonition:: Details
            :class: dropdown
        
            This function combines the outputs of the functions :func:`DW_z` and :func:`DW_tau`

            .. math::
                D_I W = \bigl(D_{1} W,\ldots, D_{h^{1,2}} W,D_{\tau} W\bigl)\, .

            This output is later used in the :math:`F`-term scalar potential :func:`scalar_potential`.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): Axio-dilaton value.
            tau_c (complex): Complex conjugate value for axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Values of the :math:`F`-term conditions for the complex structure moduli :math:`z^{i}`
                and the axio-dilaton :math:`\tau`.

        See also: :func:`DW_z`
    
        See also: :func:`DW_tau`
        
        """
        
        FTM = self.DW_z(moduli, moduli_c, tau, tau_c, fluxes, conj=conj)
        FTT = self.DW_tau(moduli, moduli_c, tau, tau_c, fluxes, conj=conj)
        
        return jnp.append(FTM, FTT)
         
        
    @partial(jit, static_argnums = (0,6,))
    def DW_x_canonical(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, conj: bool = False
        ) -> Tuple[Array,Array]:
        r"""
        
        **Description:**
        Returns the canonically normalised :math:`F`-term conditions :math:`F_I,\, F^I`.
        
        .. admonition:: Details
            :class: dropdown

            The canonically normalised :math:`F`-term conditions are defined as

            .. math::
                F_I = \mathrm{e}^{K/2}\, D_IW\, ,\quad F^I = \mathrm{e}^{K/2}\, K^{I\bar{J}}\, D_{\bar{J}}\overline{W}\, .

            They parametrise the contribution to the flux scalar potential :func:`scalar_potential`

            .. math::
                F^I F_I = \mathrm{e}^{K}\, K^{I\bar{J}}\, D_IW\, D_{\bar{J}} \overline{W} \, .

        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            Array: Canonically normalised :math:`F`-term conditions :math:`F_I`.
            Array: Canonically normalised :math:`F`-term conditions :math:`F^I`.
        
        """
        
        FT = self.DW(moduli, moduli_c, tau, tau_c, fluxes)
        cFT = self.DW(moduli, moduli_c, tau, tau_c, fluxes,conj=True)
        
        Inv_KM=self.inverse_kahler_metric(moduli, moduli_c, tau, tau_c)
        
        KP=self.kahler_potential(moduli, moduli_c, tau, tau_c)

        if conj:
            return jnp.exp(KP/2.)*cFT, jnp.exp(KP/2.)*jnp.matmul(Inv_KM,FT)
        else:
            return jnp.exp(KP/2.)*FT, jnp.exp(KP/2.)*jnp.matmul(cFT,Inv_KM)


    @partial(jit, static_argnums = (0,))
    def DW_x(
        self, x: ArrayLike, fluxes: ArrayLike
        ) -> Array:
        r"""
        
        **Description:**
        Calculates the real and imaginary parts of the :math:`F`-term conditions.
        
        .. admonition:: Details
            :class: dropdown
        
            This function is used for finding roots of the :math:`F`-term conditions.
            It takes as input a real array of values for the complex structure moduli :math:`z^i`
            and the axio-dilaton :math:`\tau`

            .. math::
                \bigl (\mathrm{Re}(z^1),\mathrm{Im}(z^1),\ldots,\mathrm{Re}(z^{h^{1,2}}),
                \mathrm{Im}(z^{h^{1,2}}),\mathrm{Re}(\tau),\mathrm{Im}(\tau)\bigl )\, .

            It then assembles the output of :func:`DW` 
            in a :math:`(2h^{1,2}+2)`-dimensional real vector
        
            .. math::
                    \bigl (\text{Re}(D_1 W), \text{Im}(D_1 W),\ldots, \text{Re}(D_{h^{1,2}} W),
                    \text{Im}(D_{h^{1,2}} W),\text{Re}(D_{\tau} W), \text{Im}(D_{\tau} W)\bigl )\, .
        
        Args:
            x (ArrayLike): Array of shape (:math:`2(h^{1,2}+1)`, ) containing real and imaginary parts of
                the complex structure moduli :math:`z^i` and axio-dilaton :math:`\tau`.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
        
        Returns:
            ArrayLike: Vector of shape (:math:`2(h^{1,2}+1)`, ) containing real and imaginary parts 
                of the :math:`F`-term conditions in alternating order.       

        See also: :func:`DW`

        """

        moduli,moduli_c,tau,tau_c = self._convert_real_to_complex(x)
    
        FT = self.DW(moduli,moduli_c,tau,tau_c,fluxes)
    
        return jnp.column_stack((jnp.real(FT), jnp.imag(FT))).flatten()
    
    @partial(jit, static_argnums = (0,))
    def dDW_x(
        self, x: ArrayLike, fluxes: ArrayLike
        ) -> Array:
        r"""
        
        **Description:**
        Returns the first derivatives of the F-term conditions by differentiating with respect to the real fields.
        
        Args:
            x (ArrayLike): Array of shape (:math:`2(h^{1,2}+1)`, ) containing real and imaginary parts of
                the complex structure moduli :math:`z^i` and axio-dilaton :math:`\tau`.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
        
        Returns:
            ArrayLike: First derivatives of the real :math:`F`-term conditions with respect to the 
                real and imaginary parts of the complex structure moduli :math:`z^i` and axio-dilaton :math:`\tau`.

        See also: :func:`DW_x`
        
        """
        
        return jax.jacrev(self.DW_x,argnums=0,holomorphic=False)(x,fluxes)

    @partial(jit, static_argnums = (0,))
    def ddDW_x(
        self, x: ArrayLike, fluxes: ArrayLike
        ) -> Array:
        r"""
        
        **Description:**
        Returns the second derivatives of the F-term conditions by differentiating with respect to the real fields.
        
        Args:
            x (ArrayLike): Array of shape (:math:`2(h^{1,2}+1)`, ) containing real and imaginary parts of
                the complex structure moduli :math:`z^i` and axio-dilaton :math:`\tau`.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
        
        Returns:
            ArrayLike: Second derivatives of the real :math:`F`-term conditions with respect to the 
                real and imaginary parts of the complex structure moduli :math:`z^i` and axio-dilaton :math:`\tau`.

        See also: :func:`DW_x`
        
        """
        
        return jax.jacrev(self.dDW_x,argnums=0,holomorphic=False)(x,fluxes)
        
    ###################################################################################################################
    ############################## 2nd holomorphic derivatives of F-term CONDITIONS ###################################
    ###################################################################################################################

    @partial(jit, static_argnums = (0,6,))
    def dDW_z_z(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the holomorphic derivative :math:`\partial_{z^i}D_{z^j}W` of the :math:`F`-term
        :math:`D_{z^j}W` with respect to the complex structure moduli :math:`z^i`.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Value of :math:`\partial_{z^i}D_{z^j}W`.
        
        """
        
        if conj:
            return jax.jacrev(self.DW_z,argnums=1,holomorphic=True)(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
        else:
            return jax.jacrev(self.DW_z,argnums=0,holomorphic=True)(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)

    
    @partial(jit, static_argnums = (0,6,))
    def dDW_tau_tau(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, conj: bool = False
        ) -> complex:
        r"""
        
        **Description:**
        Returns the holomorphic derivative :math:`\partial_{\tau}D_{\tau}W` of the :math:`F`-term
        :math:`D_{\tau}W` with respect to the axio-dilaton :math:`\tau`.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Value of :math:`\partial_{\tau}D_{\tau}W`.
        
        """

        if conj:
            return jax.grad(self.DW_tau,argnums=3,holomorphic=True)(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
        else:
            return jax.grad(self.DW_tau,argnums=2,holomorphic=True)(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
        
    @partial(jit, static_argnums = (0,6,))
    def dDW_z_tau(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the holomorphic derivative :math:`\partial_{\tau}D_{z^j}W` of the :math:`F`-term
        :math:`D_{z^j}W` with respect to the axio-dilaton :math:`\tau`.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Value of :math:`\partial_{\tau}D_{z^j}W`.
        
        """
        if conj:
            return jax.jacrev(self.DW_z,argnums=3,holomorphic=True)(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
        else:
            return jax.jacrev(self.DW_z,argnums=2,holomorphic=True)(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
        
    @partial(jit, static_argnums = (0,6,))
    def dDW_tau_z(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the holomorphic derivative :math:`\partial_{z^i}D_{\tau}W` of the :math:`F`-term
        :math:`D_{\tau}W` with respect to the complex structure moduli :math:`z^i`.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Value of :math:`\partial_{z^i}D_{\tau}W`.
        
        """

        if conj:
            return jax.grad(self.DW_tau,argnums=1,holomorphic=True)(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
        else:
            return jax.grad(self.DW_tau,argnums=0,holomorphic=True)(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)

    @partial(jit, static_argnums = (0,6,))
    def dDW_z(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the holomorphic derivative :math:`\partial_{z^i}D_{J}W` of the :math:`F`-terms
        :math:`D_{J}W` with respect to the complex structure moduli :math:`z^i`.
        
        .. note::
            The output shape is such that ``output[J-1][i-1]`` corresponds to :math:`\partial_{i}D_{J}W`
            with :math:`J=1,\ldots, h^{1,2}+1` and :math:`i=1,\ldots, h^{1,2}`.

        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Value of :math:`\partial_{I}D_{z^j}W`.
        
        """

        if conj:
            return jax.jacrev(self.DW,argnums=1,holomorphic=True)(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
        else:
            return jax.jacrev(self.DW,argnums=0,holomorphic=True)(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)


    @partial(jit, static_argnums = (0,6,))
    def dDW_tau(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the holomorphic derivative :math:`\partial_{\tau}D_{J}W` of the :math:`F`-terms
        :math:`D_{J}W` with respect to the the axio-dilaton :math:`\tau`.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Value of :math:`\partial_{\tau}D_{J}W`.
        
        """
        if conj:
            return jax.jacrev(self.DW,argnums=3,holomorphic=True)(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
        else:
            return jax.jacrev(self.DW,argnums=2,holomorphic=True)(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)

    @partial(jit, static_argnums = (0,6,))
    def dDW(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the holomorphic derivative :math:`\partial_{I}D_{J}W` of the :math:`F`-terms
        :math:`D_{J}W` with respect to the complex structure moduli :math:`z^i` and  the axio-dilaton :math:`\tau`.

        .. note::
            The output shape is such that ``output[J-1][I-1]`` corresponds to :math:`\partial_{I}D_{J}W`
            with :math:`I,J=1,\ldots, h^{1,2}+1`.

        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Value of :math:`\partial_{I}D_{J}W`.
        
        """

        dDWz = self.dDW_z(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
        dDWtau = self.dDW_tau(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
        
        return jnp.append(dDWz,jnp.array([dDWtau]).T,axis=1)

    ###################################################################################################################
    ############################ 2nd anti-holomorphic derivatives of F-term CONDITIONS ################################
    ###################################################################################################################


    @partial(jit, static_argnums = (0,6,))
    def dDW_z_cz(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the holomorphic derivative :math:`\partial_{\overline{z}^i}D_{z^j}W` of the :math:`F`-term
        :math:`D_{z^j}W` with respect to the complex conjugate complex structure moduli :math:`\overline{z}^i`.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Value of :math:`\partial_{\overline{z}^i}D_{z^j}W`.
        
        """

        km = self.ddK_z_cz(moduli, moduli_c, tau, tau_c)

        if conj:
            return km.T*self.superpotential(moduli_c,tau_c,fluxes,conj=conj)
        else:
            return km*self.superpotential(moduli,tau,fluxes,conj=conj)

    
    @partial(jit, static_argnums = (0,6,))
    def dDW_tau_ctau(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the holomorphic derivative :math:`\partial_{\overline{\tau}}D_{\tau}W` of the :math:`F`-term
        :math:`D_{\overline{\tau}}W` with respect to the complex conjugate axio-dilaton :math:`\overline{\tau}`.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Value of :math:`\partial_{\overline{\tau}}D_{\tau}W`.
        
        """

        # The Käher metric component is real and a scalar...
        km = self.ddK_tau_ctau(moduli, moduli_c, tau, tau_c)

        if conj:
            return km*self.superpotential(moduli_c,tau_c,fluxes,conj=conj)
        else:
            return km*self.superpotential(moduli,tau,fluxes,conj=conj)
        
    
    @partial(jit, static_argnums = (0,6,))
    def dDW_z_ctau(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the holomorphic derivative :math:`\partial_{\overline{\tau}}D_{z^j}W` of the :math:`F`-term
        :math:`D_{z^j}W` with respect to the complex conjugate axio-dilaton :math:`\overline{\tau}`.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Value of :math:`\partial_{\overline{\tau}}D_{z^j}W`.
        
        """

        if conj:
            km = self.ddK_cz_tau(moduli, moduli_c, tau, tau_c)
            return km*self.superpotential(moduli_c,tau_c,fluxes,conj=conj)
        else:
            km = self.ddK_z_ctau(moduli, moduli_c, tau, tau_c)
            return km*self.superpotential(moduli,tau,fluxes,conj=conj)
        
    @partial(jit, static_argnums = (0,6,))
    def dDW_tau_cz(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the holomorphic derivative :math:`\partial_{\overline{z}^i}D_{\tau}W` of the :math:`F`-term
        :math:`D_{\tau}W` with respect to the complex conjugate complex structure moduli :math:`\overline{z}^i`.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Value of :math:`\partial_{\overline{z}^i}D_{\tau}W`.
        
        """

        if conj:
            km = self.ddK_z_ctau(moduli, moduli_c, tau, tau_c)
            return km*self.superpotential(moduli_c,tau_c,fluxes,conj=conj)
        else:
            km = self.ddK_cz_tau(moduli, moduli_c, tau, tau_c)
            return km*self.superpotential(moduli,tau,fluxes,conj=conj)


    @partial(jit, static_argnums = (0,6,))
    def dDW_cz(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the holomorphic derivative :math:`\partial_{\overline{z}^i}D_{J}W` of the :math:`F`-term
        :math:`D_{J}W` with respect to the complex conjugate complex structure moduli :math:`\overline{z}^i`.
        
        .. note::
            The output shape is such that ``output[J-1][i-1]`` corresponds to :math:`\partial_{\bar{\imath}}D_{J}W`
            with :math:`J=1,\ldots, h^{1,2}+1` and :math:`i=1,\ldots, h^{1,2}`.

        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Value of :math:`\partial_{\overline{z}^i}D_{J}W`.

        See also: :func:`dDW_z`
        
        """
        if conj:
            return jax.jacrev(self.DW,argnums=0,holomorphic=True)(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
        else:
            return jax.jacrev(self.DW,argnums=1,holomorphic=True)(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)

    @partial(jit, static_argnums = (0,6,))
    def dDW_ctau(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the holomorphic derivative :math:`\partial_{\overline{\tau}}D_{J}W` of the :math:`F`-term
        :math:`D_{J}W` with respect to the complex conjugate axio-dilaton :math:`\overline{\tau}`.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Value of :math:`\partial_{\overline{\tau}}D_{J}W`.

        See also: :func:`dDW_tau`
        
        """
        
        if conj:
            return jax.jacrev(self.DW,argnums=2,holomorphic=True)(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
        else:
            return jax.jacrev(self.DW,argnums=3,holomorphic=True)(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)

    @partial(jit, static_argnums = (0,6,))
    def dDW_c(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the holomorphic derivative :math:`\partial_{\overline{I}}D_{J}W` of the :math:`F`-terms
        :math:`D_{J}W` with respect to the complex conjugate complex structure moduli :math:`\overline{z}^i` 
        and the axio-dilaton :math:`\overline{\tau}`.

        .. note::
            The output shape is such that ``output[J-1][I-1]`` corresponds to :math:`\partial_{\overline{I}}D_{J}W`
            with :math:`I,J=1,\ldots, h^{1,2}+1`.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Value of :math:`\partial_{\overline{I}}D_{J}W`.

        See also: :func:`dDW`
        
        """

        dDWcz = self.dDW_cz(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
        dDWctau = self.dDW_ctau(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
        
        return jnp.append(dDWcz,jnp.array([dDWctau]).T,axis=1)

    @partial(jit, static_argnums = (0,))
    def dDW_real(
        self, moduli: ArrayLike, tau: complex, fluxes: ArrayLike
        ) -> Array:
        r"""
        
        **Description:**
        Returns the first derivative :math:`\partial_{\phi^{\alpha}}D_{J}W` of the :math:`F`-terms
        :math:`D_{J}W` with respect to the real fields :math:`\phi^{\alpha}`.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            tau (complex): : Value of axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            
        Returns:
            ArrayLike: Value of :math:`\partial_{\overline{I}}D_{J}W`.

        See also: :func:`dDW`

        See also: :func:`dDW_c`

        See also: :func:`_newton_method_flux_vacua_complex`
        
        """

        dDW = self.dDW(moduli,jnp.conj(moduli),tau,jnp.conj(tau),fluxes)
        dcDW = self.dDW_c(moduli,jnp.conj(moduli),tau,jnp.conj(tau),fluxes)

        dt_DW = 1j*(dDW-dcDW)
        dphi_DW = dDW+dcDW

        return jnp.block([[jnp.real(dt_DW),jnp.real(dphi_DW)],[jnp.imag(dt_DW),jnp.imag(dphi_DW)]])
        
    

    ###################################################################################################################
    ############################ SECOND KÄHLER COVARIANT DERIVATIVES OF F-TERMS #######################################
    ###################################################################################################################
        
    ###################################################################################################################
    ######################################## Holomorphic derivatives  #################################################
    ###################################################################################################################

    @partial(jit, static_argnums = (0,6,))
    def DDW_z_z(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the Kähler covariant derivative :math:`D_{z^i}D_{z^j}W` of the :math:`F`-term
        :math:`D_{z^j}W` with respect to the complex structure moduli :math:`z^i`.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Value of :math:`D_{z^i}D_{z^j}W`.
        
        """
        

        DWz = self.DW_z(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
        if conj:
            DWz_Kz = jnp.outer(DWz,self.dK_cz(moduli,moduli_c,tau,tau_c))
        else:
            DWz_Kz = jnp.outer(DWz,self.dK_z(moduli,moduli_c,tau,tau_c))

        return self.dDW_z_z(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)+DWz_Kz
    
    @partial(jit, static_argnums = (0,6,))
    def DDW_tau_tau(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the Kähler covariant derivative :math:`D_{\tau}D_{\tau}W` of the :math:`F`-term
        :math:`D_{\tau}W` with respect to the axio-dilaton :math:`\tau`.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Value of :math:`D_{\tau}D_{\tau}W`.
        
        """
        
        if conj:
            Kt = self.dK_ctau(moduli,moduli_c,tau,tau_c)
        else:
            Kt = self.dK_tau(moduli,moduli_c,tau,tau_c)

        return self.dDW_tau_tau(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)+Kt*self.DW_tau(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
            
    @partial(jit, static_argnums = (0,6,))
    def DDW_z_tau(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the Kähler covariant derivative :math:`D_{\tau}D_{z^j}W` of the :math:`F`-term
        :math:`D_{z^j}W` with respect to the axio-dilaton :math:`\tau`.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Value of :math:`D_{\tau}D_{z^j}W`.
        
        """
        
        
        if conj:
            Kt = self.dK_ctau(moduli,moduli_c,tau,tau_c)
        else:
            Kt = self.dK_tau(moduli,moduli_c,tau,tau_c)

        return self.dDW_z_tau(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)+Kt*self.DW_z(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
    
    @partial(jit, static_argnums = (0,6,))
    def DDW_tau_z(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the Kähler covariant derivative :math:`D_{z^i}D_{\tau}W` of the :math:`F`-term
        :math:`D_{\tau}W` with respect to the complex structure moduli :math:`z^i`.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Value of :math:`D_{z^i}D_{\tau}W`.
        
        """
        

        if conj:
            Kz = self.dK_cz(moduli,moduli_c,tau,tau_c)
        else:
            Kz = self.dK_z(moduli,moduli_c,tau,tau_c)

        return self.dDW_tau_z(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)+Kz*self.DW_tau(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
        
    
    @partial(jit, static_argnums = (0,6,))
    def DDW_general(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the second Kähler covariant derivatives :math:`D_I D_J W` of the superpotential :math:`W`
        with respect to the axio-dilaton :math:`\tau` and the complex structure moduli :math:`z^{i}`.
        
        .. admonition:: Details
            :class: dropdown

            We compute the Kähler covariant derivatives of the :math:`F`-terms :math:`D_J W`
        
            .. math::
                D_I D_J W = \partial_I D_J W + (\partial_I K) D_J W

            using automatic differentiation. To do so, we combine the outputs of :func:`DDW_z_z`,
            :func:`DDW_tau_tau`, :func:`DDW_z_tau` and :func:`DDW_tau_z`.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Second Kähler covariant derivatives :math:`D_I D_J W` of the superpotential :math:`W`
                with respect to the axio-dilaton :math:`\tau` and  the moduli :math:`z^{i}`.
        
        """

        DWval = self.DW(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)

        if conj:
            DW_dK = jnp.outer(DWval,self.dK_c(moduli,moduli_c,tau,tau_c))
        else:
            DW_dK = jnp.outer(DWval,self.dK(moduli,moduli_c,tau,tau_c))

        return self.dDW(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)+DW_dK
        

        
    ###################################################################################################################
    ##################################### Anti-holomorphic derivatives  ###############################################
    ###################################################################################################################

    @partial(jit, static_argnums = (0,6,))
    def DDW_z_cz(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the Kähler covariant derivative :math:`D_{\overline{z}^i}D_{z^j}W` of the :math:`F`-term
        :math:`D_{z^j}W` with respect to the complex conjugate complex structure moduli :math:`\overline{z}^i`.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Value of :math:`D_{\overline{z}^i}D_{z^j}W`.
        
        """

        DW = self.DW_z(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)

        if conj:
            KDW = jnp.outer(DW,self.dK_z(moduli,moduli_c,tau,tau_c))
        else:
            KDW = jnp.outer(DW,self.dK_cz(moduli,moduli_c,tau,tau_c))
        
        return self.dDW_z_cz(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)+KDW
    
    @partial(jit, static_argnums = (0,6,))
    def DDW_tau_ctau(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the Kähler covariant derivative :math:`D_{\overline{\tau}}D_{\tau}W` of the :math:`F`-term
        :math:`D_{\tau}W` with respect to the complex conjugate axio-dilaton :math:`\overline{\tau}`.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Value of :math:`D_{\overline{\tau}}D_{\tau}W`.
        
        """
        

        DW = self.DW_tau(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)

        if conj:
            KDW = self.dK_tau(moduli,moduli_c,tau,tau_c)*DW
        else:
            KDW = self.dK_ctau(moduli,moduli_c,tau,tau_c)*DW

        return self.dDW_tau_ctau(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)+KDW
            
    @partial(jit, static_argnums = (0,6,))
    def DDW_z_ctau(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the Kähler covariant derivative :math:`D_{\overline{\tau}}D_{z^j}W` of the :math:`F`-term
        :math:`D_{z^j}W` with respect to the complex conjugate axio-dilaton :math:`\overline{\tau}`.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Value of :math:`D_{\overline{\tau}}D_{z^j}W`.
        
        """
        

        DW = self.DW_z(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)

        if conj:
            KDW = self.dK_tau(moduli,moduli_c,tau,tau_c)*DW
        else:
            KDW = self.dK_ctau(moduli,moduli_c,tau,tau_c)*DW

        return self.dDW_z_ctau(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)+KDW
    
    @partial(jit, static_argnums = (0,6,))
    def DDW_tau_cz(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the Kähler covariant derivative :math:`D_{\overline{z}^i}D_{\tau}W` of the :math:`F`-term
        :math:`D_{\tau}W` with respect to the complex conjugate complex structure moduli :math:`\overline{z}^i`.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Value of :math:`D_{\overline{z}^i}D_{\tau}W`.
        
        """

        DW = self.DW_tau(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)

        if conj:
            KDW = self.dK_z(moduli,moduli_c,tau,tau_c)*DW
        else:
            KDW = self.dK_cz(moduli,moduli_c,tau,tau_c)*DW

        return self.dDW_tau_cz(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)+KDW


    
    @partial(jit, static_argnums = (0,6,))
    def DcDW(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns a matrix containing the mixed second Kähler derivatives :math:`D_{\overline{I}}D_{J}W`.
        
        .. admonition:: Details
            :class: dropdown

            More explicitly, we compute

            .. math::
                D_A D_{\overline{B}} W=K_{\overline{B}}D_{A}W+K_{A\overline{B}}W

            Generically, this function is *not* symmetric, but it is in the SUSY case where :math:`D_AW=0` since

            .. math::
                D_A D_{\overline{B}} W\bigl |_{D_AW=0}=K_{A\overline{B}}W\, .

            In particular, this is symmetric.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Values of :math:`D_{\overline{I}}D_{J}W`.
        
        """

        """
        # OLD VERSION
        DcDMod=self.DDW_z_cz(moduli,moduli_c,tau,tau_c,fluxes)
        DcDTau=self.DDW_tau_ctau(moduli,moduli_c,tau,tau_c,fluxes)
        DTaucDMod=self.DDW_z_ctau(moduli,moduli_c,tau,tau_c,fluxes)
        DModcDTau=self.DDW_tau_cz(moduli,moduli_c,tau,tau_c,fluxes)
    
        a = jnp.hstack((jnp.asarray(DcDMod), jnp.asarray(DTaucDMod).reshape(self.h12, 1)))

        b = jnp.hstack((jnp.asarray(DModcDTau), jnp.asarray(DcDTau)))

        return jnp.vstack((a, b))
        """

        DW = self.DW(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
        
        if conj:
            KDW = jnp.outer(DW,self.dK(moduli,moduli_c,tau,tau_c))
        else:
            KDW = jnp.outer(DW,self.dK_c(moduli,moduli_c,tau,tau_c))

        return self.dDW_c(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)+KDW
        
    @partial(jit, static_argnums = (0,6,7,))
    def DDW_SUSY(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, conj: bool = False, mode: str = "block diagonal"
        ) -> Array:
        r"""
        
        **Description:**
        Returns a matrix containing the second holomorphic Kähler derivative of the superpotential 
        assuming that the F-flatness conditions :math:`D_{i}W=D_{\tau}W=0` are satisfied.
        
        
        .. admonition:: Details
            :class: dropdown
        
            Returns a matrix containing the second holomorphic Kähler derivative of the superpotential 
            assuming that the F-flatness conditions :math:`D_{i}W=D_{\tau}W=0` are satisfied. 
            More explicitly, we compute

            .. math::
                D_I D_I W = \partial_I D_J W + K_I D_J W 
                          = \partial_I \partial_J W + (K_{IJ}-K_I K_J) W + K_I D_J W + K_J D_I W \, .
            
            At the SUSY minimum, this suggests

            .. math::
                D_I D_I W\bigl |_{D_I W=0} = \partial_I \partial_J W + (K_{IJ}-K_I K_J) W\, .

            For the axio-dilaton, we have :math:`W_{\tau\tau}=0` for the GVW superpotential and thus

            .. math::
                D_\tau D_\tau W\bigl |_{D_\tau W=0} =  (K_{\tau\tau} - K_\tau K_\tau) W\, .

            For the standard Kähler potential :math:`K\supset -\log(-\mathrm{i}(\tau-\bar{\tau})), 
            this automatically implies

            .. math::
                D_\tau D_\tau W\bigl |_{D_\tau W=0} =  0\; , \quad D_\tau D_i W\bigl |_{D_I W=0} =  0\, .
        
        .. warning::
            We assume that there is a block-diagonal structure in the Kahler potential such that the mixed second derivatives vanish :math:`K_{i\tau}=K_{\tau i}=0`. This might not necessarily be true once further corrections to the Kahler potential are included.
        
        .. warning::
            Further, we use that :math:`W_{\tau\tau}=0` because the flux superpotential :func:`superpotential` is linear in :math:`\tau`. Again, this might not necessarily be true anymore once non-perturbative instanton effects are present in the superpotential!
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            mode (string, optional): Whether to assume that the Kähler metric is block-diagonal. Defaults to ``"block diagonal"``.
        
        Returns:
            ArrayLike: Value of :math:`D_I D_J W` assuming :math:`D_IW=0`.
        
        
        """

        if conj:
            K_ZZ=self.ddK_cz_cz(moduli, moduli_c, tau, tau_c)
        
            if mode!="block diagonal":
                K_tau_tau=self.ddK_ctau_ctau(moduli, moduli_c, tau, tau_c)

            K_Z=self.dK_cz(moduli, moduli_c, tau, tau_c)
            K_tau=self.dK_ctau(moduli, moduli_c, tau, tau_c)
            
            W_mod_tau=self.ddW_z_tau(moduli_c,tau_c,fluxes,conj=conj)
            
            W_mod_mod=self.ddW_z_z(moduli_c,tau_c,fluxes,conj=conj)
            
            W_val=self.superpotential(moduli_c,tau_c,fluxes,conj=conj)
            
            
            if mode!="block diagonal":
                ddK_z_tau = self.ddK_cz_ctau(moduli, moduli_c, tau, tau_c)

        else:
            K_ZZ=self.ddK_z_z(moduli, moduli_c, tau, tau_c)
        
            if mode!="block diagonal":
                K_tau_tau=self.ddK_tau_tau(moduli, moduli_c, tau, tau_c)
                
            K_Z=self.dK_z(moduli, moduli_c, tau, tau_c)
            K_tau=self.dK_tau(moduli, moduli_c, tau, tau_c)
            
            W_mod_tau=self.ddW_z_tau(moduli,tau,fluxes)
            
            W_mod_mod=self.ddW_z_z(moduli,tau,fluxes)
            
            W_val=self.superpotential(moduli,tau,fluxes)
            
            if mode!="block diagonal":
                
                ddK_z_tau = self.ddK_z_tau(moduli, moduli_c, tau, tau_c)
                


        if mode=="block diagonal":
            # DDTau = 0 because K_tau_tau-K_tau*K_tau = 0 for the standard Kähler potential
            DDTau = 0.+1j*0.
            DTauDMod=W_mod_tau.reshape(self.h12,1)-jnp.outer(K_Z,K_tau)*W_val
            DModDTau=W_mod_tau-jnp.outer(K_tau,K_Z)[0]*W_val
        else:
            warnings.warn("TODO: NOT TESTED!")
            DDTau=(K_tau_tau-K_tau*K_tau)*W_val
            DTauDMod=W_mod_tau.reshape(self.h12,1)+(ddK_z_tau-jnp.outer(K_Z,K_tau))*W_val
            DModDTau=W_mod_tau+(ddK_z_tau-jnp.outer(K_tau,K_Z)[0])*W_val
            
        DDMod=W_mod_mod+(K_ZZ-jnp.outer(K_Z,K_Z))*W_val

        a = jnp.hstack((jnp.asarray(DDMod), jnp.asarray(DTauDMod).reshape(self.h12, 1)))

        b = jnp.append(jnp.asarray(DModDTau), jnp.asarray([DDTau]),axis=0)

        return jnp.vstack((a, b))
        
    @partial(jit, static_argnums = (0,6,7,))
    def DDW(
            self, 
            moduli: ArrayLike, 
            moduli_c: ArrayLike, 
            tau: complex, 
            tau_c: complex, 
            fluxes: ArrayLike, 
            conj: bool = False, 
            mode: str = None
            ) -> Array:
        r"""
        
        **Description:**
        Returns a matrix the second holomorphic Kähler derivatives :math:`D_I D_J W` of the superpotential.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            mode (string, optional): The mode to :math:`DDW`. Currently implemented modes are:
                                            * None: general expression from explicit second derivatives.
                                            * "SUSY": computes at a SUSY locus where DW=0 using standard SUGRA formulas.
        
        Returns:
            ArrayLike: Value of :math:`D_I D_J W`.
        
        """

        modes = [None, "SUSY"]
        if mode not in modes:
            raise ValueError(f"Mode must be one of {modes}, but is {mode}!")
        
        
        if mode is None:
            return self.DDW_general(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)
        elif mode == "SUSY":
            return self.DDW_SUSY(moduli,moduli_c,tau,tau_c,fluxes,conj=conj)


    @partial(jit, static_argnums = (0,))
    def DDW_matrix_SUSY(
                        self, 
                        moduli: ArrayLike, 
                        moduli_c: ArrayLike, 
                        tau: complex, 
                        tau_c: complex, 
                        fluxes: ArrayLike
                        ) -> Array:
        r"""
        
        **Description:**
        Returns the matrix of second Kähler derivatives of the superpotential assuming that the F-term conditions are satisfied for all fields.
        
        .. admonition:: Details
            :class: dropdown
        
            Explicitly, we compute
        
            .. math::
                M=\left (\begin{array}{cc}\partial_{A}\partial_{B} W + \left ( K_{A B} - K_{A} K_{B} \right ) W & K_{A\overline{B}} W\ K_{A\overline{B}} \overline{W} & \partial_{\overline{A}}\partial_{\overline{B}} \overline{W} + \left ( K_{\overline{A} \overline{B}} - K_{\overline{A}} K_{\overline{B}} \right ) \overline{W} \end{array}\right )
        
            which is equivalent to the matrix of second derivatives of :math:`W` (see :func:`massive_directions_general`) provided that the F-flatness conditions
        
            .. math::
                D_{Z^{i}} W = D_{\tau} W = 0
        
            are satisfied. In this case, it turns out that
        
            .. math::
                D_{A} D_{\overline{B}} W = K_{A\overline{B}} W
        
            as well as
        
            .. math::
                D_{A} D_{B} W = \partial_{A}\partial_{B} W + \left ( K_{A B} - K_{A} K_{B} \right ) W
        
            as computed via :func:`DDW_SUSY`.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
        
        Returns:
            See also: :func:`massive_directions_general`
            See also: :func:`DDW_SUSY`
        
        """

        
        
        DDW_val=self.DDW(moduli, moduli_c, tau, tau_c,fluxes,mode="SUSY")
        W_val=self.superpotential(moduli, tau,fluxes)
        KM_val=jnp.array(self.kahler_metric(moduli,moduli_c,tau,tau_c))
        
        a = jnp.hstack((DDW_val, KM_val * W_val))

        b = jnp.hstack((jnp.conj(KM_val * W_val), jnp.conj(DDW_val)))

        return jnp.vstack((a, b))
        
    @partial(jit, static_argnums = (0,))
    def DDW_matrix_general(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, fluxes: ArrayLike
        ) -> Array:
        r"""
        **Description:**
        Returns the matrix of second Kähler derivatives of the superpotential.
        
        .. admonition:: Details
            :class: dropdown
        
            Explicitly, we compute
        
            .. math::
                M=\left (\begin{array}{cc} D_{A}D_{B} W & D_{A}D_{\overline{B}} W \ D_{\overline{A}}D_{B} \overline{W} & D_{\overline{A}}D_{\overline{B}} \overline{W} \end{array}\right )
        
            which is encodes the mass terms for chiral fermions.
        
            In this case, it turns out that
        
            .. math::
                D_{A} D_{\overline{B}} W =K_{\overline{B}}W_{A}+\left ( K_{A\overline{B}}+K_{A}K_{\overline{B}}\right ) W
        
            TODO: WHAT DOES THIS COMPUTE?
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
        
        Returns:
            (ArrayLike):
        
        """

        DDW_val=self.DDW(moduli,moduli_c,tau,tau_c,fluxes)
        DcDW_val=self.DcDW(moduli,moduli_c,tau,tau_c,fluxes)
        
        a = jnp.hstack((DDW_val, DcDW_val))

        b = jnp.hstack((jnp.conj(DcDW_val), jnp.conj(DDW_val)))

        return jnp.vstack((a, b))
        
    @partial(jit, static_argnums = (0,6))
    def DDW_matrix(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, fluxes: ArrayLike, mode: str = "SUSY"
        ) -> Array:
        r"""
        
        **Description:**
        Returns the matrix of masses for chiral fermions.
        
        
        .. admonition:: Details
            :class: dropdown
        
            The diagonal entries of this matrix (evaluated at a minimum) are proportional to the mass matrix of chiral fermions, see equation (2.5) in `1312.5659 <https://arxiv.org/pdf/1312.5659.pdf>`_. By abuse of terminology, we call this matrix the ''fermionic mass matrix''.
        
        .. note::
        
            To speed up the computation, it can be advantageous to compute this function using the ``mode="SUSY"`` option provided that :math:`DW=0` for *all* fields.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            mode (string, optional): Whether or not the point at which the mass matrix is evaluated corresponds to a minimum with :math:`DW=0` for *all* fields. Default is ``mode="SUSY"``.
        
        Returns:
            (ArrayLike):
        
        """

        modes=[None, "SUSY"]
        if mode not in modes:
            raise ValueError(f"Mode must be one of {modes}, but is {mode}!")
        
        if mode=="SUSY":
            return self.DDW_matrix_SUSY(moduli, moduli_c, tau, tau_c, fluxes)
        elif mode is None:
            return self.DDW_matrix_general(moduli, moduli_c, tau, tau_c, fluxes)

    ###################################################################################################################
    ######################################### F-TERM SCALAR POTENTIAL #################################################
    ###################################################################################################################
    

    @partial(jit, static_argnums = (0,6,7,))
    def scalar_potential(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, noscale: bool = True, normalise: bool = False
        ) -> complex:
        r"""
        
        **Description:**
        Returns the value of the :math:`F`-term scalar potential.
        
        .. admonition:: Details
            :class: dropdown
        
            The :math:`F`-term scalar potential in 4D :math:`\mathcal{N}=1` supergravity is given by

            .. math::
                V_F=\mathrm{e}^{K}\,\bigl (K^{I\bar{J}}\, D_IW\, D_{\bar{J}} \overline{W}-3|W|^2\bigl )\, .

            Here, the indices :math:`I = (1,\ldots,h^{1,2}+1)` run only over the axio-dilaton 
            and the complex structure moduli ignoring the Kähler moduli.
            Due to the no-scale structure of the classical Kähler potential, the term :math:`-3|W|^2`
            is cancelled by the Kähler moduli sector giving rise to the no-scale flux scalar potential

            .. math::
                V_F=\mathrm{e}^{K}\, K^{I\bar{J}}\, D_IW\, D_{\bar{J}} \overline{W}\, .

            In more detail, we have at the level of the classical :math:`\mathcal{N}=2` Kähler potential
            for the axio-dilaton :math:`\tau` and the complex structure moduli :math:`z^{i}`

            .. math::
                V_F=\mathrm{e}^{K}\, \bigl (K^{i\bar{\jmath}}\, D_iW\, D_{\bar{\jmath}} \overline{W}
                +K^{\tau\bar{\tau}}\, D_\tau W\, D_{\bar{\tau}} \overline{W}\bigl )\, .
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): Axio-dilaton value.
            tau_c (complex): Complex conjugate value for axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            noscale (bool, optional): If ``True``, uses the no-scale flux scalar potential. Defaults to ``True``.
            normalise (bool, optional): If ``True``, rescales superpotential by :math:`\sqrt{2/\pi}`. Defaults to ``False``.

        Returns:
            complex: Value of the :math:`F`-term scalar potential induced by 3-form fluxes.

        Aliases:
            :func:`V`, :func:`scalar_potential`

        See also: :func:`DW`
    
        See also: :func:`inverse_kahler_metric`
    
        See also: :func:`kahler_potential`
        
        """

        FT = self.DW(moduli, moduli_c, tau, tau_c, fluxes)
        cFT = self.DW(moduli, moduli_c, tau, tau_c, fluxes,conj=True)
    
        Inv_KM = self.inverse_kahler_metric(moduli, moduli_c, tau, tau_c)
        
        KP = self.kahler_potential(moduli, moduli_c, tau, tau_c)

        if noscale:
            gravitino_mass_square=0.
        else:
            W=self.superpotential(moduli,tau,fluxes)
            cW=self.superpotential(moduli_c,tau_c,fluxes,conj=True)
            gravitino_mass_square=jnp.exp(KP) * W * cW

        V0 = jnp.exp(KP) * jnp.matmul(cFT, jnp.matmul(Inv_KM, FT))-3.*gravitino_mass_square

        # Normalise from normalisation of W!
        if normalise:
            V0 = 2./jnp.pi*V0
        
        return V0

    V = scalar_potential


    @partial(jit, static_argnums = (0,6,7,))
    def dV_z(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, noscale: bool = True, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the first holomorphic derivative :math:`\partial_{z^i}V` of the 
        :math:`F`-term scalar potential :math:`V` with respect to the complex structure moduli :math:`z^i`.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): Axio-dilaton value.
            tau_c (complex): Complex conjugate value for axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            noscale (bool, optional): If ``True``, uses the no-scale flux scalar potential. Defaults to ``True``.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Value of :math:`\partial_{z^i}V`.

        
        """
        if conj:
            return jax.jacrev(self.V, argnums = 1, holomorphic = True)(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale)
        else:
            return jax.jacrev(self.V, argnums = 0, holomorphic = True)(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale)

    @partial(jit, static_argnums = (0,6,7,))
    def dV_tau(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, noscale: bool = True, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the first holomorphic derivative :math:`\partial_{\tau}V` of the 
        :math:`F`-term scalar potential :math:`V` with respect to the axio-dilaton :math:`\tau`.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): Axio-dilaton value.
            tau_c (complex): Complex conjugate value for axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            noscale (bool, optional): If ``True``, uses the no-scale flux scalar potential. Defaults to ``True``.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Value of :math:`\partial_{\tau}V`.

        
        """
        if conj:
            return jax.jacrev(self.V, argnums = 3, holomorphic = True)(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale)
        else:
            return jax.jacrev(self.V, argnums = 2, holomorphic = True)(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale)
        

    @partial(jit, static_argnums = (0,6,7,8,))
    def dV(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, noscale: bool = True, conj: bool = False, mode="complex"
        ) -> Array:
        r"""
        
        **Description:**
        Returns the first holomorphic derivative :math:`\partial_{I}V` of the 
        :math:`F`-term scalar potential :math:`V`.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): Axio-dilaton value.
            tau_c (complex): Complex conjugate value for axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            noscale (bool, optional): If ``True``, uses the no-scale flux scalar potential. Defaults to ``True``.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``. Becomes void if ``mode="real"``.
            mode (str, optional): String specifying whether to return complex or real derivatives. Defaults to ``"complex"``.
        
        Returns:
            ArrayLike: Value of :math:`\partial_{I}V`.

        
        """

        modes = ["complex","real"]
        if mode not in modes:
            raise ValueError(f"Unknown value for `mode`. Should be one of {modes}, but is {mode}!")

        if mode=="complex":

            dV_z = self.dV_z(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale,conj=conj)

            dV_tau = self.dV_tau(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale,conj=conj)

            return jnp.append(dV_z,dV_tau)

        else:

            dV = self.dV(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale,conj=False)
            
            dV_c = self.dV(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale,conj=True)

            dV_t = 1j*(dV-dV_c)
            
            dV_phi = dV+dV_c

            return jnp.append(dV_t,dV_phi)+1j*0.


    


    
    ###################################################################################################################
    ############################### SCALAR POTENTIAL AND DERIVATIVES WITH REAL INPUTS #################################
    ###################################################################################################################

    
    @partial(jit, static_argnums = (0,3))
    def V_x(
        self, x: ArrayLike, fluxes: ArrayLike, noscale: bool = True
        ) -> float:
        r"""
        
        **Description:**
        Returns the value of the real part of the F-term scalar potential :math:`V` for input of the real scalar fields.
        
        .. note::
            This function is a wrapper which is used to find non-SUSY minima by looking at gradients of the scalar potential directly,
            see for example :func:`_newton_method_flux_vacua_real`.
        
        Args:
            x (ArrayLike): JAX array of shape (:math:`2(h^{1,2}+1)`,) containing the moduli and axio-dilaton as real and imaginary parts.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            noscale (bool, optional): If ``True``, uses the no-scale flux scalar potential. Defaults to ``True``.
        
        Returns:
            float: Value of :math:`V`.
            

        See also: :func:`scalar_potential`
        """
        
        moduli,moduli_c,tau,tau_c = self._convert_real_to_complex(x)
        
        return self.V(moduli,moduli_c,tau,tau_c,fluxes,noscale=noscale).real
        
    @partial(jit, static_argnums = (0,3))
    def dV_x(
        self, x: ArrayLike, fluxes: ArrayLike, noscale: bool = True
        ) -> Array:
        r"""
        
        **Description:**
        Returns the gradients of the F-term scalar potential :math:`V` with respect to the real scalar fields for real inputs.
        
        .. admonition:: Details
            :class: dropdown
        
            Let us denote the complex scalar fields as

            .. math::
                z^i = a^i + \text{i} v^i\, ,\quad \tau = c_0 + \text{i} s\, .

            Then this function computes

            .. math::
                \nabla V = (\partial_{a^1}V, \partial_{v^1}V,\ldots ,\partial_{a^{h^{1,2}}}V, \partial_{v^{h^{1,2}}}V,\partial_{c_0}V, \partial_{s}V)\, .
        
        
        Args:
            x (ArrayLike): JAX array of shape (:math:`2(h^{1,2}+1)`,) containing the moduli and axio-dilaton as real and imaginary parts.
            fluxes (ArrayLike): Flux vector of shape (:math:`4(h^{1,2}+1)`,)
            noscale (bool, optional): If ``True``, uses the no-scale flux scalar potential. Defaults to ``True``.
        
        Returns:
            Array: Value of :math:`(\partial_{a^1}V, \partial_{v^1}V,\ldots ,\partial_{a^{h^{1,2}}}V, \partial_{v^{h^{1,2}}}V,\partial_{c_0}V, \partial_{s}V)`.


        See also: :func:`V_x`
        """

        g= jax.grad(self.V_x, argnums = 0,holomorphic=False)(x,fluxes,noscale=noscale)
        
        return jnp.real(g).flatten()
        
    
    @partial(jit, static_argnums = (0,3))
    def ddV_x(
        self, x: ArrayLike, fluxes: ArrayLike, noscale: bool = True
        ) -> Array:
        r"""
        
        **Description:**
        Returns the second derivatives of the F-term scalar potential :math:`V` with respect to the real scalar fields for real inputs.
        
        .. admonition:: Details
            :class: dropdown

            Let us denote the complex scalar fields as

            .. math::
                z^i = a^i + \text{i} v^i\, ,\quad \tau = c_0 + \text{i} s\, .

            We introduce the fields

            .. math::
                \phi^\alpha = (a^1,v^1,\ldots ,a^{h^{1,2}},v^{h^{1,2}},c_0,s)\, .
            
            Then this function computes

            .. math::
                (\nabla \nabla V)_{\alpha\beta} = \partial_{\phi^\alpha}\partial_{\phi^\beta}V\, .

            Up to permutations, this function is equivalent to :func:`_hessian_real`, but slightly faster.
        
        Args:
            x (ArrayLike): JAX array of shape (:math:`2(h^{1,2}+1)`,) containing the moduli and axio-dilaton as real and imaginary parts.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            noscale (bool, optional): If ``True``, uses the no-scale flux scalar potential. Defaults to ``True``.
        
        Returns:
            Array: Value of :math:`\partial_{\phi^\alpha}\partial_{\phi^\beta}V`.
            

        See also: :func:`V_x`
        """
        
        return jax.jacrev(self.dV_x,argnums=0,holomorphic=False)(x,fluxes,noscale=noscale)

    
    ###################################################################################################################
    ##################################### HESSIAN AND CANONICAL NORMALISATION #########################################
    ###################################################################################################################


    @partial(jit, static_argnums = (0,6,7,))
    def ddV_z(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, noscale: bool = True, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the second derivatives :math:`\partial_{I}\partial_{z^j}V` of the :math:`F`-term scalar potential.

        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): Axio-dilaton value.
            tau_c (complex): Complex conjugate value for axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            noscale (bool, optional): If ``True``, uses the no-scale flux scalar potential. Defaults to ``True``.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Value of :math:`\partial_{I}\partial_{z^j}V`.

        """

        if conj:

            return jax.jacrev(self.dV, argnums = 1, holomorphic = True)(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale,conj=True)

        else:

            return jax.jacrev(self.dV, argnums = 0, holomorphic = True)(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale,conj=False)


    @partial(jit, static_argnums = (0,6,7,))
    def ddV_cz(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, noscale: bool = True, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the second derivatives :math:`\partial_{I}\partial_{\overline{z}^j}V` of the :math:`F`-term scalar potential.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): Axio-dilaton value.
            tau_c (complex): Complex conjugate value for axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            noscale (bool, optional): If ``True``, uses the no-scale flux scalar potential. Defaults to ``True``.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Value of :math:`\partial_{I}\partial_{\overline{z}^j}V`.

        """

        if conj:

            return jax.jacrev(self.dV, argnums = 0, holomorphic = True)(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale,conj=True)

        else:

            return jax.jacrev(self.dV, argnums = 1, holomorphic = True)(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale,conj=False)

    @partial(jit, static_argnums = (0,6,7,))
    def ddV_tau(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, noscale: bool = True, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the second derivatives :math:`\partial_{I}\partial_{\tau}V` of the :math:`F`-term scalar potential.

        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): Axio-dilaton value.
            tau_c (complex): Complex conjugate value for axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            noscale (bool, optional): If ``True``, uses the no-scale flux scalar potential. Defaults to ``True``.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Value of :math:`\partial_{I}\partial_{\tau}V`.

        """

        if conj:

            return jax.jacrev(self.dV, argnums = 3, holomorphic = True)(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale,conj=True)

        else:

            return jax.jacrev(self.dV, argnums = 2, holomorphic = True)(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale,conj=False)


    @partial(jit, static_argnums = (0,6,7,))
    def ddV_ctau(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, noscale: bool = True, conj: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the second derivatives :math:`\partial_{I}\partial_{\overline{\tau}}V` of the :math:`F`-term scalar potential.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): Axio-dilaton value.
            tau_c (complex): Complex conjugate value for axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            noscale (bool, optional): If ``True``, uses the no-scale flux scalar potential. Defaults to ``True``.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``.
        
        Returns:
            ArrayLike: Value of :math:`\partial_{I}\partial_{\overline{\tau}}V`.

        """

        if conj:

            return jax.jacrev(self.dV, argnums = 2, holomorphic = True)(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale,conj=True)

        else:

            return jax.jacrev(self.dV, argnums = 3, holomorphic = True)(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale,conj=False)

    


    @partial(jit, static_argnums = (0,6,7,8,))
    def ddV(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, noscale: bool = True, conj: bool = False, mode="complex"
        ) -> Array:
        r"""
        
        **Description:**
        Returns the second derivatives :math:`(\partial_{I}\partial_{J}V,\partial_{I}\partial_{\overline{J}}V)` of the :math:`F`-term scalar potential.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): Axio-dilaton value.
            tau_c (complex): Complex conjugate value for axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            noscale (bool, optional): If ``True``, uses the no-scale flux scalar potential. Defaults to ``True``.
            conj (bool, optional): If ``True``, computes the complex conjugate. Defaults to ``False``. Becomes void if ``mode="real"``.
            mode (str, optional): String specifying whether to return complex or real derivatives. Defaults to ``"complex"``.
        
        Returns:
            ArrayLike: Value of :math:`(\partial_{I}\partial_{J}V,\partial_{I}\partial_{\overline{J}}V)`.

        """

        modes = ["complex","real"]

        if mode not in modes:
            raise ValueError(f"Unknown value for `mode`. Should be one of {modes}, but is {mode}!")

        if mode=="complex":

            ddV_z = self.ddV_z(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale,conj=conj)

            ddV_cz = self.ddV_cz(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale,conj=conj)

            ddV_tau = self.ddV_tau(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale,conj=conj)

            ddV_ctau = self.ddV_ctau(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale,conj=conj)

            ddV = jnp.append(ddV_z,ddV_tau.reshape(-1,1),axis=1)
            ddV_mixed = jnp.append(ddV_cz,ddV_ctau.reshape(-1,1),axis=1)

            return jnp.hstack((ddV,ddV_mixed))

        else:

            ddV = self.ddV(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale,conj=conj)

            ddV_c = ddV[:,self.h12+1:]
            ddV = ddV[:,:self.h12+1]

            dt_dV = 1j*(ddV-ddV_c)
            dphi_dV = ddV+ddV_c

            return jnp.block([[jnp.real(dt_dV),jnp.real(dphi_dV)],[jnp.imag(dt_dV),jnp.imag(dphi_dV)]])


    

    
    
    @partial(jit, static_argnums = (0,6,))
    def _hessian_general(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, noscale: bool = True
        ) -> Array:
        r"""
        
        **Description:**
        Returns the Hessian of the scalar potential at generic points for the complex structure moduli
        :math:`z^{i}` and the axio-dilaton :math:`\tau`.

        .. warning::
            This function is generically slower than the equivalent function :func:`ddV_x` which can also be accessed via
            :func:`hessian` with ``mode="real"``.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): Axio-dilaton value.
            tau_c (complex): Complex conjugate value for axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            noscale (bool, optional): If ``True``, uses the no-scale flux scalar potential. Defaults to ``True``.
        
        Returns:
            Array: Second derivatives of the Hessian.

        See also: :func:`scalar_potential`
        
        """

        h = jax.jacrev(self.dV, argnums = (0,1,2,3), holomorphic = True)

        hessian_val = h(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale)

        ddV_tz_z = hessian_val[0]
        ddV_tz_cz = hessian_val[1]
        ddV_tz_tau = hessian_val[2]
        ddV_tz_ctau = hessian_val[3]

        hess_mixed=jnp.hstack((ddV_tz_cz,ddV_tz_ctau.reshape(self.h12+1, 1)))
        hess_holom=jnp.hstack((ddV_tz_z,ddV_tz_tau.reshape(self.h12+1, 1)))
        
        e = jnp.hstack((jnp.asarray(hess_mixed), jnp.asarray(hess_holom)))

        f = jnp.hstack((jnp.asarray(jnp.conj(hess_holom)), jnp.asarray(jnp.conj(hess_mixed))))

        return jnp.vstack((e, f))
        
        
    
    
    @partial(jit, static_argnums = (0,6))
    def _hessian_SUSY(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, noscale: bool = True
        ) -> Array:
        r"""
        
        **Description:**
        Returns the Hessian at SUSY loci where :math:`D_A W=0` for the complex structure moduli :math:`z^{i}`
        and the axio-dilaton :math:`\tau`.
        
        .. admonition:: Details
            :class: dropdown
        
            This implementation of the Hessian assumes that the input of complex structure moduli :math:`z^{i}`,
            axio-dilaton :math:`\tau` and fluxes satisfies the F-term conditions :math:`D_AW=0`. We use 
            standard supergravity formulas as spelled out in e.g. `hep-th/0404116 <https://inspirehep.net/literature/648462>`_. 
            That is, we compute the mixed second derivatives
        
            .. math::
                \partial_{A}\partial_{\overline{B}} V|_{D_A W=0} 
                        = \mathrm{e}^{K} \bigl (M_{AC} K^{C\overline{D}} \overline{M_{BD}}
                        -(\lambda -1) K_{A\overline{B}} |W|^2 \bigl )

            and the holomorphic second derivatives

            .. math::
                \partial_{A}\partial_{B} V|_{D_A W=0} =-(\lambda - 2) \mathrm{e}^{K} \,\overline{W}\, M_{AB}

            where we introduced

            .. math::
                M_{AB} = D_AD_BW\, , \quad \lambda = \begin{cases} 3 & \text{full potential}\\ 
                                                                0 & \text{no-scale potential}\end{cases}\; \, .

        .. warning::
            When using this function, one needs to ensure that F-term conditions :math:`D_AW=0` are satisfied
            with sufficiently small numerical tolerance. In general, this function only provides an approximation
            for the Hessian at a minimum where F-term conditions :math:`|D_A W|<\epsilon`.
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            noscale (bool, optional): If ``True``, uses the no-scale flux scalar potential. Defaults to ``True``.
        
        Returns:
            ArrayLike: Hessian matrix with entries :math:`\partial_{A}\partial_{\bar{B}} V`
                and :math:`\partial_{A}\partial_{B} V` assuming :math:`D_A W=0`.
        
        """
        
        DDW_val=self.DDW(moduli, moduli_c, tau, tau_c,fluxes,mode="SUSY")
        W_val=self.superpotential(moduli, tau,fluxes)
        KM_val=jnp.array(self.kahler_metric(moduli,moduli_c,tau,tau_c))
        KP=self.kahler_potential(moduli, moduli_c, tau, tau_c)
        Inv_KM=self.inverse_kahler_metric(moduli, moduli_c, tau, tau_c)
        
        if noscale:
            lam=0.
        else:
            lam=3.
        
        term1 = jnp.matmul(DDW_val, jnp.matmul(Inv_KM.T,jnp.conj(DDW_val)))
        term2 = -(lam-1.)*W_val*jnp.conj(W_val)*KM_val
        V_I_barJ = jnp.exp(KP) * (term1+term2)
        V_I_J = jnp.exp(KP) * (2.-lam)*DDW_val*jnp.conj(W_val)

        hess_mixed,hess_holom=V_I_barJ,V_I_J
            
        a = jnp.hstack((jnp.asarray(hess_mixed), jnp.asarray(hess_holom)))

        b = jnp.hstack((jnp.asarray(jnp.conj(hess_holom)), jnp.asarray(jnp.conj(hess_mixed))))

        return jnp.vstack((a, b))
            
    
    @partial(jit, static_argnums = (0,6,7,))
    def hessian(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, 
        fluxes: ArrayLike, noscale: bool = True, mode: str = None
        ) -> Array:
        r"""
        
        **Description:**
        Returns the Hessian of the scalar potential.
        
        .. admonition:: Details
            :class: dropdown
        
            Explicitly, we compute the following second derivatives of the scalar potential,
        
            .. math::
                V_{A\overline{B}}=\partial_{A}\partial_{\bar{B}} V \; ,\quad V_{AB}=\partial_{A}\partial_{B} V\, .
        
            We return the Hessian matrix
        
            .. math::
                H = \left (\begin{array}{cc}V_{A\overline{B}} & V_{AC}\ V_{\overline{D}\overline{B}} &  V_{\overline{D}C} \end{array}\right )
        
            This matrix is used to compute the mass spectrum from :func:`mass_matrix`.
        
        .. note::
            We provide two computational modes which can be set via the optional argument ``mode``.
            If we compute the Hessian at generic points in moduli space,
            then we should use ``mode=None`` which is also the default.
            At SUSY minimum where :math:`D_I W=0` for all fields,
            we can use a simplified version of the Hessian which is faster to evaluate.
            To do so, we have to use ``mode="SUSY"`` instead.

        .. warning::
            When using ``mode="SUSY"``, one should be cautious regarding numerical errors and noise.
            
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            noscale (bool, optional): If ``True``, uses the no-scale flux scalar potential. Defaults to ``True``.
            mode (string, optional): The mode to compute the Hessian. For ``mode=None``, returns the 
                                    general Hessian from explicit second derivatives of the scalar potential.
                                    For ``"SUSY"``, computes Hessian at a SUSY locus assuming 
                                    :math:`D_I W=0` using standard SUGRA formulas for the Hessian.
                                    For ``"real"``, computes Hessian for the real and imaginary components 
                                    of the complex scalar fields.
                                    Defaults to ``None``.
        
        
        Returns:
            ArrayLike: Hessian matrix with entries :math:`\partial_{A}\partial_{\bar{B}} V` and :math:`\partial_{A}\partial_{B} V`.

        Aliases:
            :func:`H`, :func:`hessian`
            
        
        """
        modes=[None,"SUSY","real"]
        if mode not in modes:
            raise ValueError(f"Cannot determine `mode` to compute Hessian!\
                    `mode` should be one of {modes}, but got {mode}.")
                
        if mode is None:
            return self._hessian_general(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale)
        elif mode=="SUSY":
            return self._hessian_SUSY(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale)
        elif mode=="real":
            x = self._convert_complex_to_real(moduli, moduli_c, tau, tau_c)
            return self.ddV_x(x,fluxes,noscale=noscale)
        
    H = hessian

    @partial(jit, static_argnums = (0,6,7,8,))
    def mass_matrix(
        self, moduli: ArrayLike, moduli_c: ArrayLike, tau: complex, tau_c: complex, fluxes: ArrayLike,
        mode: str = None, noscale: bool = True, use_real_derivatives: bool = False
        ) -> Array:
        r"""
        
        **Description:**
        Returns the mass matrix after canonical normalisation of the kinetic terms.
        
        .. admonition:: Details
            :class: dropdown

            In the action, we expand the scalar potential :math:`V(\phi^A,\overline{\phi}^A)` in terms of the fields :math:`\phi^A =(z^i, \tau)`
            around a reference point :math:`\phi^A_0=(z^i_0,\tau_0)`
        
            .. math::
                V(\phi^A_0+\phi^A , \overline{\phi}^A_0+\overline{\phi}^A )=V_0+ (V_{B})_0 \phi^B+(V_{AB})_0 \phi^A\phi^B
                    +(V_{\overline{A}B})_0 \phi^\overline{A}\phi^B+\text{c.c.} +\ldots
        
            where, by abuse of notation, we call the fluctuations collectively :math:`\phi^A\in\lbrace z^i,\tau\rbrace`,
            :math:`A=1,\ldots, h^{1,2}+1`. Further, we defined
        
            .. math::
                V_0=V(\phi^A_0 , \overline{\phi}^A_0 )\; ,\quad (V_{B})_0 = \partial_{B}V(\phi^A_0 , \overline{\phi}^A_0 )\; ,\quad \ldots
            
            We are primarily interested in evaluating this function for vaccum solutions :math:`\phi^A_0=(z^i_0,\tau_0)` 
            in which case
        
            .. math::
                (\partial_{B}V)_0 = 0 \; ,\quad (\partial_{\overline{B}}V)_0 = 0 \, .
        
            Further, in the case of vacua satisfying the :math:`F`-term conditions :math:`D_A W=0`, we compute the vacuum energy
        
            .. math::
                V_0= \begin{cases} -3\mathrm{e}^{K}\, |W|^2 & \text{full potential}\\
                                                      0 & \text{no-scale potential}\, .
                                        \end{cases}
        
            In particular, for no-scale models like for type IIB flux vacua, we find a vanishing cosmological constant.
            In this case, by introducing :math:`\Phi^{\Lambda}=(\phi^A,\overline{\phi}^{\overline{B}})`, the above becomes
        
            .. math::
                V(\phi^A_0+\phi^A , \overline{\phi}^A_0+\overline{\phi}^A )=\Phi^{\Lambda}H_{\Lambda\overline{\Delta}}(\phi^A_0 , \overline{\phi}^A_0 )\overline{\Phi}^{\overline{\Delta}} +\ldots
        
            where :math:`H_{\Lambda\overline{\Delta}}(\phi^A_0 , \overline{\phi}^A_0 )` is the Hessian evaluated at the minimum
            which is determined as (see :func:`hessian`)

            .. math::
                H_{\Lambda\overline{\Delta}}(\phi^A_0,\overline{\phi}^A_0) = \left (\begin{array}{cc}V_{A\overline{B}} \;&\; V_{AC}\\[0.3em] 
                V_{\overline{D}\overline{B}} \;&\;  V_{\overline{D}C} \end{array}\right ) \, .

            To obtain the mass matrix, we first perform a change of basis by diagonalising :math:`K_{A\overline{B}}`.
            That is, we compute the matrix :math:`\tilde{U}` of normalised eigenvectors of :math:`K_{A\overline{B}}`.
            Then, in order to define a canonically normalised basis of fields :math:`\psi^A`, we define
        
            .. math::
                U=\text{diag}(\sqrt{\lambda_1},\ldots ,\sqrt{\lambda_{h^{1,2}+1}})\, \tilde{U}

            so that

            .. math::
                \phi^A = U^{A}\,_{B} \psi^B

            and thus

            .. math::
                K_{A\overline{B}}\partial \phi^A\partial \overline{\phi}^B = \delta_{A\overline{B}}\partial \psi^A\partial \overline{\psi}^B \, .
        
            .. warning::
                Notice that by definition :math:`U` is clearly not a unitary matrix anymore because its determinant 
                does not necessarily satisfy :math:`|\text{det}(U)|=1`.


            Then, the mass matrix for the canonically normalised fields is defined as
        
            .. math::
                M = \mathbb{U} H \mathbb{U}^{\dagger}\, , \quad \mathbb{U} = \left (\begin{array}{cc}U^{A\overline{B}} \;&\; 0\[0.3em] 0 \;&\;  U^{\overline{D}C} \end{array}\right ) 

            or, in components,
        
            .. math::
                M = \left (\begin{array}{cc}U_{E}\,^{A} V_{A\overline{B}}U^{\overline{B}}\,_{\overline{F}} \quad&\quad U_{E}\,^{A}V_{AC}U^{C}\,_{F}\\[0.8em]
                    U_{\overline{E}}\,^{\overline{D}}V_{\overline{D}\overline{B}}U^{\overline{B}}\,_{\overline{F}} \quad&\quad  U_{\overline{E}}\,^{\overline{D}}V_{\overline{D}C}U^{C}\,_{F} \end{array}\right ) \, .
        
        
        
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            mode (string, optional): The mode to compute the mass matrix. For ``mode=None``, uses the 
                                    general Hessian from explicit second derivatives of the scalar potential.
                                    For ``mode="SUSY"``, computes Hessian at a SUSY locus assuming 
                                    :math:`D_I W=0` using standard SUGRA formulas for the Hessian.
                                    For ``"real"``, calculates Hessian for the real and imaginary components 
                                    of the complex scalar fields.
                                    Defaults to ``None``.
            
        
        Returns:
            ArrayLike: Mass matrix of the canonically normalised fields.
        
        """
        
        modes=[None, "SUSY", "real"]
        if mode not in modes:
            raise ValueError(f"Cannot determine `mode` to compute Hessian!\
                    `mode` should be one of {modes}, but got {mode}.")
        
        # Alternative method:
        # THIS IS OFF BY FACTOR OF 2. WHY? -> Normalisation of complex vs. real scalar:
        # Complex scalar: phi = (a+i*b)/sqrt(2)
        # This implies: m^2 phi*phi^dagger = 1/2 m^2 * (a^2+b^2).
        if mode=="real":
            KM = self.kahler_metric(moduli, moduli_c, tau, tau_c)
            diagonalizer = jnp.linalg.cholesky(jnp.linalg.inv(KM))
            SP_hessian = self.hessian(moduli, moduli_c, tau, tau_c,fluxes,noscale=noscale,mode="real")

            return 1/2*jnp.block([[diagonalizer.T@SP_hessian[:self.dimension_H3,:self.dimension_H3]@diagonalizer,diagonalizer.T@SP_hessian[:self.dimension_H3,self.dimension_H3:]@diagonalizer],
                             [diagonalizer.T@SP_hessian[self.dimension_H3:,:self.dimension_H3]@diagonalizer,diagonalizer.T@SP_hessian[self.dimension_H3:,self.dimension_H3:]@diagonalizer]])

            
        
        KM = self.kahler_metric(moduli, moduli_c, tau, tau_c)
        
        eigvals, eigvecs = jnp.linalg.eigh(KM)
        
        # For canonical normalisation, we have to also absorb the eigenvalue in the basis change
        eigenvec_mat = jnp.matmul(eigvecs,jnp.sqrt(eigvals)*jnp.identity(self.dimension_H3))

        # The new normalised eigenvectors are the columns of `eigenvec_mat`: eigenvec_mat[:,i]

        # eigvecs is in fact a unitary matrix, but `eigenvec_mat` is not anymore because of the change of normalisation!
        eigenvec_mat_inv = jnp.linalg.inv(eigenvec_mat)

        # Stack the transformation matrix together with its conjugate -> we work with a field vector Phi = (z^i, \overline{z}^i)
        eigenvec_mat_stacked1 = jnp.hstack( ( eigenvec_mat_inv.T, jnp.zeros((self.dimension_H3,self.dimension_H3)) ) )
        eigenvec_mat_stacked2=jnp.hstack( ( jnp.zeros((self.dimension_H3,self.dimension_H3)) , jnp.conj(eigenvec_mat_inv.T) ) )
        eigenvec_mat_stacked=jnp.vstack( ( eigenvec_mat_stacked1, eigenvec_mat_stacked2 ) )


        # Evaluate the Hessian
        SP_hessian = self.hessian(moduli, moduli_c, tau, tau_c, fluxes,mode=mode,noscale=noscale)
        
        #Rotate the Scalar potential hessian using the normalised eigenvectors of the Kähler Metric
        return jnp.matmul( eigenvec_mat_stacked.T , jnp.matmul(SP_hessian, eigenvec_mat_stacked))
    
    @partial(jit,static_argnums=(0,))
    def compute_tau_vev(self,moduli,flux):
        r"""
        **Description:**
        Computes the value of the axio-dilaton :math:`\tau` from the fluxes and complex structure moduli as a solution to the corresponding :math:`F`-term condition :math:`D_{\tau}W=0`.
        
        .. admonition:: Details
            :class: dropdown

            The axio-dilaton :math:`\tau` can be computed from the fluxes and complex structure moduli as a solution to the :math:`F`-term condition
            
            .. math::
                D_{\tau}W = \partial_{\tau}W + W \partial_{\tau}K = 0\, .
                
            This condition can be written as
            
            .. math::
                D_{\tau}W = -\frac{1}{\tau-\overline{\tau}} (f - \overline{\tau} h)\cdot \Sigma \cdot \overline{\Pi}(z^i) = 0\, ,
            
            and the corresponding solution is given by

            .. math::
                \tau = \frac{f\cdot \Sigma \cdot \overline{\Pi}(z^i)}{h\cdot \Sigma \cdot \overline{\Pi}(z^i)}\, ,

            where :math:`f=(f_1,f_2)` and :math:`h=(h_1,h_2)` are the RR- and NSNS-fluxes respectively with :math:`f_i,h_i \in \mathbb{Z}^{h^{1,2}+1}`.
            Further, :math:`\Pi(z^i)` is the period vector depending on the complex structure moduli and :math:`\Sigma` is the symplectic matrix defined in :func:`period_matrix`.
            
        .. note::
        
            Notice that this equation only has a solution if the denominator is non-zero, i.e. if

            .. math::
                h\cdot \Sigma \cdot \overline{\Pi}(z^i) \neq 0\, .

            If this condition is not satisfied, then this function raises a ``ZeroDivisionError``.
            
        Args:
            moduli (ArrayLike): Complex structure moduli values.
            flux (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            
        Returns:
            complex: Value of the axio-dilaton :math:`\tau`.

        """
        
        f = flux[:self.n_fluxes]
        h = flux[self.n_fluxes:]
            
        Pi = self.period_vector(jnp.conj(moduli),conj=True)
        
        SigmaPi = jnp.matmul(self.periods.sigma(),Pi)
        
        return jnp.matmul(f,SigmaPi)/jnp.matmul(h,SigmaPi)

    
    @partial(jit, static_argnums = (0,6,))
    def ISD_condition(self,moduli,moduli_c,tau,tau_c,fluxes, mode="complex"):
        r"""
        **Description:**
        Checks whether the fluxes satisfy the ISD condition :math:`\star G_3=\text{i}G_3`.

        .. admonition:: Details
            :class: dropdown

            If the 3-form fluxes are ISD, then the RR-fluxes :math:`f=(f_1,f_2)` and NSNS-fluxes :math:`h=(h_1,h_2)` 
            with :math:`f_i,h_i \in \mathbb{Z}^{h^{1,2}+1}` satisfy

            .. math::
                f_1-\tau h_1=\overline{\mathcal{N}}(z^i,\overline{z}^i)\, (f_2-\tau h_2)\, ,

            where the matrix :math:`\mathcal{N}` corresponding to the gauge kinetic function is computed using :func:`gauge_kinetic_matrix`. 
            Alternatively, this condition can also be written as

            .. math::
                f=(s\,  M(z^i,\overline{z}^i)\Sigma + c_0)\, h\; ,\quad \tau=c_0 + \text{i} s \, .
            
            where the matrix :math:`M(z^i,\overline{z}^i)` (also called ISD-matrix) is computed via :func:`ISD_matrix`.
            
            This function computes the difference between left and right hand side of either of these equations depending on the input.
            If ``mode="complex"``, then it returns

            .. math::
                f_1-\tau h_1 - \overline{\mathcal{N}}(z^i,\overline{z}^i)\, (f_2-\tau h_2)\, ,
                
            and for ``mode="real"``
            
            .. math::
                f-(s \,  M(z^i,\overline{z}^i)\Sigma +  c_0)\, h\, .

        Args:
            moduli (ArrayLike): Complex structure moduli values.
            moduli_c (ArrayLike): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            mode (str, optional): String specifying which version of the ISD equation to use. Options are :c:var`"complex"` or :c:var`"real"`. Defaults to :c:var`mode = "complex"`.

        Returns:
            Array: Difference between the left and right hand side of the ISD equation.

        """
            
        modes = ["complex","real"]
        if mode not in modes:
            raise ValueError(f"Unknown value for `mode`. Should be one of {modes}, but is {mode}!")
        
        FF = fluxes[:self.n_fluxes]
        HF = fluxes[self.n_fluxes:]
        
        if mode == "real":
            MMatVal=self.ISD_matrix(moduli,moduli_c)
            c0 = (tau+tau_c)/2
            s = -1j*(tau-tau_c)/2

            # Initialise residual as F-flux
            res = FF

            # Subtract LHS of ISD condition
            res = res - (jnp.matmul(MMatVal,jnp.matmul(self.periods.sigma(),HF))*s+c0*HF)

            return res
        else:
            # Compute matrix
            cNMatVal=self.gauge_kinetic_matrix(moduli,moduli_c,conj=True)

            return FF[:self.dimension_H3]-tau*HF[:self.dimension_H3]-jnp.matmul(cNMatVal,(FF[self.dimension_H3:]-tau*HF[self.dimension_H3:]))    


    @partial(jit, static_argnums = (0,))
    def _get_hf_from_N(
                      self,
                      N: ArrayLike,
                      tau: complex
                      ) -> Tuple[Array,Array]:
        r"""
        **Description:**
        Dummy function to get NSNS- and RR-fluxes from value of array :var:`N`
        and axio-dilaton values :math:`\tau`.
        
        Args:
            N (ArrayLike): Array.
            tau (complex): Value of the axio-dilaton.
        
        Returns:
            Array: NSNS-flux.
            Array: RR-flux.
        """

        h = -jnp.imag(N)/jnp.imag(tau)

        f = jnp.real(N)+jnp.real(tau)*h

        return h,f

    @partial(jit, static_argnums = (0,))
    def _projection_fluxes_0_3(
                              self,
                              moduli: ArrayLike,
                              tau: complex,
                              fluxes: ArrayLike,
                              Kcs: complex,
                              cpi_vec: ArrayLike):
        r"""
        **Description:**
        Computes the :math:`(0,3)`-component of the 3-form flux.
        
        Args:
            moduli (ArrayLike): Values for the complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            Kcs (complex): Value of the complex structure Kähler potential :math:`K_{\mathrm{cs}}`.
            pi_vec (ArrayLike): Complex conjugate period vector :math:`\overline{\Pi}`.
        
        Returns:
            Array: :math:`(0,3)`-component of the 3-form flux.
        """
        W0 = self.superpotential(moduli,tau,fluxes)
        N = -1j*jnp.exp(Kcs)*W0*cpi_vec
        
        h,f = self._get_hf_from_N(N,tau)
        
        return jnp.append(f,h)

    @partial(jit, static_argnums = (0,))
    def _projection_fluxes_2_1(
                              self,
                              moduli: ArrayLike,
                              tau: complex,
                              fluxes: ArrayLike,
                              Kcs: complex,
                              pi_vec: ArrayLike,
                              IKM: ArrayLike
                              ):
        r"""
        **Description:**
        Computes the :math:`(2,1)`-component of the 3-form flux.
        
        Args:
            moduli (ArrayLike): Values for the complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            Kcs (complex): Value of the complex structure Kähler potential :math:`K_{\mathrm{cs}}`.
            pi_vec (ArrayLike): Period vector :math:`\Pi`.
            IKM (ArrayLike): Inverse Kähler metric :math:`K^{i\bar{\jmath}}`.
        
        Returns:
            Array: :math:`(2,1)`-component of the 3-form flux.

        """
        DDW = self.DDW_general(moduli,jnp.conj(moduli),tau,jnp.conj(tau),fluxes)
        cDDW = jnp.conj(DDW)
        
        DPi = self.periods.D_period_vector_per(self.moduli_to_periods(moduli),self.moduli_to_periods(jnp.conj(moduli),conj=True),conj=False)
        DPi = DPi[:,1:].T

        N = -1j*jnp.exp(Kcs)*jnp.matmul(jnp.matmul(jnp.conj(IKM),cDDW)[:-1].T,DPi)[-1]*(tau-jnp.conj(tau))
        h,f = self._get_hf_from_N(N,tau)
        
        return jnp.append(f,h)

    @partial(jit, static_argnums = (0,))
    def _projection_fluxes_3_0(
                              self,
                              moduli: ArrayLike,
                              tau: complex,
                              fluxes: ArrayLike,
                              Kcs: complex,
                              pi_vec: ArrayLike):
        r"""
        **Description:**
        Computes the :math:`(3,0)`-component of the 3-form flux.
        
        Args:
            moduli (ArrayLike): Values for the complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            Kcs (complex): Value of the complex structure Kähler potential :math:`K_{\mathrm{cs}}`.
            pi_vec (ArrayLike): Period vector :math:`\Pi`.
        
        Returns:
            Array: :math:`(3,0)`-component of the 3-form flux.

        """

        Ftau = self.DW_tau(moduli,jnp.conj(moduli),tau,jnp.conj(tau),fluxes)
        
        N = 1j*jnp.exp(Kcs)*jnp.conj(Ftau)*(tau-jnp.conj(tau))*pi_vec
        
        h,f = self._get_hf_from_N(N,tau)
        
        return jnp.append(f,h)

    @partial(jit, static_argnums = (0,))
    def _projection_fluxes_1_2(
                              self,
                              moduli: ArrayLike,
                              tau: complex,
                              fluxes: ArrayLike,
                              Kcs: complex,
                              IKM: ArrayLike,
                              pi_vec: ArrayLike
                              ):
        r"""
        **Description:**
        Computes the :math:`(1,2)`-component of the 3-form flux.
        
        Args:
            moduli (ArrayLike): Values for the complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            Kcs (complex): Value of the complex structure Kähler potential :math:`K_{\mathrm{cs}}`.
            IKM (ArrayLike): Inverse Kähler metric :math:`K^{i\bar{\jmath}}`.
            pi_vec (ArrayLike): Period vector :math:`\Pi`.
        
        Returns:
            Array: :math:`(1,2)`-component of the 3-form flux.

        """
        fterms_mod = self.DW_z(moduli,jnp.conj(moduli),tau,jnp.conj(tau),fluxes)
        
        Fi_bar = jnp.matmul(IKM[:-1,:-1],fterms_mod)

        DPi = self.periods.D_period_vector_per(self.moduli_to_periods(moduli),self.moduli_to_periods(jnp.conj(moduli),conj=True),conj=False)
        DPi = DPi[:,1:].T

        cDPi = jnp.conj(DPi)
        
        N = 1j*jnp.exp(Kcs)*jnp.matmul(Fi_bar,cDPi)
        h,f = self._get_hf_from_N(N,tau)
        
        return jnp.append(f,h)

        
    @partial(jit, static_argnums = (0,4,))
    def projection_fluxes(
                          self,
                          moduli: ArrayLike,
                          tau: complex,
                          fluxes: ArrayLike,
                          mode: str = None
                          ):
        r"""
        **Description:**
        Computes projection of fluxes onto their ISD and AISD components.
        
        Args:
            moduli (ArrayLike): Values for the complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            mode (str, optional): If :var:`mode="SUSY"`, then only returns :math:`(2,1)`-component and :math:`(0,3)`-component.
        
        Returns:
            Array: :math:`(3,0)`-component of the 3-form flux.
            Array: :math:`(2,1)`-component of the 3-form flux.
            Array: :math:`(1,2)`-component of the 3-form flux.
            Array: :math:`(0,3)`-component of the 3-form flux.

        """
        pi_vec = self.period_vector(moduli)
        cpi_vec = jnp.conj(pi_vec)
        
        Kcs = -jnp.log(-1j*jnp.matmul(cpi_vec,jnp.matmul(self.periods.sigma(),pi_vec)))
        
        IKM = self.inverse_kahler_metric(moduli,jnp.conj(moduli),tau,jnp.conj(tau))
        
        N_0_3 = self._projection_fluxes_0_3(moduli,tau,fluxes,Kcs,cpi_vec)
        N_2_1 = self._projection_fluxes_2_1(moduli,tau,fluxes,Kcs,pi_vec,IKM)
        
        if mode=="SUSY":
            return N_2_1,N_0_3
        else:
            
            N_3_0 = self._projection_fluxes_3_0(moduli,tau,fluxes,Kcs,pi_vec)
            N_1_2 = self._projection_fluxes_1_2(moduli,tau,fluxes,Kcs,IKM,pi_vec)
            
            return N_3_0,N_2_1,N_1_2,N_0_3


    @partial(jit, static_argnums = (0,4,5,6,7,8,))
    def _newton_method_flux_vacua_complex(
        self, moduli: ArrayLike, tau: complex, fluxes: ArrayLike, mode: str = None, step_size_Newton: float = 1e-1,
        tol: float = 1e-10,max_iters: int = 100,print_progress: bool = False
        ) -> Tuple[int,Array,Array,float]:
        r"""

        **Description:**
        Solves the minimum conditions for flux vacua in terms of complex fields using Newton's method.

        Args:
            moduli (ArrayLike): Complex structure moduli values.
            tau (complex): Value of axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            mode (str, optional): Whether to solve F-term conditions :math:`D_IW=0` for SUSY minima or instead :math:`\partial_IV=0` for general minima.
                Defaults to ``None``.
            step_size_Newton (float, optional): Step size to be used in Newton's method. Defaults to ``0.1``.
            tol (float, optional): Tolerance for the residual. Defaults to ``1e-10``.
            max_iters (int, optional): Maximum number of iterations. Defaults to ``100``.
            print_progress (bool, optional): Whether to print the progress of the minimisation. Defaults to ``False``.
            
        Returns:
            int: Number of iterations.
            ArrayLike: Complex structure moduli values.
            ArrayLike: Value of axio-dilaton.
            float: Residual :math:`\sum_A |D_A W|` of the :math:`F`-terms.

        """

        modes=[None, "SUSY"]
        if mode not in modes:
            raise ValueError(f"Cannot determine mode for Newton's method!\
                    `mode` must be one of {modes}, but is {mode}!")

        def cond(arg):
            step, moduli, tau, res = arg
            return (step <max_iters) & (res > tol)

        def body(arg):
            step, moduli, tau, res = arg

            if mode=="SUSY":
                m2 = self.dDW_real(moduli,tau,fluxes)
                DW = self.DW(moduli,jnp.conj(moduli),tau,jnp.conj(tau),fluxes)
                m1 = jnp.concatenate([jnp.real(DW),jnp.imag(DW)])
            else:
                m2 = self.ddV(moduli,jnp.conj(moduli),tau,jnp.conj(tau),fluxes,mode="real")
                m1 = self.dV(moduli,jnp.conj(moduli),tau,jnp.conj(tau),fluxes,mode="real")
                

            deltaPhi = -jnp.linalg.inv(m2)@m1

            moduli = moduli + step_size_Newton*(deltaPhi[self.h12+1:2*self.h12+1]+1j*deltaPhi[0:self.h12])
            tau = tau + step_size_Newton*(deltaPhi[2*self.h12+1]+1j*deltaPhi[self.h12])

            res = jnp.sum(jnp.abs(m1))

            if print_progress:
                out_type = jax.ShapeDtypeStruct(jnp.shape(step), jnp.result_type(step))
                step = jax.pure_callback(progress_bar_jax, out_type,(step, max_iters, res),step)
                
            return (step + 1,  moduli, tau, res)

        return jax.lax.while_loop(cond,body,(0,moduli,tau,10.))


    @partial(jit, static_argnums = (0,3,4,5,6,7,))
    def _newton_method_flux_vacua_real(
        self, x: ArrayLike, fluxes: ArrayLike, mode: str = None, step_size_Newton: float = 1e-1,
        tol: float = 1e-10, max_iters: int = 100, print_progress: bool = False
        ) -> Tuple[int,Array,float]:
        r"""
        **Description:**
        Solves the minimum conditions for flux vacua in terms of real fields using Newton's method.

        Args:
            x (ArrayLike): Array of shape (:math:`2(h^{1,2}+1)`, ) containing real and imaginary parts of
                the complex structure moduli :math:`z^i` and axio-dilaton :math:`\tau`.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            mode (str, optional): Whether to solve F-term conditions :math:`D_IW=0` for SUSY minima or instead :math:`\partial_I V=0` for general minima.
                Defaults to ``None``.
            step_size_Newton (float, optional): Step size to be used in Newton's method. Defaults to ``0.1``.
            tol (float, optional): Tolerance for the residual. Defaults to ``1e-10``.
            max_iters (int, optional): Maximum number of iterations. Defaults to ``100``.
            print_progress (bool, optional): Whether to print the progress of the minimisation. Defaults to ``False``.

        Returns:
            int: Number of iterations.
            ArrayLike: Values of real and imaginary parts of the complex structure moduli :math:`z^i`
                and axio-dilaton :math:`\tau`.
            float: Residual :math:`\sum_A |\partial_A V|`.

        """

        modes=[None, "SUSY"]
        if mode not in modes:
            raise ValueError(f"Cannot determine mode for Newton's method!\
                    `mode` must be one of {modes}, but is {mode}!")

        def cond(arg):
            step, x, res = arg
            return (step <max_iters) & (res > tol)

        def body(arg):
            step, x, res = arg

            if mode=="SUSY":
                m2 = self.dDW_x(x,fluxes)
                m1 = self.DW_x(x,fluxes)
            else:
                m2 = self.ddV_x(x,fluxes)
                m1 = self.dV_x(x,fluxes)

            deltaPhi = -jnp.linalg.inv(m2)@m1

            x = x + step_size_Newton*deltaPhi

            res = jnp.sum(jnp.abs(m1))

            if print_progress:
                out_type = jax.ShapeDtypeStruct(jnp.shape(step), jnp.result_type(step))
                step = jax.pure_callback(progress_bar_jax, out_type,(step, max_iters, res),step)

            return (step + 1,  x, res)

        return jax.lax.while_loop(cond,body,(0,x,10.))

    @partial(jit, static_argnums = (0,4,5,6,7,8,9,))
    def newton_method_flux_vacua(
        self, moduli: ArrayLike, tau: complex, fluxes: ArrayLike, mode: str = None, step_size_Newton: float = 1e-1,
        tol: float = 1e-10,max_iters: int = 100,print_progress: bool = False, solver_mode: str = "complex"
        ) -> Tuple[int,Array,Array,float]:
        r"""
        **Description:**
        Solves the minimum conditions for flux vacua using Newton's method.

        Args:
            moduli (ArrayLike): Complex structure moduli values.
            tau (complex): Value of axio-dilaton.
            fluxes (ArrayLike): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            mode (str, optional): Whether to solve F-term conditions :math:`D_IW=0` for SUSY minima or instead :math:`\partial_IV=0` for general minima.
                Defaults to ``None``.
            step_size_Newton (float, optional): Step size to be used in Newton's method. Defaults to ``0.1``.
            tol (float, optional): Tolerance for the residual. Defaults to ``1e-10``.
            max_iters (int, optional): Maximum number of iterations. Defaults to ``100``.
            print_progress (bool, optional): Whether to print the progress of the minimisation. Defaults to ``False``.
            solver_mode (str, optional): Solver mode to use for the computation. Available options are ``"real"``
                or ``"complex"`` to solve equations in terms of real or complex fields. Defaults to ``"complex"``.

        Returns:
            ArrayLike: Complex structure moduli values.
            ArrayLike: Value of axio-dilaton.
            float: Residual :math:`\sum_A |D_A W|` of the :math:`F`-terms.
        """
        
        solver_modes=["real", "complex"]
        if solver_mode not in solver_modes:
            raise ValueError(f"Cannot determine solver mode for Newton's method!\
                    `solver_mode` must be one of {solver_modes}, but is {solver_mode}!")
            
        if solver_mode=="complex":
            
            _,moduli,tau,res = self._newton_method_flux_vacua_complex(moduli,tau,fluxes,mode=mode, step_size_Newton=step_size_Newton,tol=tol,max_iters=max_iters,print_progress=print_progress)
        
        elif solver_mode=="real":
            
            x = self._convert_complex_to_real(moduli,jnp.conj(moduli),tau,jnp.conj(tau))
            
            _, x, res = self._newton_method_flux_vacua_real(x,fluxes,mode=mode, step_size_Newton=step_size_Newton,tol=tol,max_iters=max_iters,print_progress=print_progress)
            
            moduli,_,tau,_ = self._convert_real_to_complex(x)
            
        return moduli,tau,res




