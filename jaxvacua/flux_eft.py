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
from jax import pure_callback
#from jax.typing import Array
#from numpy.typing import Array

os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Enable 64 bit precision
config.update("jax_enable_x64", True)

# CYTools import
try:
    from cytools import Polytope, Cone
except ImportError:
    #raise Exception("Cannot import CYTools!")
    warnings.warn("Cannot import CYTools!")

# JAXVacua custom imports
from .util import *
from .flux_sector import flux_sector

from .sampling import data_sampler


class flux_eft(flux_sector):
    r"""
    **Description:**
    Class representing the effective field theory of the flux sector in 4D Type IIB string theory compactified on Calabi-Yau threefolds with 3-form flux backgrounds. 
    This class provides tools for computing and analyzing flux vacua, including superpotentials, Kähler potentials, F-term conditions, and various properties of the moduli space.
    
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
        basis_transformation: Optional[Array] = None,
        ncf: Optional[int] = None,
        conifold_curve: Optional[Any] = None,
        grading_vector: Optional[Array] = None,
        period_input: Optional[Callable] = None,
        Q: Optional[int] = None,
        prepotential_input: Optional[Callable] = None,
        gauge_choice: complex = 1.+0.*1j,
        prange: int = 500,
        use_gvs: bool = False,
        save_file: bool = False,
        **kwargs
    ) -> None:
        r"""

        **Description:**
        Initializes a flux EFT class representing the effective field theory of the flux sector 
        in 4D Type IIB string theory compactified on Calabi-Yau threefolds with 3-form flux backgrounds.
        
        This class provides tools for computing and analyzing flux vacua, including superpotentials,
        Kähler potentials, F-term conditions, and various properties of the moduli space.
        
        Args:
            h12 (int, optional): Number of complex structure moduli :math:`h^{1,2}` for the compactified geometry.
            model_type (str, optional): Type of Calabi-Yau manifold. Currently supports ``"KS"`` (Kreuzer-Skarke) 
            and ``"CICY"`` (Complete Intersection Calabi-Yau). Defaults to ``"KS"``.
            model_ID (int, optional): Identifier for a specific model within the database.
            moduli_space_limit (str, optional): Type of moduli space limit for the periods. Currently supports ``"LCS"`` 
            (Large Complex Structure limit). Defaults to ``"LCS"``.
            model_data (dict, optional): Dictionary containing topological data of the Calabi-Yau, such as 
            triple intersection numbers, second Chern class, etc.
            instanton_data (list, optional): List of Gopakumar-Vafa (GV) and Gromov-Witten (GW) invariants 
            for instanton corrections.
            maximum_degree (int, optional): Maximum degree cutoff for instanton sum expansions. Defaults to ``0``.
            use_cytools (bool, optional): Whether to use CYTools library to compute topological data of the 
            Calabi-Yau automatically. Defaults to ``False``.
            mirror_cy (cytools.CalabiYau, optional): Mirror Calabi-Yau threefold object (from CYTools).
            basis_transformation (Array, optional): Basis transformation matrix to be applied to the topological 
            data of the Calabi-Yau.
            ncf (int, optional): Conifold number for conifold-enhanced models.
            conifold_curve (Array, optional): Specification of the conifold curve in the moduli space.
            grading_vector (Array, optional): Grading vector used for computing Gopakumar-Vafa invariants.
            period_input (Callable, optional): Custom function for computing periods.
            Q (int, optional): D3-brane tadpole bound constraint.
            prepotential_input (Callable, optional): Custom function for computing the prepotential.
            gauge_choice (complex, optional): Gauge choice parameter for the periods. Defaults to ``1.0+0.0j``.
            prange (int, optional): Period range parameter for convergence. Defaults to ``500``.
            use_gvs (bool, optional): Whether to use Gopakumar-Vafa invariants for instanton corrections. 
            Defaults to ``False``.
            save_file (bool, optional): Whether to save model data to disk for future use. Defaults to ``False``.
            **kwargs: Additional keyword arguments passed to parent class :class:`flux_sector`.

        """

        
        # Initialise private flux sector class!
        super(flux_eft, self).__init__(
            h12=h12,
            Q=Q,
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

    def __repr__(self) -> str:
        r"""
        **Description:**
        Returns a string representation of the flux sector class.
        
        Returns:
            str: Description of the class.
        """
        if self.periods.name!="":
            return f"Flux EFT for {self.periods.name} with h12={self.h12} complex structure moduli in the {self.periods.moduli_space_limit} limit."
        else:
            return f"Flux EFT for h12={self.h12} complex structure moduli in the {self.periods.moduli_space_limit} limit."
        
    ############# CALABI-YAU AND KÄHLER CONE HANDLING #############
    
    def eft_to_poly(self):
        r"""
        **Description:**
        Extracts the polytope from the given model.

        Returns:
            cytools.Polytope: The polytope of the Calabi-Yau threefold.
        """
        
        return Polytope(self.periods.polytope_points)
    
    def eft_to_cy(self):
        r"""
        **Description:**
        Returns the Calabi-Yau manifold from the given model.
            
        Returns:
            cytools.CalabiYau: The Calabi-Yau manifold.
        """
        
        # Create a Polytope object using the polytope points from the periods
        poly = Polytope(self.periods.polytope_points)
        
        # Retrieve the heights associated with the periods
        heights = self.periods.heights
        
        # Triangulate the polytope using the specified heights and return the corresponding Calabi-Yau manifold
        return poly.triangulate(heights=heights).get_cy()

    def eft_to_coninop(self):
        r"""
        **Description:**
        Returns the basis transformation and conifold curve from the given model.

        Returns:
            tuple: A tuple containing the basis transformation and conifold curve.
        """
        return self.periods.basis_transformation, self.periods.conifold_curve

    
    def eft_to_cone(self,timeout: int = 60, generators: Optional[Array] = None):
        r"""
        **Description:**
        Extracts the Kähler cone from the given model.
        
        Args:
            timeout (int, optional): Timeout in seconds for CYTools computation if generators are not provided. Defaults to ``60``.
            generators (Array, optional): Generators of the Kähler cone. If provided, these will be used directly. Defaults to ``None``.
        
        Returns:
            Cone: The Kähler cone.
        """
        
        if generators is None:
            # Retrieve the generators of the Kähler cone from the periods
            generators = self.periods.generators_kahler_cone
        
        if generators is None:
            
            @exit_after(timeout)
            def get_cone():
                cy = self.eft_to_cy()
                mcap = cy.mori_cone_cap(in_basis=True)
                xray = mcap.extremal_rays()
                return Cone(xray).dual()
            
            return get_cone()
        
        # Return a Cone object constructed from the generators
        return Cone(generators)

    def eft_to_facet(self):
        r"""
        **Description:**
        Extracts the facet of the Kähler cone from the given model.
        
        Returns:
            Cone: The facet of the Kähler cone.
        """
        # Retrieve the generators of the Kähler cone from the periods
        generators = self.periods.generators_kahler_cone
        
        # Exclude the first column of generators, which is typically not part of the facet
        generators_facet = generators[:,1:]
        
        # Create a boolean flag array indicating which rows have non-zero entries
        flag = np.all(generators_facet==0, axis=1) == False
        
        # Filter the generators_facet to keep only those rows that are not all zeros
        generators_facet = generators_facet[flag]
        
        # Return a Cone object constructed from the filtered generators_facet
        return Cone(generators_facet)
    
    ############# FLUX VECTOR HANDLING #############
    
    @partial(jit, static_argnums = (0,))
    def _split_fluxes(
                    self,
                    fluxes: Array
                    ) -> Tuple[Array,Array,Array,Array]:
        r"""
        **Description:**
        Split flux vector into individual components.
        
        Args:
            fluxes (Array): Flux vector.
        
        Returns:
            Array: Flux vector :math:`f_1`.
            Array: Flux vector :math:`f_2`.
            Array: Flux vector :math:`h_1`.
            Array: Flux vector :math:`h_2`.
        
        """

        f = fluxes[:self.n_fluxes] # Extract the first part of the flux vector, which contains the RR-fluxes.
        h = fluxes[self.n_fluxes:] # Extract the second part of the flux vector, which contains the NSNS-fluxes.
        f1 = f[:self.dimension_H3] # Assign the first set of fluxes (f1) from the RR-fluxes, limited to the dimension of H3.
        f2 = f[self.dimension_H3:] # Assign the second set of fluxes (f2) from the RR-fluxes, starting from the dimension of H3.
        h1 = h[:self.dimension_H3] # Assign the first set of fluxes (h1) from the NSNS-fluxes, limited to the dimension of H3.
        h2 = h[self.dimension_H3:] # Assign the second set of fluxes (h2) from the NSNS-fluxes, starting from the dimension of H3.

        return f1,f2,h1,h2 # Return the split flux components as a tuple.

    @partial(jit, static_argnums = (0,))
    def flux_to_pfv(
                    self, 
                    flux: Array, 
                    ) -> Tuple[Array,Array]:
        r"""
        **Description:**
        Returns M- and K-vector specifying PFV from full flux vector.
        
        Args:
            flux (Array): Flux-vector.           
        
        Returns:
            M (Array): M-vector.
            K (Array): K-vector.  
        
        """
        _,f2,h1,_ = self._split_fluxes(flux)

        Mvec = f2[1:]
        Kvec = h1[1:]

        return Mvec,Kvec

    @partial(jit, static_argnums = (0,))
    def pfv_to_flux(
                    self, 
                    M: Array, 
                    K: Array
                    ) -> Array:
        r"""
        **Description:**
        Returns full flux vector from M- and K-vector specifying PFV.
        
        Args:
            M (Array): M-vector.
            K (Array): K-vector.            
        
        Returns:
            Array: Flux vector.
        
        """

        a = self.periods.a_matrix
        b = self.periods.b_vector

        if "coniLCS" in self.periods.moduli_space_limit:
            tmp = jnp.zeros(b.shape[0])
            tmp = tmp.at[0].set(1.)
            b = b+tmp*self.periods.ncf/24

        f0 = M@b
        f2 = (a.T@M)
        f2 = jnp.append(jnp.ones(1)*f0,f2)
        f2 = jnp.append(f2,jnp.zeros(1))
        f = jnp.append(f2,M)

        h = jnp.append(jnp.zeros(1),K)
        h = jnp.append(h,jnp.zeros(K.shape[0]+1))

        return jnp.append(f,h)


    @partial(jit, static_argnums = (0,))
    def pfv_to_moduli(
                    self, 
                    M: Array, 
                    K: Array,
                    tau: complex
                    ) -> Array:
        r"""
        **Description:**
        Returns values of the complex structure moduli at the level of the PFV.
        
        Args:
            M (Array): M-vector.
            K (Array): K-vector.
            tau (complex): Axio-dilaton value.
        
        Returns:
            Array: Values of complex structure moduli.
        
        """

        N = self.periods.mirror_intersection_numbers@M
        if "coniLCS" in self.periods.moduli_space_limit:
            gs = 1/tau.imag
            c0 = tau.real
            Nhat = N[1:,1:]
            phat = jnp.linalg.inv(Nhat)@K[1:]
            zbulk = phat*tau
            Kprime = K[0]-N[0,1:]@phat
            #zcf = jnp.exp(2*jnp.pi*Kprime/self.periods.ncf/gs/M[0])/2/jnp.pi

            # amatrix[0]@M-P1==0???
            phase_comb = -1j*(+c0*Kprime)
            radial = Kprime/gs
            zcf = -1/(2*jnp.pi*1j)*jnp.exp(2*jnp.pi/self.periods.ncf/M[0]*(phase_comb+radial))

            z0 = jnp.append(jnp.ones(1)*zcf,zbulk)
        else:
            p = jnp.linalg.inv(N)@K
            z0 = p*tau

        return z0

    @partial(jit, static_argnums = (0,7,))
    def W1_tilde(self,zbulk,tau,f2,h2,P1,K1,conj=False):
        r"""
        **Description:**
        Computes :math:`\widetilde{W}_1` as defined .
        
        Args:
            zbulk (Array): Values of the complex structure moduli excluding the conifold modulus.
            tau (complex): Value of the axio-dilaton :math:`\tau`.
            f2 (Array): Flux vector :math:`f_2`.
            h2 (Array): Flux vector :math:`h_2`.
            P1 (int): Flux :math:`P^1`.
            K1 (int): Flux :math:`K^1`.
            conj (bool, optional): Whether or not to conjugate the expression. Defaults to ``False``.
        
        Returns:
            complex: Value of :math:`\widetilde{W}_1`.
        
        """
    
        M0 = f2[0]
        H0 = h2[0]
        M1 = f2[1]
        H1 = h2[1]
        Malpha = f2[2:]
        Halpha = h2[2:]
        
        kappa = self.periods.mirror_intersection_numbers
        
        ncf = 2
        coeff = 2*jnp.pi*1j
        
        if conj:
            coeff = -coeff
            
        F1b = ((kappa[0,1:,1:]@zbulk)@zbulk)/2+self.periods.b_vector[0]-jnp.pi**2/6*ncf/coeff**2
        
        F2b = -((kappa[0,0,1:]@zbulk))+self.periods.a_matrix[0,0]
        
        dF1b = -((kappa[0,1:,1:]@zbulk))+self.periods.a_matrix[0,1:]
        
        tmp = (M0-tau*H0)*F1b
        
        tmp += (M1-tau*H1)*F2b
        
        tmp += (Malpha-tau*Halpha)@dF1b
        
        tmp += (-1)*(P1-tau*K1)
        
        return tmp

    @partial(jax.jit,static_argnums=(0,3,4,))
    def compute_zcf(self,x,flux,mode=None,conj=False):
        r"""
        **Description:**
        Computes the value of the conifold modulus :math:`z_{cf}`.
        Args:
            x (Array): Real variables.
            flux (Array): Flux vector.
            mode (string, optional): Mode to be used. If ``"pfv"``, uses the PFV approximation. If ``None``, uses the full expression. Defaults to ``None``.
            conj (bool, optional): Whether or not to conjugate the expression. Defaults to ``False``.
            
        Returns:
            complex: Value of the conifold modulus :math:`z_{cf}`.
        
        """
        
        if mode not in [None,"pfv","analytic"]:
            raise ValueError('Mode must be one of [None,"pfv"]!')
        
        f1,f2,h1,h2 = self._split_fluxes(flux)

        coeff = 2*jnp.pi*1j
            
        if conj:
            coeff = -coeff

        ncf = 2

        if mode == "pfv":
            
            Mvec = f2[1:]
            Kvec = h1[1:]
            M = Mvec[0]
            P1 = f1[1]
            N = self.periods.mirror_intersection_numbers@Mvec
            pvec = jnp.linalg.inv(N[1:,1:])@Kvec[1:]
            Kprime = Kvec[0]-N[0,1:]@pvec
            c0 = x[-2]
            s = x[-1]
            gs = 1/s
            #ncf = self.periods.ncf
            
            phase_comb = -1j*(self.periods.a_matrix[0]@Mvec-P1+c0*Kprime)
            if conj:
                phase_comb = -phase_comb

            radial = Kprime/gs

            exponent = 2*jnp.pi/ncf/M*(phase_comb+radial)
            
        else:

            x_out = jnp.append(jnp.ones(2),x)

            z,_,tau,_=self._convert_real_to_complex(x_out)

            zbulk = z[1:]
            f1,f2,h1,h2=self._split_fluxes(flux)
            
            M1 = f2[1]
            H1 = h2[1]
            
            P1 = f1[1]
            K1 = h1[1]
            
            W1 = self.W1_tilde(zbulk,tau,f2,h2,P1,K1,conj=conj)
            
            exponent = -coeff*(W1)/ncf/(M1-tau*H1)    
            
        zcf = (-1)/coeff * jnp.exp(exponent)
        
        return zcf

    @partial(jax.jit,static_argnums=(0,3,))
    def zcf_handling(self,x,flux,mode=None):
        r"""
        **Description:**
        Handles the conifold modulus depending on the mode chosen.
        
        Args:
            x (Array): Real variables.
            flux (Array): Flux vector.
            mode (string, optional): Mode to be used. If ``"pfv"``, uses the PFV approximation. If ``None``, uses the full expression. Defaults to ``None``.
            
        Returns:
            Array: Real variables including the conifold modulus.
        
        """
        
        if mode is None:
            xcz=jnp.zeros(2)
        else:
            zcf = self.compute_zcf(x,flux,mode=mode)
            xcz = jnp.array([zcf.real,zcf.imag])
            #if mode != "pfv":
            #    x = x[2:]
            
        return jnp.append(xcz,x)

    @partial(jax.jit,static_argnums=(0,3,))
    def DW_x_bulk(self,x,flux,mode=None):
        r"""
        **Description:**
        Computes the gradient of the flux superpotential with respect to the bulk moduli.
        
        Args:
            x (Array): Real variables.
            flux (Array): Flux vector.
            mode (string, optional): Mode to be used. If ``"pfv"``, uses the PFV approximation. If ``None``, uses the full expression. Defaults to ``None``.
            
        Returns:
            Array: Gradient of the flux superpotential with respect to the bulk moduli.
        """
        
        X = self.zcf_handling(x,flux,mode=mode)

        return self.DW_x(X,flux)[2:]

    @partial(jax.jit,static_argnums=(0,3,))
    def dDW_x_bulk(self,x,flux,mode=None):
        r"""
        **Description:**
        Computes the Hessian of the flux superpotential with respect to the bulk moduli.
        
        Args:
            x (Array): Real variables.
            flux (Array): Flux vector.
            mode (string, optional): Mode to be used. If ``"pfv"``, uses the PFV approximation. If ``None``, uses the full expression. Defaults to ``None``.
            
        Returns:
            Array: Hessian of the flux superpotential with respect to the bulk moduli.
        
        """

        X = self.zcf_handling(x,flux,mode=mode)

        return self.dDW_x(X,flux)[2:,2:]

            
    
    
    @partial(jit, static_argnums = (0,6,7))
    def codim_vacuum_moduli_space(self, moduli, moduli_c, tau, tau_c, fluxes,mode="SUSY",tolerance=1e-10):
        r"""
        
        **Description:**
        Determines the number of massive directions in SUSY flux vacua by computing the rank of the matrix returned by :func:`massive_directions`.
        
        .. admonition:: Details
            :class: dropdown
            
            This function computes the rank of the fermionic mass matrix in SUSY vacua or the Hessian of the scalar potential in non-SUSY vacua.
            The number of massive directions is then given by the rank of this matrix. The number of flat directions is then given by :math:`2(h^{2,1}+1)-\text{rank}`. 
            
        
        .. warning::
        
            A tolerance needs to be provided to compute the rank of the fermionic mass matrix. Ideally, this tolerance should be given by the one used for the minimisation itself. If the tolerance is chosen too large, the number of massive directions will be underestimated. If the tolerance is chosen too small, numerical noise will lead to an overestimation of the number of massive directions.
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Value of the axio-dilaton :math:`\tau`.
            tau_c (complex): Value of the complex conjugate axio-dilaton :math:`\overline{\tau}`.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            mode (string, optional): Default is ``mode="SUSY"``.
            tolerance (float, optional): Tolerance used to compute the rank of the fermionic mass matrix. Defaults to ``1e-10``.
        
        Returns:
            (Array): Number of massive directions.
        
        """
        modes=[None, "SUSY"]
        if mode not in modes:
            raise ValueError(f"Mode must be one of {modes}, but is {mode}!")

        if mode=="SUSY":
            return jnp.linalg.matrix_rank(self.DDW_matrix(moduli, moduli_c, tau, tau_c, fluxes,mode=mode),tol=tolerance)
        elif mode is None:
            return jnp.linalg.matrix_rank(self.scalar_potential_hessian(moduli, moduli_c, tau, tau_c, fluxes,mode=mode),tol=tolerance)
                
            
        
    ###################################################################################################################
    ######################################## IMAGINARY SELF-DUAL CHECK  ###############################################
    ###################################################################################################################
    
    


    

    ###################################################################################################################
    ################################## ALL BELOW IS UPDATED IN NOTEBOOK!!!  ###########################################
    ###################################################################################################################
    ###################################################################################################################
    ################################## ALL BELOW IS UPDATED IN NOTEBOOK!!!  ###########################################
    ###################################################################################################################
    ###################################################################################################################
    ################################## ALL BELOW IS UPDATED IN NOTEBOOK!!!  ###########################################
    ###################################################################################################################
    ###################################################################################################################
    ################################## ALL BELOW IS UPDATED IN NOTEBOOK!!!  ###########################################
    ###################################################################################################################
    ###################################################################################################################
    ################################## ALL BELOW IS UPDATED IN NOTEBOOK!!!  ###########################################
    ###################################################################################################################
    ###################################################################################################################
    ################################## ALL BELOW IS UPDATED IN NOTEBOOK!!!  ###########################################
    ###################################################################################################################
        
        
    
    
    
    
    @partial(jit, static_argnums = (0,3,))
    def W0_ISD_coefficients(self,moduli,tau,solve_mode="upper"):
        r"""
        
        **Description:**
        Computes coefficients in the superpotential.
        
        .. admonition:: Details
            :class: dropdown
        
            This function returns the coefficients
        
            .. math::
                C_{I}=\mathcal{N}(Z^i,\overline{Z}^i)_{IJ}X^J - F_I

            TODO CONTINUE DESCTIPTION
        
        
        
        Args:
            moduli (Array): Complex structure moduli values.
            tau (complex): : Value of axio-dilaton.
            solve_mode (str, optional): Description
        
        Returns:
            See also: :func:`small_W0_init`
        
        """

        print("TODO: FUNCTION IS BROKEN!!! DO WE NEED IT?")

        solve_modes=["upper","lower"]
        if solve_mode not in solve_modes:
            warnings.warn(f"Solve mode should be one of {solve_modes}, but is {solve_mode}!")
            sys.exit()
            
            
        # Compute matrix
        NMatVal=self.gauge_kinetic_matrix(moduli)
        
        # Construct half of the period vector
        XPer=jnp.concatenate((jnp.ones(1),moduli))
        
        # Compute gradient of
        grad_F_Per = self.prepot_grad_periods(XPer)
        
        if solve_mode=="upper":
            return jnp.matmul(XPer,jnp.conj(NMatVal))-grad_F_Per
        elif solve_mode=="lower":
            return XPer-jnp.matmul(jnp.linalg.inv(jnp.conj(NMatVal)),grad_F_Per)
            
    
    @partial(jit, static_argnums = (0,4))
    def small_W0_init(self, moduli,tau,fluxes,solve_mode="upper"):
        r"""
        
        **Description:**
        Computes an approximate value for :math:`W_0` based on ISD conditions
        
        TODO: CHANGE NOTATION FOR f_1 and h_1 as well as f_2 and h_2
        
        .. admonition:: Details
            :class: dropdown
        
            Using the identity
        
            .. math::
                \mathbf{f}_1-\tau \mathbf{h}_1=\mathcal{N}(Z^i,\overline{Z}^i)\, (\mathbf{f}_2-\tau\mathbf{h}_2)
        
            we can write the superpotential as
        
            .. math::
                W_0 = (\mathbf{f}_2-\tau\mathbf{h}_2)^I\, (\mathcal{N}(Z^i,\overline{Z}^i)_{IJ}X^J - F_I) \, .
        
            This value depends only on values that we randomly sample.
        
        Args:
            moduli (Array): Complex structure moduli values.
            tau (complex): : Value of axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            solve_mode (str, optional): Description
        
        Returns:
            See also: :func:`ISD_cond_complex`
        
            See also: :func:`ISD_cond_complex_solvable`
        
            See also: :func:`W0_ISD_coefficients`
        
        """

        print("TODO: RENAME!")

        print("TODO: WHAT'S THE INPUT HERE????")

        solve_modes=["upper","lower"]
        if solve_mode not in solve_modes:
            warnings.warn(f"Solve mode should be one of {solve_modes}, but is {solve_mode}!")
            sys.exit()
        
        # Split fluxes
        FF = fluxes[:self.dimension_H3]
        HF = fluxes[self.dimension_H3:]
        
        # Get coefficients
        matrix_coeff=self.W0_ISD_coefficients(moduli,tau,fluxes,solve_mode=solve_mode)
        
        # Compute Kähler potential
        KP=self.kahler_potential(moduli,jnp.conj(moduli),tau,jnp.conj(tau))
        
        if solve_mode=="upper":
            res = jnp.matmul((FF-tau*HF),matrix_coeff)
        elif solve_mode=="lower":
            res = jnp.matmul((FF-tau*HF),matrix_coeff)
            
        return matrix_coeff,res,jnp.exp(KP.real/2.)*jnp.sqrt(2./jnp.pi)*res
    
    
        
    
    
    
    
    
    


     ############# NOT WORKING YET!!!!

    @partial(jit, static_argnums = (0,))
    def scalar_potential_mod(
        self,
        moduli: Array,
        moduli_c: Array,
        tau: complex,
        tau_c: complex,
        fluxes: Array
    ) -> complex:
        r"""
        
        **Description:**
        Computes 
        
        .. admonition:: Details
            :class: dropdown
        
            From ??? eq. A.11 in `2306.01059 <https://arxiv.org/pdf/2306.01059.pdf>`_
        
        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values of the complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            tau_c (complex): Complex conjugate value for axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
        
        Returns:
            complex: Value of the scalar potential in terms of ISD matrix.
        
        """
        
        FF = fluxes[:self.n_fluxes]
        HF = fluxes[self.n_fluxes:]
        
        G = FF-tau*HF
        cG = FF-tau_c*HF
        
        tau2 = -1j*(tau-tau_c)/2.
        
        NFlux = self.tadpole(fluxes)
        
        M = self.ISD_matrix(moduli,moduli_c)
        # Why do we have to take the inverse here?
        print("TODO: Why do we have to take the inverse here? Because we defined the ISD matrix as in https://arxiv.org/pdf/2310.06040, \
            but we use the inverse in our convention due to reordering of the periods!!")
        M = jnp.linalg.inv(M)
        
        #return -jnp.matmul(cG,jnp.matmul(M,G))/tau2,2.*NFlux,1j*jnp.matmul(cG,jnp.matmul(self.periods.sigma(),G))/tau2
        return -jnp.matmul(cG,jnp.matmul(M,G))/tau2+2.*NFlux

    
    @partial(jit, static_argnums = (0,6))
    def _hessian_real(
        self,
        moduli: Array,
        moduli_c: Array,
        tau: complex,
        tau_c: complex,
        fluxes: Array,
        noscale: bool = True
    ) -> Array:
        r"""
        
        **Description:**
        Returns the second derivatives :math:`\partial_{\phi^\alpha}\partial_{\phi^\beta}V` of the :math:`F`-term scalar potential
        with respect to the real fields :math:`\phi^\alpha`.

        .. note::

            The complex fields :math:`\Phi^I` can be decomposed as
            
            .. math::
                \Phi^I = \phi^I + \mathrm{i}\phi^{I+h^{1,2}+1}

            where :math:`\phi^\alpha`, :math:`\alpha=0,\ldots,2h^{1,2}+1`, are the real and imaginary parts respectively.

            Up to permutations, this function is equivalent to :func:`ddV_x`, but slightly slower.

        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): Axio-dilaton value.
            tau_c (complex): Complex conjugate value for axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            noscale (bool, optional): If ``True``, uses the no-scale flux scalar potential. Defaults to ``True``.
        
        Returns:
            Array: Value of :math:`\partial_{\phi^\alpha}\partial_{\phi^\beta}V`.

        """

        ddV = jax.jacrev(self.dV, argnums = (0,2), holomorphic = True)(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale,mode="real")
        ddV_c = jax.jacrev(self.dV, argnums = (1,3), holomorphic = True)(moduli, moduli_c, tau, tau_c, fluxes,noscale=noscale,mode="real")

        ddV = jnp.append(ddV[0],jnp.array([ddV[1]]).T,axis=1)
        ddV_c = jnp.append(ddV_c[0],jnp.array([ddV_c[1]]).T,axis=1)

        dt_dV = 1j*(ddV-ddV_c)
        dphi_dV = ddV+ddV_c

        return jnp.append(jnp.real(dt_dV),jnp.real(dphi_dV),axis=1)
    
    @partial(jit, static_argnums = (0,6))
    def hessian_SUGRA(
        self, 
        moduli: Array, 
        moduli_c: Array, 
        tau: complex, 
        tau_c: complex, 
        fluxes: Array, 
        noscale: bool = True
        ) -> Array:
        r"""
        
        **Description:**
        Returns the Hessian  for the complex structure moduli :math:`z^{i}`
        and the axio-dilaton :math:`\tau`.
        
        .. admonition:: Details
            :class: dropdown
        
            This implementation of the Hessian uses standard supergravity formulas for :math:`\mathcal{N}=1`
            supergravity in four dimensions as spelled out in e.g. `hep-th/0404116 <https://inspirehep.net/literature/648462>`_.
            That is, we compute the mixed 
            second derivatives
        
            .. math::
                    \partial_{A}\partial_{\overline{B}} V
                            = \mathrm{e}^{K} \bigl (M_{AC} K^{C\overline{D}} \overline{M_{BD}}
                            -(\lambda -1) K_{A\overline{B}} |W|^2 \bigl )

            TODO: TERMS MISSING!!!!

            and the holomorphic second derivatives

            .. math::
                    \partial_{A}\partial_{B} V =-(\lambda - 2) \mathrm{e}^{K} \,\overline{W}\, M_{AB}

            where we introduced

            .. math::
                    M_{AB} = D_AD_BW\, , \quad \lambda = \begin{cases} 3 & \text{full potential}\\ 
                                                                0 & \text{no-scale potential}\end{cases}\; \, .

            TODO: ADD DETAILS HERE!!!

        
        Args:
            moduli (Array): Complex structure moduli values.
            moduli_c (Array): Complex conjugate values for complex structure moduli.
            tau (complex): : Value of axio-dilaton.
            tau_c (complex): Value of complex conjugate axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            noscale (bool, optional): If ``True``, uses the no-scale flux scalar potential. Defaults to ``True``.
        
        Returns:
            Array: Hessian matrix with entries :math:`\partial_{A}\partial_{\bar{B}} V`
                and :math:`\partial_{A}\partial_{B} V` assuming :math:`D_A W=0`.
        
        """
        
        DDW_val=self.DDW(moduli, moduli_c, tau, tau_c,fluxes,mode=None,check_SUSY=False)
        
        DW_val=self.DW(moduli, moduli_c, tau, tau_c,fluxes)
        
        W_val=self.superpotential(moduli, tau,fluxes)
        KM_val=jnp.array(self.kahler_metric(moduli,moduli_c,tau,tau_c))
        KP=self.kahler_potential(moduli, moduli_c, tau, tau_c)
        Inv_KM=self.inverse_kahler_metric(moduli, moduli_c, tau, tau_c)


        dDDW = jax.jacrev(self.DDW,argnums=(0,2),holomorphic=True)
        dIKM = jax.jacrev(self.inverse_kahler_metric,argnums=(0,2),holomorphic=True)

        dDDW_val0= dDDW(moduli, moduli_c, tau, tau_c,fluxes,mode=None,check_SUSY=False)

        dDDW_val = jnp.append(dDDW_val0[0],dDDW_val0[1].reshape(self.h12+1,self.h12+1,1),axis=2)

        dIKM_val0= dIKM(moduli, moduli_c, tau, tau_c)
        dIKM_val = jnp.append(dIKM_val0[0],dIKM_val0[1].reshape(self.h12+1,self.h12+1,1),axis=2)
           
        DcDW_val = self.DcDW(moduli, moduli_c, tau, tau_c,fluxes)


        if noscale:
            lam=0.
        else:
            lam=3.
        
        term1 = jnp.matmul(DDW_val, jnp.matmul(Inv_KM.T,jnp.conj(DDW_val)))
        term2 = -(lam-1.)*W_val*jnp.conj(W_val)*KM_val
        V_I_barJ = jnp.exp(KP) * (term1+term2)

        term1 = jnp.matmul(dDDW_val.T,jnp.matmul(Inv_KM.T,jnp.conj(DW_val)))
        term2 = jnp.matmul(DDW_val,jnp.matmul(dIKM_val.transpose(2,0,1),jnp.conj(DW_val)).T).T
        term3 = (1.-lam)*DDW_val*jnp.conj(W_val)
        term4 =  jnp.matmul(DDW_val,jnp.matmul(Inv_KM.T,jnp.conj(DcDW_val)).T)
        V_I_J = jnp.exp(KP) * (term1+term2+term3+term4)

        print("TODO: FINISH IMPLEMENTATION!!!")

        hess_mixed,hess_holom=V_I_barJ,V_I_J
            
        a = jnp.hstack((jnp.asarray(hess_mixed), jnp.asarray(hess_holom)))

        b = jnp.hstack((jnp.asarray(jnp.conj(hess_holom)), jnp.asarray(jnp.conj(hess_mixed))))

        return jnp.vstack((a, b))


    @partial(jit, static_argnums = (0,4,5,6,))
    def linearised_shifts_ISD(
        self,
        moduli: Array,
        tau: complex,
        fluxes: Array,
        mode: str = "ISD",
        return_shifts: bool = False,
        remove_NANs: bool = False
    ) -> Union[Tuple[Array, complex, Array], Tuple[Array, complex, Array, Array]]:
        """
        **Description:**
        Computes the linearised shifts for the complex structure moduli and the axio-dilaton based on :math:`ISD_+`-sampling for fluxes.

        Args:
            moduli (Array): Values for the complex structure moduli.
            tau (complex): Value of the axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            mode (str, optional): Computation mode for obtaining the linearised shifts.
            return_shifts (bool, optional): Whether to return the shifts for the moduli or the shifted moduli. Defaults to ``False``.
            remove_NANs (bool, optional): Whether to remove NaN values from input. Defaults to ``False``.

        Returns:
            Array: Moduli shifts or shifted values of the complex structure moduli :math:`Z^i`.
            complex: Axio-dilaton shift or shifted value of the axio-dilaton :math:`\tau`.
            Array: Values of the fluxes.
        """

        modes = ["ISD","random"]

        if mode not in modes:
            raise ValueError(f"Mode must be one of {modes}, but `mode = {mode}` was given!")
            
        if remove_NANs:
            moduli = jnp.nan_to_num(moduli,nan=1.)
            tau = jnp.nan_to_num(tau,nan=1.)
            fluxes = jnp.nan_to_num(fluxes,nan=1.)

        NMatVal=self.gauge_kinetic_matrix(moduli,jnp.conj(moduli))
        IMat = jnp.imag(NMatVal)
        RMat = jnp.real(NMatVal)
        FF = fluxes[:self.n_fluxes]
        HF = fluxes[self.n_fluxes:]

        mu = FF[self.dimension_H3:]
        nu = HF[self.dimension_H3:]
        RHS = mu-tau*nu

        Lambda=jnp.matmul(jnp.conj(NMatVal),RHS)

        Re_Lambda=jnp.real(Lambda)
        Im_Lambda=jnp.imag(Lambda)
        c0=jnp.real(tau)
        s=jnp.imag(tau)

        ml_cont,nl_cont = Re_Lambda-c0/s*Im_Lambda, -1./s*Im_Lambda

        if mode=="ISD":
            ml = jnp.around(ml_cont,0)
            nl = jnp.around(nl_cont,0)
        else:
            ml = FF[:self.dimension_H3]
            nl = HF[:self.dimension_H3]

        delta_ml = ml-ml_cont
        delta_nl = nl-nl_cont

        delta_LHS = delta_ml-tau*delta_nl
        
        Lambdak = jnp.matmul(self.dN(moduli,jnp.conj(moduli),conj=True).transpose(2,0,1),RHS)
        tilde_Lambdak = jnp.matmul(self.dN_c(moduli,jnp.conj(moduli),conj=True).transpose(2,0,1),RHS)
        
        re_Lambdak = jnp.real(Lambdak)
        im_Lambdak = jnp.imag(Lambdak)
        re_tilde_Lambdak = jnp.real(tilde_Lambdak)
        im_tilde_Lambdak = jnp.imag(tilde_Lambdak)

        re_ak = re_Lambdak+re_tilde_Lambdak
        re_vk = -im_Lambdak+im_tilde_Lambdak
        im_ak = im_Lambdak+im_tilde_Lambdak
        im_vk = re_Lambdak-re_tilde_Lambdak
        re_c = nl-RMat@nu
        re_s = -IMat@nu
        im_c = IMat@nu
        im_s = nl-RMat@nu
        re_delta_LHS = jnp.real(delta_LHS)
        im_delta_LHS = jnp.imag(delta_LHS)
        b = jnp.append(re_delta_LHS,im_delta_LHS)


        A1 = jnp.append(jnp.append(jnp.append(re_ak,re_vk,axis=0),jnp.array([re_c]),axis=0),jnp.array([re_s]),axis=0)
        A2 = jnp.append(jnp.append(jnp.append(im_ak,im_vk,axis=0),jnp.array([im_c]),axis=0),jnp.array([im_s]),axis=0)
        A = jnp.append(A1.T,A2.T,axis=0)
        # warning: not guaranteed to be inveritble!!!
        # TODO: safer option to invert?

        # Solve for the shifts
        #shift = jnp.matmul(jnp.linalg.inv(A),b)
        shift = jnp.linalg.solve(A,b)
        
        flux_new = jnp.append(jnp.append(ml,mu),jnp.append(nl,nu))
        
        if return_shifts:
            return shift[:self.h12]+1j*shift[self.h12:2*self.h12],shift[-2]+1j*shift[-1],flux_new
        else:
            moduli_new = moduli+shift[:self.h12]+1j*shift[self.h12:2*self.h12]
            
            tau_new = tau+shift[-2]+1j*shift[-1]
            
            return moduli_new,tau_new,flux_new

    @partial(jit, static_argnums = (0,4,5,6,))
    def linearised_shifts_H(
        self,
        moduli: Array,
        tau: complex,
        fluxes: Array,
        mode: str = "Hflux",
        return_shifts: bool = False,
        remove_NANs: bool = False
    ) -> Union[Tuple[Array, complex, Array], Tuple[Array, complex, Array, Array]]:
        """
        **Description:**
        Computes linearised shifts for the complex structure moduli and axio-dilaton based on given H-flux.

        Args:
            moduli (Array): Starting point for the complex structure moduli.
            tau (complex): Starting point for the axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            mode (str, optional): Computation mode for obtaining the linearised shifts.
            return_shifts (bool, optional): Whether to return the shifts for the moduli or the shifted moduli. Defaults to ``False``.
            remove_NANs (bool, optional): Whether to remove NaN values from input. Defaults to ``False``.

        Returns:
            Array: Moduli shifts or shifted values of the complex structure moduli :math:`Z^i`.
            complex: Axio-dilaton shift or shifted value of the axio-dilaton :math:`\tau`.
            Array: Values of the fluxes.
        """

        modes = ["Hflux","Hflux_random"]

        if mode not in modes:
            raise ValueError(f"Mode must be one of {modes}, but `mode = {mode}` was given!")

        if remove_NANs:
            moduli = jnp.nan_to_num(moduli,nan=1.)
            tau = jnp.nan_to_num(tau,nan=1.)
            fluxes = jnp.nan_to_num(fluxes,nan=1.)

        FF = fluxes[:self.n_fluxes]
        HF = fluxes[self.n_fluxes:]

        c0=jnp.real(tau)
        s=jnp.imag(tau)

        SigmaH = jnp.matmul(self.periods.sigma(),HF)
        RHS = s*SigmaH

        M0 = self.ISD_matrix(moduli,jnp.conj(moduli))

        M0_SigmaH = jnp.matmul(M0,SigmaH)

        f_cont = jnp.real(M0_SigmaH*s+HF*c0)

        # jnp.matmul((jnp.matmul(MMatVal,self.periods.sigma())*jnp.imag(tau)+jnp.identity(len(MMatVal))*jnp.real(tau)),HF)

        if mode=="Hflux":
            f = jnp.around(f_cont,0)
        else:
            f = FF     

        delta_f = f-f_cont

        delta_LHS = delta_f

        Theta_k = jnp.matmul(self.dM(moduli,jnp.conj(moduli)).transpose(2,0,1),RHS)
        Theta_bk = jnp.matmul(self.dM_c(moduli,jnp.conj(moduli)).transpose(2,0,1),RHS)


        re_Theta_k = jnp.real(Theta_k)
        im_Theta_k = jnp.imag(Theta_k)
        re_Theta_bk = jnp.real(Theta_bk)
        im_Theta_bk = jnp.imag(Theta_bk)

        re_ak = re_Theta_k+re_Theta_bk
        re_vk = -im_Theta_k+im_Theta_bk
        #im_ak = im_Theta_k+im_Theta_bk
        #im_vk = re_Theta_k-re_Theta_bk

        re_c = HF
        re_s = M0_SigmaH
        #im_c = jnp.zeros(re_ak.shape[1])
        #im_s = jnp.zeros(re_ak.shape[1])

        re_delta_LHS = jnp.real(delta_LHS)
        #im_delta_LHS = jnp.imag(delta_LHS) # THIS IS ZERO HERE!!!

        #b = jnp.append(re_delta_LHS,im_delta_LHS)

        b = re_delta_LHS

        A1 = jnp.append(jnp.append(jnp.append(re_ak,re_vk,axis=0),jnp.array([re_c]),axis=0),jnp.array([re_s]),axis=0)

        # SHOULD BE ZERO!!!
        #A2 = jnp.append(jnp.append(jnp.append(im_ak,im_vk,axis=0),jnp.array([im_c]),axis=0),jnp.array([im_s]),axis=0)
        
        A = A1.T

        # warning: not guaranteed to be inveritble!!!
        # TODO: Other version for inversion?
        #shift = jnp.matmul(jnp.linalg.inv(A),b)
        shift = jnp.linalg.solve(A,b)
        #print("TODO: CHANGED INVERSE!")
        #shift = jax.scipy.linalg.solve(A,b)
        #cfac = jax.scipy.linalg.lu_factor(A.astype(jnp.float64))
        #shift = jax.scipy.linalg.lu_solve(cfac, b)
        
        
        flux_new = jnp.append(f,HF)

        if return_shifts:
            return shift[:self.h12]+1j*shift[self.h12:2*self.h12],shift[-2]+1j*shift[-1],flux_new
        else:
            moduli_new = moduli+shift[:self.h12]+1j*shift[self.h12:2*self.h12]
            tau_new = tau+shift[-2]+1j*shift[-1]
            return moduli_new,tau_new,flux_new
        
    @partial(jit, static_argnums = (0,4,5,6,))
    @partial(jit, static_argnums=(0, 4, 5, 6,))
    def linearised_shifts_F(
        self,
        moduli: Array,
        tau: complex,
        fluxes: Array,
        mode: str = "Fflux",
        return_shifts: bool = False,
        remove_NANs: bool = False
    ) -> Union[Tuple[Array, complex, Array], Tuple[Array, complex, Array, Array]]:
        r"""
        **Description:**
        Computes linearised shifts for the complex structure moduli and axio-dilaton based on given F-flux.

        Args:
            moduli (Array): Starting point for the complex structure moduli.
            tau (complex): Starting point for the axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            mode (str, optional): Computation mode for obtaining the linearised shifts. Defaults to ``"Fflux"``.
            return_shifts (bool, optional): Whether to return the shifts for the moduli or the shifted moduli. Defaults to ``False``.
            remove_NANs (bool, optional): Whether to remove NaN values from input. Defaults to ``False``.

        Returns:
            Array: Moduli shifts or shifted values of the complex structure moduli :math:`Z^i`.
            complex: Axio-dilaton shift or shifted value of the axio-dilaton :math:`\tau`.
            Array: Values of the fluxes.
        """
        
        modes: list = ["Fflux", "Fflux_random"]
        if mode not in modes:
            raise ValueError(f"Mode must be one of {modes}, but `mode = {mode}` was given!")

        # Handle NaN values if requested
        if remove_NANs:
            moduli = jnp.nan_to_num(moduli, nan=1.)
            tau = jnp.nan_to_num(tau, nan=1.)
            fluxes = jnp.nan_to_num(fluxes, nan=1.)
            
        FF = fluxes[:self.n_fluxes]
        HF = fluxes[self.n_fluxes:]

        c0=jnp.real(tau)
        s=jnp.imag(tau)
        
        tau_abs = c0**2+s**2

        SigmaF = jnp.matmul(self.periods.sigma(),FF)
        RHS = s*SigmaF

        M0 = self.ISD_matrix(moduli,jnp.conj(moduli))

        M0_SigmaF = jnp.matmul(M0,SigmaF)
        
        h_cont = jnp.real(FF*c0-M0_SigmaF*s)/tau_abs

        if mode=="Fflux":
            h = jnp.around(h_cont,0)
        else:
            h = HF     

        delta_h = h-h_cont

        delta_LHS = delta_h

        Theta_k = jnp.matmul(self.dM(moduli,jnp.conj(moduli)).transpose(2,0,1),RHS)
        Theta_bk = jnp.matmul(self.dM_c(moduli,jnp.conj(moduli)).transpose(2,0,1),RHS)


        re_Theta_k = jnp.real(Theta_k)
        im_Theta_k = jnp.imag(Theta_k)
        re_Theta_bk = jnp.real(Theta_bk)
        im_Theta_bk = jnp.imag(Theta_bk)

        re_ak = re_Theta_k+re_Theta_bk
        re_vk = -im_Theta_k+im_Theta_bk
        #im_ak = im_Theta_k+im_Theta_bk
        #im_vk = re_Theta_k-re_Theta_bk

        re_c = FF-2*h_cont*c0
        re_s = -M0_SigmaF-2*h_cont*s
        #im_c = jnp.zeros(re_ak.shape[1])
        #im_s = jnp.zeros(re_ak.shape[1])

        re_delta_LHS = jnp.real(delta_LHS)
        #im_delta_LHS = jnp.imag(delta_LHS) # THIS IS ZERO HERE!!!

        #b = jnp.append(re_delta_LHS,im_delta_LHS)

        b = re_delta_LHS*tau_abs

        A1 = jnp.append(jnp.append(jnp.append(-re_ak,-re_vk,axis=0),jnp.array([re_c]),axis=0),jnp.array([re_s]),axis=0)

        # SHOULD BE ZERO!!!
        #A2 = jnp.append(jnp.append(jnp.append(im_ak,im_vk,axis=0),jnp.array([im_c]),axis=0),jnp.array([im_s]),axis=0)
        
        A = A1.T

        # warning: not guaranteed to be inveritble!!!
        # TODO: Other version for inversion?
        #shift = jnp.matmul(jnp.linalg.inv(A),b)
        shift = jnp.linalg.solve(A,b)
        #print("TODO: CHANGED INVERSE!")
        
        flux_new = jnp.append(FF,h)

        if return_shifts:
            return shift[:self.h12]+1j*shift[self.h12:2*self.h12],shift[-2]+1j*shift[-1],flux_new
        else:
            moduli_new = moduli+shift[:self.h12]+1j*shift[self.h12:2*self.h12]
            tau_new = tau+shift[-2]+1j*shift[-1]
            return moduli_new,tau_new,flux_new
        

    @partial(jit, static_argnums = (0,4,5,6,7,8,))
    def linearised_shifts(
        self,
        moduli: Array,
        tau: complex,
        fluxes: Array,
        Q: Optional[int] = None,
        mode: str = "ISD",
        return_flag: bool = True,
        constraints: Optional[Callable] = None,
        remove_NANs: bool = False
    ) -> Union[Tuple[Array, complex, Array], Tuple[Array, complex, Array, Array]]:
        r"""
        **Description:**
        Computes the linearised shifts for the complex structure moduli and the axio-dilaton.

        Args:
            moduli (Array): Starting point for the complex structure moduli.
            tau (complex): Starting point for the axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            Q (int, optional): Tadpole bound to be used in the computation for the constraints.
            mode (str, optional): Computation mode for obtaining the linearised shifts.
            constraints (None, optional): Additional constraints to be imposed on the solutions.
            remove_NANs (bool, optional): Whether to remove NaN values from input. Defaults to ``False``.

        Returns:
            Array: Shifted values for the complex structure moduli :math:`Z^i`.
            complex: Shifted value for the axio-dilaton :math:`\tau`.
            Array: Updated values for the fluxes based on the ISD-condition.
            bool, optional: Boolean specifying whether the constraints are satisfied.

        """


        modes = ["ISD","random","Hflux","Hflux_random","Fflux","Fflux_random"]

        if mode not in modes:
            raise ValueError(f"Mode must be one of {modes}, but `mode = {mode}` was given!")

        if mode == "ISD" or mode == "random":
            moduli_new,tau_new,flux_new = self.linearised_shifts_ISD(moduli,tau,fluxes,mode=mode,remove_NANs=remove_NANs)
        elif mode == "Hflux" or mode == "Hflux_random":
            moduli_new,tau_new,flux_new = self.linearised_shifts_H(moduli,tau,fluxes,mode=mode,remove_NANs=remove_NANs)
        elif mode == "Fflux" or mode == "Fflux_random":
            moduli_new,tau_new,flux_new = self.linearised_shifts_F(moduli,tau,fluxes,mode=mode,remove_NANs=remove_NANs)


        if return_flag:

            # Checks boundary conditions like Kähler cone conditions or tadpole
            hyperplane_dist=self.periods.hyperplanes@jnp.imag(moduli_new)
            hyperplane_check=jnp.all(hyperplane_dist>=0.,axis=0)

            filter_conditions = (hyperplane_check==True)

            NFlux = self.tadpole(flux_new)

            filter_conditions &= (NFlux>0)

            if Q is not None:

                filter_conditions &= (NFlux<=Q)

            if constraints is not None:
                filter_conditions &= (constraints(moduli_new,tau_new,flux_new))

            return moduli_new,tau_new,flux_new,filter_conditions
        else:

            return moduli_new,tau_new,flux_new

    @partial(jit, static_argnums = (0,2,))
    def compute_residual(self,x,axis=1):
        r"""
        **Description:**
        Computes residual.
        
        Args:
            x (Array): Array. 
        
        Returns: 
            Array: Sum of absolute values.
        
        """

        return jnp.sum(jnp.abs(x),axis=axis)

    @partial(jit, static_argnums = (0,4,5,6,7,8,9,))
    def fterm_solver(
                     self, 
                     moduli: Array, 
                     tau: complex, 
                     fluxes: Array, 
                     objective_fct: Optional[Callable] = None, 
                     optimiser: Optional[Callable] = None,
                     tol: float = 1e-10, 
                     max_iters: int = 100, 
                     print_progress: bool = False, 
                     mode: str = None
                     ) -> Tuple[int,Array,Array,Array,Array,Array]:
        r"""
        **Description:**
        Solves F-term conditions for a given optimiser.
        
        Args:
            moduli (Array): Complex structure moduli values.
            tau (complex): Value of axio-dilaton.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            objective_fct (Callable, optional): Objective function to miminise.
            optimiser (Callable, optional): Optimiser to be used for the minimisation.
            tol (float, optional): Tolerance for the residual. Defaults to :c:var:`1e-10`.
            max_iters (int, optional): Maximum number of iterations. Defaults to :c:var:`100`.
            print_progress (bool, optional): Whether to print the progress of the minimisation. Defaults to ``False``.
            mode (str, optional): Mode for running the solver. Defaults to ``None``.    
        
        Returns: 
            int: Number of iterations.
            Array: Complex structure moduli values.
            Array: Value of axio-dilaton.
            Array: Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            Array: Boolean array specifying whether constraints are satisfied.
            Array: Residuals :math:`\sum_A |D_A W|` of the :math:`F`-terms.
        
        """

        modes=[None, "fluxes"]
        if mode not in modes:
            raise ValueError(f"CANNOT DETERMINE MODE FOR F-TERM SOLVER!\
                    Mode for filter computation must be one of {modes}!")
            
        if objective_fct is None:
            if mode is None:
                objective_fct = jax.vmap(self.DW)
            elif mode =="fluxes":
                objective_fct = jax.vmap(jax.vmap(self.DW))

        if mode is None:
            axis=1
                
        elif mode == "fluxes":
            axis=2

        def cond(arg):
            step, moduli, tau, fluxes, checks, res = arg
            return (step <max_iters) & (jnp.any(res > tol)) & (jnp.any(checks))


        def body(arg):
            #step, moduli, tau, fluxes, checks, res = arg
            step, moduli, tau, fluxes, _ , res = arg

            moduli, tau, fluxes, checks = optimiser(moduli, tau, fluxes)

            obj = objective_fct(moduli,jnp.conj(moduli),tau,jnp.conj(tau),fluxes)

            res = self.compute_residual(obj,axis=axis)

            if print_progress:
                # DEPRECATED
                #step = host_callback.id_tap(progress_bar_jax, (step, max_iters, jnp.min(jnp.nan_to_num(res,nan=1.))), result=step)

                out_type = jax.ShapeDtypeStruct(jnp.shape(step), jnp.result_type(step))
                step = pure_callback(progress_bar_jax, out_type,(step, max_iters, jnp.min(jnp.nan_to_num(res,nan=1.))),step)

            return (step + 1,  moduli, tau, fluxes, checks, res)

        if mode is None:
            return jax.lax.while_loop(cond,body,(0,moduli,tau,fluxes,jnp.ones(moduli.shape[0], dtype=bool),jnp.ones(moduli.shape[0])*10.))
        elif mode == "fluxes":
            input_shape = (moduli.shape[0],moduli.shape[1])
            return jax.lax.while_loop(cond,body,(0,moduli,tau,fluxes,jnp.ones(input_shape, dtype=bool),jnp.ones(input_shape)*10.))
        

    def sample_SUSY_flux_vacua(
                               self, 
                               N: int = 100, 
                               sampler = None, 
                               rns_key = None, 
                               max_iters: int = 10, 
                               moduli_sampling_mode: str = "cone", 
                               max_tadpole: Optional[int] = None,
                               objective_fct: Optional[Callable] = None, 
                               optimiser: Optional[Callable] = None, 
                               optimisers: Optional[list] = None, 
                               constraints: Optional[Callable] = None, 
                               mode: Optional[str] = None, 
                               tol: float = 1e-10, 
                               vmap_dim: int = 10**2,
                               print_progress: bool = False
                               ) -> Tuple[Array, Array, Array, Array]:
        r"""
        **Description:**
        Samples SUSY flux vacua.
        
        Args:
            N (int, optional): Defaults to :c:`100`.
            sampler (data_sampler, optional): Defaults to :c:`None`.
            rns_key (PRNGKey, optional): PRNG random key. Defaults to :c:`None`.
            max_iters (int, optional): Maximum number of iterations. Defaults to :c:var:`100`.
            moduli_sampling_mode (str, optional): Sampling mode for the moduli values. Defaults to :c:`"cone"`.
            max_tadpole (int, optional): Maximum tadpole to use for the sampling. Defaults to :c:`None`.
            objective_fct (Callable, optional): Objective function to miminise.
            optimiser (Callable, optional): Optimiser to be used for the minimisation.
            optimisers (list, optional): List of optimisers to be used for the minimisation.
            tol (float, optional): Tolerance for the residual. Defaults to :c:var:`1e-10`.
            print_progress (bool, optional): Whether to print the progress of the minimisation. Defaults to ``False``.
            constraints (Callable, optional): Defaults to :c:`None`.
            mode (str, optional): Solving mode specifying we want to solve F-term conditions with random fluxes (:c:var:`mode="random"`)
                or using ISD bias (:c:var:`mode="ISD"`) for the fluxes. Defaults to :c:`None`.
            vmap_dim (int, optional): Array dimension to use in vmapping. Defaults to :c:`100`.
        
        Returns: 
            Tuple[Array, Array, Array, Array]: Complex structure moduli, axio-dilaton values, fluxes, and residuals.
        
        """
        
        
        modes = [None,"ISD","random"]

        if mode not in modes:
            raise ValueError(f"Mode must be one of {modes}, but `mode = {mode}` was given!")

        if max_tadpole is None:
            max_tadpole = self.D3_tadpole
            
        if sampler is None:
            sampler = init_data_sampler(self)
        
        if optimisers is None:
            if optimiser is None:
                print("TODO: Should we make these separate functions?")

                kwargs = {"Q":max_tadpole,"return_flag":True,"constraints":constraints,"remove_NANs":True,"in_axes":(0,0,0)}
                
                if mode == "ISD":

                    linearised_shifts_ISD_v = vmapping_func(self.linearised_shifts,mode="ISD",**kwargs)
                    linearised_shifts_H_v = vmapping_func(self.linearised_shifts,mode="Hflux",**kwargs)
                    linearised_shifts_F_v = vmapping_func(self.linearised_shifts,mode="Fflux",**kwargs)

                elif mode == "random":

                    linearised_shifts_ISD_v = vmapping_func(self.linearised_shifts,mode="random",**kwargs)
                    linearised_shifts_H_v = vmapping_func(self.linearised_shifts,mode="Hflux_random",**kwargs)
                    linearised_shifts_F_v = vmapping_func(self.linearised_shifts,mode="Fflux_random",**kwargs)
                
                optimisers = [linearised_shifts_ISD_v,linearised_shifts_H_v,linearised_shifts_F_v]
                
            
            else:
                optimisers = [optimiser]
        
        if objective_fct is None:
            objective_fct = jax.vmap(self.DW)
        
        
        if rns_key is None:
            seed = 42
            rns_key = PRNGSequence(seed)
        
        moduli_sols = []
        tau_sols = []
        fluxes_sols = []
        res_sols = []
        count = 0
        num_vacua = 0
        tic = time.time()
        warnings.warn("TODO: Change the way we use checks. Seems like things are slowing done because we have to pull them back from CPU?")
        try:
            while num_vacua<N:
                
                
                for optimiser in optimisers:
                    
                    flag = True
                    tic0 = time.time()
                    c = 0
                    while flag:
                        moduli,tau,fluxes = sampler.initial_guesses(vmap_dim,rns_key=rns_key,moduli_sampling_mode=moduli_sampling_mode)
                        
                        _, moduli, tau, fluxes, checks,_ = self.fterm_solver(moduli,tau,fluxes.astype(float),objective_fct=objective_fct,
                                                            optimiser=optimiser,tol=tol,max_iters=1)

                        if jnp.any(checks):
                            flag = False

                        toc0 = time.time()
                        if np.abs(toc0-tic0)>60:
                            if c==0:
                                print(f"Sampling suitable initial guesses seems to take a long time.")


                            print(f"Iteration: {c}",flush=True,end="\r")
                            c += 1

                    _, moduli, tau, fluxes, checks,res = self.fterm_solver(moduli,tau,fluxes.astype(float),objective_fct=objective_fct,
                                                            optimiser=optimiser,tol=tol,max_iters=max_iters)
                    
                    checks &= (res<tol)
                    
                    moduli, tau, fluxes, res = moduli[checks], tau[checks], fluxes[checks], res[checks]
                    
                    
                    if len(moduli_sols)==0:
                        moduli_sols,tau_sols,fluxes_sols,res_sols = moduli, tau, fluxes, res
                    else:
                        moduli_sols = jnp.append(moduli_sols,moduli,axis=0)
                        tau_sols = jnp.append(tau_sols,tau)
                        fluxes_sols = jnp.append(fluxes_sols,fluxes,axis=0)
                        res_sols = jnp.append(res_sols,res)
                    
                    num_vacua += len(moduli)

                    toc = time.time()

                    if print_progress:
                    
                        print(f"Number vacua: {num_vacua}/{N}      finishing rate: {np.around(num_vacua/N*100,2)}%       counter: {count}         time: {np.around(toc-tic,2)}s           ",end="\r",flush=False)
                    
                    if num_vacua>N:
                        break
                    
                count += 1
        except KeyboardInterrupt:
            print("")
            print("Stopped sampling due to KeyboardInterrupt.")
            return moduli_sols,tau_sols,fluxes_sols,res_sols
        except Exception as error:
            print("")
            print('Caught this error: ' + repr(error))
            return moduli_sols,tau_sols,fluxes_sols,res_sols

        
        return moduli_sols,tau_sols,fluxes_sols,res_sols

    def sample_SUSY_vacua_from_fluxes(
                                      self,
                                      fluxes_init=None,
                                      initial_guesses = None,
                                      N=100,
                                      sampler = None,
                                      rns_key=None,
                                      max_iters = 10,
                                      moduli_sampling_mode = "cone",
                                      max_tadpole = None,
                                      objective_fct = None,
                                      optimiser_init: Optional[Callable] = None,
                                      optimiser_steps: Optional[Callable] = None,
                                      constraints=None,
                                      mode=None,
                                      tol = 1e-10,
                                      vmap_dim_flux = 10,
                                      vmap_dim_pts = 10,
                                      print_progress=False
                                      ) -> Tuple[Array,Array,Array,Array]:
        r"""
        **Description:**
        Samples SUSY flux vacua for given input fluxes.
        
        Args:
            fluxes_init (Array, optional): Input fluxes.
            initial_guesses (Tuple[Array,Array], optional): Initial guesses. Format: (moduli,tau).
            N (int, optional): Defaults to :c:`100`.
            sampler (data_sampler, optional): Defaults to :c:`None`.
            rns_key (PRNGKey, optional): PRNG random key. Defaults to :c:`None`.
            max_iters (int, optional): Maximum number of iterations. Defaults to :c:var:`100`.
            moduli_sampling_mode (str, optional): Sampling mode for the moduli values. Defaults to :c:`"cone"`.
            max_tadpole (int, optional): Maximum tadpole to use for the sampling. Defaults to :c:`None`.
            objective_fct (Callable, optional): Objective function to miminise.
            optimiser (Callable, optional): Optimiser to be used for the minimisation.
            optimisers (list, optional): List of optimisers to be used for the minimisation.
            tol (float, optional): Tolerance for the residual. Defaults to :c:var:`1e-10`.
            print_progress (bool, optional): Whether to print the progress of the minimisation. Defaults to ``False``.
            constraints (Callable, optional): Defaults to :c:`None`.
            mode (str, optional): Solving mode specifying we want to solve F-term conditions with random fluxes (:c:var:`mode="random"`)
                or using ISD bias (:c:var:`mode="ISD"`) for the fluxes. Defaults to :c:`None`.
            vmap_dim_flux (int, optional): Array dimension to use in vmapping over fluxes. Defaults to :c:`10`.
            vmap_dim_pts (int, optional): Array dimension to use in vmapping over initial guesses. Defaults to :c:`10`.
        
        Returns: 
            int: Number of iterations.
            Array: Complex structure moduli values.
            Array: Value of axio-dilaton.
            Array: Boolean array specifying whether constraints are satisfied.
            Array: Residuals :math:`\sum_A |D_A W|` of the :math:`F`-terms.
        
        """
        
        modes = ["ISD","random","Hflux","Hflux_random","Fflux","Fflux_random"]

        if mode not in modes:
            raise ValueError(f"Mode must be one of {modes}, but `mode = {mode}` was given!")

        if max_tadpole is None:
            max_tadpole = self.D3_tadpole
            
        if sampler is None and initial_guesses is None:
            sampler = data_sampler(self)
        
        if objective_fct is None:
            objective_fct = jax.vmap(jax.vmap(self.DW))
        
        if rns_key is None:
            seed = 42
            rns_key = PRNGSequence(seed)
        
        if optimiser_init is None:
            kwargs = {"mode":mode,"Q":max_tadpole,"constraints":constraints,"remove_NANs":True}
            find_solution_init = vmapping_func(self.linearised_shifts,in_axes=(0,0,None),return_flag=False,**kwargs)
            optimiser_init = vmapping_func(find_solution_init,in_axes=(None,None,0))
        
        if optimiser_steps is None:
            kwargs = {"mode":mode,"Q":max_tadpole,"constraints":constraints,"remove_NANs":True}
            find_solution_steps = vmapping_func(self.linearised_shifts,in_axes=(0,0,0),return_flag=True,**kwargs)
            optimiser_steps = vmapping_func(find_solution_steps,in_axes=(0,0,0))

        #if not (fluxes_init is None):
        #    N = 0

        moduli_sols = []
        tau_sols = []
        fluxes_sols = []
        res_sols = []
        count = 0
        num_vacua = 0
        tic = time.time()
        try:
            while num_vacua<N:
                
                if fluxes_init is None:
                    fluxes = sampler.get_fluxes(vmap_dim_flux)
                else:
                    fluxes = fluxes_init
                    
                    
                if initial_guesses is None:
                    moduli,tau = sampler.initial_guesses(vmap_dim_pts,rns_key=rns_key,moduli_sampling_mode=moduli_sampling_mode,include_fluxes=False)
                else:
                    moduli,tau = initial_guesses
                
                # Initial step:
                ## Maps initial guesses for a single choice of half of the fluxes
                ## onto full choices of fluxes via ISD sampling.
                ## We use the linearised shift computation to also return new initial guesses
                ## closer to the actual F-term minimum.
                moduli0,tau0,fluxes0 = optimiser_init(moduli,tau,fluxes)
                
                # Optimisation step:
                ## Using linearised shift computation to solve the F-term conditions
                ## for the above choices of fluxes.
                _, moduli, tau, fluxes, checks, res = self.fterm_solver(moduli0,tau0,fluxes0.astype(float),objective_fct=objective_fct,
                                                        optimiser=optimiser_steps,tol=tol,max_iters=max_iters,mode="fluxes")#,print_progress=print_progress

                ## Add to the checks that DW<tol
                checks &= (res<tol)

                ## Use the checks as flags to collect final results.
                moduli, tau, fluxes, res = moduli[checks], tau[checks], fluxes[checks], res[checks]


                if len(moduli_sols)==0:
                    moduli_sols,tau_sols,fluxes_sols,res_sols = moduli, tau, fluxes, res
                else:
                    moduli_sols = jnp.append(moduli_sols,moduli,axis=0)
                    tau_sols = jnp.append(tau_sols,tau)
                    fluxes_sols = jnp.append(fluxes_sols,fluxes,axis=0)
                    res_sols = jnp.append(res_sols,res)



                num_vacua += len(moduli)

                toc = time.time()
                if print_progress:
                    if N>0:
                        print(f"Number vacua: {num_vacua}/{N}      finishing rate: {np.around(num_vacua/N*100,2)}%       counter: {count}         time: {np.around(toc-tic,2)}s           ",end="\r",flush=False)
                    else:
                        print(f"Number vacua: {num_vacua}/{len(fluxes0)}      success rate: {np.around(num_vacua/len(fluxes0)*100,2)}%       time: {np.around(toc-tic,2)}s           ",end="\r",flush=False)
                    
                count += 1
        except KeyboardInterrupt:
            print("")
            print("Stopped sampling due to KeyboardInterrupt.")
            return moduli_sols,tau_sols,fluxes_sols,res_sols
        except Exception as error:
            print("")
            print('Caught this error: ' + repr(error))
            return moduli_sols,tau_sols,fluxes_sols,res_sols

        return moduli_sols,tau_sols,fluxes_sols,res_sols


    
    
    
    
    
    def CheckConstraints(model, roots, fluxes,flux_ids=None,uni_pts_ids=None,max_deg=1,mode=None,verbose=False,tolerance=1e-10):
        r"""
        
        **Description:**
        Tests filter conditions on the obtained roots of the F-term conditions for given fluxes. We currently check the following conditions
        
        
        .. admonition:: Details
            :class: dropdown
        
            1.) Positivity of the string coupling
        
            2.) Kahler cone conditions: real Kähler potential and positive definiteness of the Kähler metric
        
            3.) Positive definiteness of the Hessian of the Scalar potential
        
            To check the positive definiteness of the Hermitian matrices obtained from the Kähler metric and Hessian, we compute the Cholesky decomposition via `jnp.linalg.cholesky`.
        
            Flat directions or, stated otherwise, a non-trivial dimension of the vacuum moduli space are accounted for by computing `model.codim_vacuum_moduli_space`. Here, we use a tolerance set by the one used for checking roots of the F-term conditions `model.fterms`.
        
            Provide example here?
        
            TODO
        
            Let :math:`t^i` be the generators of the Kähler cone. For our vacuum solution :math:`Z` to be inside the Kähler cone, we demand that it can be written as
        
            .. math::
                Z=\sum_{i=1}^{h^{1,2}}\, c_i\, t^{i}\; ,\quad c_i\geq 0\, .
        
            The coefficients :math:`c_i` are a measure of the distance to one of the walls for the Kähler cone.
        
            If we search only for SUSY vacua, we look at 1st order obstructions to DW=0 for the fluctuations (see https://inspirehep.net/files/6bbb139c9328fbcd16e4b2ce50a8a567)
        
            TODO CONTINUE DESCRIPTIOn
        
        Args:
            model (class)
            roots (JAX array): Values of moduli at point in moduli space to compute Hessian.
            fluxes (Array): Array of fluxes. The ordering starts with RR-fluxes, then the NSNS-fluxes.
            flux_ids (JAX array, optional): IDs of initial fluxes.
            uni_pts_ids (JAX array, optional): ID of unique start points for which roots have been obtained
            max_deg (int, optional): Description
            mode (None, optional): Description
            verbose (string, optional)
            tolerance (float, optional): Description
        
        Returns:
            filtered_roots (numpy array): Array of roots satisfying the filter conditions.
            filtered_fluxes (numpy array): Array of fluxes satisfying the filter conditions.
        
        """

        check_nan_V=vmap(check_nan)

        cholesky_V=vmap(jnp.linalg.cholesky)




        print("NOT WORKING YET!!!!")


        modes = [None, "SUSY"]
        if mode not in modes:
            warnings.warn(f"Mode for filter computation must be one of {modes}!")
            sys.exit()
            
            
        # Define vmap functions:
        F_inst_V = vmap(model.F_inst)
        
        Prepot_V = vmap(model.prepot)
        
        Kahler_Metric_V = vmap(model.kahler_metric)
        
        Pi_dagger_Sigma_Pi_V=vmap(model.A)
        
        # If we search only for SUSY vacua, we look at 1st order obstructions to DW=0 for the fluctuations (see https://inspirehep.net/files/6bbb139c9328fbcd16e4b2ce50a8a567)
        if mode=="SUSY":
            codim_vacuum_moduli_space_V=vmap(lambda mod, cmod, tau, ctau, f: model.codim_vacuum_moduli_space(mod,cmod,tau,ctau,f,mode=mode,tolerance=tolerance))
            rank_matrix_V=vmap(lambda x: rank_matrix(x,tolerance=tolerance*tolerance)) # Take tolerance^2 because we have F-term squared for SUSY minima which should be of order less than tolerance!
        else:
            rank_matrix_V=vmap(lambda x: rank_matrix(x,tolerance=tolerance))
            codim_vacuum_moduli_space_V=rank_matrix_V
        
        Scalar_Potential_Hess_V = vmap(lambda mod, cmod, tau, ctau, f: model.scalar_potential_hessian(mod,cmod,tau,ctau,f,mode=mode))
        
        
        # Define complex moduli from real roots
        mod = jnp.asarray(roots[:,0:-2:2]) + 1.j*jnp.asarray(roots[:,1:-2:2])
        
        # Define complex axio-dilaton from real roots
        tau = roots[:,-2] + 1.j*roots[:,-1]
        
        # Get only imaginary part
        imag_mod=jnp.imag(mod)

        Pi_D_Pi=Pi_dagger_Sigma_Pi_V(mod, jnp.conj(mod))
        
        Finstb = F_inst_V(mod)
        Fpertb = Prepot_V(mod) - Finstb
        
        
        filter_conditions=(jnp.imag(tau)>0) & (Pi_D_Pi>0)& (jnp.abs(Finstb/Fpertb)< 1e-1)
        
        if verbose:
            print("HAVE TO STILL IMPLEMENT THE RADIUS OF CONVERGENCE CHECK!!!")
            
        # TODO: Implement radius of convergence check!
        """
        if model.max_deg>0:
            
            lamCs=model.check_radius_conv(max_deg)
            radius_conv_check1 = jnp.all(imag_mod-jnp.array(lamCs)>0,axis=0)
            filter_conditions&=(radius_conv_check1==True)
            
            # Vmap the radius convergence check
            check_radius_conv_KC_V=vmap(model.check_radius_conv_KC, in_axes = (0,None))
            
            lamC=check_radius_conv_KC_V(imag_mod,max_deg)
            radius_conv_check2 = jnp.all(imag_mod-lamC.reshape(len(imag_mod),1)>0)
            
            filter_conditions&=(radius_conv_check2==True)
        """
            
        
        # FILTER FLAG TO SPECIFY HOW WE CHECK THE FILTERS!
        filter_flag=""
        
        if len(model.generators_kahler_cone)!=model.NumMod:
            # IF WE DON'T HAVE THE GENERTORS; WE TEST HYPERPLANES OR KAHLER METRIC
            if len(model.hyperplanes)==0:
                filter_flag="naive"
                KMM=Kahler_Metric_V(mod,jnp.conj(mod),tau,jnp.conj(tau))
                CKMM=cholesky_V(KMM)
                
                filter_conditions &= (check_nan_V(KMM)==False)& (check_nan_V(CKMM)==False)
            else:
                filter_flag="hyperplane"
                imaginary_parts=jnp.imag(mod)
                hyperplane_dist=jnp.einsum("ij,kj",model.hyperplanes,imaginary_parts)
                hyperplane_check=jnp.all(hyperplane_dist>=0.,axis=0)
                
                filter_conditions &= (hyperplane_check==True)
            
        else:
            filter_flag="span"
            inv_KC_generator=jnp.linalg.inv(model.generators_kahler_cone)
            
            imaginary_parts=jnp.imag(mod)
            span_dist=jnp.einsum("ji,kj",inv_KC_generator,imaginary_parts)
            
            # We demand that the coefficients are all positive.
            span_check=jnp.all(span_dist>=0.,axis=0)
            
            filter_conditions &= (span_check==True)
            
            
        print("Using flag: ",filter_flag)

        SP_Hess = Scalar_Potential_Hess_V(mod, jnp.conj(mod), tau, jnp.conj(tau), fluxes)
        
        CSP=cholesky_V(SP_Hess)
        
        if mode=="SUSY":
            codims_mod_space=codim_vacuum_moduli_space_V(mod, jnp.conj(mod), tau, jnp.conj(tau), fluxes)
            # If verbose is turned on, we make an additional test by looking
            if verbose:
                rank_hessian_test=rank_matrix_V(SP_Hess)
                if np.max(np.abs(codims_mod_space-rank_hessian_test))>0:
                    print("Rank test on Hessian does not agree with rank test on vacuum moduli space!")
                    print("Vacuum moduli space: ",codims_mod_space)
                    print("Hessian: ",rank_hessian_test)
        else:
            codims_mod_space=codim_vacuum_moduli_space_V(SP_Hess)
            
        
        filter_conditions &= (check_nan_V(SP_Hess)==False)& (check_nan_V(CSP)==False)& (codims_mod_space==2*(model.NumMod+1))
        
        
        ind0=jnp.where(filter_conditions)[0]
        
        roots_ret=roots[ind0]
        flux_ret=fluxes[ind0]
        
        
        if flux_ids is not None:
            flux_ids_ret=jnp.array(flux_ids)[ind0]
        else:
            flux_ids_ret=None
        
        if uni_pts_ids is not None:
            uni_pts_ids_ret=jnp.array(uni_pts_ids)[ind0]
        else:
            uni_pts_ids_ret=None
        
        if verbose:
        
            print("Number of solutions passing...")
            ind1=jnp.where(jnp.imag(tau)>0)
            print("... Im(tau)>0: ",len(ind1[0]))
            ind2=jnp.where(Pi_D_Pi>0)
            print("... Pi^{dagger}*Pi>0: ",len(ind2[0]))
            ind3=jnp.where(jnp.abs(Finstb/Fpertb)< 1e-1)
            print("... |F_inst/F_pert|<1e-1: ",len(ind3[0]))
            
            if filter_flag=="naive":
                ind4=jnp.where(check_nan_V(KMM)==False)
                print("... Nan Kähler Metric: ",len(ind4[0]))
                ind5=jnp.where(check_nan_V(CKMM)==False)
                print("... Nan Kähler Cholesky: ",len(ind5[0]))
                
            ind6=jnp.where(check_nan_V(SP_Hess)==False)
            print("... Nan Hessian: ",len(ind6[0]))
            ind7=jnp.where(check_nan_V(CSP)==False)
            print("... Nan Hessian Cholesky: ",len(ind7[0]))
            ind8=jnp.where(codims_mod_space==2*(model.NumMod+1))
            print("... flat direction: ",len(ind8[0]))
            
            if filter_flag=="hyperplane":
                
                ind9 = jnp.where(hyperplane_check==True)
                print("... hyperplane test: ",len(ind9[0]))
                
            if filter_flag=="span":
            
                ind9 = jnp.where(span_check==True)
                print("... span test: ",len(ind9[0]))
            
            """
            if model.max_deg>0:
                ind10 = jnp.where(radius_conv_check1==True)
                print("... radius convergence asymptotic: ",len(ind10[0]))
                ind11 = jnp.where(radius_conv_check1==True)
                print("... radius convergence point: ",len(ind11[0]))
            """
        
        return jnp.array(roots_ret),jnp.array(flux_ret),flux_ids_ret,uni_pts_ids_ret
    

