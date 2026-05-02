# ==============================================================================
# Utility functions for flux vector manipulations.
#
# All functions take an lcs_tree object as their geometry input, keeping
# signatures clean and consistent.
# ==============================================================================

import jax.numpy as jnp
from jax import Array
from typing import Tuple, Any
from jax import jit
from functools import partial

# ---------------------------------------------------------------------------
# PFV algebra
# ---------------------------------------------------------------------------

@partial(jit, static_argnums = ())
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

@partial(jit, static_argnums = ())
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

    a = self.lcs_tree.a_matrix
    b = self.lcs_tree.b_vector

    if "coniLCS" in self.periods.limit:
        #tmp = jnp.zeros(b.shape[0])
        #tmp = tmp.at[0].set(1.)
        if self.lcs_tree.conifold_basis:
            tmp = self.lcs_tree.conifold.conifold_curve0
        else:
            tmp = self.lcs_tree.conifold.conifold_curve
            
        b = b+tmp*self.lcs_tree.conifold.ncf/24

    f0 = M@b
    f2 = (a.T@M)
    f2 = jnp.append(jnp.ones(1)*f0,f2)
    f2 = jnp.append(f2,jnp.zeros(1))
    f = jnp.append(f2,M)

    h = jnp.append(jnp.zeros(1),K)
    h = jnp.append(h,jnp.zeros(K.shape[0]+1))

    return jnp.append(f,h)


@partial(jit, static_argnums = ())
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

    N = self.lcs_tree.intnums@M
    if "coniLCS" in self.periods.limit:
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
        zcf = -1/(2*jnp.pi*1j)*jnp.exp(2*jnp.pi/self.lcs_tree.conifold.ncf/M[0]*(phase_comb+radial))

        z0 = jnp.append(jnp.ones(1)*zcf,zbulk)
    else:
        p = jnp.linalg.inv(N)@K
        z0 = p*tau

    return z0



