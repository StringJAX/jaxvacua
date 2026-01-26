# ==============================================================================
# This code is written by Andreas Schachner.
#
# If any questions arise, please feel free to reach out to me (Andreas) either at
# andreas.schachner@gmx.net or at as3475@cornell.edu .
# ==============================================================================
#
# ------------------------------------------------------------------------------
# This file holds functions for ....
# ------------------------------------------------------------------------------

#Standard libraries
# Important standard libraries
import os, sys, warnings
import numpy as np
import itertools
import getopt
from functools import partial

# Import jax modules
import jax
import jax.numpy as jnp
from jax import jit
from jax import Array
from numpy.typing import ArrayLike


# To load pickle files
import pickle
import gzip

jax.config.update("jax_enable_x64", True)


class bounded_fluxes():
    
    def __init__(self,model,Nmax=None,sampler=None,sample=None,dil_min=None,sample_size=None,Nmax=None,target_region=None):
        
        modes = ["local","global"]
        
        self.model = model
        
        if Nmax is None:
            self.Nmax = model.D3_tadpole
        else:
            self.Nmax = Nmax
        
        if dil_min is None:
            self.dil_min = jnp.sqrt(3)/2
        else:
            self.dil_min = dil_min


        
        self.lambda_max,self.mu_min,self.mu_max,self.tilde_mu_min,self.tilde_mu_max = self.compute_evs_vmap(moduli)
        
        
        self.dil_max = lambda_max*Nmax
        
        
        
        print("TODO: What are the ratios of mu_min/mu_max and tilde_mu_min/tilde_mu_max?")

        print("TODO: Use voxalized box for generation h1 and h2?")
        
    
    @partial(jit, static_argnums = (0,))
    def compute_evs(self,moduli):
        
        N = self.model.gauge_kinetic_matrix(moduli,jnp.conj(moduli))
        
        I = N.imag
        evs = jnp.linalg.eigvals(-I)
        mu_min = jnp.min(evs.real)
        mu_max = jnp.max(evs.real)

        Ninv = jnp.linalg.inv(N)
        tilde_I = Ninv.imag
        tilde_evs = jnp.linalg.eigvals(tilde_I)
        tilde_mu_min = jnp.min(tilde_evs.real)
        tilde_mu_max = jnp.max(tilde_evs.real)
        
        M = self.model.ISD_matrix(moduli,jnp.conj(moduli))
        evs = jnp.linalg.eigvals(M)
        lambda_max = jnp.max(evs.real)
        
        return lambda_max,mu_min,mu_max,tilde_mu_min,tilde_mu_max


    def update_global(self,moduli,tau,flux):

        if self.lambda_max>self.lambda_max_gl:
            self.lambda_max_gl = self.lambda_max

        if self.mu_max>self.mu_max_gl:
            self.mu_max_gl = self.mu_max

        if self.tilde_mu_max>self.tilde_mu_max_gl:
            self.tilde_mu_max_gl  = self.tilde_mu_max

        if self.mu_min<self.mu_min_gl:
            self.mu_min_gl = self.mu_min
        
        if self.tilde_mu_min<self.tilde_mu_min_gl:
            self.tilde_mu_min_gl = self.tilde_mu_min

    def update_evs(self,moduli):
        
        lambda_max,mu_min,mu_max,tilde_mu_min,tilde_mu_max = self.compute_evs(moduli)

        self.lambda_max    = lambda_max
        self.mu_min        = mu_min
        self.mu_max        = mu_max
        self.tilde_mu_min  = tilde_mu_min
        self.tilde_mu_max  = tilde_mu_max

        self.update_global()

    def update_local(self,moduli,tau,flux):
        
        lambda_max,mu_min,mu_max,tilde_mu_min,tilde_mu_max = self.compute_evs(moduli)

        self.lambda_max    = lambda_max
        self.mu_min        = mu_min
        self.mu_max        = mu_max
        self.tilde_mu_min  = tilde_mu_min
        self.tilde_mu_max  = tilde_mu_max
        self.c0            = tau.real
        self.s             = tau.imag
        self.Nflux         = self.model.tadpole(flux)

        self.update_global()

        self.f,self.h = self.get_fh(flux)
        self.h1,self.h2 = self.get_flux_split(self.h)
        self.f1,self.f2 = self.get_flux_split(self.f)

        self.hnorm = self.compute_norm(self.h)
        self.fnorm = self.compute_norm(self.f)
        
        self.h1norm = self.compute_norm(self.h1)
        self.h2norm = self.compute_norm(self.h2)
        self.f1norm = self.compute_norm(self.f1)
        self.f2norm = self.compute_norm(self.f2)
        
        self.f2tilde = self.f2-self.c0*self.h2
        self.f2tilde_norm = self.compute_norm(self.f2tilde)
        
        self.f1tilde = self.f1-self.c0*self.h1
        self.f1tilde_norm = self.compute_norm(self.f1tilde)
        
    
    @partial(jit, static_argnums = (0,))
    def compute_evs_vmap_tmp(self,moduli):
        
        return vmap(self.compute_evs)

    @partial(jit, static_argnums = (0,))
    def compute_evs_vmap(self,moduli):
        
        return self.compute_evs_vmap_tmp(moduli)
    
        
        
    def __repr__(self) -> str:
        r"""
        **Description:**
        Returns a string representation of the flux bounding class.
        
        Returns:
            str: Description of the class.
        """
        return f"Flux bounding class."
    
    
    def get_subvector(self,fluxes):

        print("Can use private function instead!")
        
        h1 = fluxes[self.model.NumFluxes:][:self.model.dimension_H3].real
        h2 = fluxes[self.model.NumFluxes:][self.model.dimension_H3:].real
        f1 = fluxes[:self.model.NumFluxes][:self.model.dimension_H3].real
        f2 = fluxes[:self.model.NumFluxes][self.model.dimension_H3:].real
        
        return h1,h2,f1,f2
    
    
    
    def compute_bounding_box(self,):
        
        
        h1_box = jnp.sqrt(Nflux/(s*tilde_mu_min))
        h2_box = jnp.sqrt(Nflux/(s*mu_min))
        
        f1_box = lambda_max*Nflux*2/jnp.sqrt(3)
        f2_box = f1_box
        

    def bound_h1_local(self):
        bound1 = (self.s*self.tilde_mu_min*self.h1norm<=self.Nflux)
        return (bound1),"h1 local"
        
    def bound_h1_global(self):
        bound2 = (self.dil_min*self.tilde_mu_min_gl*self.h1norm<=self.Nmax)
        return (bound2),"h1 global"
    
    def bound_h2_local(self):
        bound1 = (self.s*self.mu_min*self.h2norm<=self.Nflux)
        return (bound1),"h2 local"
    
    def bound_h2_global(self):
        bound2 = (self.dil_min*self.mu_min_gl*self.h2norm<=self.Nmax)

        return (bound2),"h2 global"
    
    def bound_f1_local(self):
        bound1 = (self.tilde_mu_min*self.f1tilde_norm<=self.s*self.Nflux)
        bound2 = (self.tilde_mu_min*(self.f1norm+self.c0**2*self.h1norm)-2*jnp.abs(self.c0)*self.tilde_mu_max*jnp.sqrt(self.f1norm*self.h1norm)<=self.s*self.Nflux)
        bound5 = (self.tilde_mu_min*(self.f1norm+(self.c0**2+3/4)*self.h1norm)-2*jnp.abs(self.c0)*self.tilde_mu_max*jnp.sqrt(self.f1norm*self.h1norm)<=self.s*self.Nflux)
        bound6 = (self.tilde_mu_min*(self.f1tilde_norm+(self.c0**2+3/4)*self.h1norm)<=self.s*self.Nflux)
    
        return (bound1,bound2,bound5,bound6),"f1 local"

    def bound_f1_global(self):
        bound3 = (self.tilde_mu_min_gl*(self.f1norm+3/4*self.h1norm)-self.tilde_mu_max_gl*jnp.sqrt(self.f1norm*self.h1norm)<=self.dil_max*self.Nmax)
        bound4 = (self.tilde_mu_min_gl*(self.f1tilde_norm+3/4*self.h1norm)<=self.dil_max*self.Nmax)
    
        return (bound3,bound4),"f1 global"
    
    def bound_f2_local(self):
        
        bound1 = (self.mu_min*self.f2tilde_norm<=self.s*self.Nflux)
        bound2 = (self.mu_min*(self.f2norm+self.c0**2*self.h2norm)-2*jnp.abs(self.c0)*self.mu_max*jnp.sqrt(self.f2norm*self.h2norm)<=self.s*self.Nflux)
        bound5 = (self.mu_min*(self.f2norm+(self.c0**2+3/4)*self.h2norm)-self.mu_max*jnp.abs(self.c0)*jnp.sqrt(self.f2norm*self.h2norm)<=self.s*self.Nflux)
        bound6 = (self.mu_min*(self.f2tilde_norm+(self.c0**2+3/4)*self.h2norm)<=self.s*self.Nflux)

        return (bound1,bound2,bound5,bound6),"f2 local"
    
    def bound_f2_global(self):
        bound3 = (self.mu_min_gl*(self.f2norm+3/4*self.h2norm)-self.mu_max_gl*jnp.sqrt(self.f2norm*self.h2norm)<=self.dil_max*self.Nmax)
        bound4 = (self.mu_min_gl*(self.f2tilde_norm+3/4*self.h2norm)<=self.dil_max*self.Nmax)
    
        return (bound3,bound4),"f2 global"    
    
    
    
    def bound_s_local(self):
    
        bound1 = (self.s>=self.dil_min)
        bound2 = (self.s<=self.lambda_max*self.Nflux)
        bound3 = (self.s<=self.lambda_max*self.Nflux/self.hnorm+self.hnorm/4/self.lambda_max)

        return (bound1,bound2,bound3),"s local"
        
    def bound_s_global(self):
        
        bound1 = (self.s>=self.dil_min)
        bound2 = (self.s<=self.lambda_max_gl*self.Nmax)
        # Globally not defined?
        #bound3 = (self.s<=self.lambda_max*self.Nmax/self.hnorm+self.hnorm/4/self.lambda_max)
        
        #return (bound1,bound2,bound3),"s global"
        return (bound1,bound2),"s global"
        
    def bound_h_local(self):
        bound2 = (self.hnorm<=self.Nflux*self.lambda_max/self.s)
        return (bound2),"h local"
        
        
    def bound_h_global(self):
        bound1 = (self.hnorm<=self.Nmax*self.lambda_max_gl/self.dil_min)

        return (bound1),"h global"
    
    
    def bound_f_local(self):
        bound2 = (self.fnorm>=self.s*self.Nflux/self.lambda_max)
        # (3.23) in https://arxiv.org/pdf/2310.06040
        bound3 = (self.fnorm<=self.lambda_max**2*self.Nflux**2/self.hnorm*(1+self.c0**2/self.s**2)) 
        
        return (bound2,bound3),"f local"

    def bound_f_global(self):
        
        bound1 = (self.fnorm>=self.dil_min*self.Nmax/self.lambda_max_gl)
        bound3 = (self.fnorm<=self.lambda_max_gl**2*self.Nmax**2*4/3)
        # using Nflux here because this bound uses hnorm explicitly -> H is indeed known!
        bound4 = (self.fnorm<=self.lambda_max_gl**2*self.Nflux**2/self.hnorm+self.hnorm/4)

        return (bound1,bound3,bound4),"f global"

    
    def get_nflux(self,fluxes):
        Nflux = self.model.tadpole(fluxes).real
        return Nflux
    
    
    def get_fh(self,flux):
        return flux[:self.n_fluxes].real,flux[self.n_fluxes:].real

    def get_flux_split(self,flux_half):
        return flux_half[:self.dimension_H3].real,flux_half[:self.dimension_H3].real

    def compute_norm(self,flux):
        return flux@flux


    def check_bounds(self,moduli,tau,flux):
        r"""
        Checks all bounds (methods starting with 'bound_') using the given flux and tau.

        Returns:
            bool: True if all bounds are satisfied, False otherwise.
        """
        
        print("TODO: How do these eigenvalues behave under rescaling? \
            Based on that, can we say whether along a certain ray in Kähler cone we can satisfy all bounds?")
        self.update_local(moduli,tau,flux)
        
        results = []
        for attr_name in dir(self):
            if attr_name.startswith("bound_"):
                bound_func = getattr(self, attr_name)
                if callable(bound_func):
                    # Call the bound check method with flux and tau
                    result = bound_func(moduli,tau,flux)
                    results.append(result)

        return results




    """
    f,h = self.get_hf(flux)
    h1,h2 = self.get_flux_split(h)
    f1,f2 = self.get_flux_split(f)

    hnorm = self.compute_norm(h)
    fnorm = self.compute_norm(f)
    
    c0 = tau.real
    s = tau.imag
    
    h1norm = self.compute_norm(h1)
    h2norm = self.compute_norm(h2)
    f1norm = self.compute_norm(f1)
    f2norm = self.compute_norm(f2)
    
    f2tilde = f2-c0*h2
    f2tilde_norm = self.compute_norm(f2tilde)
    
    f1tilde = f1-c0*h1
    f1tilde_norm = self.compute_norm(f1tilde)
    """


    """
    hnorm = h@h
    fnorm = f@f
    
    c0 = tau.real
    s = tau.imag
    
    
    h1norm = h1@h1
    h2norm = h2@h2
    f1norm = f1@f1
    f2norm = f2@f2
    
    f2tilde = f2-c0*h2
    f2tilde_norm = (f2tilde@f2tilde)
    
    f1tilde = f1-c0*h1
    f1tilde_norm = (f1tilde@f1tilde)
    """




