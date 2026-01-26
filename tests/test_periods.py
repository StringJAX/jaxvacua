# Copyright 2024 Andreas Schachner
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys, os, warnings
import jax
#from jaxlib.xla_extension import ArrayImpl
from functools import partial
from scipy.optimize import root
from util import *

sys.path.append("./../")
import jaxvacua

class TestPeriodSector(TestCase):
    
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        
        h12 = 2

        cls.model = jaxvacua.periods(h12=h12,model_ID=1,model_type="KS",maximum_degree=5)
        cls.z = jnp.array(np.random.uniform(-1,1,h12+1)+1j*np.random.uniform(0,10,h12+1))
        cls.z = cls.z.at[0].set(1.+0.*1j)
        cls.cz = jnp.conj(cls.z)
        
        
        cls.z0 = jnp.zeros(h12+1)
        cls.z0 = cls.z0.at[0].set(1.+0.*1j)
        cls.cz0 = jnp.conj(cls.z0)
        
        
    print("TODO: Choose one specific value for which we know the answer (e.g. for one of the minima which relates fluxes by gauge_kin fct!")
    print("Test model attributes?")

    @chex.variants(with_jit=True, without_jit=True)
    def test_ISD(self):
        """
        Test ISD condition.
        """
        
        # ISD matrix symmetric and symplectiv
        # M = M.T
        # M.T@sigma@M=sigma
        
        # Inverse for ISD:
        # jnp.linalg.inv(M) = sigma.T@M@sigma

        M = self.variant(self.model.ISD_matrix)(self.z,self.cz)
        
        chex.assert_type(M, complex)
        chex.assert_shape(M, (2*(self.model.h12+1),2*(self.model.h12+1)))
        # Imaginary part vanishes
        self.assertAllClose(M.imag,0., rtol=1e-11, atol=1e-11)
        
        
        self.assertAllClose(M , M.T, rtol=1e-11, atol=1e-11)
        self.assertAllClose(jnp.matmul(M.T,jnp.matmul(self.model.sigma(),M)), self.model.sigma(), rtol=1e-11, atol=1e-11)
        self.assertAllClose(jnp.linalg.inv(M), jnp.matmul(self.model.sigma().T,jnp.matmul(M,self.model.sigma())), rtol=1e-11, atol=1e-11)

        dM = self.variant(self.model.dM)(self.z,self.cz)
        dM_c = self.variant(self.model.dM_c)(self.z,self.cz)

        chex.assert_type(dM, complex)
        chex.assert_shape(dM, (2*(self.model.h12+1),2*(self.model.h12+1),self.model.h12+1))
        chex.assert_type(dM_c, complex)
        chex.assert_shape(dM_c, (2*(self.model.h12+1),2*(self.model.h12+1),self.model.h12+1))

        self.assertAllClose(dM_c,jnp.conj(dM))

        
    @chex.variants(with_jit=True, without_jit=True)
    def test_gauge_kinetic_matrix(self):
        
        conj=False
        N = self.variant(lambda x,y: self.model.gauge_kinetic_matrix(x,y,conj=conj))(self.z,self.cz)
        N_periods = self.variant(lambda x,y: self.model.gauge_kinetic_matrix_periods(x,y,conj=conj))(self.z,self.cz)
        N_prepotential = self.variant(lambda x,y: self.model.gauge_kinetic_matrix_prepotential(x,y,conj=conj))(self.z,self.cz)

        chex.assert_type(N, complex)
        chex.assert_shape(N, (self.model.h12+1,self.model.h12+1))
        chex.assert_type(N_periods, complex)
        chex.assert_shape(N_periods, (self.model.h12+1,self.model.h12+1))
        chex.assert_type(N_prepotential, complex)
        chex.assert_shape(N_prepotential, (self.model.h12+1,self.model.h12+1))

        dN_X = self.variant(lambda x,y: self.model.dN(x,y,conj=conj))(self.z,self.cz)
        dN_cX = self.variant(lambda x,y: self.model.dN_c(x,y,conj=conj))(self.z,self.cz)

        chex.assert_type(dN_X, complex)
        chex.assert_shape(dN_X, (self.model.h12+1,self.model.h12+1,self.model.h12+1))
        chex.assert_type(dN_cX, complex)
        chex.assert_shape(dN_cX, (self.model.h12+1,self.model.h12+1,self.model.h12+1))

        
        conj=True
        N_c = self.variant(lambda x,y: self.model.gauge_kinetic_matrix(x,y,conj=conj))(self.z,self.cz)
        N_periods_c = self.variant(lambda x,y: self.model.gauge_kinetic_matrix_periods(x,y,conj=conj))(self.z,self.cz)
        N_prepotential_c = self.variant(lambda x,y: self.model.gauge_kinetic_matrix_prepotential(x,y,conj=conj))(self.z,self.cz)

        chex.assert_type(N_c, complex)
        chex.assert_shape(N_c, (self.model.h12+1,self.model.h12+1))
        chex.assert_type(N_periods_c, complex)
        chex.assert_shape(N_periods_c, (self.model.h12+1,self.model.h12+1))
        chex.assert_type(N_prepotential_c, complex)
        chex.assert_shape(N_prepotential_c, (self.model.h12+1,self.model.h12+1))

        dN_X_c = self.variant(lambda x,y: self.model.dN(x,y,conj=conj))(self.z,self.cz)
        dN_cX_c = self.variant(lambda x,y: self.model.dN_c(x,y,conj=conj))(self.z,self.cz)

        chex.assert_type(dN_X_c, complex)
        chex.assert_shape(dN_X_c, (self.model.h12+1,self.model.h12+1,self.model.h12+1))
        chex.assert_type(dN_cX_c, complex)
        chex.assert_shape(dN_cX_c, (self.model.h12+1,self.model.h12+1,self.model.h12+1))

        self.assertAllClose(N_c,jnp.conj(N))
        self.assertAllClose(dN_cX_c,jnp.conj(dN_X))
        self.assertAllClose(dN_X_c,jnp.conj(dN_cX))


        self.assertAllClose(N_periods, N_prepotential)
        self.assertAllClose(N_periods_c, N_prepotential_c)
        self.assertAllClose(N_periods_c,jnp.conj(N_periods))
        self.assertAllClose(N_prepotential_c,jnp.conj(N_prepotential))

        

        
       

    @chex.variants(with_jit=True, without_jit=True)
    def test_mirror_volume(self):
        """

        Test

        """
        Vtilde = self.variant(self.model.A_per)(self.z,self.cz)
        
        chex.assert_type(Vtilde, complex)
        chex.assert_shape(Vtilde, ())
        self.assertAllClose(Vtilde.imag,0.)

        

    @chex.variants(with_jit=True, without_jit=True)
    def test_kahler_potential(self):
        KP = self.variant(self.model.kahler_potential_per)(self.z,self.cz)
        
        chex.assert_type(KP, complex)
        chex.assert_shape(KP, ())
        self.assertAllClose(KP.imag,0.)

    @chex.variants(with_jit=True, without_jit=True)
    def test_dK(self):
        
        dK = self.variant(lambda x,y: self.model.grad_kahler_potential_per(x,y,conj=False))(self.z,self.cz)
        
        chex.assert_type(dK, complex)
        chex.assert_shape(dK, (self.model.h12+1,))

        dK_c = self.variant(lambda x,y: self.model.grad_kahler_potential_per(x,y,conj=True))(self.z,self.cz)
        
        chex.assert_type(dK_c, complex)
        chex.assert_shape(dK_c, (self.model.h12+1,))

        self.assertAllClose(dK_c,jnp.conj(dK))
        

    @chex.variants(with_jit=True, without_jit=True)
    def test_rest(self):

        conj = False
        P = self.variant(lambda x,y: self.model.P_per(x,y,conj=conj))(self.z,self.cz)
        Q = self.variant(lambda x,y: self.model.Q_per(x,y,conj=conj))(self.z,self.cz)
        Qinv = self.variant(lambda x,y: self.model.Q_inv_per(x,y,conj=conj))(self.z,self.cz)
        P_mod,Q_mod = self.variant(lambda x,y: self.model.PQ_per(x,y,conj=conj))(self.z,self.cz)


        conj = True
        P_c = self.variant(lambda x,y: self.model.P_per(x,y,conj=conj))(self.z,self.cz)
        Q_c = self.variant(lambda x,y: self.model.Q_per(x,y,conj=conj))(self.z,self.cz)
        Qinv_c = self.variant(lambda x,y: self.model.Q_inv_per(x,y,conj=conj))(self.z,self.cz)
        P_mod_c,Q_mod_c = self.variant(lambda x,y: self.model.PQ_per(x,y,conj=conj))(self.z,self.cz)

        
        chex.assert_shape(P, (self.model.h12+1,self.model.h12+1))
        chex.assert_shape(P_c, (self.model.h12+1,self.model.h12+1))
        chex.assert_shape(Q, (self.model.h12+1,self.model.h12+1))
        chex.assert_shape(Q_c, (self.model.h12+1,self.model.h12+1))
        chex.assert_shape(Qinv, (self.model.h12+1,self.model.h12+1))
        chex.assert_shape(Qinv_c, (self.model.h12+1,self.model.h12+1))
        chex.assert_shape(P_mod, (self.model.h12+1,self.model.h12+1))
        chex.assert_shape(P_mod_c, (self.model.h12+1,self.model.h12+1))
        chex.assert_shape(Q_mod, (self.model.h12+1,self.model.h12+1))
        chex.assert_shape(Q_mod_c, (self.model.h12+1,self.model.h12+1))

        self.assertAllClose(Qinv,jnp.linalg.inv(Q))
        self.assertAllClose(Qinv_c,jnp.linalg.inv(Q_c))

        self.assertAllClose(Qinv,jnp.conj(Qinv_c))
        self.assertAllClose(Q,jnp.conj(Q_c))
        self.assertAllClose(P,jnp.conj(P_c))
        self.assertAllClose(Q_mod,jnp.conj(Q_mod_c))
        self.assertAllClose(P_mod,jnp.conj(P_mod_c))
        self.assertAllClose(Q_mod,Q)
        self.assertAllClose(P_mod,P)
        

        

        

    @chex.variants(with_jit=True, without_jit=True)
    def test_prepotential(self):
        """
        Test prepotential.

        Aliases:
        'prepot'
        'F'

        """

        conj = False
        F = self.variant(lambda x: self.model.prepot_per(x,conj=conj))(self.z)
        F_LCS = self.variant(lambda x: self.model.F_LCS_per(x,conj=conj))(self.z)
        F_LCS_poly = self.variant(lambda x: self.model.F_LCS_poly_per(x,conj=conj))(self.z)
        F_inst = self.variant(lambda x: self.model.F_inst_per(x,conj=conj))(self.z)
        dF = self.variant(lambda x: self.model.prepot_grad_per(x,conj=conj))(self.z)
        ddF = self.variant(lambda x: self.model.prepot_grad_grad_per(x,conj=conj))(self.z)


        conj = True
        F_c = self.variant(lambda x: self.model.prepot_per(x,conj=conj))(self.cz)
        F_LCS_c = self.variant(lambda x: self.model.F_LCS_per(x,conj=conj))(self.cz)
        F_LCS_poly_c = self.variant(lambda x: self.model.F_LCS_poly_per(x,conj=conj))(self.cz)
        F_inst_c = self.variant(lambda x: self.model.F_inst_per(x,conj=conj))(self.cz)
        dF_c = self.variant(lambda x: self.model.prepot_grad_per(x,conj=conj))(self.cz)
        ddF_c = self.variant(lambda x: self.model.prepot_grad_grad_per(x,conj=conj))(self.cz)

        chex.assert_shape(dF, (self.model.h12+1,))
        chex.assert_shape(dF_c, (self.model.h12+1,))
        chex.assert_shape(ddF, (self.model.h12+1,self.model.h12+1))
        chex.assert_shape(ddF_c, (self.model.h12+1,self.model.h12+1))
        self.assertAllClose(F_c,jnp.conj(F))
        self.assertAllClose(F_LCS_c,jnp.conj(F_LCS))
        self.assertAllClose(F_LCS_poly_c,jnp.conj(F_LCS_poly))
        self.assertAllClose(F_inst_c,jnp.conj(F_inst))
        self.assertAllClose(dF_c,jnp.conj(dF))
        self.assertAllClose(ddF_c,jnp.conj(ddF))

        #At LCS, we can test that for the polynomial F, F(0)=xi, F_i(0)=b, F_ij(0)=a, F_ijk(0)=kappa!
        conj = False
        zzero = jnp.append(jnp.ones(1),jnp.zeros(self.z.shape[0]-1)*1j)
        F_0 = self.variant(lambda x: self.model.F_LCS_poly_per(x,conj=conj))(zzero)
        dF_0 = self.variant(lambda x: jax.grad(self.model.F_LCS_poly_per,holomorphic=True)(x,conj=conj))(zzero)
        ddF_0 = self.variant(lambda x: jax.jacfwd(jax.grad(self.model.F_LCS_poly_per,holomorphic=True),holomorphic=True)(x,conj=conj))(zzero)
        dddF_0 = self.variant(lambda x: jax.jacfwd(jax.jacfwd(jax.grad(self.model.F_LCS_poly_per,holomorphic=True),holomorphic=True),holomorphic=True)(x,conj=conj))(zzero)

        print("Should we test also the derivative along X0?")
        chex.assert_shape(F_0, ())
        self.assertAllClose(F_0, self.model.K0/2.)
        chex.assert_shape(dF_0, (self.model.h12+1,))
        self.assertAllClose(dF_0[1:], self.model.b_vector)
        chex.assert_shape(ddF_0, (self.model.h12+1,self.model.h12+1))
        self.assertAllClose(ddF_0[1:,1:], self.model.a_matrix)
        chex.assert_shape(dddF_0, (self.model.h12+1,self.model.h12+1,self.model.h12+1))
        self.assertAllClose(dddF_0[1:,1:,1:], -self.model.mirror_intersection_numbers)



    @chex.variants(with_jit=True, without_jit=True)
    def test_period_vector(self):
        """
        Test period vector.

        """

        conj = False
        Pi = self.variant(lambda x: self.model.period_vector_per(x,conj=conj))(self.z)
        dPi = self.variant(lambda x: self.model.grad_period_vector_per(x,conj=conj))(self.z)
        DPi = self.variant(lambda x,y: self.model.D_period_vector_per(x,y,conj=conj))(self.z,self.cz)

        conj = True
        Pi_c = self.variant(lambda x: self.model.period_vector_per(x,conj=conj))(self.cz)
        dPi_c = self.variant(lambda x: self.model.grad_period_vector_per(x,conj=conj))(self.cz)
        DPi_c = self.variant(lambda x,y: self.model.D_period_vector_per(x,y,conj=conj))(self.z,self.cz)
        

        chex.assert_shape(Pi, (2*(self.model.h12+1),))
        chex.assert_shape(Pi_c, (2*(self.model.h12+1),))
        self.assertAllClose(Pi_c,jnp.conj(Pi))

        print("Not sure if the tests below are correct!")
        chex.assert_shape(dPi, (2*(self.model.h12+1),self.model.h12+1))
        chex.assert_shape(dPi_c, (2*(self.model.h12+1),self.model.h12+1))
        self.assertAllClose(dPi_c,jnp.conj(dPi))

        chex.assert_shape(DPi, (2*(self.model.h12+1),self.model.h12+1))
        chex.assert_shape(DPi_c, (2*(self.model.h12+1),self.model.h12+1))
        self.assertAllClose(DPi_c,jnp.conj(DPi))





