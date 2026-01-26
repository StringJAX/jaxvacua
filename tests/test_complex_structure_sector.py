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

class TestCSSector(TestCase):
    
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        
        h12 = 2

        cls.model = jaxvacua.flux_sector(h12=h12,model_ID=1,model_type="KS",maximum_degree=5)
        cls.z = jnp.array(np.random.uniform(-1,1,h12)+1j*np.random.uniform(1.,10.,h12))
        cls.cz = jnp.conj(cls.z)
        cls.tau = np.random.uniform(1,2)+1j*np.random.uniform(0.1,0.5)
        cls.ctau = jnp.conj(cls.tau)
        cls.f = jnp.array(np.random.randint(-10,11,4*(h12+1))).astype(float)
        
        cls.x = jnp.array(np.append(np.append([cls.z.real],[cls.z.imag],axis=0).T.flatten(),[cls.tau.real,cls.tau.imag]))
        
        cls.tau_fd,cls.f_fd = cls.model.map_to_FD_tau(cls.tau,cls.f)
        
        cls.f_fd = jnp.array(cls.f_fd).astype(float)
        
        cls.ctau_fd = jnp.conj(cls.tau_fd)
        
        cls.sigma = cls.model.periods.sigma()
        
        
        # Special solution
        cls.f_solution = jnp.array([7, 3, -24, 0, -16, 50,0, 3, -4, 0, 0, 0])
        u1sol = 2.74215479602462524879172086700112955631003945168828832743217138983767*1j
        u2sol = 2.05661613496943436323419976712599580262262253939859294519039244649420*1j
        tausol = 6.85540179778358427172610564536555609784128313762349971439377181031816*1j
        
        x0 = jnp.array([0.,u1sol.imag,0.,u2sol.imag,0.,tausol.imag])
        res = root(cls.model.DW_x,x0=x0,args=(cls.f_solution,),method="hybr",jac=cls.model.dDW_x)
        
        if not res.success==True:
            raise ValueError("Unable to find minimum using `scipy.optimize.root`!")
            
        x = res.x
        
        cls.tausol = x[4]+1j*x[5]
        
        cls.zsol = jnp.array([x[0]+1j*x[1],x[2]+1j*x[3]])
        cls.czsol = jnp.conj(cls.zsol)
        cls.ctausol = jnp.conj(cls.tausol)
        cls.solution = jnp.array(x)#jnp.array([cls.zsol[0].real,cls.zsol[0].imag,cls.zsol[1].real,cls.zsol[1].imag,jnp.real(cls.tausol),jnp.imag(cls.tausol)])
        
         
    print("Test model attributes?")
    
    @chex.variants(with_jit=True, without_jit=True)
    def test_mirror_volume(self):
        """

        Aliases:
        'A',
        'V_tilde',
        'mirror_volume'

        """
        Vtilde = self.variant(self.model.A)(self.z,self.cz)
        
        chex.assert_type(Vtilde, complex)
        chex.assert_shape(Vtilde, ())
        self.assertAllClose(Vtilde.imag,0.)

        

    @chex.variants(with_jit=True, without_jit=True)
    def test_kahler_potential(self):
        KP = self.variant(self.model.kahler_potential)(self.z,self.cz,self.tau,self.ctau)
        
        chex.assert_type(KP, complex)
        chex.assert_shape(KP, ())
        self.assertAllClose(KP.imag,0.)

        
        

    @chex.variants(with_jit=True, without_jit=True)
    def test_dK_z(self):
        
        dK_z = self.variant(self.model.dK_z)(self.z,self.cz,self.tau,self.ctau)
        
        chex.assert_type(dK_z, complex)
        chex.assert_shape(dK_z, (self.model.h12,))

        dK_cz = self.variant(self.model.dK_cz)(self.z,self.cz,self.tau,self.ctau)
        
        chex.assert_type(dK_cz, complex)
        chex.assert_shape(dK_cz, (self.model.h12,))

        self.assertAllClose(dK_cz,jnp.conj(dK_z))

    @chex.variants(with_jit=True, without_jit=True)
    def test_dK_tau(self):

        dK_tau = self.variant(self.model.dK_tau)(self.z,self.cz,self.tau,self.ctau)
        
        chex.assert_type(dK_tau, complex)
        chex.assert_shape(dK_tau, ())

        dK_ctau = self.variant(self.model.dK_ctau)(self.z,self.cz,self.tau,self.ctau)
        
        chex.assert_type(dK_ctau, complex)
        chex.assert_shape(dK_ctau, ())

        self.assertAllClose(dK_ctau,jnp.conj(dK_tau))

    @chex.variants(with_jit=True, without_jit=True)
    def test_dK(self):
        
        dK = self.variant(self.model.dK)(self.z,self.cz,self.tau,self.ctau)
        
        chex.assert_type(dK, complex)
        chex.assert_shape(dK, (self.model.h12+1,))

        dK_c = self.variant(self.model.dK_c)(self.z,self.cz,self.tau,self.ctau)
        
        chex.assert_type(dK_c, complex)
        chex.assert_shape(dK_c, (self.model.h12+1,))

        self.assertAllClose(dK_c,jnp.conj(dK))

    @chex.variants(with_jit=True, without_jit=True)
    def test_ddK(self):
        
        ddK_z_cz = self.variant(self.model.ddK_z_cz)(self.z,self.cz,self.tau,self.ctau)
        
        chex.assert_type(ddK_z_cz, complex)
        chex.assert_shape(ddK_z_cz, (self.model.h12,self.model.h12))

        ddK_cz_z = self.variant(self.model.ddK_cz_z)(self.z,self.cz,self.tau,self.ctau)
        
        chex.assert_type(ddK_cz_z, complex)
        chex.assert_shape(ddK_cz_z, (self.model.h12,self.model.h12))

        self.assertAllClose(ddK_cz_z,jnp.conj(ddK_z_cz))

        ddK_z_z = self.variant(self.model.ddK_z_z)(self.z,self.cz,self.tau,self.ctau)
        
        chex.assert_type(ddK_z_z, complex)
        chex.assert_shape(ddK_z_z, (self.model.h12,self.model.h12))

        ddK_cz_cz = self.variant(self.model.ddK_cz_cz)(self.z,self.cz,self.tau,self.ctau)
        
        chex.assert_type(ddK_cz_cz, complex)
        chex.assert_shape(ddK_cz_cz, (self.model.h12,self.model.h12))

        self.assertAllClose(ddK_cz_cz,jnp.conj(ddK_z_z))


        ddK_z_ctau = self.variant(self.model.ddK_z_ctau)(self.z,self.cz,self.tau,self.ctau)
        
        chex.assert_type(ddK_z_ctau, complex)
        chex.assert_shape(ddK_z_ctau, (self.model.h12,))

        ddK_cz_tau = self.variant(self.model.ddK_cz_tau)(self.z,self.cz,self.tau,self.ctau)
        
        chex.assert_type(ddK_cz_tau, complex)
        chex.assert_shape(ddK_cz_tau, (self.model.h12,))

        self.assertAllClose(ddK_cz_tau,jnp.conj(ddK_z_ctau))

        ddK_z_tau = self.variant(self.model.ddK_z_tau)(self.z,self.cz,self.tau,self.ctau)
        
        chex.assert_type(ddK_z_tau, complex)
        chex.assert_shape(ddK_z_tau, (self.model.h12,))

        ddK_cz_ctau = self.variant(self.model.ddK_cz_ctau)(self.z,self.cz,self.tau,self.ctau)
        
        chex.assert_type(ddK_cz_ctau, complex)
        chex.assert_shape(ddK_cz_ctau, (self.model.h12,))

        self.assertAllClose(ddK_cz_ctau,jnp.conj(ddK_z_tau))


        ddK_tau_tau = self.variant(self.model.ddK_tau_tau)(self.z,self.cz,self.tau,self.ctau)
        
        chex.assert_type(ddK_tau_tau, complex)
        chex.assert_shape(ddK_tau_tau, ())

        ddK_ctau_ctau = self.variant(self.model.ddK_ctau_ctau)(self.z,self.cz,self.tau,self.ctau)
        
        chex.assert_type(ddK_ctau_ctau, complex)
        chex.assert_shape(ddK_ctau_ctau, ())

        self.assertAllClose(ddK_ctau_ctau,jnp.conj(ddK_tau_tau))

        ddK_tau_ctau = self.variant(self.model.ddK_tau_ctau)(self.z,self.cz,self.tau,self.ctau)
        
        chex.assert_type(ddK_tau_ctau, complex)
        chex.assert_shape(ddK_tau_ctau, ())

        self.assertAllClose(jnp.diag(ddK_z_cz).imag,0.)
        self.assertAllClose(ddK_z_cz,jnp.conj(ddK_z_cz.T))
        self.assertAllClose(ddK_z_ctau,jnp.conj(ddK_z_ctau.T))
        self.assertAllClose(ddK_tau_ctau,jnp.conj(ddK_tau_ctau))


        
    @chex.variants(with_jit=True, without_jit=True)
    def test_kahler_metric(self):
        KM = self.variant(lambda x,y,z,u: self.model.kahler_metric(x,y,z,u,mode=None))(self.z,self.cz,self.tau,self.ctau)
        KM_bd = self.variant(lambda x,y,z,u: self.model.kahler_metric(x,y,z,u,mode="block diagonal"))(self.z,self.cz,self.tau,self.ctau)
        chex.assert_shape(KM, (self.model.h12+1,self.model.h12+1))
        chex.assert_shape(KM_bd, (self.model.h12+1,self.model.h12+1))
        self.assertAllClose(KM,KM_bd)

        self.assertAllClose(jnp.diag(KM).imag,0.)
        self.assertAllClose(KM,jnp.conj(KM.T))

        eigvals = jnp.linalg.eigvals(KM)
        # Real eigenvalues
        self.assertAllClose(eigvals.imag, 0., rtol=1e-11, atol=1e-11)
        # Positive eigenvalues
        self.assertAllClose(jnp.min(eigvals.real)/jnp.min(jnp.abs(eigvals.real)), 1., rtol=1e-11, atol=1e-11)
        self.assertAllClose(jnp.sign(eigvals.real), jnp.ones(len(eigvals)), rtol=1e-11, atol=1e-11)
        
        IKM = self.variant(lambda x,y,z,u: self.model.inverse_kahler_metric(x,y,z,u,mode=None))(self.z,self.cz,self.tau,self.ctau)

        chex.assert_shape(IKM, (self.model.h12+1,self.model.h12+1))

        self.assertAllClose(jnp.diag(IKM).imag,0.)
        self.assertAllClose(IKM,jnp.conj(IKM.T))
        self.assertAllClose(IKM,jnp.linalg.inv(KM))

        dIKM = self.variant(lambda x,y,z,u: self.model.inverse_kahler_metric_grad(x,y,z,u,mode=None))(self.z,self.cz,self.tau,self.ctau)

        chex.assert_shape(dIKM, (self.model.h12+1,self.model.h12+1,self.model.h12))

        
        
        
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
        
        print("Can be copied to periods!")

        M = self.variant(self.model.ISD_matrix)(self.z,self.cz)
        
        chex.assert_type(M, complex)
        chex.assert_shape(M, (2*(self.model.h12+1),2*(self.model.h12+1)))
        # Imaginary part vanishes
        self.assertAllClose(M.imag,0., rtol=1e-11, atol=1e-11)
        
        
        self.assertAllClose(M , M.T, rtol=1e-11, atol=1e-11)
        self.assertAllClose(jnp.matmul(M.T,jnp.matmul(self.sigma,M)), self.sigma, rtol=1e-11, atol=1e-11)
        self.assertAllClose(jnp.linalg.inv(M), jnp.matmul(self.sigma.T,jnp.matmul(M,self.sigma)), rtol=1e-11, atol=1e-11)

        dM = self.variant(self.model.dM_X)(self.z,self.cz)
        dM_c = self.variant(self.model.dM_cX)(self.z,self.cz)

        chex.assert_type(dM, complex)
        chex.assert_shape(dM, (2*(self.model.h12+1),2*(self.model.h12+1),self.model.h12+1))
        chex.assert_type(dM_c, complex)
        chex.assert_shape(dM_c, (2*(self.model.h12+1),2*(self.model.h12+1),self.model.h12+1))

        self.assertAllClose(dM_c,jnp.conj(dM))

        dM = self.variant(self.model.dM)(self.z,self.cz)
        dM_c = self.variant(self.model.dM_c)(self.z,self.cz)

        chex.assert_type(dM, complex)
        chex.assert_shape(dM, (2*(self.model.h12+1),2*(self.model.h12+1),self.model.h12))
        chex.assert_type(dM_c, complex)
        chex.assert_shape(dM_c, (2*(self.model.h12+1),2*(self.model.h12+1),self.model.h12))

        self.assertAllClose(dM_c,jnp.conj(dM))
        
        


    
    @chex.variants(with_jit=True, without_jit=True)
    def test_gauge_kinetic_matrix(self):


        conj=False
        N = self.variant(lambda x,y: self.model.gauge_kinetic_matrix(x,y,conj=conj))(self.z,self.cz)

        chex.assert_type(N, complex)
        chex.assert_shape(N, (self.model.h12+1,self.model.h12+1))

        dN_X = self.variant(lambda x,y: self.model.dN_X(x,y,conj=conj))(self.z,self.cz)
        dN_cX = self.variant(lambda x,y: self.model.dN_cX(x,y,conj=conj))(self.z,self.cz)

        chex.assert_type(dN_X, complex)
        chex.assert_shape(dN_X, (self.model.h12+1,self.model.h12+1,self.model.h12+1))
        chex.assert_type(dN_cX, complex)
        chex.assert_shape(dN_cX, (self.model.h12+1,self.model.h12+1,self.model.h12+1))

        

        dN_z = self.variant(lambda x,y: self.model.dN(x,y,conj=conj))(self.z,self.cz)
        dN_cz = self.variant(lambda x,y: self.model.dN_c(x,y,conj=conj))(self.z,self.cz)

        chex.assert_type(dN_z, complex)
        chex.assert_shape(dN_z, (self.model.h12+1,self.model.h12+1,self.model.h12))
        chex.assert_type(dN_cz, complex)
        chex.assert_shape(dN_cz, (self.model.h12+1,self.model.h12+1,self.model.h12))

        conj=True
        N_c = self.variant(lambda x,y: self.model.gauge_kinetic_matrix(x,y,conj=conj))(self.z,self.cz)

        chex.assert_type(N_c, complex)
        chex.assert_shape(N_c, (self.model.h12+1,self.model.h12+1))

        dN_X_c = self.variant(lambda x,y: self.model.dN_X(x,y,conj=conj))(self.z,self.cz)
        dN_cX_c = self.variant(lambda x,y: self.model.dN_cX(x,y,conj=conj))(self.z,self.cz)

        chex.assert_type(dN_X_c, complex)
        chex.assert_shape(dN_X_c, (self.model.h12+1,self.model.h12+1,self.model.h12+1))
        chex.assert_type(dN_cX_c, complex)
        chex.assert_shape(dN_cX_c, (self.model.h12+1,self.model.h12+1,self.model.h12+1))

        

        dN_z_c = self.variant(lambda x,y: self.model.dN(x,y,conj=conj))(self.z,self.cz)
        dN_cz_c = self.variant(lambda x,y: self.model.dN_c(x,y,conj=conj))(self.z,self.cz)

        chex.assert_type(dN_z_c, complex)
        chex.assert_shape(dN_z_c, (self.model.h12+1,self.model.h12+1,self.model.h12))
        chex.assert_type(dN_cz_c, complex)
        chex.assert_shape(dN_cz_c, (self.model.h12+1,self.model.h12+1,self.model.h12))


        self.assertAllClose(N_c,jnp.conj(N))

        self.assertAllClose(dN_cX_c,jnp.conj(dN_X))
        self.assertAllClose(dN_cz_c,jnp.conj(dN_z))

        self.assertAllClose(dN_X_c,jnp.conj(dN_cX))
        self.assertAllClose(dN_z_c,jnp.conj(dN_cz))


    
    @chex.variants(with_jit=True, without_jit=True)
    def test_prepotential(self):
        """
        Test prepotential.

        Aliases:
        'prepot'
        'F'

        """
        
        conj = False
        F = self.variant(lambda x: self.model.F(x,conj=conj))(self.z)
        F_LCS = self.variant(lambda x: self.model.F_LCS(x,conj=conj))(self.z)
        F_LCS_poly = self.variant(lambda x: self.model.F_LCS_poly(x,conj=conj))(self.z)
        F_inst = self.variant(lambda x: self.model.F_inst(x,conj=conj))(self.z)
        dF = self.variant(lambda x: self.model.dF(x,conj=conj))(self.z)


        conj = True
        F_c = self.variant(lambda x: self.model.F(x,conj=conj))(self.cz)
        F_LCS_c = self.variant(lambda x: self.model.F_LCS(x,conj=conj))(self.cz)
        F_LCS_poly_c = self.variant(lambda x: self.model.F_LCS_poly(x,conj=conj))(self.cz)
        F_inst_c = self.variant(lambda x: self.model.F_inst(x,conj=conj))(self.cz)
        dF_c = self.variant(lambda x: self.model.dF(x,conj=conj))(self.cz)

        chex.assert_shape(dF, (self.model.h12,))
        chex.assert_shape(dF_c, (self.model.h12,))
        self.assertAllClose(F_c,jnp.conj(F))
        self.assertAllClose(F_LCS_c,jnp.conj(F_LCS))
        self.assertAllClose(F_LCS_poly_c,jnp.conj(F_LCS_poly))
        self.assertAllClose(F_inst_c,jnp.conj(F_inst))
        self.assertAllClose(dF_c,jnp.conj(dF))

        #At LCS, we can test that for the polynomial F, F(0)=xi, F_i(0)=b, F_ij(0)=a, F_ijk(0)=kappa!
        conj = False
        zzero = jnp.zeros(self.z.shape[0])*1j
        F_0 = self.variant(lambda x: self.model.F_LCS_poly(x,conj=conj))(zzero)
        dF_0 = self.variant(lambda x: jax.grad(self.model.F_LCS_poly,holomorphic=True)(x,conj=conj))(zzero)
        ddF_0 = self.variant(lambda x: jax.jacfwd(jax.grad(self.model.F_LCS_poly,holomorphic=True),holomorphic=True)(x,conj=conj))(zzero)
        dddF_0 = self.variant(lambda x: jax.jacfwd(jax.jacfwd(jax.grad(self.model.F_LCS_poly,holomorphic=True),holomorphic=True),holomorphic=True)(x,conj=conj))(zzero)

        chex.assert_shape(F_0, ())
        self.assertAllClose(F_0, self.model.periods.K0/2.)
        chex.assert_shape(dF_0, (self.model.h12,))
        self.assertAllClose(dF_0, self.model.periods.b_vector)
        chex.assert_shape(ddF_0, (self.model.h12,self.model.h12))
        self.assertAllClose(ddF_0, self.model.periods.a_matrix)
        chex.assert_shape(dddF_0, (self.model.h12,self.model.h12,self.model.h12))
        self.assertAllClose(dddF_0, -self.model.periods.mirror_intersection_numbers)

    @chex.variants(with_jit=True, without_jit=True)
    def test_period_vector(self):
        """
        Test period vector.

        """
        
        conj = False
        Pi = self.variant(lambda x: self.model.period_vector(x,conj=conj))(self.z)

        conj = True
        Pi_c = self.variant(lambda x: self.model.period_vector(x,conj=conj))(self.cz)
        

        chex.assert_shape(Pi, (2*(self.model.h12+1),))
        chex.assert_shape(Pi_c, (2*(self.model.h12+1),))
        self.assertAllClose(Pi_c,jnp.conj(Pi))

        
        
        
        
        

