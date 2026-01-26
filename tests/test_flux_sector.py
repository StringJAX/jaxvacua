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

class TestFluxSector(TestCase):
    
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        
        h12 = 2

        cls.model = jaxvacua.flux_sector(h12=h12,model_ID=1,model_type="KS",maximum_degree=5)
        cls.z = jnp.array(np.random.uniform(-1,1,h12)+1j*np.random.uniform(0,10,h12))
        cls.cz = jnp.conj(cls.z)
        cls.tau = np.random.uniform(1,2)+1j*np.random.uniform(0.1,0.5)
        cls.ctau = jnp.conj(cls.tau)
        cls.f = jnp.array(np.random.randint(-10,11,4*(h12+1))).astype(float)
        
        cls.x = jnp.array(np.append(np.append([cls.z.real],[cls.z.imag],axis=0).T.flatten(),[cls.tau.real,cls.tau.imag]))
        
        cls.tau_fd,cls.f_fd = cls.model.map_to_FD_tau(cls.tau,cls.f)
        
        cls.f_fd = jnp.array(cls.f_fd).astype(float)
        
        cls.ctau_fd = jnp.conj(cls.tau_fd)
        
        
        # Special solution
        cls.f_solution = jnp.array([7, 3, -24, 0, -16, 50,0, 3, -4, 0, 0, 0])
        u1sol = 2.74215479602462524879172086700112955631003945168828832743217138983767*1j
        u2sol = 2.05661613496943436323419976712599580262262253939859294519039244649420*1j
        tausol = 6.85540179778358427172610564536555609784128313762349971439377181031816*1j
        
        x0 = jnp.array([0.,u1sol.imag,0.,u2sol.imag,0.,tausol.imag])
        res = root(cls.model.fterms,x0=x0,args=(cls.f_solution,),method="hybr",jac=cls.model.fterms_jacobian)
        
        if not res.success==True:
            raise ValueError("Unable to find minimum using `scipy.optimize.root`!")
            
        x = res.x
        
        cls.tausol = x[4]+1j*x[5]
        
        cls.zsol = jnp.array([x[0]+1j*x[1],x[2]+1j*x[3]])
        cls.czsol = jnp.conj(cls.zsol)
        cls.ctausol = jnp.conj(cls.tausol)
        cls.solution = jnp.array(x)#jnp.array([cls.zsol[0].real,cls.zsol[0].imag,cls.zsol[1].real,cls.zsol[1].imag,jnp.real(cls.tausol),jnp.imag(cls.tausol)])
        
        
    print("Test model attributes?")
    
    print("ADD TEST MASS MATRIX!!!!!")

    @chex.variants(with_jit=True, without_jit=True)
    def test_kahler_metric(self):
        KM = self.variant(lambda x,y,z,u: self.model.kahler_metric(x,y,z,u,mode=None))(self.z,self.cz,self.tau,self.ctau)
        KM_bd = self.variant(lambda x,y,z,u: self.model.kahler_metric(x,y,z,u,mode="block diagonal"))(self.z,self.cz,self.tau,self.ctau)
        chex.assert_shape(KM, (self.model.h12+1,self.model.h12+1))
        chex.assert_shape(KM_bd, (self.model.h12+1,self.model.h12+1))
        self.assertAllClose(KM,KM_bd)
        
    @chex.variants(with_jit=True, without_jit=True)
    def test_W(self):
        """
        Test the superpotential calculation.
        """
        W = self.variant(lambda x,y,z: self.model.superpotential(x,y,z,conj=False))(self.z,self.tau,self.f)
        cW = self.variant(lambda x,y,z: self.model.superpotential(x,y,z,conj=True))(self.cz,self.ctau,self.f)
        
        chex.assert_type(W, complex)
        chex.assert_shape(W, ())
        chex.assert_equal(jnp.conj(W), cW)
        
        
    @chex.variants(with_jit=True, without_jit=True)
    def test_W_gradients(self):
        """Test the superpotential calculation."""
        
        W = self.variant(lambda x,y: self.model.superpotential(x,y,self.f,conj=False))
        cW = self.variant(lambda x,y: self.model.superpotential(x,y,self.f,conj=True))
        
        chex.assert_numerical_grads(f = W, f_args = (self.z,self.tau),order=1,atol=1e-10)
        chex.assert_numerical_grads(f = cW, f_args = (self.cz,self.ctau),order=1,atol=1e-10)
        
        chex.assert_numerical_grads(f = W, f_args = (self.z,self.tau),order=2,atol=1e-10)
        chex.assert_numerical_grads(f = cW, f_args = (self.cz,self.ctau),order=2,atol=1e-10)
        
    @chex.variants(with_jit=True, without_jit=True)
    def test_W_gauge_invariant(self):
        """
        Test the gauge invariant superpotential calculation.
        """
        W = self.variant(lambda x,y,z: self.model.superpotential_gauge_invariant(x,y,z,conj=False))(self.z,self.tau,self.f)
        cW = self.variant(lambda x,y,z: self.model.superpotential_gauge_invariant(x,y,z,conj=True))(self.cz,self.ctau,self.f)
        
        chex.assert_type(W, complex)
        chex.assert_shape(W, ())
        chex.assert_equal(jnp.conj(W), cW)
        
        W_fd = self.variant(lambda x,y,z: self.model.superpotential_gauge_invariant(x,y,z,conj=False))(self.z,self.tau_fd,self.f_fd)
        cW_fd = self.variant(lambda x,y,z: self.model.superpotential_gauge_invariant(x,y,z,conj=True))(self.cz,self.ctau_fd,self.f_fd)
        
        # Test gauge invariance
        self.assertAllClose(jnp.abs(W), jnp.abs(W_fd))
        self.assertAllClose(jnp.abs(cW), jnp.abs(cW_fd))
        
        
        Wsol = self.variant(lambda x,y,z: self.model.superpotential_gauge_invariant(x,y,z,conj=False))(self.zsol,self.tausol,self.f_solution)
        
        
        self.assertAllClose(Wsol, -2.037e-08, rtol=1e-11, atol=1e-11)
        
        
    @chex.variants(with_jit=True, without_jit=True)
    def test_dW(self):
        """
        Test the gradients of superpotential.
        """
        
        dWz = self.variant(lambda x,y,z: self.model.dW_z(x,y,z,conj=False))(self.z,self.tau,self.f)
        dWcz = self.variant(lambda x,y,z: self.model.dW_z(x,y,z,conj=True))(self.cz,self.ctau,self.f)
        
        chex.assert_shape(dWz, (self.model.h12,))
        chex.assert_shape(dWcz, (self.model.h12,))
        self.assertAllClose(jnp.conj(dWz), dWcz)
        
        
        dWtau = self.variant(lambda x,y,z: self.model.dW_tau(x,y,z,conj=False))(self.z,self.tau,self.f)
        dWctau = self.variant(lambda x,y,z: self.model.dW_tau(x,y,z,conj=True))(self.cz,self.ctau,self.f)
        
        chex.assert_type(dWtau, complex)
        chex.assert_type(dWctau, complex)
        chex.assert_shape(dWtau, ())
        chex.assert_shape(dWctau, ())
        chex.assert_equal(jnp.conj(dWtau), dWctau)
        
        
        dW = self.variant(lambda x,y,z: self.model.dW(x,y,z,conj=False))(self.z,self.tau,self.f)
        dWc = self.variant(lambda x,y,z: self.model.dW(x,y,z,conj=True))(self.cz,self.ctau,self.f)
        
        chex.assert_shape(dW, (self.model.h12+1,))
        chex.assert_shape(dWc, (self.model.h12+1,))
        self.assertAllClose(jnp.conj(dW), dWc)
        
        self.assertAllClose(dW, jnp.append(dWz,dWtau))
        self.assertAllClose(dWc, jnp.append(dWcz,dWctau))
        
        
        
        
        
    @chex.variants(with_jit=True, without_jit=True)
    def test_ddW(self):
        """
        Test the second derivatives of superpotential.
        """
        
        ddWzz = self.variant(lambda x,y,z: self.model.ddW_z_z(x,y,z,conj=False))(self.z,self.tau,self.f)
        ddWczcz = self.variant(lambda x,y,z: self.model.ddW_z_z(x,y,z,conj=True))(self.cz,self.ctau,self.f)
        
        ddWztau = self.variant(lambda x,y,z: self.model.ddW_z_tau(x,y,z,conj=False))(self.z,self.tau,self.f)
        ddWczctau = self.variant(lambda x,y,z: self.model.ddW_z_tau(x,y,z,conj=True))(self.cz,self.ctau,self.f)
        
        
        chex.assert_shape(ddWzz, (self.model.h12,self.model.h12))
        chex.assert_shape(ddWczcz, (self.model.h12,self.model.h12))
        self.assertAllClose(jnp.conj(ddWzz), ddWczcz)
        
        
        chex.assert_shape(ddWztau, (self.model.h12,))
        chex.assert_shape(ddWczctau, (self.model.h12,))
        self.assertAllClose(jnp.conj(ddWztau), ddWczctau)
        
        
    @chex.variants(with_jit=True, without_jit=True)
    def test_DW(self):
        """
        Test the F-terms.
        """
        args = (self.z,self.cz,self.tau,self.ctau,self.f)
        
        DWz = self.variant(lambda x,y,z,u,v: self.model.DW_z(x,y,z,u,v,conj=False))(*args)
        DWcz = self.variant(lambda x,y,z,u,v: self.model.DW_z(x,y,z,u,v,conj=True))(*args)
        
        chex.assert_shape(DWz, (self.model.h12,))
        chex.assert_shape(DWcz, (self.model.h12,))
        self.assertAllClose(jnp.conj(DWz), DWcz)
        
        
        DWtau = self.variant(lambda x,y,z,u,v: self.model.DW_tau(x,y,z,u,v,conj=False))(*args)
        DWctau = self.variant(lambda x,y,z,u,v: self.model.DW_tau(x,y,z,u,v,conj=True))(*args)
        
        chex.assert_type(DWtau, complex)
        chex.assert_type(DWctau, complex)
        chex.assert_shape(DWtau, ())
        chex.assert_shape(DWctau, ())
        chex.assert_equal(jnp.conj(DWtau), DWctau)
        
        
        DW = self.variant(lambda x,y,z,u,v: self.model.DW(x,y,z,u,v,conj=False))(*args)
        DWc = self.variant(lambda x,y,z,u,v: self.model.DW(x,y,z,u,v,conj=True))(*args)
        
        chex.assert_shape(DW, (self.model.h12+1,))
        chex.assert_shape(DWc, (self.model.h12+1,))
        self.assertAllClose(jnp.conj(DW), DWc)
        self.assertAllClose(DW, jnp.append(DWz,DWtau))
        self.assertAllClose(DWc, jnp.append(DWcz,DWctau))
        
        
        cDW1,cDW2 = self.variant(lambda x,y,z,u,v: self.model.fterms_canonical(x,y,z,u,v,conj=False))(*args)
        cDWc1,cDWc2 = self.variant(lambda x,y,z,u,v: self.model.fterms_canonical(x,y,z,u,v,conj=True))(*args)
        
        chex.assert_shape(cDW1, (self.model.h12+1,))
        chex.assert_shape(cDWc1, (self.model.h12+1,))
        chex.assert_shape(cDW2, (self.model.h12+1,))
        chex.assert_shape(cDWc2, (self.model.h12+1,))
        self.assertAllClose(jnp.conj(cDW1), cDWc1)
        self.assertAllClose(jnp.conj(cDW2), cDWc2)
        
        
        DW_real = self.variant(self.model.fterms)(self.x,self.f)
        DW_test = jnp.array(np.append([DW.real],[DW.imag],axis=0).T.flatten())
                            
        chex.assert_type(DW_real, float)
        chex.assert_shape(DW_real, (2*(self.model.h12+1),))
        self.assertAllClose(DW_real, DW_test)
        
        
        dDW_real = self.variant(self.model.fterms_jacobian)(self.x,self.f)
        
        chex.assert_type(dDW_real, float)
        chex.assert_shape(dDW_real, (2*(self.model.h12+1),2*(self.model.h12+1)))
        
        
        # Test F-term solution
        DW_sol = self.variant(self.model.fterms)(self.solution,self.f_solution)
        self.assertAllClose(DW_sol, jnp.zeros(2*(self.model.h12+1)))
        
    @chex.variants(with_jit=True, without_jit=True)
    def test_dDW(self):
        """
        Test the derivatives of the F-terms.
        """
        
        args = (self.z,self.cz,self.tau,self.ctau,self.f)
        
        conj = False
        dDW_tau_ctau = self.variant(lambda x,y,z,u,v: self.model.dDW_tau_ctau(x,y,z,u,v,conj=conj))(*args)
        dDW_tau_tau = self.variant(lambda x,y,z,u,v: self.model.dDW_tau_tau(x,y,z,u,v,conj=conj))(*args)
        
        chex.assert_type(dDW_tau_ctau, complex)
        chex.assert_shape(dDW_tau_ctau, ())
        chex.assert_type(dDW_tau_tau, complex)
        chex.assert_shape(dDW_tau_tau, ())
        
        dDW_tau_z = self.variant(lambda x,y,z,u,v: self.model.dDW_tau_z(x,y,z,u,v,conj=conj))(*args)
        dDW_tau_cz = self.variant(lambda x,y,z,u,v: self.model.dDW_tau_cz(x,y,z,u,v,conj=conj))(*args)
        
        chex.assert_type(dDW_tau_z, complex)
        chex.assert_shape(dDW_tau_z, (self.model.h12,))
        chex.assert_type(dDW_tau_cz, complex)
        chex.assert_shape(dDW_tau_cz, (self.model.h12,))
        
        dDW_z_tau = self.variant(lambda x,y,z,u,v: self.model.dDW_z_tau(x,y,z,u,v,conj=conj))(*args)
        dDW_z_ctau = self.variant(lambda x,y,z,u,v: self.model.dDW_z_ctau(x,y,z,u,v,conj=conj))(*args)
        
        chex.assert_type(dDW_z_tau, complex)
        chex.assert_shape(dDW_z_tau, (self.model.h12,))
        chex.assert_type(dDW_z_ctau, complex)
        chex.assert_shape(dDW_z_ctau, (self.model.h12,))
        
        dDW_z_cz = self.variant(lambda x,y,z,u,v: self.model.dDW_z_cz(x,y,z,u,v,conj=conj))(*args)
        dDW_z_z = self.variant(lambda x,y,z,u,v: self.model.dDW_z_z(x,y,z,u,v,conj=conj))(*args)
        
        chex.assert_type(dDW_z_cz, complex)
        chex.assert_shape(dDW_z_cz, (self.model.h12,self.model.h12))
        chex.assert_type(dDW_z_z, complex)
        chex.assert_shape(dDW_z_z, (self.model.h12,self.model.h12))
        
        dDW_z = self.variant(lambda x,y,z,u,v: self.model.dDW_z(x,y,z,u,v,conj=conj))(*args)
        dDW_cz = self.variant(lambda x,y,z,u,v: self.model.dDW_cz(x,y,z,u,v,conj=conj))(*args)
        
        chex.assert_type(dDW_z, complex)
        chex.assert_shape(dDW_z, (self.model.h12+1,self.model.h12))
        chex.assert_type(dDW_cz, complex)
        chex.assert_shape(dDW_cz, (self.model.h12+1,self.model.h12))
        
        dDW_tau = self.variant(lambda x,y,z,u,v: self.model.dDW_tau(x,y,z,u,v,conj=conj))(*args)
        dDW_ctau = self.variant(lambda x,y,z,u,v: self.model.dDW_ctau(x,y,z,u,v,conj=conj))(*args)
        
        chex.assert_type(dDW_tau, complex)
        chex.assert_shape(dDW_tau, (self.model.h12+1,))
        chex.assert_type(dDW_ctau, complex)
        chex.assert_shape(dDW_ctau, (self.model.h12+1,))
        
        dDW = self.variant(lambda x,y,z,u,v: self.model.dDW(x,y,z,u,v,conj=conj))(*args)
        dDW_c = self.variant(lambda x,y,z,u,v: self.model.dDW_c(x,y,z,u,v,conj=conj))(*args)
        
        chex.assert_type(dDW, complex)
        chex.assert_shape(dDW, (self.model.h12+1,self.model.h12+1))
        chex.assert_type(dDW_c, complex)
        chex.assert_shape(dDW_c, (self.model.h12+1,self.model.h12+1))
        
        ########################################################################################################
        ########################################################################################################
        
        conj = True
        dDW_ctau_tau = self.variant(lambda x,y,z,u,v: self.model.dDW_tau_ctau(x,y,z,u,v,conj=conj))(*args)
        dDW_ctau_ctau = self.variant(lambda x,y,z,u,v: self.model.dDW_tau_tau(x,y,z,u,v,conj=conj))(*args)
        
        chex.assert_type(dDW_ctau_tau, complex)
        chex.assert_shape(dDW_ctau_tau, ())
        chex.assert_type(dDW_ctau_ctau, complex)
        chex.assert_shape(dDW_ctau_ctau, ())
        
        dDW_ctau_cz = self.variant(lambda x,y,z,u,v: self.model.dDW_tau_z(x,y,z,u,v,conj=conj))(*args)
        dDW_ctau_z = self.variant(lambda x,y,z,u,v: self.model.dDW_tau_cz(x,y,z,u,v,conj=conj))(*args)
        
        chex.assert_type(dDW_ctau_cz, complex)
        chex.assert_shape(dDW_ctau_cz, (self.model.h12,))
        chex.assert_type(dDW_ctau_z, complex)
        chex.assert_shape(dDW_ctau_z, (self.model.h12,))
        
        dDW_cz_ctau = self.variant(lambda x,y,z,u,v: self.model.dDW_z_tau(x,y,z,u,v,conj=conj))(*args)
        dDW_cz_tau = self.variant(lambda x,y,z,u,v: self.model.dDW_z_ctau(x,y,z,u,v,conj=conj))(*args)
        
        chex.assert_type(dDW_cz_ctau, complex)
        chex.assert_shape(dDW_cz_ctau, (self.model.h12,))
        chex.assert_type(dDW_cz_tau, complex)
        chex.assert_shape(dDW_cz_tau, (self.model.h12,))
        
        dDW_cz_z = self.variant(lambda x,y,z,u,v: self.model.dDW_z_cz(x,y,z,u,v,conj=conj))(*args)
        dDW_cz_cz = self.variant(lambda x,y,z,u,v: self.model.dDW_z_z(x,y,z,u,v,conj=conj))(*args)
        
        chex.assert_type(dDW_cz_z, complex)
        chex.assert_shape(dDW_cz_z, (self.model.h12,self.model.h12))
        chex.assert_type(dDW_cz_cz, complex)
        chex.assert_shape(dDW_cz_cz, (self.model.h12,self.model.h12))
        
        dDW_cz_c = self.variant(lambda x,y,z,u,v: self.model.dDW_z(x,y,z,u,v,conj=conj))(*args)
        dDW_z_c = self.variant(lambda x,y,z,u,v: self.model.dDW_cz(x,y,z,u,v,conj=conj))(*args)
        
        chex.assert_type(dDW_cz_c, complex)
        chex.assert_shape(dDW_cz_c, (self.model.h12+1,self.model.h12))
        chex.assert_type(dDW_z_c, complex)
        chex.assert_shape(dDW_z_c, (self.model.h12+1,self.model.h12))
        
        dDW_ctau_c = self.variant(lambda x,y,z,u,v: self.model.dDW_tau(x,y,z,u,v,conj=conj))(*args)
        dDW_tau_c = self.variant(lambda x,y,z,u,v: self.model.dDW_ctau(x,y,z,u,v,conj=conj))(*args)
        
        chex.assert_type(dDW_ctau_c, complex)
        chex.assert_shape(dDW_ctau_c, (self.model.h12+1,))
        chex.assert_type(dDW_tau_c, complex)
        chex.assert_shape(dDW_tau_c, (self.model.h12+1,))
        
        dDW_cc = self.variant(lambda x,y,z,u,v: self.model.dDW(x,y,z,u,v,conj=conj))(*args)
        dDW_c_cc = self.variant(lambda x,y,z,u,v: self.model.dDW_c(x,y,z,u,v,conj=conj))(*args)
        
        chex.assert_type(dDW_cc, complex)
        chex.assert_shape(dDW_cc, (self.model.h12+1,self.model.h12+1))
        chex.assert_type(dDW_c_cc, complex)
        chex.assert_shape(dDW_c_cc, (self.model.h12+1,self.model.h12+1))
        
        ########################################################################################################
        ########################################################################################################
        
        self.assertAllClose(dDW_cz_z, jnp.conj(dDW_z_cz), rtol=1e-11, atol=1e-11)
        self.assertAllClose(dDW_cz_cz, jnp.conj(dDW_z_z), rtol=1e-11, atol=1e-11)
        self.assertAllClose(dDW_cz_tau, jnp.conj(dDW_z_ctau), rtol=1e-11, atol=1e-11)
        self.assertAllClose(dDW_cz_ctau, jnp.conj(dDW_z_tau), rtol=1e-11, atol=1e-11)
        self.assertAllClose(dDW_ctau_z, jnp.conj(dDW_tau_cz), rtol=1e-11, atol=1e-11)
        self.assertAllClose(dDW_ctau_cz, jnp.conj(dDW_tau_z), rtol=1e-11, atol=1e-11)
        self.assertAllClose(dDW_ctau_tau, jnp.conj(dDW_tau_ctau), rtol=1e-11, atol=1e-11)
        self.assertAllClose(dDW_ctau_ctau, jnp.conj(dDW_tau_tau), rtol=1e-11, atol=1e-11)
        
        print("ARE THESE CORRECT? IF SO; WHAT DOES IT MEAN?")
        self.assertAllClose(dDW_cz_c, jnp.conj(dDW_z), rtol=1e-11, atol=1e-11)
        self.assertAllClose(dDW_z_c, jnp.conj(dDW_cz), rtol=1e-11, atol=1e-11)
        self.assertAllClose(dDW_ctau_c, jnp.conj(dDW_tau), rtol=1e-11, atol=1e-11)
        self.assertAllClose(dDW_tau_c, jnp.conj(dDW_ctau), rtol=1e-11, atol=1e-11)
        self.assertAllClose(dDW_cc, jnp.conj(dDW), rtol=1e-11, atol=1e-11)
        self.assertAllClose(dDW_c_cc, jnp.conj(dDW_c), rtol=1e-11, atol=1e-11)
        
        
    @chex.variants(with_jit=True, without_jit=True)
    def test_DDW(self):
        """
        Test the Kähler covariant derivatives of the F-terms.
        """
        args = (self.z,self.cz,self.tau,self.ctau,self.f)
        
        conj = False
        DDW_tau_ctau = self.variant(lambda x,y,z,u,v: self.model.DDW_tau_ctau(x,y,z,u,v,conj=conj))(*args)
        DDW_tau_tau = self.variant(lambda x,y,z,u,v: self.model.DDW_tau_tau(x,y,z,u,v,conj=conj))(*args)
        
        chex.assert_type(DDW_tau_ctau, complex)
        chex.assert_shape(DDW_tau_ctau, ())
        chex.assert_type(DDW_tau_tau, complex)
        chex.assert_shape(DDW_tau_tau, ())
        
        DDW_tau_z = self.variant(lambda x,y,z,u,v: self.model.DDW_tau_z(x,y,z,u,v,conj=conj))(*args)
        DDW_tau_cz = self.variant(lambda x,y,z,u,v: self.model.DDW_tau_cz(x,y,z,u,v,conj=conj))(*args)
        
        chex.assert_type(DDW_tau_z, complex)
        chex.assert_shape(DDW_tau_z, (self.model.h12,))
        chex.assert_type(DDW_tau_cz, complex)
        chex.assert_shape(DDW_tau_cz, (self.model.h12,))
        
        DDW_z_tau = self.variant(lambda x,y,z,u,v: self.model.DDW_z_tau(x,y,z,u,v,conj=conj))(*args)
        DDW_z_ctau = self.variant(lambda x,y,z,u,v: self.model.DDW_z_ctau(x,y,z,u,v,conj=conj))(*args)
        
        chex.assert_type(DDW_z_tau, complex)
        chex.assert_shape(DDW_z_tau, (self.model.h12,))
        chex.assert_type(DDW_z_ctau, complex)
        chex.assert_shape(DDW_z_ctau, (self.model.h12,))
        
        DDW_z_cz = self.variant(lambda x,y,z,u,v: self.model.DDW_z_cz(x,y,z,u,v,conj=conj))(*args)
        DDW_z_z = self.variant(lambda x,y,z,u,v: self.model.DDW_z_z(x,y,z,u,v,conj=conj))(*args)
        
        chex.assert_type(DDW_z_cz, complex)
        chex.assert_shape(DDW_z_cz, (self.model.h12,self.model.h12))
        chex.assert_type(DDW_z_z, complex)
        chex.assert_shape(DDW_z_z, (self.model.h12,self.model.h12))
        
        DDW_gen = self.variant(lambda x,y,z,u,v: self.model.DDW_general(x,y,z,u,v,conj=conj))(*args)
        DDW_SUSY = self.variant(lambda x,y,z,u,v: self.model.DDW_SUSY(x,y,z,u,v,conj=conj))(*args)
        
        chex.assert_type(DDW_gen, complex)
        chex.assert_shape(DDW_gen, (self.model.h12+1,self.model.h12+1))
        chex.assert_type(DDW_SUSY, complex)
        chex.assert_shape(DDW_SUSY, (self.model.h12+1,self.model.h12+1))
        
        DDW_wrap_SUSY = self.variant(lambda x,y,z,u,v: self.model.DDW(x,y,z,u,v,conj=conj,mode="SUSY"))(*args)
        DDW_wrap_gen = self.variant(lambda x,y,z,u,v: self.model.DDW(x,y,z,u,v,conj=conj,mode=None))(*args)
        
        chex.assert_type(DDW_wrap_SUSY, complex)
        chex.assert_shape(DDW_wrap_SUSY, (self.model.h12+1,self.model.h12+1))
        chex.assert_type(DDW_wrap_gen, complex)
        chex.assert_shape(DDW_wrap_gen, (self.model.h12+1,self.model.h12+1))
        
        self.assertAllClose(DDW_wrap_SUSY, DDW_SUSY, rtol=1e-11, atol=1e-11)
        self.assertAllClose(DDW_wrap_gen, DDW_gen, rtol=1e-11, atol=1e-11)
        
        # Test that jnp.conj(DDW_z_cz) = DDW_z_cz.T
        # THIS IS NOT TRUE!
        #self.assertAllClose(jnp.conj(DDW_z_cz), DDW_z_cz.T, rtol=1e-11, atol=1e-11)
        
        
        # TO TEST DDW:    
        a = jnp.hstack((jnp.asarray(DDW_z_z), jnp.asarray(DDW_z_tau).reshape(self.model.h12, 1)))

        b = jnp.hstack((jnp.asarray(DDW_tau_z), jnp.asarray(DDW_tau_tau)))

        DDW_gen_comp = jnp.vstack((a, b))
        
        chex.assert_type(DDW_gen_comp, complex)
        chex.assert_shape(DDW_gen_comp, (self.model.h12+1,self.model.h12+1))
        
        self.assertAllClose(DDW_gen_comp, DDW_gen, rtol=1e-11, atol=1e-11)
        
        DcDW = self.variant(lambda x,y,z,u,v: self.model.DcDW_general(x,y,z,u,v,conj=conj))(*args)
        
        chex.assert_type(DcDW, complex)
        chex.assert_shape(DcDW, (self.model.h12+1,self.model.h12+1))
        
    
        a = jnp.hstack((jnp.asarray(DDW_z_cz), jnp.asarray(DDW_z_ctau).reshape(self.model.h12, 1)))

        b = jnp.hstack((jnp.asarray(DDW_tau_cz), jnp.asarray(DDW_tau_ctau)))

        DcDW_gen = jnp.vstack((a, b))
        
        chex.assert_type(DcDW_gen, complex)
        chex.assert_shape(DcDW_gen, (self.model.h12+1,self.model.h12+1))
        
        # Test naive (autodiff) vs. general computation
        self.assertAllClose(DcDW_gen, DcDW, rtol=1e-11, atol=1e-11)
        
        ########################################################################################################
        ########################################################################################################
        
        # CONJUGATE:
        conj = True
        
        DDW_ctau_tau = self.variant(lambda x,y,z,u,v: self.model.DDW_tau_ctau(x,y,z,u,v,conj=conj))(*args)
        DDW_ctau_ctau = self.variant(lambda x,y,z,u,v: self.model.DDW_tau_tau(x,y,z,u,v,conj=conj))(*args)
        
        chex.assert_type(DDW_ctau_tau, complex)
        chex.assert_shape(DDW_ctau_tau, ())
        chex.assert_type(DDW_ctau_ctau, complex)
        chex.assert_shape(DDW_ctau_ctau, ())
        
        DDW_ctau_cz = self.variant(lambda x,y,z,u,v: self.model.DDW_tau_z(x,y,z,u,v,conj=conj))(*args)
        DDW_ctau_z = self.variant(lambda x,y,z,u,v: self.model.DDW_tau_cz(x,y,z,u,v,conj=conj))(*args)
        
        chex.assert_type(DDW_ctau_cz, complex)
        chex.assert_shape(DDW_ctau_cz, (self.model.h12,))
        chex.assert_type(DDW_ctau_z, complex)
        chex.assert_shape(DDW_ctau_z, (self.model.h12,))
        
        DDW_cz_ctau = self.variant(lambda x,y,z,u,v: self.model.DDW_z_tau(x,y,z,u,v,conj=conj))(*args)
        DDW_cz_tau = self.variant(lambda x,y,z,u,v: self.model.DDW_z_ctau(x,y,z,u,v,conj=conj))(*args)
        
        chex.assert_type(DDW_cz_ctau, complex)
        chex.assert_shape(DDW_cz_ctau, (self.model.h12,))
        chex.assert_type(DDW_cz_tau, complex)
        chex.assert_shape(DDW_cz_tau, (self.model.h12,))
        
        DDW_cz_z = self.variant(lambda x,y,z,u,v: self.model.DDW_z_cz(x,y,z,u,v,conj=conj))(*args)
        DDW_cz_cz = self.variant(lambda x,y,z,u,v: self.model.DDW_z_z(x,y,z,u,v,conj=conj))(*args)
        
        chex.assert_type(DDW_cz_z, complex)
        chex.assert_shape(DDW_cz_z, (self.model.h12,self.model.h12))
        chex.assert_type(DDW_cz_cz, complex)
        chex.assert_shape(DDW_cz_cz, (self.model.h12,self.model.h12))
        
        DDW_gen_c = self.variant(lambda x,y,z,u,v: self.model.DDW_general(x,y,z,u,v,conj=conj))(*args)
        DDW_SUSY_c = self.variant(lambda x,y,z,u,v: self.model.DDW_SUSY(x,y,z,u,v,conj=conj))(*args)
        
        chex.assert_type(DDW_gen_c, complex)
        chex.assert_shape(DDW_gen_c, (self.model.h12+1,self.model.h12+1))
        chex.assert_type(DDW_SUSY_c, complex)
        chex.assert_shape(DDW_SUSY_c, (self.model.h12+1,self.model.h12+1))
        
        DDW_wrap_SUSY_c = self.variant(lambda x,y,z,u,v: self.model.DDW(x,y,z,u,v,conj=conj,mode="SUSY"))(*args)
        DDW_wrap_gen_c = self.variant(lambda x,y,z,u,v: self.model.DDW(x,y,z,u,v,conj=conj,mode=None))(*args)
        
        chex.assert_type(DDW_wrap_SUSY_c, complex)
        chex.assert_shape(DDW_wrap_SUSY_c, (self.model.h12+1,self.model.h12+1))
        chex.assert_type(DDW_wrap_gen_c, complex)
        chex.assert_shape(DDW_wrap_gen_c, (self.model.h12+1,self.model.h12+1))
        
        #print("DDW: ",(DDW_wrap_gen_c, DDW_gen_c))
        
        self.assertAllClose(DDW_wrap_SUSY_c, DDW_SUSY_c, rtol=1e-11, atol=1e-11)
        self.assertAllClose(DDW_wrap_gen_c, DDW_gen_c, rtol=1e-11, atol=1e-11)
        
    
        a = jnp.hstack((jnp.asarray(DDW_cz_cz), jnp.asarray(DDW_cz_ctau).reshape(self.model.h12, 1)))

        b = jnp.hstack((jnp.asarray(DDW_ctau_cz), jnp.asarray(DDW_ctau_ctau)))

        DDW_gen_comp_c = jnp.vstack((a, b))
        
        chex.assert_type(DDW_gen_comp_c, complex)
        chex.assert_shape(DDW_gen_comp_c, (self.model.h12+1,self.model.h12+1))
        
        self.assertAllClose(DDW_gen_comp_c, DDW_gen_c, rtol=1e-11, atol=1e-11)
        
        DcDW_c = self.variant(lambda x,y,z,u,v: self.model.DcDW_general(x,y,z,u,v,conj=conj))(*args)
        
        chex.assert_type(DcDW_c, complex)
        chex.assert_shape(DcDW_c, (self.model.h12+1,self.model.h12+1))
    
        a = jnp.hstack((jnp.asarray(DDW_cz_z), jnp.asarray(DDW_cz_tau).reshape(self.model.h12, 1)))

        b = jnp.hstack((jnp.asarray(DDW_ctau_z), jnp.asarray(DDW_ctau_tau)))

        DcDW_gen_c = jnp.vstack((a, b))
        
        chex.assert_type(DcDW_gen_c, complex)
        chex.assert_shape(DcDW_gen_c, (self.model.h12+1,self.model.h12+1))
        
        # Test naive (autodiff) vs. general computation
        self.assertAllClose(DcDW_gen_c, DcDW_c, rtol=1e-11, atol=1e-11)
        
        
        
        
        ########################################################################################################
        ########################################################################################################
        
        # Test conjugation
        self.assertAllClose(DDW_cz_z, jnp.conj(DDW_z_cz), rtol=1e-11, atol=1e-11)
        self.assertAllClose(DDW_cz_cz, jnp.conj(DDW_z_z), rtol=1e-11, atol=1e-11)
        self.assertAllClose(DDW_cz_tau, jnp.conj(DDW_z_ctau), rtol=1e-11, atol=1e-11)
        self.assertAllClose(DDW_cz_ctau, jnp.conj(DDW_z_tau), rtol=1e-11, atol=1e-11)
        self.assertAllClose(DDW_ctau_z, jnp.conj(DDW_tau_cz), rtol=1e-11, atol=1e-11)
        self.assertAllClose(DDW_ctau_cz, jnp.conj(DDW_tau_z), rtol=1e-11, atol=1e-11)
        self.assertAllClose(DDW_ctau_tau, jnp.conj(DDW_tau_ctau), rtol=1e-11, atol=1e-11)
        self.assertAllClose(DDW_ctau_ctau, jnp.conj(DDW_tau_tau), rtol=1e-11, atol=1e-11)
        # THIS IS NOT TRUE!
        #self.assertAllClose(DDW_cz_z, DDW_z_cz.T, rtol=1e-11, atol=1e-11)
        
        self.assertAllClose(DDW_SUSY_c, jnp.conj(DDW_SUSY), rtol=1e-11, atol=1e-11)
        self.assertAllClose(DDW_gen_c, jnp.conj(DDW_gen), rtol=1e-11, atol=1e-11)
        self.assertAllClose(DDW_wrap_SUSY_c, jnp.conj(DDW_wrap_SUSY), rtol=1e-11, atol=1e-11)
        self.assertAllClose(DDW_wrap_gen_c, jnp.conj(DDW_wrap_gen), rtol=1e-11, atol=1e-11)
        
        self.assertAllClose(DDW_gen_comp_c, jnp.conj(DDW_gen_comp), rtol=1e-11, atol=1e-11)
        self.assertAllClose(DDW_gen_c, jnp.conj(DDW_gen), rtol=1e-11, atol=1e-11)
        
        
        self.assertAllClose(DcDW_gen_c, jnp.conj(DcDW_gen), rtol=1e-11, atol=1e-11)
        self.assertAllClose(DcDW_c, jnp.conj(DcDW), rtol=1e-11, atol=1e-11)
        
        
        ########################################################################################################
        ########################################################################################################
        
        print("SUSY TESTS MISSING!")
        
        # Test SUSY version of model.DcDW_general???
        # DzDczW = K_zcz*W+K_cz*DzW
        #DDW_z_cz-model.kahler_metric(z,cz,tau,ctau)[:-1,:-1]*model.superpotential(z,tau,f)
        
        
    @chex.variants(with_jit=True, without_jit=True)
    def test_V(self):
        """
        Test the scalar potential.
        """
        
        args = (self.z,self.cz,self.tau,self.ctau,self.f)
        
        V = self.variant(self.model.scalar_potential)(*args)
        
        # Test output is scalar
        # Test output is real
        chex.assert_type(V, complex)
        chex.assert_shape(V, ())
        self.assertAllClose(V.imag, 0., rtol=1e-11, atol=1e-11)
        
        # From ISD matrix
        V_mod = self.variant(self.model.scalar_potential_mod)(*args)
        
        chex.assert_type(V_mod, complex)
        chex.assert_shape(V_mod, ())
        self.assertAllClose(V_mod.imag, 0., rtol=1e-11, atol=1e-11)
        
        V_real = self.variant(self.model.scalar_potential_real)(self.x,self.f)
        
        chex.assert_type(V_real, float)
        chex.assert_shape(V_real, ())
        
        dV_real = self.variant(self.model.gradV)(self.x,self.f)
        
        chex.assert_type(dV_real, float)
        chex.assert_shape(dV_real, (2*(self.model.h12+1),))
        
        ddV_real = self.variant(self.model.gradV_jacobian)(self.x,self.f)
        
        chex.assert_type(ddV_real, float)
        chex.assert_shape(ddV_real, (2*(self.model.h12+1),2*(self.model.h12+1)))
        
        
        hess = self.variant(self.model.scalar_potential_hessian)(*args)
        hess_SUSY = self.variant(self.model._scalar_potential_hessian_SUSY)(*args)
        hess_gen = self.variant(self.model._scalar_potential_hessian_general)(*args)
        #hess_SUGRA = 
        
        
        # Check that Hessian is Hermitian
        self.assertAllClose(hess, jnp.conj(hess.T), rtol=1e-11, atol=1e-11)
        self.assertAllClose(hess_SUSY, jnp.conj(hess_SUSY.T), rtol=1e-11, atol=1e-11)
        self.assertAllClose(hess_gen, jnp.conj(hess_gen.T), rtol=1e-11, atol=1e-11)
        #self.assertAllClose(hess_SUGRA, jnp.conj(hess_SUGRA.T), rtol=1e-11, atol=1e-11)
        
        
        eigvals = jnp.linalg.eigvals(ddV_real)
        eigvals_SUSY = jnp.linalg.eigvals(hess_SUSY)
        eigvals_gen = jnp.linalg.eigvals(hess_gen)
        
        #print("Eigvals SUSY: ",eigvals_SUSY)
        
        self.assertAllClose(eigvals.imag, 0., rtol=1e-11, atol=1e-11)
        self.assertAllClose(eigvals_SUSY.imag, 0., rtol=1e-11, atol=1e-11)
        self.assertAllClose(eigvals_gen.imag, 0., rtol=1e-11, atol=1e-11)
        
        eigvals_gen = jnp.sort(eigvals_gen.real)
        eigvals = jnp.sort(eigvals.real)
        
        # Test equivalence of eigenvalues
        self.assertAllClose(eigvals_gen*2,eigvals, rtol=1e-09, atol=1e-09)
        
        
        ########################################################################################################
        ########################################################################################################
        
        # Test V on solutions...
        args = (self.zsol,self.czsol,self.tausol,self.ctausol,self.f_solution)
        V = self.variant(self.model.scalar_potential)(*args)
        
        # Test output is scalar
        # Test output is real
        chex.assert_type(V, complex)
        chex.assert_shape(V, ())
        self.assertAllClose(V.imag, 0., rtol=1e-11, atol=1e-11)
        self.assertAllClose(V.real, 0., rtol=1e-11, atol=1e-11)
        
        # From ISD matrix
        V_mod = self.variant(self.model.scalar_potential_mod)(*args)
        
        chex.assert_type(V_mod, complex)
        chex.assert_shape(V_mod, ())
        self.assertAllClose(V_mod.imag, 0., rtol=1e-11, atol=1e-11)
        self.assertAllClose(V_mod.real, 0., rtol=1e-11, atol=1e-11)
        
        V_real = self.variant(self.model.scalar_potential_real)(self.solution,self.f_solution)
        
        chex.assert_type(V_real, float)
        chex.assert_shape(V_real, ())
        self.assertAllClose(V_real, 0., rtol=1e-11, atol=1e-11)
        
        dV_real = self.variant(self.model.gradV)(self.solution,self.f_solution)
        
        chex.assert_type(dV_real, float)
        chex.assert_shape(dV_real, (2*(self.model.h12+1),))
        self.assertAllClose(dV_real, 0., rtol=1e-08, atol=1e-08)
        
        ddV_real = self.variant(self.model.gradV_jacobian)(self.solution,self.f_solution)
        
        chex.assert_type(ddV_real, float)
        chex.assert_shape(ddV_real, (2*(self.model.h12+1),2*(self.model.h12+1)))
        
        # TEST HESSIAN????
        eigvals = jnp.linalg.eigvals(ddV_real)
        self.assertAllClose(eigvals.imag, 0., rtol=1e-11, atol=1e-11)
        self.assertAllClose(jnp.min(eigvals.real)/jnp.min(jnp.abs(eigvals.real)), 1., rtol=1e-11, atol=1e-11)
        self.assertAllClose(jnp.sign(eigvals.real), jnp.ones(len(eigvals)), rtol=1e-11, atol=1e-11)
        
        # Check that at SUSY minimum, the Hessian implementations agree
        hess_SUSY = self.variant(self.model._scalar_potential_hessian_SUSY)(*args)
        hess_gen = self.variant(self.model._scalar_potential_hessian_general)(*args)
        
        self.assertAllClose(hess_SUSY,hess_gen, rtol=1e-09, atol=1e-09)
        
        eigvals_SUSY = jnp.linalg.eigvals(hess_SUSY)
        eigvals_gen = jnp.linalg.eigvals(hess_gen)
        
        self.assertAllClose(eigvals.imag, 0., rtol=1e-11, atol=1e-11)
        self.assertAllClose(eigvals_SUSY.imag, 0., rtol=1e-11, atol=1e-11)
        self.assertAllClose(eigvals_gen.imag, 0., rtol=1e-11, atol=1e-11)
        
        #print("gen: ",eigvals_gen*2)
        #print("SUSY: ",eigvals_SUSY*2)
        #print("real: ",eigvals)
        
        eigvals_SUSY = jnp.sort(eigvals_SUSY.real)
        eigvals_gen = jnp.sort(eigvals_gen.real)
        eigvals = jnp.sort(eigvals.real)
        
        self.assertAllClose(eigvals_SUSY*2,eigvals, rtol=1e-09, atol=1e-09)
        self.assertAllClose(eigvals_SUSY,eigvals_gen, rtol=1e-09, atol=1e-09)
        self.assertAllClose(eigvals_gen*2,eigvals, rtol=1e-09, atol=1e-09)
        
        print("Add Tests for dV_real, ddV_real, ddV_real_real!")








        