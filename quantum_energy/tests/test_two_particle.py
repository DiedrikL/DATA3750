import unittest
import numpy as np
import physics_two_particle
import physics

class TwoParticleSystemTest(unittest.TestCase):

    xi_2 = np.linspace(-10,10,500)
    xi_1 = np.linspace(-10,10,500)
    w0 = 0
    x0 = 0
    a = 0.7071

    def test1(self):
        print('hei')

        psi = physics_two_particle.create_psi_matrix(self.__class__.x0, self.__class__.a, self.__class__.xi_1, self.__class__.xi_2)

        #print(psi)

        np.testing.assert_almost_equal(psi, psi.T)

    def test2(self):

        W = physics_two_particle.create_w_matrix(self.__class__.xi_1, self.__class__.xi_2, self.__class__.w0) 

        print(W.shape)
        #print(W)

    def test3(self):

        phi = physics_two_particle.create_phi_matrix(self.__class__.x0, self.__class__.a, self.__class__.xi_1, self.__class__.xi_2, self.__class__.w0)
        print(phi.shape)
        #print(phi)

    def test4(self):

        print(physics_two_particle.calculate_e(self.__class__.x0, self.__class__.a, self.__class__.xi_1, self.__class__.xi_2, self.__class__.w0))

if __name__ == '__main__':
    unittest.main()