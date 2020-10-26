import unittest
import numpy as np
import physics_two_particle
import physics

class TwoParticleSystemTest(unittest.TestCase):

    xi_2 = np.linspace(-2,2,4)
    xi_1 = np.linspace(-3,3,4)
    x0 = 0
    a = 1

    def test1(self):
        print('hei')
       

        m = physics_two_particle.create_psi_matrix(self.__class__.xi_1, self.__class__.xi_2, self.__class__.x0, self.__class__.a)

        print(m)

        np.testing.assert_almost_equal(m, m.T)

    def test2(self):

        W = physics_two_particle.w_matrix(self.__class__.xi_1, self.__class__.xi_2) 

        print(W.shape)
        print(W)

    def test3(self):

        m = physics_two_particle.create_phi(self.__class__.xi_1, self.__class__.xi_2, self.__class__.x0, self.__class__.a)
        print(m.shape)
        print(m)

if __name__ == '__main__':
    unittest.main()