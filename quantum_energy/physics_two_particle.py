import numpy as np
import physics


def create_psi_matrix(x1, x2, x0, a):

    psi_pluss= physics.psi_func(x1, x0, a).reshape(-1, 1)
    psi_minus = physics.psi_func(x2, -x0, a).reshape(-1, 1)

    psi_total = psi_pluss @ psi_minus.T + psi_minus @ psi_pluss.T
    
    return psi_total

def w_matrix(x1, x2):

    X1, X2 = np.meshgrid(x1, x2)

    W = 1/(np.sqrt((X1-X2)**2 + 1**2))

    return W

def create_phi( x1, x2, x0, a):
    L = 20 # Length of interval
    N = 4 # No. of subintervals
    h = L / N # Stepsize

    fdm = physics.create_2nd_order_finite_difference_scheme(N, h)

    H_1 = -1/2*(fdm) + (np.diagflat(physics.get_v_vector(x1, 'func1')))
    H_2 = -1/2*(fdm) + (np.diagflat(physics.get_v_vector(x2, 'func1')))
    psi = create_psi_matrix(x1, x2, x0, a)
    W = w_matrix(x1, x2)

    return H_1 @ psi + psi @ H_2 + W * psi