import numpy as np
import quantum_energy.physics.one_particle as physics


def create_psi_matrix(x0, a, xi_1, xi_2):
  
    psi_pluss= physics.psi_func(xi_1, x0, a).reshape(-1, 1)
    psi_minus = physics.psi_func(xi_2, -x0, a).reshape(-1, 1)

    psi_total = psi_pluss @ psi_minus.T + psi_minus @ psi_pluss.T
    
    return psi_total

def create_w_matrix(xi_1, xi_2, w0):

    X1, X2 = np.meshgrid(xi_1, xi_2)

    W = w0/(np.sqrt((X1-X2)**2 + 1**2))

    return W

def create_H_matrix(fdm, xi, func = 'func1'):

    H = -1/2*(fdm) + (np.diagflat(physics.get_v_vector(xi, func)))
    return H

def create_phi_matrix(W, H, psi):

    return H @ psi + psi @ H + W * psi

def calculate_e(x0, a, xi_1, xi_2, W, H):

    psi = create_psi_matrix(x0, a, xi_1, xi_2)
    phi = create_phi_matrix(W, H, psi)

    return sum(sum(psi * phi))/(sum(sum(psi * psi)))

def partial_difference_quotient2(params, i, dx, xi_1, xi_2, H, W):
    """
    This function calculates the central partial difference quotient approximation with respect to the ith parameter.
    
    Argument:
    params -- List of the functions parameters
    i -- ith paramer
    dx -- step length
    
    Returns:
    d_e -- A scalar, the central partial difference quotient approximation.
    """
    
    x0_pluss, a_pluss = [param + (dx if j == i else 0) for j, param in enumerate(params)]
    x0_minus, a_minus = [param - (dx if j == i else 0) for j, param in enumerate(params)]
    

    d_e = (calculate_e(x0_pluss, a_pluss, xi_1, xi_2, W, H) - calculate_e(x0_minus, a_minus, xi_1, xi_2, W, H))/2*dx
    return d_e