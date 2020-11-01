import numpy as np

import quantum_energy.physics.one_particle as physics


def create_psi_matrix(x0, a, xi):
  
    psi_pluss= physics.psi_func(xi, x0, a).reshape(-1, 1)
    psi_minus = physics.psi_func(xi, -x0, a).reshape(-1, 1)

    psi_total = psi_pluss @ psi_minus.T + psi_minus @ psi_pluss.T
    
    return psi_total

def create_w_matrix(xi, w0):

    X1, X2 = np.meshgrid(xi, xi)

    W = w0/(np.sqrt((X1-X2)**2 + 1**2))

    return W

def create_H_matrix(fdm, xi, func = 'func1'):

    H = -1/2*(fdm) + (np.diagflat(physics.get_v_vector(xi, func)))
    return H

def create_phi_matrix(W, H, psi):

    return H @ psi + psi @ H + W * psi

def calculate_e(params, xi, W, H):
    x0, a = params

    psi = create_psi_matrix(x0, a, xi)
    phi = create_phi_matrix(W, H, psi)

    return sum(sum(psi * phi))/(sum(sum(psi * psi)))
