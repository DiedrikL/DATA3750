import numpy as np
import quantum_energy.physics.one_particle as physics


def create_psi_matrix(params, xi):
    """
    This function creates and returns a matrix populated with values from the wave funtion for two particles

    psi_func = np.exp(-a*(x-x0)**2)

    psi_matrix = psi_func(x; x0, a) * psi_func(x; -x0, a) + psi_func(x; x0, a) * psi_func(x; -x0, a)

    Arguments:
    x0 -- parameter in function
    a -- parameter in function
    xi -- list of x-values 

    Returns:
    psi_total --  The matrix of values for two particle wave function
    """
    psi_pluss= physics.psi_func(xi, params)
    psi_minus = physics.psi_func(xi, params)

    psi_total = psi_pluss @ psi_minus.T + psi_minus @ psi_pluss.T
    
    return psi_total

def create_w_matrix(xi, w0):
    """
    This function creates and returns the particle interaction matrix
    
    Arguments:
    xi -- list of x-values
    w0 -- interaction force

    
    Returns:
    W -- the particle interaction matrix
    """
    X1, X2 = np.meshgrid(xi, xi)

    W = w0/(np.sqrt((X1-X2)**2 + 1**2))

    return W

def create_H_matrix(fdm, xi, func = 'func1'):
    """
    This function creates and returns the Hamiltonian matrix 
    
    Arguments:
    fdm -- matrix representation of a second order central finite difference scheme
    xi -- list of x-values
    func -- function to differentiate

    Returns:
    H -- the Hamiltonian matrix
    """
    H = -1/2*(fdm) + (np.diagflat(physics.get_v_vector(xi, func)))
    return H

def create_phi_matrix(W, H, psi):
    """
    This function creates and returns the phi matrix
    
    Arguments:
    W -- the particle interaction matrix
    H -- the Hamiltonian matrix
    psi -- The matrix of values for two particle wave function
    
    
    Returns:
    phi -- the phi matrix
    """
    phi = H @ psi + psi @ H + W * psi

    return phi
def calculate_e(params, xi, W, H):
    """
    This function calculates the energy level for a wave function given by the parameters.
    
    Arguments:
    params -- list of parameters for the psi matrix
    xi -- list of x-values
    W -- the particle interaction matrix
    H -- the Hamiltonian matrix
    
    Returns:
    e -- the energy for the wave function calculated
    """
    #x0, a = params

    psi = create_psi_matrix(params, xi)
    phi = create_phi_matrix(W, H, psi)

    e = sum(sum(psi * phi))/(sum(sum(psi * psi)))

    return e
