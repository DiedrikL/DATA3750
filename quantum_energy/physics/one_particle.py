import numpy as np
import matplotlib.pyplot as plt


def most_accurate_e(finite_difference_matrix, v_vector):
    H = -1/2*(finite_difference_matrix) + (np.diagflat(v_vector))
    E, u = np.linalg.eig(H)
    E_min = np.amin(E)
    index = np.where(E == E_min)[0][0]
    return E_min, u[:,index].reshape(-1, 1)

def get_v_vector(x, func, k = 1):
    if (func == 'func1'):
        return np.array(k*x**2/2).reshape(-1, 1)
    elif (func == 'func2'):
        return np.array(1 - np.exp(-((1/2)*k*x**2))).reshape(-1, 1)

def psi_func(x, *args):
    """Returns the test wave function"""
    if len(args) == 2:
        x0, a = args
        return np.exp(-a*(x-x0)**2)
    elif len(args) == 3:
        x0, a, b = args
        return np.exp(-abs(a)*(x-x0)**2/(np.sqrt(1 + abs(b)*x**2)))

def create_psi_vector(xi, params):
    return np.array(psi_func(xi, *params)).reshape(-1,1)

def create_2nd_order_finite_difference_scheme(N, h):
    """Returns a matrix representation of a second order central finite difference scheme"""
    m = np.zeros((N,N))
    for i in range(N):
        m[i,i] = -2
        if i+1 < N:
            m[i,i+1] = 1
        if i-1 >= 0:
            m[i, i-1] = 1
            
    m = 1/(h**2)*m
    return m

def compute_e(params, h, finite_difference_matrix, v_vector, xi):
    """
    Evaluate and returns the energy at the given point.

    Arguments:
    params -- list of wavefunction parameters
    h -- step size
    finite_difference_matrix -- matrix representation of a second order central finite difference scheme
    v_vector -- vector representation of the potential term in the hamiltonian
    xi -- list of x-values

    Returns:
    e -- scalar value of the energy
    """
    psi_vector = create_psi_vector(xi, params)

    h_psi = -1/2*(finite_difference_matrix @ psi_vector) + (v_vector * psi_vector)
     
    e = h*(psi_vector.T @ h_psi) / (h*(psi_vector.T @ psi_vector))
    
    return e[0][0]

