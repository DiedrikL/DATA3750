import numpy as np
import matplotlib.pyplot as plt


def most_accurate_e(finite_difference_matrix, v_vector):
    """
    Solves the time independent schrödinger equation as an eigeinvalue equation to find
    the "true" (within our numerics) ground state and wavefunction.

    Arguments:
    finite_difference_matrix -- matrix representation of a second order central finite difference scheme
    v_vector -- vector representing the potential term in the hamiltonian

    Returns:
    E_min -- energy eigenvalue
    eigenfunction -- eigenfunction/wavefunction corresponding to the eigenvalue E_min
    """
    H = -1/2*(finite_difference_matrix) + (np.diagflat(v_vector))
    E, u = np.linalg.eig(H)
    E_min = np.amin(E)
    index = np.where(E == E_min)[0][0]
    eigenfunction = u[:,index].reshape(-1, 1)
    return E_min, eigenfunction

def get_v_vector(x, func, k = 1):
    """
    Returns the term for potential energy in the hamiltonian.

    Arguments:
    x -- list of x-values
    func -- string representing one of two functions (func1/func2)
    k -- integer (default = 1) 
    """

    if (func == 'func1'):
        return np.array(k*x**2/2).reshape(-1, 1)
    elif (func == 'func2'):
        return np.array(1 - np.exp(-((1/2)*k*x**2))).reshape(-1, 1)

def psi_func(x, params):
    """
    Returns the test wave function

    Arguments:
    x -- list of x-values
    params -- list of parameters, x0, a and eventually b
    
    """
    if len(params) == 2:
        x0, a = params
        return np.exp(-a*(x-x0)**2).reshape(-1, 1)
    elif len(params) == 3:
        x0, a, b = params
        return np.exp(-abs(a)*(x-x0)**2/(np.sqrt(1 + abs(b)*x**2))).reshape(-1, 1)

def create_2nd_order_finite_difference_scheme(N, h):
    """
    Returns a matrix representation of a second order central finite difference scheme.

    Arguments:
    N -- number of subintervals
    h -- step length

    Returns:
    m -- matrix representation of a second order central finite difference scheme
    """
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
    psi_vector = psi_func(xi, params)

    h_psi = -1/2*(finite_difference_matrix @ psi_vector) + (v_vector * psi_vector)
     
    e = h*(psi_vector.T @ h_psi) / (h*(psi_vector.T @ psi_vector))
    
    return e[0][0]

