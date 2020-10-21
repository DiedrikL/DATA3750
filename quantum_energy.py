#!/usr/bin/env python
# coding: utf-8

# The Energy of a Quantum Physical Two-Body System

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import argparse
from plot_results import plot_gradient_descent, plot_wave_functions
#get_ipython().run_line_magic('matplotlib', 'notebook')

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
        return np.exp(-a*(x-x0)**2-b*(x-x0)**4)

def create_psi_vector(params):
    return np.array(psi_func(xi, *params)).reshape(-1,1)
    
def most_accurate_e():
    H = -1/2*(finite_difference_matrix) + (np.diagflat(v_vector))
    E, u = np.linalg.eig(H)
    E_min = np.amin(E)
    index = np.where(E == E_min)[0][0]
    return E_min, u[:,index].reshape(-1, 1)

def create_2nd_order_finite_difference_scheme():
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

def compute_e(params):
    """Evaluate and returns the energy at the give point"""
    psi_vector = create_psi_vector(params)

    h_psi = -1/2*(finite_difference_matrix @ psi_vector) + (v_vector * psi_vector)
     
    e = h*(psi_vector.T @ h_psi) / (h*(psi_vector.T @ psi_vector))
    return e[0][0]

def partial_difference_quotient(params, i, dx):
    """
    This function calculates the central partial difference quotient approximation with respect to the ith parameter.
    
    Argument:
    params -- List of the functions parameters
    i -- ith paramer
    dx -- step length
    
    Returns:
    d_e -- A scalar, the central partial difference quotient approximation.
    """
    
    plus_dx = [param + (dx if j == i else 0) for j, param in enumerate(params)]
    minus_dx = [param - (dx if j == i else 0) for j, param in enumerate(params)]
    
    d_e = (compute_e(plus_dx) - compute_e(minus_dx))/2*dx
    return d_e

def gradient_step(params, lr):
    new_params = []
    for i, param in enumerate(params):
        new_value = param - lr * partial_difference_quotient(params, i, dx=lr)
        new_params.append(new_value)
    return new_params

def gradient_descent(params, max_iterations, lr, plot):
    number_of_iterations = 0 
    e = compute_e(params) # Initial calculation of energy level
    gradient_path = []
    
    def add_plot():
        one_step = params.copy()
        one_step.append(e)
        gradient_path.append(one_step) # saving values for plotting
        
    while (number_of_iterations < max_iterations): # Breaks loop if maximum iterations is reached
        new_params = gradient_step(params, lr) # New values for parameters
        new_e = compute_e(new_params) # New value for energy level
            
        if lr < 0.0005: 
            if plot:
                add_plot()
            break 
        elif new_e > e: 
            lr = lr/2
        else:
            params, e =  new_params, new_e # updates the variables with the new values
            if (number_of_iterations % 100 == 0) and plot:
                add_plot()
        
        number_of_iterations += 1

    return params, gradient_path, number_of_iterations

def estimate_lowest_energy(params, plot, max_iterations = 10000, lr = 1, ):
    # Running gradient descent
    params, gradient_path, iterations_used = gradient_descent(params=params, max_iterations=max_iterations, lr=lr, plot=plot)
    return params, gradient_path, iterations_used
    

def create_plot_axes(x_min, x_max, x_step, y_min, y_max, y_step):
    """Creating surface for plotting"""

    X = np.arange(x_min, x_max, x_step)
    Y = np.arange(y_min, y_max, y_step)
    E = np.array([[compute_e([x, y]) for y in Y] for x in X])
    X, Y = np.meshgrid(X, Y)
    return X, Y, E

def plot_gradient_descent(gradient_path, step_size = 1):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    # Surface plot
    X, Y, E = create_plot_axes(-L/3, L/3, h*10, 0.5, 5, 0.1) 
    ax.plot_surface(X, Y, Z=E.T, rstride=1, cstride=1, cmap='viridis', alpha = 0.6)

    # Path
    gradient_path = np.array(gradient_path) # transforms into a numpy array
    ax.plot(gradient_path[::step_size,0], gradient_path[::step_size,1], gradient_path[::step_size, 2], 'bx-', label='path')

    ax.plot(gradient_path[-1:,0], gradient_path[-1:,1], gradient_path[-1:,2], markerfacecolor='r', marker='o', markersize=5, label='endpoint')

    # Labeling
    ax.set_title('Energy(x0, a)', fontsize=20)
    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('a', fontsize=15)
    ax.set_zlabel('e', fontsize=15)
    ax.view_init(elev=35, azim=300)
    fig.legend(loc='upper left')
    plt.show()

def plot_wave_functions(old_params, new_params):
    fig, ax = plt.subplots(figsize=(10,6))
    plt.title('Psi')

    plt.plot(xi, (u/np.sqrt(h))**2, label = 'Fasit')

    psi = psi_func(xi, *old_params)
    psi_norm = psi/np.sqrt(norm_vector(psi, h))
    ax.plot(xi, psi_norm**2, 'r--', label = 'Start')

    psi = psi_func(xi, *new_params)
    psi_norm = psi/np.sqrt(norm_vector(psi, h))
    ax.plot(xi, psi_norm**2, 'y--', label = 'Slutt')

    plt.legend()
    plt.show()

def norm_vector(vector, h):
    return h*(vector.T @ vector)

# Handling input variables
parser = argparse.ArgumentParser(description='A script that estimates the energy of a quantum physical two body system by implementing gradient descent')
parser.add_argument('-x', type=float, default=1.0, metavar='x0',help='initial value for x', required=True)
parser.add_argument('-a', type=float, default=1.0, metavar='a',help='initial value for a', required=True)
parser.add_argument('-b', type=float, default=1.0, metavar='b',help='initial value for b', required=False)
parser.add_argument('-lr', type=float, default=1.0, metavar='learning rate', help='value for initial learning rate used in gradient descent', required=False)
parser.add_argument('-i', '--max_iter', type=int, default=2000, metavar='max iterations',
help='number of maximum iterations in gradient descent', required=False)
parser.add_argument('-f', '--function', dest='func', choices=['func1', 'func2'], required=False,
                    help='Choose between the functions. Default: func1', default='func1')
parser.add_argument('-p', '--plot', dest='plot', action='store_true', default=False,
                    help='Option for plotting the result')

# Forklaring: type=type parser konverterer til, metavar=dummy variabelnavn i help og feilmeldinger,
# dest=variabelnavn for lagring

args = parser.parse_args()
print(args)

x0 = args.x
a = args.a
b = args.b
max_iter = args.max_iter
lr = args.lr
func = args.func
plot_res = args.plot
    
# Constants
L = 20 # Length of interval
N = 500 # No. of subintervals
h = L / N # Stepsize

# Argument vector
xi = np.linspace(-L/2, L/2, N)

#v_vector = np.array(get_v_func(xi, func).reshape(-1, 1))
v_vector = get_v_vector(xi, func)
finite_difference_matrix = create_2nd_order_finite_difference_scheme()

params, gradient_path, iterations_used = estimate_lowest_energy(params=[x0, a], max_iterations=max_iter, lr=lr, plot=plot_res)

new_x0, new_sigma = params
E, u = most_accurate_e()

print(f"Initial energy at x0 = {x0} and a = {a}: {compute_e([x0, a])}")
print('Estimations:')
print(f"x0: {new_x0}")
print(f"a: {new_sigma}")
print(f"Found energy: {compute_e([new_x0, new_sigma])}")
print(f"Most accurate answer: {E}")
print(f"Used {iterations_used} out of {max_iter} iterations")


if (plot_res):
    plot_wave_functions(old_params = [x0, a], new_params = [new_x0, new_sigma])
    #plot_gradient_descent(gradient_path, step_size=1)