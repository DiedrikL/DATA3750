import random

import numpy as np
import matplotlib.pyplot as plt

from quantum_energy.physics.one_particle import psi_func, compute_e
from quantum_energy.physics.two_particles import create_psi_matrix, calculate_e
from quantum_energy.optimization import gradient_descent


def create_plot_axes(x_min, x_max, x_step, y_min, y_max, y_step, e_func, e_args):
    """
    This function creates axes and evaluates a given function, 'e_func', at all points
    within bounds specified by 'x_min', 'x_max', 'y_min', and 'y_max' in order to create
    surface plot of e_func.
    
    Arguments:
    x_min -- minimum value for x-axes
    x_max -- maximum values for x-axes
    x_step -- step length for x-axes
    y_min -- minimum value for y-axes
    y_max -- maximum values for y-axes
    y_step -- step length for y-axes
    e_func -- function to plot
    e_args -- list of arguments passed to 'e_func'

    Returns:
    X -- coordinate matrix for x-axes
    Y -- coordinate matrix for y-axes
    E -- coordinate matrix for 'e_func'
    """

    X = np.arange(x_min, x_max, x_step)
    Y = np.arange(y_min, y_max, y_step)
    print('Drawing surface plot. This might take a while...')
    E = np.array([[e_func([x, y], *e_args) for y in Y] for x in X])
    X, Y = np.meshgrid(X, Y)
    return X, Y, E

def plot_psi_matrix(guess_params, new_params, xi, plot_zoom=4):
    """
    This function plots a surface plot of the wavefunction for two-particle systems.

    Arguments:
    guess_params -- list of parameters before gradient descent
    new_params -- list of parameters after gradient descent
    xi -- list of x-values.
    """

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(xi, xi)
    step_size = (np.max(xi) - np.min(xi))/len(xi)

    psi_guess =  create_psi_matrix(guess_params, xi)
    psi_guess_norm = psi_guess/np.sqrt(norm_matrix(psi_guess, step_size))

    size_x, size_y = X.shape
    start = round(size_x/plot_zoom)
    stop = round(size_x - size_x/plot_zoom)

    psi_guess_plot = psi_guess_norm[start:stop, start:stop]

    Xn = X[start:stop, start:stop]
    Yn = Y[start:stop, start:stop]


    surface = ax.plot_surface(Xn, Yn, Z=psi_guess_plot, rstride=10, cstride=10, cmap='plasma',alpha = 0.5, label = 'psi_old')
    surface._facecolors2d=surface._facecolors3d
    surface._edgecolors2d=surface._edgecolors3d

    psi_new =  create_psi_matrix(new_params, xi)
    psi_new_norm = psi_new/np.sqrt(norm_matrix(psi_new, step_size))

    psi_new_plot = psi_new_norm[start:stop, start:stop]

    surface = ax.plot_surface(Xn, Yn, Z=psi_new_plot,rstride=10, cstride=10, cmap='viridis',alpha = 0.8, label = 'psi_new')
    surface._facecolors2d=surface._facecolors3d
    surface._edgecolors2d=surface._edgecolors3d
    ax.set_title('Psi(x0, a)', fontsize=20)
    ax.set_xlabel(r'${x_1}$', fontsize=15)
    ax.set_ylabel(r'${x_2}$', fontsize=15)
    ax.set_zlabel('psi', fontsize=15)
    ax.view_init(elev=10, azim=45)
    
    ax.legend(loc='upper left')
    leg = ax.get_legend()
    leg.legendHandles[0].set_color('salmon')
    leg.legendHandles[1].set_color('lawngreen') #psi_new

    plt.show()
    

def plot_gradient_descent(gradient_path, L, h, e_func, e_func_args, block_plot):
    """
    Plots surface plot of energy function 'e_func' along with the path taken
    by gradient descent.

    Arguments:
    gradient_path -- list of points
    L -- length of interval
    h -- step size
    e_func -- function to plot
    e_func_args -- list of arguments to e_func
    block_plot -- boolean specifying whether plot shall block rest of script or not
    """
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    # Surface plot
    X, Y, E = create_plot_axes(-1, 2.2, h*10, 0.1, 2, 0.1, e_func, e_func_args) 

    ax.plot_surface(X, Y, Z=E.T, rstride=1, cstride=1, cmap='viridis', alpha = 0.6)

    # Path
    gradient_path = np.array(gradient_path) # transforms into a numpy array
    ax.plot(gradient_path[::1,0], gradient_path[::1,1], gradient_path[::1,2], 'bx-', label='path')
    ax.plot(gradient_path[-1:,0], gradient_path[-1:,1], gradient_path[-1:,2], markerfacecolor='r', marker='o', markersize=5, label='endpoint')

    # Labeling
    ax.set_title('Energy(x0, a)', fontsize=20)
    ax.set_xlabel(r'${x_0}$', fontsize=15)
    ax.set_ylabel('a', fontsize=15)
    ax.set_zlabel('e', fontsize=15)
    ax.view_init(elev=20, azim=300)
    fig.legend(loc='upper left')
    plt.show(block = block_plot)
    return (fig,ax)

def plot_new_path(ax, gradient_path):
    """
    Plots path on a given axes-object.

    Arguments:
    ax -- axes object to plot on
    gradient_path -- list of points
    """
    colors = ['g', 'r', 'c', 'm']
    line_style = random.choice(colors) + 'x-'
    gradient_path = np.array(gradient_path) # transforms into a numpy array
    ax.plot(gradient_path[::1,0], gradient_path[::1,1], gradient_path[::1,2], line_style, label='path')
    ax.plot(gradient_path[-1:,0], gradient_path[-1:,1], gradient_path[-1:,2], markerfacecolor='r', marker='o', markersize=5, label='endpoint')
    plt.show(block = False)


def plot_wave_functions(old_params, new_params, xi, u , h):
    """
    Plots test-function before and after gradient descent
    and the 'true' wavefunction on a single plot.

    Arguments:
    old_params -- list of params before gradient descent
    new_params -- list of parameters after gradient descent
    xi -- list of x-values
    u -- eigenvector representing the 'true' wavefunction
    h -- step size
    """
    fig, ax = plt.subplots(figsize=(10,6))
    plt.title('Psi')

    plt.plot(xi, (u/np.sqrt(h))**2, label = 'Fasit')

    psi = psi_func(xi, old_params)
    psi_norm = psi/np.sqrt(norm_vector(psi, h))
    ax.plot(xi, psi_norm**2, 'r--', label = 'Start')

    psi = psi_func(xi, new_params)
    psi_norm = psi/np.sqrt(norm_vector(psi, h))
    ax.plot(xi, psi_norm**2, 'y--', label = 'Slutt')

    plt.legend()
    plt.show()

def interactive_plot(ax, gd_args):
    """
    Plots consecutive paths from gradient descent if "interactive" is set to True in config.ini

    Arguments:
    ax -- axes object to plot on
    gd_args -- list arguments to gradient_descent()
    """

    while True:
        plot_again = input('\nDo you want to plot another path? y/n: ')

        if plot_again.lower() == 'y':
            print('Choose parameters to initialize gradient descent:\n')
            x0 = float(input('Initial guess for x0: '))
            a = float(input('Initial guess for a/sigma: '))

            _, gradient_path, _= gradient_descent([x0, a], *gd_args)

            plot_new_path(ax, gradient_path)

        else:
            break

def norm_vector(vector, h):
    """Returns the norm of a vector"""
    return h*(vector.T @ vector)

def norm_matrix(matrix, h):
    """Returns the norm of a matrix"""
    return h**2 * np.sum(np.sum(np.abs(matrix)**2))

