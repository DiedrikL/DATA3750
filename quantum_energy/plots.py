import numpy as np
import matplotlib.pyplot as plt
from quantum_energy.physics.one_particle import psi_func, compute_e
from quantum_energy.physics.two_particles import create_psi_matrix, calculate_e

def create_plot_axes(x_min, x_max, x_step, y_min, y_max, y_step, e_func, e_args):
    """Creating surface for plotting"""

    X = np.arange(x_min, x_max, x_step)
    Y = np.arange(y_min, y_max, y_step)
    print('Drawing surface plot. This might take a while...')
    E = np.array([[e_func([x, y], *e_args) for y in Y] for x in X])
    X, Y = np.meshgrid(X, Y)
    return X, Y, E

def two_particle_gradient_path(gradient_path, h, L, W, H, xi, block_plot):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    # Surface plot
    X, Y, E = create_plot_axes(-L/2, L/2, h*10, 0.2, 5, 0.1, calculate_e, [xi, W, H]) 
    ax.plot_surface(X, Y, Z=E.T, rstride=1, cstride=1, cmap='viridis', alpha = 0.6)

    # Path
    gradient_path = np.array(gradient_path) # transforms into a numpy array
    ax.plot(gradient_path[::1,0], gradient_path[::1,1], gradient_path[::1,2], 'bx-', label='path')
    ax.plot(gradient_path[-1:,0], gradient_path[-1:,1], gradient_path[-1:,2], markerfacecolor='r', marker='o', markersize=5, label='endpoint')

    # Labeling
    ax.set_title('Energy(x0, a)', fontsize=20)
    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('a', fontsize=15)
    ax.set_zlabel('e', fontsize=15)
    ax.view_init(elev=35, azim=300)
    fig.legend(loc='upper left')
    plt.show(block = block_plot)
    return (fig,ax)

def plot_psi_matrix(guess_params, new_params, xi):

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(xi, xi)
    step_size = (np.max(xi) - np.min(xi))/len(xi)

    guess_x0, guess_a = guess_params
    psi_guess =  create_psi_matrix(guess_x0, guess_a, xi)
    psi_guess_norm = psi_guess/np.sqrt(norm_matrix(psi_guess, step_size))

    ax.plot_surface(X, Y, Z=psi_guess_norm, rstride=10, cstride=10, cmap='viridis',alpha = 0.5, label = 'psi_old')

    x0, a = new_params
    psi_new =  create_psi_matrix(x0, a, xi)
    psi_new_norm = psi_new/np.sqrt(norm_matrix(psi_new, step_size))

    ax.plot_surface(X, Y, Z=psi_new_norm,rstride=10, cstride=10, cmap='coolwarm',alpha = 0.8, label = 'psi_new')
    

    ax.set_title('Psi(x0, a)', fontsize=20)
    ax.set_xlabel('x_1', fontsize=15)
    ax.set_ylabel('x_2', fontsize=15)
    ax.set_zlabel('psi', fontsize=15)
    ax.view_init(elev=35, azim=300)
    #plt.legend(loc='upper left')

    plt.show()
    


def plot_gradient_descent(gradient_path, L, h, finite_difference_matrix, v_vector, xi, block_plot):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    # Surface plot
    X, Y, E = create_plot_axes(-L/3, L/3, h*10, 0.2, 5, 0.1, compute_e, [h, finite_difference_matrix, v_vector, xi]) 
    ax.plot_surface(X, Y, Z=E.T, rstride=1, cstride=1, cmap='viridis', alpha = 0.6)

    # Path
    gradient_path = np.array(gradient_path) # transforms into a numpy array
    ax.plot(gradient_path[::1,0], gradient_path[::1,1], gradient_path[::1,2], 'bx-', label='path')
    ax.plot(gradient_path[-1:,0], gradient_path[-1:,1], gradient_path[-1:,2], markerfacecolor='r', marker='o', markersize=5, label='endpoint')

    # Labeling
    ax.set_title('Energy(x0, a)', fontsize=20)
    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('a', fontsize=15)
    ax.set_zlabel('e', fontsize=15)
    ax.view_init(elev=35, azim=300)
    fig.legend(loc='upper left')
    plt.show(block = block_plot)
    return (fig,ax)

def plot_new_path(ax, gradient_path):
    gradient_path = np.array(gradient_path) # transforms into a numpy array
    ax.plot(gradient_path[::1,0], gradient_path[::1,1], gradient_path[::1,2], 'bx-', label='path')
    ax.plot(gradient_path[-1:,0], gradient_path[-1:,1], gradient_path[-1:,2], markerfacecolor='r', marker='o', markersize=5, label='endpoint')
    plt.show(block = False)


def plot_wave_functions(old_params, new_params, xi, u , h):
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

def interactive_plot(ax, gd_func, gd_args):
    while True:
        plot_again = input('\nDo you want to plot another path? y/n: ')

        if plot_again.lower() == 'y':
            print('Choose parameters to initialize gradient descent:\n')
            x0 = float(input('Initial guess for x0: '))
            a = float(input('Initial guess for a/sigma: '))

            _, gradient_path, _= gd_func([x0, a], *gd_args)

            plot_new_path(ax, gradient_path)

        else:
            break

def norm_vector(vector, h):
    return h*(vector.T @ vector)

def norm_matrix(matrix, h):
    return h**2 * np.sum(np.sum(np.abs(matrix)**2))

