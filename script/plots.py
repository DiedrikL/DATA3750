import numpy as np
import matplotlib.pyplot as plt
from physics import psi_func, compute_e

def create_plot_axes(x_min, x_max, x_step, y_min, y_max, y_step, h, finite_difference_matrix, v_vector, xi):
    """Creating surface for plotting"""

    X = np.arange(x_min, x_max, x_step)
    Y = np.arange(y_min, y_max, y_step)
    E = np.array([[compute_e([x, y], h, finite_difference_matrix, v_vector, xi) for y in Y] for x in X])
    X, Y = np.meshgrid(X, Y)
    return X, Y, E

def plot_gradient_descent(gradient_path, L, h, finite_difference_matrix, v_vector, xi):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    # Surface plot
    X, Y, E = create_plot_axes(-L/3, L/3, h*10, 0.5, 5, 0.1, h, finite_difference_matrix, v_vector, xi) 
    ax.plot_surface(X, Y, Z=E.T, rstride=1, cstride=1, cmap='viridis', alpha = 0.6)

    # Path
    gradient_path = np.array(gradient_path) # transforms into a numpy array
    ax.plot(gradient_path[:,0], gradient_path[:,1], gradient_path[:,2], 'bx-', label='path')
    ax.plot(gradient_path[-1:,0], gradient_path[-1:,1], gradient_path[-1:,2], markerfacecolor='r', marker='o', markersize=5, label='endpoint')

    # Labeling
    ax.set_title('Energy(x0, a)', fontsize=20)
    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('a', fontsize=15)
    ax.set_zlabel('e', fontsize=15)
    ax.view_init(elev=35, azim=300)
    fig.legend(loc='upper left')
    plt.show()

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

def norm_vector(vector, h):
    return h*(vector.T @ vector)