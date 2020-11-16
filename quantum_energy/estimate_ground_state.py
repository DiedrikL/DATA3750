#!/usr/bin/env python
# coding: utf-8

# The Energy of a Quantum Physical Two-Body System

#Importing libraries
import numpy as np

import quantum_energy.physics.one_particle as physics
import quantum_energy.physics.two_particles as physics2
import quantum_energy.plots as plots
import quantum_energy.optimization as optimization


def one_particle_estimation(args):
    # Unpacking input variables
    params = args['params']
    max_iter = args['max_iter']
    lr = args['lr']
    func = args['func']
    print_plot = args['plot']
    interactive_mode = args['interactive']
    L = args['L']
    N = args['N']

    h = L / N # stepsize

    # initializing argument vector 'xi', vector representation of potential term 'v_vector' and 
    xi = np.linspace(-L/2, L/2, N)
    v_vector = physics.get_v_vector(xi, func)
    finite_difference_matrix = physics.create_2nd_order_finite_difference_scheme(N, h)

    # Calculates the energy at inital guess
    initial_energy = physics.compute_e(params, lr, finite_difference_matrix, v_vector, xi)

    # Running gradient descent
    new_params, gradient_path, iterations_used = optimization.gradient_descent(
        params=params, max_iterations=max_iter, plot=print_plot, lr=lr, func=physics.compute_e,
        func_args=[lr, finite_difference_matrix, v_vector, xi]
        )

    # Calculates the energy with updated parameters after gradient descent
    new_energy = physics.compute_e(new_params, lr, finite_difference_matrix, v_vector, xi)

    # Solves the time-independent schrodinger equation to find the 'true' energy and wavefunction
    E, u = physics.most_accurate_e(finite_difference_matrix, v_vector)

    # Prints information about parameter values, number of iterations used, values of energy,  and percentage error
    optimization.print_estimate(params, new_params, initial_energy, new_energy, iterations_used, max_iter, E)

    # Plotting
    if (print_plot):
        plots.plot_wave_functions(old_params = params, new_params = new_params, xi=xi, u=u, h=h)
        if len(params) == 2:
            _, ax = plots.plot_gradient_descent(gradient_path, L, h, e_func = physics.compute_e, e_func_args=[h, finite_difference_matrix, v_vector, xi], block_plot = not interactive_mode)
            if interactive_mode:
                plots.interactive_plot(ax=ax, gd_args=[max_iter, True, lr, physics.compute_e, [lr, finite_difference_matrix, v_vector, xi]])
                


def two_particle_estimation(args):

    assert len(args['params']) == 2 or len(args['params']) == 3 , 'Two particle estimation only works with two or three parameters!'

    
    # Unpacking input variables
    params = args['params']
    w0 = args['w0']
    max_iter = args['max_iter']
    lr = args['lr']
    func = args['func']
    print_plot = args['plot']
    interactive_mode = args['interactive']
    L = args['L']
    N = args['N']
    
    h = L / N # Stepsize

    # Initializing argument vector 'xi', W term, finite difference matrix and H-matrix.
    xi= np.linspace(-L/2, L/2, N)
    W = physics2.create_w_matrix(xi, w0)
    fdm = physics.create_2nd_order_finite_difference_scheme(N, h)
    H = physics2.create_H_matrix(fdm, xi)

    # Calculates energy at inital guess
    initial_energy = physics2.calculate_e(params, xi, W, H)

    # Running gradient descent
    new_params, gradient_path, iterations_used = optimization.gradient_descent(
        params, max_iterations=max_iter, plot=print_plot, lr=lr, func=physics2.calculate_e, func_args=[xi, W, H])

    # Calculates energy after gradient descent
    new_energy = physics2.calculate_e(new_params, xi, W, H)

    # Prints information about parameter values, number of iterations used and energy values
    optimization.print_estimate(params, new_params, initial_energy, new_energy, iterations_used, max_iter, None)

    # Plotting
    if (print_plot):
        plots.plot_psi_matrix(params, new_params, xi)
        if len(params) == 2:
            _, ax = plots.plot_gradient_descent(gradient_path, L, h, e_func = physics2.calculate_e, e_func_args = [xi, W, H], block_plot = not interactive_mode )
            if interactive_mode:
                plots.interactive_plot(ax=ax, gd_args=[max_iter, True, lr, physics2.calculate_e,  [xi, W, H]])
            

def run(args, num_particles):
    assert num_particles in [1,2]

    if num_particles == 1:
        one_particle_estimation(args)
    elif num_particles == 2:
        two_particle_estimation(args)
