#!/usr/bin/env python
# coding: utf-8

# The Energy of a Quantum Physical Two-Body System

#Importing libraries
import numpy as np
import quantum_energy.physics as physics
import quantum_energy.plots as plots
import quantum_energy.optimization as optimization
import quantum_energy.physics_two_particle as physics2
import quantum_energy.optimization_two_particle as optimization2

def one_particle_estimation(args):
    # Unpacking input variables

    params = args['params']
    max_iter = args['max_iter']
    lr = args['lr']
    func = args['func']
    print_plot = args['plot']
    interactive_mode = args['interactive']

    # Constants
    L = 20 # Length of interval
    N = 500 # No. of subintervals
    h = L / N # Stepsize

    # Argument vector
    xi = np.linspace(-L/2, L/2, N)
    v_vector = physics.get_v_vector(xi, func)
    finite_difference_matrix = physics.create_2nd_order_finite_difference_scheme(N, h)

    # Calculates energy at inital guess
    initial_energy = physics.compute_e(params, lr, finite_difference_matrix, v_vector, xi)

    # Running gradient descent
    new_params, gradient_path, iterations_used = optimization.gradient_descent(
        params=params, max_iterations=max_iter, lr=lr, plot=print_plot,
        finite_difference_matrix=finite_difference_matrix, v_vector=v_vector, xi=xi
        )

    # Calculates energy after gradient descent
    new_energy = physics.compute_e(new_params, lr, finite_difference_matrix, v_vector, xi)

    E, u = physics.most_accurate_e(finite_difference_matrix, v_vector)

    optimization.print_estimate(params, new_params, initial_energy, new_energy, E, u, iterations_used, max_iter)

    if (print_plot):
        plots.plot_wave_functions(old_params = params, new_params = new_params, xi=xi, u=u, h=h)
        if len(params) == 2:
            _, ax = plots.plot_gradient_descent(gradient_path, L, h, finite_difference_matrix, v_vector, xi, not interactive_mode)
            if interactive_mode:
                while True:
                    plot_again = input('\nDo you want to plot another path? y/n: ')

                    if plot_again.lower() == 'y':
                        print('Choose parameters to initialize gradient descent:\n')
                        x0 = float(input('Initial guess for x0: '))
                        a = float(input('Initial guess for a/sigma: '))

                        _, gradient_path, _= optimization.gradient_descent(
                            params=[x0, a], max_iterations=max_iter, lr=lr, plot=True,
                            finite_difference_matrix=finite_difference_matrix, v_vector=v_vector, xi=xi
                            )

                        plots.plot_new_path(ax, gradient_path)

                    else:
                        break


def two_particle_estimation(args):
    # Unpacking input variables

    x0, a = args['params']
    max_iter = args['max_iter']
    lr = args['lr']
    func = args['func']
    print_plot = args['plot']

    # Constants
    L = 20 # Length of interval
    N = 500 # No. of subintervals
    h = L / N # Stepsize

    # Argument vector
    xi_1 = np.linspace(-L/2, L/2, N)
    xi_2 = xi_1.copy()
    W = physics2.create_w_matrix(xi_1, xi_2, w0 = 0)
    fdm = physics.create_2nd_order_finite_difference_scheme(N, h)
    H = physics2.create_H_matrix(fdm, xi_1)

    # Calculates energy at inital guess
    initial_energy = physics2.calculate_e(x0, a, xi_2, xi_2, H, W)

    # Running gradient descent
    new_params, gradient_path, iterations_used = optimization2.gradient_descent(
        x0, a, max_iterations=max_iter, lr=lr, plot=print_plot, H = H, W = W, xi_1 = xi_1, xi_2 = xi_2)

    # Calculates energy after gradient descent
    params = [x0, a]
    x0, a = new_params
    new_energy = initial_energy = physics2.calculate_e(x0, a, xi_2, xi_2, H, W)

    # E, u = physics.most_accurate_e(finite_difference_matrix, v_vector)

    optimization2.print_estimate(params, new_params, initial_energy, new_energy, iterations_used, max_iter)

    #if (print_plot):
        #plots.plot_wave_functions(old_params = params, new_params = new_params, xi=xi, u=u, h=h)

    #   if len(params) == 2:
            #plots.plot_gradient_descent(gradient_path, L, h, finite_difference_matrix, v_vector, xi)


def run(args, num_particles):
    assert num_particles in [1,2]

    if num_particles == 1:
        one_particle_estimation(args)
    elif num_particles == 2:
        two_particle_estimation(args)