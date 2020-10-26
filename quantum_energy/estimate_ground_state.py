#!/usr/bin/env python
# coding: utf-8

# The Energy of a Quantum Physical Two-Body System

#Importing libraries
import numpy as np
import quantum_energy.physics as physics
import quantum_energy.plots as plots
import quantum_energy.optimization as optimization

def one_particle_estimation(args):
    # Unpcking input variables

    x0 = args['x0']
    a = args['a']
    b = args['b']
    max_iter = args['max_iter']
    lr = args['lr']
    func = args['func']
    print_plot = args['plot']
    
    # Constants
    L = 20 # Length of interval
    N = 500 # No. of subintervals
    h = L / N # Stepsize

    # Argument vector
    xi = np.linspace(-L/2, L/2, N)
    #v_vector = np.array(get_v_func(xi, func).reshape(-1, 1))
    v_vector = physics.get_v_vector(xi, func)
    finite_difference_matrix = physics.create_2nd_order_finite_difference_scheme(N, h)

    # Running gradient descent
    params, gradient_path, iterations_used = optimization.gradient_descent(
        params=[x0, a], max_iterations=max_iter, lr=lr, plot=print_plot,
        finite_difference_matrix=finite_difference_matrix, v_vector=v_vector, xi=xi
        )

    new_x0, new_sigma = params
    E, u = physics.most_accurate_e(finite_difference_matrix, v_vector)

    print(f"Initial energy at x0 = {x0} and a = {a}: {physics.compute_e([x0, a], lr, finite_difference_matrix, v_vector, xi)}")
    print('Estimations:')
    print(f"x0: {new_x0}")
    print(f"a: {new_sigma}")
    print(f"Found energy: {physics.compute_e([new_x0, new_sigma], lr, finite_difference_matrix, v_vector, xi)}")
    print(f"Most accurate answer: {E}")
    print(f"Used {iterations_used} out of {max_iter} iterations")


    if (print_plot):
        plots.plot_wave_functions(old_params = [x0, a], new_params = [new_x0, new_sigma], xi=xi, u=u, h=h)
        plots.plot_gradient_descent(gradient_path, L, h, finite_difference_matrix, v_vector, xi)

def two_particle_estimation(args):
    print("Two particle")

    
def run(args, num_particles):
    assert num_particles in [1,2]

    if num_particles == 1:
        one_particle_estimation(args)
    elif num_particles == 2:
        two_particle_estimation(args)