#!/usr/bin/env python
# coding: utf-8

# The Energy of a Quantum Physical Two-Body System

#Importing libraries
import argparse
import numpy as np
import physics
import optimization
import plots

def one_particle():
        # Handling input variables
    parser = argparse.ArgumentParser(description='A script that estimates the energy of a quantum physical two body system by implementing gradient descent')
    parser.add_argument('-x', type=float, default=1.0, metavar='x0',help='initial value for x', required=True)
    parser.add_argument('-a', type=float, default=1.0, metavar='a',help='initial value for a (sigma)', required=True)
    parser.add_argument('-b', type=float, default=1.0, metavar='b',help='initial value for b', required=False)
    parser.add_argument('-lr', type=float, default=1.0, metavar='learning rate', help='value for initial learning rate used in gradient descent', required=False)
    parser.add_argument('-i', '--max_iter', type=int, default=2000, metavar='max iterations',
    help='number of maximum iterations in gradient descent', required=False)
    parser.add_argument('-f', '--function', dest='func', choices=['func1', 'func2'], required=False,
                        help='Choose between the functions. Default: func1', default='func1')
    parser.add_argument('-p', '--plot', dest='plot', action='store_true', default=False,
                        help='Option for plotting the result', required=False)
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
    print_plot = args.plot
    
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

def two_particle():
    print("Two particle")

    
def main():
    one_particle()


if __name__ == "__main__":
    main()


