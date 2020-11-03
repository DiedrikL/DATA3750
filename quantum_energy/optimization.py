from quantum_energy.physics.one_particle import compute_e
from quantum_energy.physics.two_particles import calculate_e


def gradient_step(params, lr, func, func_args):
    """
    This function updates the values of each parameter in 'params' by taking one step of size 'lr'
    in the direction of steepest descent for (energy) function 'func'.
    
    Arguments:
    params -- list of parameters to differentiate 'func' with respect to
    lr -- learning rate/step length
    func -- function to differentiate
    func_args -- args passed to func
    
    Returns:
    new_params -- A list with updated parameter values after one step in the direction of steepest descent.
    """
    new_params = []
    for i, param in enumerate(params):
        new_value = param - lr * partial_difference_quotient(params, i, lr, func = func, func_args = func_args)
        new_params.append(new_value)
    return new_params

def gradient_descent(params, max_iterations, plot, lr, func, func_args):
    """
    Implementation of gradient descent.
    
    Arguments:
    params -- list of parameters to differentiate 'func' with respect to
    max_iterations -- maximum number of iterations before break
    plot -- boolean for plotting the path of gradient descent
    lr -- learning rate/step size
    func -- function to minimize
    func_args -- arguments passed to 'func'
    
    Returns:
    params -- a list with updated parameter values after one step in the direction of steepest descent.
    gradient_path -- a list of points representing the path of gradient descent
    used_iterations -- number of iterations used
    """
    used_iterations = 0
    e = func(params, *func_args) # Initial calculation of energy level
    gradient_path = []
    
    def add_plot(params_plot, e_plot): # Saves values for plotting
        one_step = params_plot.copy()
        one_step.append(e_plot)
        gradient_path.append(one_step)
    
    def print_status():
        print(f"Iterations: {used_iterations}/{max_iterations}", end ="\r")

    add_plot(params, e)

    
    while (used_iterations < max_iterations): # Breaks loop if maximum iterations is reached
        new_params = gradient_step(params, lr, func, func_args) # New values for parameters
        new_e = func(new_params, *func_args) # New value for energy level

        print_status()

        if used_iterations % (max_iterations // 10) == 0:
            print(new_e)
            
        if lr < 0.005: 
            if plot:
                add_plot(new_params, new_e)
            break
        elif new_e > e: 
            lr = lr/2
        else:
            params, e =  new_params, new_e # updates the variables with the new values
            test = round(1/lr)
            if (used_iterations % test == 0) and plot:
                add_plot(new_params, new_e)
        
        used_iterations += 1

    return params, gradient_path, used_iterations

def print_estimate(old_params, new_params, initial_energy, energy_estimate, iterations_used, max_iter, E):
    """
    Prints estimates after minimizing the energy-function.

    Arguments:
    old_params -- parameters
    new_params -- updated parameters after minimization with gradient descent
    initial_energy -- energy at guess 'old_params'
    energy_estimate -- energy at updated parameters 'new_params'
    iterations_used -- number of iterations in gradient descent
    max_iter -- maximum number of iterations in gradient descent
    E -- true value of energy. Only applicable for one-particle systems.
    """
    if len(old_params) == 2:
        x0, a = old_params
        new_x0, new_a = new_params
        print('************************************************************')
        print(f"Initial energy at x0 = {x0} and a = {a}: {initial_energy}\n")
        print('Estimated parameter values:')
        print(f"x0: {new_x0}")
        print(f"a: {new_a}")
        
    else:
        x0, a, b = old_params
        new_x0, new_a, new_b = new_params
        print(f"Initial energy at x0 = {x0}, a = {a} and b = {b}: {initial_energy}")
        print('Estimations:')
        print(f"x0: {new_x0}")
        print(f"a: {new_a}")
        print(f'b: {new_b}')
    
    print(f"New energy: {energy_estimate}")
    if E:
        print(f"Most accurate answer: {E}")
        error = round(abs(E-energy_estimate)/E*100, 4)
        print(f'Percentage error: {error} %')
    print(f"Used {iterations_used} out of {max_iter} iterations")
    print('************************************************************')


def partial_difference_quotient(params, i, dx, func, func_args):
    """
    This function calculates the central partial difference quotient approximation with respect to the ith parameter.
    
    Arguments:
    params -- List of the functions parameters
    i -- ith parameter
    dx -- step length
    func -- function to differentiate
    func_args -- args passed to func
    
    Returns:
    d_e -- A scalar, the central partial difference quotient approximation.
    """
    
    plus_dx = [param + (dx if j == i else 0) for j, param in enumerate(params)]
    minus_dx = [param - (dx if j == i else 0) for j, param in enumerate(params)]
    
    d_e = (func(plus_dx, *func_args) - func(minus_dx, *func_args))/(2*dx)
    return d_e
