import quantum_energy.physics.two_particles as physics2

def gradient_step(params, lr, xi, H, W):
    new_params = []
    for i, param in enumerate(params):
        new_value = param - lr * physics2.partial_difference_quotient2(params, i, lr, xi, H, W)
        new_params.append(new_value)
    return new_params

def gradient_descent(params, max_iterations, lr, plot, H, W, xi):
    used_iterations = 0
    x0, a = params
    e = physics2.calculate_e(x0, a, xi, W, H) # Initial calculation of energy level
    gradient_path = []

    def print_status():
        print(f"Iterations: {used_iterations}/{max_iterations}", end ="\r")

    def add_plot(params_plot, e_plot): # Saves values for plotting
        one_step = params_plot.copy()
        one_step.append(e_plot)
        gradient_path.append(one_step)
        
    while (used_iterations < max_iterations): # Breaks loop if maximum iterations is reached
        new_params = gradient_step(params, lr, xi, H, W) # New values for parameters
        x0, a = new_params
        new_e = physics2.calculate_e(x0, a, xi, W, H) # New value for energy level

        print_status()

        if (used_iterations % 10 == 0):
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

def print_estimate(old_params, new_params, initial_energy, energy_estimate, iterations_used, max_iter):

    x0, a = old_params
    new_x0, new_a = new_params
    print('************************************************************')
    print(f"Initial energy at x0 = {x0} and a = {a}: {initial_energy}\n")
    print('Estimated parameter values:')
    print(f"x0: {new_x0}")
    print(f"a: {new_a}")
    print(f"New energy: {energy_estimate}")
    print(f"Used {iterations_used} out of {max_iter} iterations")
    print('************************************************************')