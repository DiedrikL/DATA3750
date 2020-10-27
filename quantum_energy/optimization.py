from quantum_energy.physics import partial_difference_quotient, compute_e

def gradient_step(params, lr, finite_difference_matrix, v_vector, xi):
    new_params = []
    for i, param in enumerate(params):
        new_value = param - lr * partial_difference_quotient(params, i, lr, finite_difference_matrix, v_vector, xi)
        new_params.append(new_value)
    return new_params

def gradient_descent(params, max_iterations, lr, plot, finite_difference_matrix, v_vector, xi):
    used_iterations = 0
    e = compute_e(params, lr, finite_difference_matrix, v_vector, xi) # Initial calculation of energy level
    gradient_path = []
    
    def add_plot(params_plot, e_plot): # Saves values for plotting
        one_step = params_plot.copy()
        one_step.append(e_plot)
        gradient_path.append(one_step)
        
    while (used_iterations < max_iterations): # Breaks loop if maximum iterations is reached
        new_params = gradient_step(params, lr, finite_difference_matrix, v_vector, xi) # New values for parameters
        new_e = compute_e(new_params, lr, finite_difference_matrix, v_vector, xi) # New value for energy level
            
        if lr < 0.005: 
            if plot:
                add_plot(new_params, new_e)
            break
        elif new_e > e: 
            lr = lr/2
        else:
            params, e =  new_params, new_e # updates the variables with the new values
            if (used_iterations % 10 == 0) and plot:
                add_plot(new_params, new_e)
        
        used_iterations += 1

    return params, gradient_path, used_iterations

def print_estimate(old_params, new_params, initial_energy, energy_estimate, E, u, iterations_used, max_iter):
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
    print(f"Most accurate answer: {E}")
    print(f"Used {iterations_used} out of {max_iter} iterations")
    print('************************************************************')