from physics import partial_difference_quotient, compute_e

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