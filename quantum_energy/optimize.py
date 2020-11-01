


def partial_difference_quotient(params, i, dx, func, func_args):
    """
    This function calculates the central partial difference quotient approximation with respect to the ith parameter.
    
    Arguments:
    params -- List of the functions parameters
    i -- ith paramer
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
