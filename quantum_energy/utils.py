import argparse
import configparser
import os.path
import sys

CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config.ini')

def parse_cli_arguments():
    parser = argparse.ArgumentParser(description='A script that estimates the energy of a quantum physical two body system by implementing gradient descent')
    parser.add_argument('-x0', type=float, default=1.0, metavar='x0',help='initial value for x', required=False)
    parser.add_argument('-a', type=float, default=1.0, metavar='a',help='initial value for a (sigma)', required=False)
    parser.add_argument('-b', type=float, default=0.0, metavar='b',help='initial value for b', required=False)
    parser.add_argument('-lr', type=float, default=0.5, metavar='learning rate', help='value for initial learning rate used in gradient descent', required=False)
    parser.add_argument('-i', '--max_iter', type=int, default=20000, metavar='max iterations',
    help='number of maximum iterations in gradient descent', required=False)
    parser.add_argument('-l', type=int, dest='L', default=20, help='Lenght of interval', required=False)
    parser.add_argument('-n', type=int, dest='N', default=500, help='Number of subintervals', required=False)
    parser.add_argument('-w0', type=int, dest='w0', default=0, help='w0 must be positive', required=False)
    parser.add_argument('-f', '--function', dest='func', choices=['func1', 'func2'], required=False,
                        help='Choose between the functions. Default: func1', default='func1')
    parser.add_argument('-p', '--plot', dest='plot', action='store_true', default=False,
                        help='Option for plotting the result', required=False)
    parser.add_argument('-np', '--num_particles', type=int, dest='num_particles', default = 1, help='number of particles in quantum system', choices=[1,2], required = False)
    parser.add_argument('-it', '--interactive', dest='interactive', action='store_true', default=False,
                        help='Lets you run gradient descent several times to plot multiple paths on a surface. Only works for test functions with two parameters', 
                        required=False)
    # Forklaring: type=type parser konverterer til, metavar=dummy variabelnavn i help og feilmeldinger,
    # dest=variabelnavn for lagring

    args = parser.parse_args()
    args_dict = gather_params_in_list(vars(args))
    return args_dict

def gather_params_in_list(args):
    args_dict = args.copy()
    x0 = args_dict.pop('x0')
    a = args_dict.pop('a')
    b = args_dict.pop('b', 0)
    params = [x0, a]
    if b:
        params.append(b)
    args_dict['params'] = params

    return args_dict
    
def parse_config_file():
    assert os.path.isfile(CONFIG_FILE)
    
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)

    params = []
    
    for (key, val) in config.items('PARAMS'):
        params.append(float(val))
    w0 = config['TWO-PARTICLE'].getint('w0')
    lr = config['NUMERICS'].getfloat('lr')
    max_iter = config['NUMERICS'].getint('max_iter')
    L = config['NUMERICS'].getint('L')
    N = config['NUMERICS'].getint('N')
    plot = config['CONFIGURATION'].getboolean('plot')
    func = config['CONFIGURATION']['function']
    num_particles = config['CONFIGURATION'].getint('num_particles')
    interactive = config['CONFIGURATION'].getboolean('interactive')

    args = {
        'params':params,
        'w0':w0,
        'lr':lr, 
        'max_iter':max_iter,
        'L':L,
        'N':N,
        'plot': plot,
        'func':func, 
        'num_particles':num_particles,
        'interactive': interactive
        }

    return args



