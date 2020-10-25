import sys
import quantum_energy2.estimate_ground_state as estimate
from quantum_energy2.utils import parse_cli_arguments, parse_config_file


if __name__ == '__main__':
    
    args = parse_cli_arguments() if (len(sys.argv) > 1) else parse_config_file()
    estimate.run(args = args, num_particles = args['num_particles'])
