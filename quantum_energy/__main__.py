import sys

import quantum_energy.estimate_ground_state as estimate
from quantum_energy.utils import parse_cli_arguments, parse_config_file

# If no command-line arguments are given, args will be loaded from config.ini
args = parse_cli_arguments() if (len(sys.argv) > 1) else parse_config_file()

# Estimates the ground state with the given arguments
estimate.run(args = args, num_particles = args['num_particles'])
