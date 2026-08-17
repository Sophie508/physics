import argparse
import json

from mhw_solver import run_pilot


parser = argparse.ArgumentParser()
parser.add_argument("output_directory")
parser.add_argument("--adiabaticity", type=float, required=True)
args = parser.parse_args()
print(json.dumps(run_pilot(args.output_directory, args.adiabaticity), indent=2, sort_keys=True))
