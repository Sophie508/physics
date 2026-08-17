import argparse
import json

from mhw_solver_gpu import run_gpu_pilot


parser = argparse.ArgumentParser()
parser.add_argument("output_directory")
parser.add_argument("--adiabaticity", type=float, required=True)
parser.add_argument("--grid-points", type=int, default=512)
args = parser.parse_args()
print(
    json.dumps(
        run_gpu_pilot(
            args.output_directory,
            args.adiabaticity,
            grid_points=args.grid_points,
        ),
        indent=2,
        sort_keys=True,
    )
)
