import argparse
import json

from mhw_solver_gpu import continue_gpu_pilot


parser = argparse.ArgumentParser()
parser.add_argument("output_directory")
parser.add_argument("source_run_directory")
parser.add_argument("--target-growth-times", type=float, default=100.0)
args = parser.parse_args()
print(
    json.dumps(
        continue_gpu_pilot(
            args.output_directory,
            args.source_run_directory,
            target_duration_in_growth_times=args.target_growth_times,
        ),
        indent=2,
        sort_keys=True,
    )
)
