"""Run the paper-scaled MHW configuration with P-FLARE's CuPy DOP853."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cupy as cp
import numpy as np

from mhw_solver import MHWSolver, fastest_linear_mode, paper_scaled_config
from mhw_solver_gpu import CuPyMHWSolver


parser = argparse.ArgumentParser()
parser.add_argument("output_directory")
parser.add_argument("pflare_directory")
parser.add_argument("--adiabaticity", type=float, required=True)
parser.add_argument("--grid-points", type=int, default=512)
parser.add_argument("--seed", type=int, default=20260813)
parser.add_argument("--target-growth-times", type=float, default=100.0)
parser.add_argument("--max-step", type=float, default=0.1)
parser.add_argument("--sample-interval", type=float, default=0.1)
args = parser.parse_args()

sys.path.insert(0, str(Path(args.pflare_directory).resolve()))
from cupy_ivp import DOP853


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


output = Path(args.output_directory)
output.mkdir(parents=True, exist_ok=False)
pflare = Path(args.pflare_directory).resolve()
source_directory = Path(__file__).resolve().parent
growth, ky0 = fastest_linear_mode(args.adiabaticity)
config = paper_scaled_config(args.adiabaticity, args.grid_points, args.max_step)
target_time = args.target_growth_times / growth
cpu_solver = MHWSolver(config)
gpu_solver = CuPyMHWSolver(config)
density_cpu, vorticity_cpu = cpu_solver.paper_initial_state(args.seed)
n = args.grid_points
initial = cp.asarray(np.concatenate([density_cpu.ravel(), vorticity_cpu.ravel()]))


def rhs(_time, packed):
    density = packed[: n * n].reshape(n, n)
    vorticity = packed[n * n :].reshape(n, n)
    density_rhs, vorticity_rhs = gpu_solver.rhs(density, vorticity)
    return cp.concatenate([density_rhs.ravel(), vorticity_rhs.ravel()])


pre_run_manifest = {
    "adiabaticity": args.adiabaticity,
    "seed": args.seed,
    "grid_points": args.grid_points,
    "target_duration_in_growth_times": args.target_growth_times,
    "target_time": target_time,
    "gamma_max": growth,
    "ky0": ky0,
    "box_size": config.box_size,
    "viscosity": config.viscosity,
    "diffusion": config.diffusion,
    "fft_normalization": "forward",
    "integrator": "P-FLARE cupy_ivp.DOP853",
    "rtol": 1.0e-10,
    "atol": 1.0e-12,
    "max_step": args.max_step,
    "sample_interval": args.sample_interval,
    "mhw_solver_sha256": sha256(source_directory / "mhw_solver.py"),
    "mhw_solver_gpu_sha256": sha256(source_directory / "mhw_solver_gpu.py"),
    "run_adaptive_gpu_sha256": sha256(source_directory / "run_adaptive_gpu.py"),
    "pflare_run_sha256": sha256(pflare / "run_P-FLARE.py"),
    "pflare_gensolver_sha256": sha256(pflare / "gensolver.py"),
    "pflare_cupy_ivp_rk_sha256": sha256(pflare / "cupy_ivp" / "rk.py"),
    "cupy_version": cp.__version__,
    "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
    "device_name": str(cp.cuda.runtime.getDeviceProperties(0)["name"]),
}
(output / "pre_run_manifest.json").write_text(
    json.dumps(pre_run_manifest, indent=2, sort_keys=True) + "\n"
)

integrator = DOP853(
    rhs,
    0.0,
    initial,
    target_time,
    max_step=args.max_step,
    rtol=1.0e-10,
    atol=1.0e-12,
)
records = []
accepted_steps = []
next_sample = 0.0
memory_pool = cp.get_default_memory_pool()


def current_fields():
    return integrator.y[: n * n].reshape(n, n), integrator.y[n * n :].reshape(n, n)


def add_record():
    density, vorticity = current_fields()
    record = {"accepted_step": len(accepted_steps), "time": float(integrator.t)}
    record.update(gpu_solver.diagnostics(density, vorticity))
    records.append(record)
    return record


add_record()
start = time.perf_counter()
while integrator.status == "running":
    previous = float(integrator.t)
    integrator.step()
    accepted_steps.append(float(integrator.t) - previous)
    if float(integrator.t) + 1.0e-12 >= next_sample + args.sample_interval:
        next_sample = float(integrator.t)
        if not add_record()["finite"]:
            break
if records[-1]["time"] != float(integrator.t):
    add_record()
cp.cuda.Stream.null.synchronize()
elapsed = time.perf_counter() - start
density, vorticity = current_fields()
density_cpu = cp.asnumpy(density)
vorticity_cpu = cp.asnumpy(vorticity)
np.savez_compressed(
    output / "final_state.npz",
    density_hat=density_cpu,
    vorticity_hat=vorticity_cpu,
)
with (output / "diagnostics.jsonl").open("w", encoding="utf-8") as handle:
    for record in records:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
digest = hashlib.sha256()
digest.update(np.ascontiguousarray(density_cpu).view(np.uint8))
digest.update(np.ascontiguousarray(vorticity_cpu).view(np.uint8))
completed = integrator.status == "finished" and records[-1]["finite"]
manifest = {
    **pre_run_manifest,
    "completed": bool(completed),
    "integrator_status": integrator.status,
    "integrator_message": getattr(integrator, "message", None),
    "accepted_step_count": len(accepted_steps),
    "rhs_evaluations": int(integrator.nfev),
    "minimum_accepted_step": min(accepted_steps) if accepted_steps else None,
    "maximum_accepted_step": max(accepted_steps) if accepted_steps else None,
    "elapsed_seconds": elapsed,
    "gpu_memory_pool_bytes": int(memory_pool.total_bytes()),
    "record_count": len(records),
    "final_time": float(integrator.t),
    "state_digest": digest.hexdigest(),
}
(output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps(manifest, indent=2, sort_keys=True))
if not completed:
    raise SystemExit("adaptive MHW run did not complete with a finite state")
