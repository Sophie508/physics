"""Cross-backend equivalence test; run only where CuPy has a CUDA device."""

import json

import cupy as cp
import numpy as np

from mhw_solver import MHWConfig, MHWSolver
from mhw_solver_gpu import CuPyMHWSolver


cfg = MHWConfig(
    grid_points=48,
    box_size=20.0 * np.pi,
    adiabaticity=0.1,
    density_gradient=1.0,
    viscosity=2.0e-3,
    diffusion=2.0e-3,
    time_step=0.01,
)
cpu = MHWSolver(cfg)
gpu = CuPyMHWSolver(cfg)
density, vorticity = cpu.paper_initial_state(20260813)
density_gpu = cp.asarray(density)
vorticity_gpu = cp.asarray(vorticity)


def errors(cpu_values, gpu_values):
    return {
        "density_max_abs": float(
            np.max(np.abs(cpu_values[0] - cp.asnumpy(gpu_values[0])))
        ),
        "vorticity_max_abs": float(
            np.max(np.abs(cpu_values[1] - cp.asnumpy(gpu_values[1])))
        ),
    }


cpu_rhs = cpu.rhs(density, vorticity)
gpu_rhs = gpu.rhs(density_gpu, vorticity_gpu)
cpu_step = cpu.step(density, vorticity)
gpu_step = gpu.step(density_gpu, vorticity_gpu)
rhs_errors = errors(cpu_rhs, gpu_rhs)
step_errors = errors(cpu_step, gpu_step)
cpu_diagnostics = cpu.diagnostics(*cpu_step)
gpu_diagnostics = gpu.diagnostics(*gpu_step)
diagnostic_errors = {
    key: abs(float(cpu_diagnostics[key]) - float(gpu_diagnostics[key]))
    for key in cpu_diagnostics
    if key != "finite"
}
report = {
    "tolerance": 1.0e-11,
    "rhs_errors": rhs_errors,
    "step_errors": step_errors,
    "diagnostic_errors": diagnostic_errors,
    "cpu_finite": cpu_diagnostics["finite"],
    "gpu_finite": gpu_diagnostics["finite"],
}
print(json.dumps(report, indent=2, sort_keys=True))
all_errors = list(rhs_errors.values()) + list(step_errors.values()) + list(
    diagnostic_errors.values()
)
if not (cpu_diagnostics["finite"] and gpu_diagnostics["finite"]):
    raise SystemExit("cross-backend state was non-finite")
if max(all_errors) > report["tolerance"]:
    raise SystemExit("CPU/CuPy equivalence tolerance failed")
