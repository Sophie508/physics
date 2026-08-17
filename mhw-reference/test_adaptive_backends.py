"""Short-time SciPy/P-FLARE-CuPy DOP853 equivalence check."""

import json
import sys

import cupy as cp
import numpy as np
from scipy.integrate import DOP853 as SciPyDOP853

from mhw_solver import MHWConfig, MHWSolver
from mhw_solver_gpu import CuPyMHWSolver


if len(sys.argv) != 2:
    raise SystemExit("usage: test_adaptive_backends.py /path/to/P-FLARE")
sys.path.insert(0, sys.argv[1])
from cupy_ivp import DOP853 as CuPyDOP853


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
n = cfg.grid_points
cpu_initial = np.concatenate([density.ravel(), vorticity.ravel()])
gpu_initial = cp.asarray(cpu_initial)


def cpu_rhs(_time, packed):
    density_state = packed[: n * n].reshape(n, n)
    vorticity_state = packed[n * n :].reshape(n, n)
    density_rhs, vorticity_rhs = cpu.rhs(density_state, vorticity_state)
    return np.concatenate([density_rhs.ravel(), vorticity_rhs.ravel()])


def gpu_rhs(_time, packed):
    density_state = packed[: n * n].reshape(n, n)
    vorticity_state = packed[n * n :].reshape(n, n)
    density_rhs, vorticity_rhs = gpu.rhs(density_state, vorticity_state)
    return cp.concatenate([density_rhs.ravel(), vorticity_rhs.ravel()])


settings = dict(t0=0.0, t_bound=1.0, max_step=0.01, rtol=1.0e-10, atol=1.0e-12)
cpu_integrator = SciPyDOP853(cpu_rhs, y0=cpu_initial, **settings)
gpu_integrator = CuPyDOP853(gpu_rhs, y0=gpu_initial, **settings)
cpu_steps = []
gpu_steps = []
while cpu_integrator.status == "running":
    previous = cpu_integrator.t
    cpu_integrator.step()
    cpu_steps.append(cpu_integrator.t - previous)
while gpu_integrator.status == "running":
    previous = float(gpu_integrator.t)
    gpu_integrator.step()
    gpu_steps.append(float(gpu_integrator.t) - previous)
cp.cuda.Stream.null.synchronize()

gpu_final = cp.asnumpy(gpu_integrator.y)
cpu_density = cpu_integrator.y[: n * n].reshape(n, n)
cpu_vorticity = cpu_integrator.y[n * n :].reshape(n, n)
gpu_density = gpu_final[: n * n].reshape(n, n)
gpu_vorticity = gpu_final[n * n :].reshape(n, n)
cpu_diagnostics = cpu.diagnostics(cpu_density, cpu_vorticity)
gpu_diagnostics = cpu.diagnostics(gpu_density, gpu_vorticity)
report = {
    "tolerance": 1.0e-9,
    "density_max_abs": float(np.max(np.abs(cpu_density - gpu_density))),
    "vorticity_max_abs": float(np.max(np.abs(cpu_vorticity - gpu_vorticity))),
    "diagnostic_max_abs": float(
        max(
            abs(float(cpu_diagnostics[key]) - float(gpu_diagnostics[key]))
            for key in cpu_diagnostics
            if key != "finite"
        )
    ),
    "cpu_nfev": int(cpu_integrator.nfev),
    "gpu_nfev": int(gpu_integrator.nfev),
    "cpu_accepted_steps": len(cpu_steps),
    "gpu_accepted_steps": len(gpu_steps),
    "cpu_min_step": min(cpu_steps),
    "cpu_max_step": max(cpu_steps),
    "gpu_min_step": min(gpu_steps),
    "gpu_max_step": max(gpu_steps),
    "cpu_status": cpu_integrator.status,
    "gpu_status": gpu_integrator.status,
}
print(json.dumps(report, indent=2, sort_keys=True))
if cpu_integrator.status != "finished" or gpu_integrator.status != "finished":
    raise SystemExit("an adaptive integrator did not finish")
if max(
    report["density_max_abs"],
    report["vorticity_max_abs"],
    report["diagnostic_max_abs"],
) > report["tolerance"]:
    raise SystemExit("adaptive backend equivalence tolerance failed")
