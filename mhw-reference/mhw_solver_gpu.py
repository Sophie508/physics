"""CuPy backend equivalent to the CPU modified-HW reference solver."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import cupy as cp
import numpy as np

from mhw_solver import MHWConfig, MHWSolver, fastest_linear_mode, paper_scaled_config


class CuPyMHWSolver:
    def __init__(self, config: MHWConfig):
        self.config = config
        cpu = MHWSolver(config)
        self.kx = cp.asarray(cpu.kx)
        self.ky = cp.asarray(cpu.ky)
        self.k2 = cp.asarray(cpu.k2)
        self.non_zonal = cp.asarray(cpu.non_zonal)
        self.dealias = cp.asarray(cpu.dealias)
        self.nonzero = cp.asarray(cpu.nonzero)

    def potential_hat(self, vorticity_hat):
        potential = cp.zeros_like(vorticity_hat, dtype=cp.complex128)
        potential[self.nonzero] = -vorticity_hat[self.nonzero] / self.k2[self.nonzero]
        return potential

    def _bracket_hat(self, potential_hat, field_hat):
        dphi_dx = cp.fft.ifft2(
            1j * self.kx * potential_hat, norm="forward"
        ).real
        dphi_dy = cp.fft.ifft2(
            1j * self.ky * potential_hat, norm="forward"
        ).real
        df_dx = cp.fft.ifft2(1j * self.kx * field_hat, norm="forward").real
        df_dy = cp.fft.ifft2(1j * self.ky * field_hat, norm="forward").real
        transformed = cp.fft.fft2(
            dphi_dx * df_dy - dphi_dy * df_dx, norm="forward"
        )
        transformed[~self.dealias] = 0.0
        return transformed

    def rhs(self, density_hat, vorticity_hat):
        cfg = self.config
        potential_hat = self.potential_hat(vorticity_hat)
        coupling = cp.zeros_like(potential_hat)
        coupling[self.non_zonal] = cfg.adiabaticity * (
            potential_hat[self.non_zonal] - density_hat[self.non_zonal]
        )
        density_rhs = -self._bracket_hat(potential_hat, density_hat)
        density_rhs += -1j * cfg.density_gradient * self.ky * potential_hat
        density_rhs += coupling
        density_rhs[self.non_zonal] += (
            -cfg.diffusion * self.k2[self.non_zonal] * density_hat[self.non_zonal]
        )
        vorticity_rhs = -self._bracket_hat(potential_hat, vorticity_hat)
        vorticity_rhs += coupling
        vorticity_rhs[self.non_zonal] += (
            -cfg.viscosity
            * self.k2[self.non_zonal]
            * vorticity_hat[self.non_zonal]
        )
        density_rhs[~self.dealias] = 0.0
        vorticity_rhs[~self.dealias] = 0.0
        density_rhs[0, 0] = 0.0
        vorticity_rhs[0, 0] = 0.0
        return density_rhs, vorticity_rhs

    def step(self, density_hat, vorticity_hat):
        dt = self.config.time_step
        k1n, k1o = self.rhs(density_hat, vorticity_hat)
        k2n, k2o = self.rhs(
            density_hat + dt * k1n / 2, vorticity_hat + dt * k1o / 2
        )
        k3n, k3o = self.rhs(
            density_hat + dt * k2n / 2, vorticity_hat + dt * k2o / 2
        )
        k4n, k4o = self.rhs(density_hat + dt * k3n, vorticity_hat + dt * k3o)
        density_new = density_hat + dt * (k1n + 2 * k2n + 2 * k3n + k4n) / 6
        vorticity_new = vorticity_hat + dt * (k1o + 2 * k2o + 2 * k3o + k4o) / 6
        density_new[~self.dealias] = 0.0
        vorticity_new[~self.dealias] = 0.0
        density_new[0, 0] = 0.0
        vorticity_new[0, 0] = 0.0
        return density_new, vorticity_new

    def diagnostics(self, density_hat, vorticity_hat):
        potential_hat = self.potential_hat(vorticity_hat)
        kinetic_spectrum = self.k2 * cp.abs(potential_hat) ** 2 / 2.0
        kinetic_energy = float(cp.asnumpy(cp.sum(kinetic_spectrum)))
        zonal_energy = float(cp.asnumpy(cp.sum(kinetic_spectrum[~self.non_zonal])))
        density = cp.fft.ifft2(density_hat, norm="forward").real
        radial_velocity = cp.fft.ifft2(
            -1j * self.ky * potential_hat, norm="forward"
        ).real
        particle_flux = float(cp.asnumpy(cp.mean(density * radial_velocity)))
        finite = bool(
            cp.asnumpy(
                cp.isfinite(density_hat).all()
                & cp.isfinite(vorticity_hat).all()
                & cp.isfinite(potential_hat).all()
            )
        )
        return {
            "finite": finite,
            "kinetic_energy": kinetic_energy,
            "zonal_kinetic_energy": zonal_energy,
            "zonal_energy_fraction": (
                zonal_energy / kinetic_energy if kinetic_energy > 0 else 0.0
            ),
            "particle_flux": particle_flux,
        }


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _device_name() -> str:
    name = cp.cuda.runtime.getDeviceProperties(0)["name"]
    return name.decode() if isinstance(name, bytes) else str(name)


def run_gpu_pilot(
    output_directory: str | Path,
    adiabaticity: float,
    seed: int = 20260813,
    grid_points: int = 512,
    time_step: float = 0.01,
    duration_in_growth_times: float = 100.0,
    diagnostic_stride: int = 10,
) -> dict:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    config = paper_scaled_config(adiabaticity, grid_points, time_step)
    growth, injection_wave_number = fastest_linear_mode(adiabaticity)
    duration = duration_in_growth_times / growth
    steps = int(np.ceil(duration / time_step))
    source_directory = Path(__file__).resolve().parent
    pre_run_manifest = {
        "adiabaticity": adiabaticity,
        "seed": seed,
        "grid_points": grid_points,
        "time_step": time_step,
        "diagnostic_stride": diagnostic_stride,
        "duration_in_growth_times": duration_in_growth_times,
        "duration": duration,
        "steps": steps,
        "gamma_max": growth,
        "ky0": injection_wave_number,
        "box_size": config.box_size,
        "viscosity": config.viscosity,
        "diffusion": config.diffusion,
        "fft_normalization": "forward",
        "initialization": "real-field phases with isotropic Gaussian spectral envelope, sigma_k=0.5, max coefficient amplitude=1e-4",
        "backend": "cupy",
        "cupy_version": cp.__version__,
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "device_name": _device_name(),
        "mhw_solver_sha256": _file_digest(source_directory / "mhw_solver.py"),
        "mhw_solver_gpu_sha256": _file_digest(source_directory / "mhw_solver_gpu.py"),
    }
    (output / "pre_run_manifest.json").write_text(
        json.dumps(pre_run_manifest, indent=2, sort_keys=True) + "\n"
    )

    cpu_solver = MHWSolver(config)
    density_cpu, vorticity_cpu = cpu_solver.paper_initial_state(seed)
    solver = CuPyMHWSolver(config)
    density = cp.asarray(density_cpu)
    vorticity = cp.asarray(vorticity_cpu)
    records: list[dict] = []
    memory_pool = cp.get_default_memory_pool()
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    for step in range(steps + 1):
        if step % diagnostic_stride == 0 or step == steps:
            record = {"step": step, "time": step * time_step}
            record.update(solver.diagnostics(density, vorticity))
            records.append(record)
            if not record["finite"]:
                break
        if step < steps:
            density, vorticity = solver.step(density, vorticity)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start
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
    completed = records[-1]["finite"] and records[-1]["step"] == steps
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(density_cpu).view(np.uint8))
    digest.update(np.ascontiguousarray(vorticity_cpu).view(np.uint8))
    manifest = {
        **pre_run_manifest,
        "elapsed_seconds": elapsed,
        "gpu_memory_pool_bytes": int(memory_pool.total_bytes()),
        "state_digest": digest.hexdigest(),
        "record_count": len(records),
        "completed": bool(completed),
        "first_nonfinite_step": None if completed else records[-1]["step"],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    if not completed:
        raise FloatingPointError(
            f"non-finite state at step {records[-1]['step']} time {records[-1]['time']}"
        )
    return manifest


def continue_gpu_pilot(
    output_directory: str | Path,
    source_run_directory: str | Path,
    target_duration_in_growth_times: float = 100.0,
) -> dict:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    source = Path(source_run_directory).resolve()
    source_manifest = json.loads((source / "manifest.json").read_text())
    if not source_manifest["completed"]:
        raise ValueError("source run did not complete")
    adiabaticity = float(source_manifest["adiabaticity"])
    grid_points = int(source_manifest["grid_points"])
    time_step = float(source_manifest["time_step"])
    diagnostic_stride = int(source_manifest["diagnostic_stride"])
    seed = int(source_manifest["seed"])
    config = paper_scaled_config(adiabaticity, grid_points, time_step)
    growth, injection_wave_number = fastest_linear_mode(adiabaticity)
    target_duration = target_duration_in_growth_times / growth
    target_steps = int(np.ceil(target_duration / time_step))
    source_steps = int(source_manifest["steps"])
    additional_steps = target_steps - source_steps
    if additional_steps <= 0:
        raise ValueError("target duration does not extend source run")
    if not np.isclose(config.box_size, source_manifest["box_size"]):
        raise ValueError("source box does not match reconstructed configuration")
    if not np.isclose(config.viscosity, source_manifest["viscosity"]):
        raise ValueError("source viscosity does not match reconstructed configuration")
    source_directory = Path(__file__).resolve().parent
    source_hashes = {
        name: _file_digest(source / name)
        for name in ["manifest.json", "pre_run_manifest.json", "diagnostics.jsonl", "final_state.npz"]
    }
    pre_run_manifest = {
        "adiabaticity": adiabaticity,
        "seed": seed,
        "grid_points": grid_points,
        "time_step": time_step,
        "diagnostic_stride": diagnostic_stride,
        "source_steps": source_steps,
        "additional_steps": additional_steps,
        "target_steps": target_steps,
        "target_duration_in_growth_times": target_duration_in_growth_times,
        "target_duration": target_duration,
        "gamma_max": growth,
        "ky0": injection_wave_number,
        "box_size": config.box_size,
        "viscosity": config.viscosity,
        "diffusion": config.diffusion,
        "fft_normalization": "forward",
        "backend": "cupy",
        "cupy_version": cp.__version__,
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "device_name": _device_name(),
        "source_run_directory": str(source),
        "source_state_digest": source_manifest["state_digest"],
        "source_artifact_sha256": source_hashes,
        "mhw_solver_sha256": _file_digest(source_directory / "mhw_solver.py"),
        "mhw_solver_gpu_sha256": _file_digest(source_directory / "mhw_solver_gpu.py"),
    }
    (output / "pre_run_manifest.json").write_text(
        json.dumps(pre_run_manifest, indent=2, sort_keys=True) + "\n"
    )
    with np.load(source / "final_state.npz") as state:
        density = cp.asarray(state["density_hat"])
        vorticity = cp.asarray(state["vorticity_hat"])
    solver = CuPyMHWSolver(config)
    segment_records: list[dict] = []
    memory_pool = cp.get_default_memory_pool()
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    for local_step in range(additional_steps + 1):
        global_step = source_steps + local_step
        if local_step % diagnostic_stride == 0 or local_step == additional_steps:
            record = {"step": global_step, "time": global_step * time_step}
            record.update(solver.diagnostics(density, vorticity))
            segment_records.append(record)
            if not record["finite"]:
                break
        if local_step < additional_steps:
            density, vorticity = solver.step(density, vorticity)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start
    density_cpu = cp.asnumpy(density)
    vorticity_cpu = cp.asnumpy(vorticity)
    np.savez_compressed(
        output / "final_state.npz",
        density_hat=density_cpu,
        vorticity_hat=vorticity_cpu,
    )
    with (output / "diagnostics_continuation.jsonl").open("w", encoding="utf-8") as handle:
        for record in segment_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    source_lines = (source / "diagnostics.jsonl").read_text().splitlines()
    with (output / "diagnostics.jsonl").open("w", encoding="utf-8") as handle:
        for line in source_lines:
            handle.write(line + "\n")
        for record in segment_records[1:]:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    completed = (
        segment_records[-1]["finite"] and segment_records[-1]["step"] == target_steps
    )
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(density_cpu).view(np.uint8))
    digest.update(np.ascontiguousarray(vorticity_cpu).view(np.uint8))
    manifest = {
        **pre_run_manifest,
        "elapsed_seconds": elapsed,
        "gpu_memory_pool_bytes": int(memory_pool.total_bytes()),
        "state_digest": digest.hexdigest(),
        "source_record_count": len(source_lines),
        "segment_record_count": len(segment_records),
        "combined_record_count": len(source_lines) + len(segment_records) - 1,
        "completed": bool(completed),
        "first_nonfinite_step": None if completed else segment_records[-1]["step"],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    if not completed:
        raise FloatingPointError(
            f"non-finite state at step {segment_records[-1]['step']} time {segment_records[-1]['time']}"
        )
    return manifest
