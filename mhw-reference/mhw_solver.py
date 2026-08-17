"""CPU pseudo-spectral solver for the fixed-gradient modified HW equations.

The equations follow Guillon & Gurcan (arXiv:2410.01406v2), equations (1)-(2):
resistive coupling and classical diffusion act only on non-zonal (ky != 0)
fluctuations. Fourier coefficients use the forward FFT normalization, matching
the Fourier-series amplitude convention in the cross-checked P-FLARE code.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class MHWConfig:
    grid_points: int = 64
    box_size: float = 32.0 * np.pi
    adiabaticity: float = 0.1
    density_gradient: float = 1.0
    viscosity: float = 1.0e-3
    diffusion: float = 1.0e-3
    time_step: float = 0.01


class MHWSolver:
    def __init__(self, config: MHWConfig):
        self.config = config
        n = config.grid_points
        spacing = config.box_size / n
        wave = 2.0 * np.pi * np.fft.fftfreq(n, d=spacing)
        self.kx, self.ky = np.meshgrid(wave, wave, indexing="ij")
        self.k2 = self.kx**2 + self.ky**2
        self.non_zonal = self.ky != 0.0
        cutoff = n // 3
        integer_wave = np.fft.fftfreq(n) * n
        ix, iy = np.meshgrid(integer_wave, integer_wave, indexing="ij")
        self.dealias = (np.abs(ix) <= cutoff) & (np.abs(iy) <= cutoff)
        self.nonzero = self.k2 > 0.0

    def potential_hat(self, vorticity_hat: Array) -> Array:
        potential = np.zeros_like(vorticity_hat, dtype=np.complex128)
        potential[self.nonzero] = -vorticity_hat[self.nonzero] / self.k2[self.nonzero]
        return potential

    def _bracket_hat(self, potential_hat: Array, field_hat: Array) -> Array:
        dphi_dx = np.fft.ifft2(1j * self.kx * potential_hat, norm="forward").real
        dphi_dy = np.fft.ifft2(1j * self.ky * potential_hat, norm="forward").real
        df_dx = np.fft.ifft2(1j * self.kx * field_hat, norm="forward").real
        df_dy = np.fft.ifft2(1j * self.ky * field_hat, norm="forward").real
        bracket = dphi_dx * df_dy - dphi_dy * df_dx
        transformed = np.fft.fft2(bracket, norm="forward")
        transformed[~self.dealias] = 0.0
        return transformed

    def rhs(self, density_hat: Array, vorticity_hat: Array) -> tuple[Array, Array]:
        cfg = self.config
        potential_hat = self.potential_hat(vorticity_hat)
        coupling = np.zeros_like(potential_hat)
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

    def step(self, density_hat: Array, vorticity_hat: Array) -> tuple[Array, Array]:
        dt = self.config.time_step
        k1n, k1o = self.rhs(density_hat, vorticity_hat)
        k2n, k2o = self.rhs(density_hat + dt * k1n / 2, vorticity_hat + dt * k1o / 2)
        k3n, k3o = self.rhs(density_hat + dt * k2n / 2, vorticity_hat + dt * k2o / 2)
        k4n, k4o = self.rhs(density_hat + dt * k3n, vorticity_hat + dt * k3o)
        density_new = density_hat + dt * (k1n + 2 * k2n + 2 * k3n + k4n) / 6
        vorticity_new = vorticity_hat + dt * (k1o + 2 * k2o + 2 * k3o + k4o) / 6
        density_new[~self.dealias] = 0.0
        vorticity_new[~self.dealias] = 0.0
        density_new[0, 0] = 0.0
        vorticity_new[0, 0] = 0.0
        return density_new, vorticity_new

    def initial_state(self, seed: int, amplitude: float = 1.0e-4) -> tuple[Array, Array]:
        rng = np.random.default_rng(seed)
        n = self.config.grid_points
        density = rng.normal(size=(n, n))
        vorticity = rng.normal(size=(n, n))
        density_hat = np.fft.fft2(density, norm="forward") * amplitude / np.std(density)
        vorticity_hat = np.fft.fft2(vorticity, norm="forward") * amplitude / np.std(vorticity)
        density_hat[~self.dealias] = 0.0
        vorticity_hat[~self.dealias] = 0.0
        density_hat[0, 0] = 0.0
        vorticity_hat[0, 0] = 0.0
        return density_hat, vorticity_hat

    def paper_initial_state(
        self, seed: int, amplitude: float = 1.0e-4, spectral_width: float = 0.5
    ) -> tuple[Array, Array]:
        """Return real-field Fourier seeds with an isotropic Gaussian envelope."""
        rng = np.random.default_rng(seed)
        n = self.config.grid_points
        density_phase_field = rng.normal(size=(n, n))
        potential_phase_field = rng.normal(size=(n, n))
        envelope = np.exp(-self.k2 / (2.0 * spectral_width**2)) * self.dealias

        def shaped(field: Array) -> Array:
            coefficients = np.fft.fft2(field, norm="forward")
            magnitude = np.abs(coefficients)
            phase = np.divide(coefficients, magnitude, out=np.ones_like(coefficients), where=magnitude > 0)
            result = phase * envelope
            result *= amplitude / np.max(np.abs(result))
            result[0, 0] = 0.0
            return result

        density_hat = shaped(density_phase_field)
        potential_hat = shaped(potential_phase_field)
        vorticity_hat = -self.k2 * potential_hat
        return density_hat, vorticity_hat

    def diagnostics(self, density_hat: Array, vorticity_hat: Array) -> dict[str, float | bool]:
        potential_hat = self.potential_hat(vorticity_hat)
        kinetic_spectrum = self.k2 * np.abs(potential_hat) ** 2 / 2.0
        kinetic_energy = float(np.sum(kinetic_spectrum))
        zonal_energy = float(np.sum(kinetic_spectrum[~self.non_zonal]))
        density = np.fft.ifft2(density_hat, norm="forward").real
        radial_velocity = np.fft.ifft2(
            -1j * self.ky * potential_hat, norm="forward"
        ).real
        particle_flux = float(np.mean(density * radial_velocity))
        finite = bool(
            np.isfinite(density_hat).all()
            and np.isfinite(vorticity_hat).all()
            and np.isfinite(potential_hat).all()
        )
        return {
            "finite": finite,
            "kinetic_energy": kinetic_energy,
            "zonal_kinetic_energy": zonal_energy,
            "zonal_energy_fraction": zonal_energy / kinetic_energy if kinetic_energy > 0 else 0.0,
            "particle_flux": particle_flux,
        }


def state_digest(density_hat: Array, vorticity_hat: Array) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(density_hat).view(np.uint8))
    digest.update(np.ascontiguousarray(vorticity_hat).view(np.uint8))
    return digest.hexdigest()


def linear_matrix(config: MHWConfig, kx: float, ky: float) -> Array:
    """Return the independent linear operator on state [phi_hat, density_hat]."""
    k2 = kx * kx + ky * ky
    if k2 == 0 or ky == 0:
        raise ValueError("linear drift-wave check requires a non-zonal nonzero mode")
    c = config.adiabaticity
    return np.array(
        [
            [-(c / k2) - config.viscosity * k2, c / k2],
            [c - 1j * config.density_gradient * ky, -c - config.diffusion * k2],
        ],
        dtype=np.complex128,
    )


def fastest_linear_mode(adiabaticity: float, density_gradient: float = 1.0) -> tuple[float, float]:
    config = MHWConfig(
        adiabaticity=adiabaticity,
        density_gradient=density_gradient,
        viscosity=0.0,
        diffusion=0.0,
    )
    wave_numbers = np.linspace(1.0e-3, 3.0, 30_000)
    growth = np.array(
        [np.max(np.linalg.eigvals(linear_matrix(config, 0.0, float(ky))).real) for ky in wave_numbers]
    )
    selected = int(np.argmax(growth))
    return float(growth[selected]), float(wave_numbers[selected])


def paper_scaled_config(adiabaticity: float, grid_points: int, time_step: float) -> MHWConfig:
    growth, injection_wave_number = fastest_linear_mode(adiabaticity)
    viscosity = 0.017 * growth / injection_wave_number**2
    return MHWConfig(
        grid_points=grid_points,
        box_size=20.0 * np.pi / injection_wave_number,
        adiabaticity=adiabaticity,
        density_gradient=1.0,
        viscosity=viscosity,
        diffusion=viscosity,
        time_step=time_step,
    )


def run_pilot(
    output_directory: str | Path,
    adiabaticity: float,
    seed: int = 20260813,
    grid_points: int = 96,
    time_step: float = 0.01,
    duration_in_growth_times: float = 50.0,
    diagnostic_stride: int = 10,
) -> dict:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    config = paper_scaled_config(adiabaticity, grid_points, time_step)
    growth, injection_wave_number = fastest_linear_mode(adiabaticity)
    duration = duration_in_growth_times / growth
    steps = int(np.ceil(duration / time_step))
    solver = MHWSolver(config)
    density, vorticity = solver.paper_initial_state(seed)
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
    }
    (output / "pre_run_manifest.json").write_text(
        json.dumps(pre_run_manifest, indent=2, sort_keys=True) + "\n"
    )
    records: list[dict] = []
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
    elapsed = time.perf_counter() - start
    np.savez_compressed(output / "final_state.npz", density_hat=density, vorticity_hat=vorticity)
    with (output / "diagnostics.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    completed = records[-1]["finite"] and records[-1]["step"] == steps
    manifest = {
        **pre_run_manifest,
        "elapsed_seconds": elapsed,
        "state_digest": state_digest(density, vorticity),
        "record_count": len(records),
        "completed": bool(completed),
        "first_nonfinite_step": None if completed else records[-1]["step"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if not completed:
        raise FloatingPointError(
            f"non-finite state at step {records[-1]['step']} time {records[-1]['time']}"
        )
    return manifest
