import unittest

import numpy as np

from mhw_solver import MHWConfig, MHWSolver, linear_matrix, state_digest


class TestMHWSolver(unittest.TestCase):
    def test_pure_zonal_has_no_linear_coupling_or_diffusion(self):
        cfg = MHWConfig(grid_points=48, density_gradient=0.0)
        solver = MHWSolver(cfg)
        density = np.zeros((48, 48), dtype=np.complex128)
        vorticity = np.zeros_like(density)
        density[3, 0] = 1.0 + 0.5j
        vorticity[3, 0] = -0.25j
        density_rhs, vorticity_rhs = solver.rhs(density, vorticity)
        self.assertLess(np.max(np.abs(density_rhs)), 1e-12)
        self.assertLess(np.max(np.abs(vorticity_rhs)), 1e-12)

    def test_dealias_mask(self):
        cfg = MHWConfig(grid_points=48)
        solver = MHWSolver(cfg)
        density, vorticity = solver.initial_state(1)
        self.assertEqual(np.count_nonzero(density[~solver.dealias]), 0)
        self.assertEqual(np.count_nonzero(vorticity[~solver.dealias]), 0)

    def test_paper_initial_state_is_real_gaussian_and_bounded(self):
        cfg = MHWConfig(grid_points=48)
        solver = MHWSolver(cfg)
        density, vorticity = solver.paper_initial_state(5)
        potential = solver.potential_hat(vorticity)
        self.assertLess(
            np.max(np.abs(np.fft.ifft2(density, norm="forward").imag)), 1e-15
        )
        self.assertLess(
            np.max(np.abs(np.fft.ifft2(potential, norm="forward").imag)), 1e-15
        )
        self.assertLessEqual(np.max(np.abs(density)), 1.0e-4 * (1 + 1e-12))
        self.assertLessEqual(np.max(np.abs(potential)), 1.0e-4 * (1 + 1e-12))
        self.assertEqual(np.count_nonzero(density[~solver.dealias]), 0)

    def test_forward_fft_round_trip_and_parseval_diagnostic(self):
        cfg = MHWConfig(grid_points=48)
        solver = MHWSolver(cfg)
        rng = np.random.default_rng(17)
        potential = rng.normal(size=(48, 48))
        potential_hat = np.fft.fft2(potential, norm="forward")
        potential_hat[~solver.dealias] = 0.0
        potential_hat[0, 0] = 0.0
        filtered = np.fft.ifft2(potential_hat, norm="forward").real
        np.testing.assert_allclose(
            np.fft.fft2(filtered, norm="forward"), potential_hat, rtol=1e-12, atol=1e-12
        )
        density_hat = np.zeros_like(potential_hat)
        vorticity_hat = -solver.k2 * potential_hat
        observed = solver.diagnostics(density_hat, vorticity_hat)["kinetic_energy"]
        dx = cfg.box_size / cfg.grid_points
        gradient_x = np.fft.ifft2(1j * solver.kx * potential_hat, norm="forward").real
        gradient_y = np.fft.ifft2(1j * solver.ky * potential_hat, norm="forward").real
        expected = 0.5 * float(np.mean(gradient_x**2 + gradient_y**2))
        self.assertAlmostEqual(observed, expected, delta=1e-12)

    def test_linear_operator_matches_independent_matrix(self):
        for c, mode_x, mode_y in [(0.03, 1, 2), (0.1, 0, 3), (0.8, 2, 4)]:
            cfg = MHWConfig(grid_points=48, box_size=16 * np.pi, adiabaticity=c)
            solver = MHWSolver(cfg)
            kx = solver.kx[mode_x, mode_y]
            ky = solver.ky[mode_x, mode_y]
            matrix = linear_matrix(cfg, float(kx), float(ky))
            state = np.array([0.7 - 0.2j, -0.3 + 0.5j])
            density = np.zeros((48, 48), dtype=np.complex128)
            vorticity = np.zeros_like(density)
            potential, density_value = state
            density[mode_x, mode_y] = density_value
            vorticity[mode_x, mode_y] = -(kx * kx + ky * ky) * potential
            density_rhs, vorticity_rhs = solver.rhs(density, vorticity)
            potential_rhs = -vorticity_rhs[mode_x, mode_y] / (kx * kx + ky * ky)
            observed = np.array([potential_rhs, density_rhs[mode_x, mode_y]])
            np.testing.assert_allclose(observed, matrix @ state, rtol=1e-12, atol=1e-12)

    def test_measured_linear_growth_matches_eigenvalue(self):
        for c, mode_x, mode_y in [(0.03, 1, 2), (0.1, 0, 3), (0.8, 2, 4)]:
            cfg = MHWConfig(
                grid_points=48,
                box_size=16 * np.pi,
                adiabaticity=c,
                time_step=0.002,
            )
            solver = MHWSolver(cfg)
            kx = float(solver.kx[mode_x, mode_y])
            ky = float(solver.ky[mode_x, mode_y])
            eigenvalues, eigenvectors = np.linalg.eig(linear_matrix(cfg, kx, ky))
            selected = int(np.argmax(eigenvalues.real))
            growth_rate = float(eigenvalues[selected].real)
            eigenstate = eigenvectors[:, selected] * 1e-10
            density = np.zeros((48, 48), dtype=np.complex128)
            vorticity = np.zeros_like(density)
            density[mode_x, mode_y] = eigenstate[1]
            vorticity[mode_x, mode_y] = -(kx * kx + ky * ky) * eigenstate[0]
            initial_amplitude = abs(eigenstate[0])
            steps = 500
            for _ in range(steps):
                density, vorticity = solver.step(density, vorticity)
            final_potential = solver.potential_hat(vorticity)[mode_x, mode_y]
            measured = np.log(abs(final_potential) / initial_amplitude) / (steps * cfg.time_step)
            self.assertAlmostEqual(measured, growth_rate, delta=max(2e-4, abs(growth_rate) * 0.02))

    def test_deterministic_finite_nonlinear_smoke(self):
        cfg = MHWConfig(grid_points=48, time_step=0.01)
        solver = MHWSolver(cfg)

        def run():
            density, vorticity = solver.initial_state(20260813)
            for _ in range(100):
                density, vorticity = solver.step(density, vorticity)
            return density, vorticity, solver.diagnostics(density, vorticity)

        first_density, first_vorticity, first_diag = run()
        second_density, second_vorticity, second_diag = run()
        self.assertEqual(state_digest(first_density, first_vorticity), state_digest(second_density, second_vorticity))
        self.assertTrue(first_diag["finite"])
        self.assertEqual(first_diag, second_diag)
        self.assertGreaterEqual(first_diag["zonal_energy_fraction"], 0.0)
        self.assertLessEqual(first_diag["zonal_energy_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
