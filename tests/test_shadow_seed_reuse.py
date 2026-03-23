import math
import unittest
from unittest import mock

import numpy as np

from scripts import gen_asym_data
from light_path_tracer.asymmetries import (
    AsymmetryMeasurements,
    _circumcircle_from_points,
)


class FakeShadowMetric:
    is_spherically_symmetric = False

    def __init__(self, *, offset=0.0):
        self.offset = float(offset)
        self.trace_calls: list[float] = []
        self.trace_records: list[tuple[float, float]] = []

    def alpha_crit(self, r_obs, theta_obs):
        del r_obs, theta_obs
        return 0.7 + self.offset

    def boundary_alpha(self, theta):
        theta = float(theta)
        return 0.42 + self.offset + 0.05 * math.sin(theta) - 0.02 * math.cos(2.0 * theta)

    def trace_ray(self, r_obs, alpha, *, theta, theta_obs, axis_refine):
        del r_obs, theta_obs, axis_refine
        alpha = float(alpha)
        self.trace_calls.append(alpha)
        self.trace_records.append((float(theta), alpha))
        outcome = "captured" if alpha < self.boundary_alpha(theta) else "escaped"
        return 0.0, 0.0, outcome


class ShadowSeedReuseTests(unittest.TestCase):
    def test_top_bottom_gap_ratio_keeps_circumcircle_geometry(self):
        top = {"x": 0.0, "y": -2.0}
        bottom = {"x": 0.0, "y": 2.0}
        left = {"x": -1.0, "y": 0.0}
        right = {"x": 3.0, "y": 0.0}
        center_x, center_y, radius = _circumcircle_from_points(top, bottom, right)
        inner_radius = float(np.hypot(left["x"] - center_x, left["y"] - center_y))
        expected = float((radius - inner_radius) / radius)

        measurements = AsymmetryMeasurements(FakeShadowMetric(offset=0.0), 15.0, 1.1)
        with mock.patch.object(
            AsymmetryMeasurements,
            "_shadow_cardinal_points",
            return_value={
                "top": top,
                "bottom": bottom,
                "left": left,
                "right": right,
            },
        ), mock.patch.object(
            AsymmetryMeasurements,
            "_shared_circle_fit_result",
            side_effect=AssertionError("top_bottom_outer_circle_gap_ratio should not use a shared fit"),
        ):
            actual = measurements.top_bottom_outer_circle_gap_ratio(
                circle_fit="global_least_squares_circle",
                n_boundary_samples=24,
            )

        self.assertAlmostEqual(actual, expected, places=12)

    def test_boundary_sampling_warm_starts_each_theta_from_previous_solution(self):
        alpha_kwargs = {"n_bracket_samples": 9, "tol": 1e-6, "max_iter": 20}
        n_boundary_samples = 24

        metric = FakeShadowMetric(offset=0.0)
        measurements = AsymmetryMeasurements(metric, 15.0, 1.1)
        measurements.global_least_squares_circle(
            n_boundary_samples=n_boundary_samples,
            **alpha_kwargs,
        )

        first_alpha_by_theta = {}
        for theta, alpha in metric.trace_records:
            first_alpha_by_theta.setdefault(theta, alpha)

        theta_samples = np.linspace(
            -math.pi,
            math.pi,
            n_boundary_samples,
            endpoint=False,
            dtype=np.float64,
        )
        wrapped_thetas = [
            math.pi if math.isclose(float(theta), -math.pi, abs_tol=1e-12) else float(theta)
            for theta in theta_samples
        ]

        self.assertAlmostEqual(first_alpha_by_theta[wrapped_thetas[0]], 0.0, places=12)
        for previous_theta, theta in zip(wrapped_thetas, wrapped_thetas[1:]):
            self.assertNotAlmostEqual(first_alpha_by_theta[theta], 0.0, places=12)
            self.assertAlmostEqual(
                first_alpha_by_theta[theta],
                metric.boundary_alpha(previous_theta),
                delta=2e-6,
            )

    def test_seeded_alpha_crit_starts_from_previous_profile(self):
        alpha_kwargs = {"n_bracket_samples": 9, "tol": 1e-6, "max_iter": 20}
        theta = 0.73

        previous_metric = FakeShadowMetric(offset=0.0)
        previous = AsymmetryMeasurements(previous_metric, 15.0, 1.1)
        previous.global_least_squares_circle(n_boundary_samples=24, **alpha_kwargs)
        seed = previous.export_shadow_solve_seed()

        self.assertIsNotNone(seed)
        self.assertAlmostEqual(
            seed.alpha_guess(theta, alpha_kwargs),
            previous_metric.boundary_alpha(theta),
            delta=0.03,
        )

        seeded_metric = FakeShadowMetric(offset=0.01)
        seeded = AsymmetryMeasurements(
            seeded_metric,
            15.0,
            1.1,
            initial_shadow_seed=seed,
        )
        seeded.alpha_crit_for_theta(theta, **alpha_kwargs)

        unseeded_metric = FakeShadowMetric(offset=0.01)
        unseeded = AsymmetryMeasurements(unseeded_metric, 15.0, 1.1)
        unseeded.alpha_crit_for_theta(theta, **alpha_kwargs)

        self.assertNotAlmostEqual(seeded_metric.trace_calls[0], 0.0)
        self.assertAlmostEqual(
            seeded_metric.trace_calls[0],
            previous_metric.boundary_alpha(theta),
            delta=0.03,
        )
        self.assertAlmostEqual(unseeded_metric.trace_calls[0], 0.0, places=12)

    def test_missing_matching_seed_key_falls_back_to_default_bracket(self):
        previous = AsymmetryMeasurements(FakeShadowMetric(offset=0.0), 15.0, 1.1)
        previous.global_least_squares_circle(
            n_boundary_samples=24,
            n_bracket_samples=7,
            tol=1e-6,
            max_iter=20,
        )
        seed = previous.export_shadow_solve_seed()

        metric = FakeShadowMetric(offset=0.01)
        measurements = AsymmetryMeasurements(
            metric,
            15.0,
            1.1,
            initial_shadow_seed=seed,
        )
        measurements.alpha_crit_for_theta(
            0.4,
            n_bracket_samples=9,
            tol=1e-6,
            max_iter=20,
        )

        self.assertAlmostEqual(metric.trace_calls[0], 0.0, places=12)

    def test_right_left_measurement_exports_seed_points(self):
        alpha_kwargs = {"n_bracket_samples": 9, "tol": 1e-6, "max_iter": 20}
        theta = 0.5 * math.pi

        previous_metric = FakeShadowMetric(offset=0.0)
        previous = AsymmetryMeasurements(previous_metric, 15.0, 1.1)
        previous.right_left_tangent_ratio(**alpha_kwargs)
        seed = previous.export_shadow_solve_seed()

        self.assertIsNotNone(seed)
        self.assertAlmostEqual(
            seed.alpha_guess(theta, alpha_kwargs),
            previous_metric.boundary_alpha(theta),
            delta=0.03,
        )

        metric = FakeShadowMetric(offset=0.01)
        measurements = AsymmetryMeasurements(
            metric,
            15.0,
            1.1,
            initial_shadow_seed=seed,
        )
        measurements.alpha_crit_for_theta(theta, **alpha_kwargs)

        self.assertAlmostEqual(
            metric.trace_calls[0],
            previous_metric.boundary_alpha(theta),
            delta=0.03,
        )


class GenerationChunkSeedReuseTests(unittest.TestCase):
    def _task(self, sample_index, line_index, spin_index):
        return gen_asym_data.GenerationSampleTask(
            sample_index=sample_index,
            line_index=line_index,
            total_lines=2,
            spin_index=spin_index,
            total_spins_for_line=2,
            spin=0.1 * sample_index,
            inclination_deg=45.0 + 5.0 * (line_index - 1),
        )

    def test_chunk_reuses_previous_seed_across_line_handoff(self):
        chunk = gen_asym_data.GenerationChunkTask(
            chunk_index=0,
            tasks=(
                self._task(0, 1, 1),
                self._task(1, 1, 2),
                self._task(2, 2, 1),
            ),
        )
        received_seeds = []

        def fake_measure(task, measurement_plan, *, M, r_obs, initial_shadow_seed=None):
            del measurement_plan, M, r_obs
            received_seeds.append(initial_shadow_seed)
            return np.asarray([float(task.sample_index)], dtype=np.float64), f"seed-{task.sample_index}"

        with mock.patch.object(gen_asym_data, "_measure_generation_sample", side_effect=fake_measure):
            value_matrix = gen_asym_data._compute_generation_chunk_value_matrix(
                chunk,
                measurement_plan=(),
                quantity_count=1,
                M=1.0,
                r_obs=30.0,
            )

        self.assertEqual(received_seeds, [None, "seed-0", "seed-1"])
        np.testing.assert_allclose(value_matrix[:, 0], [0.0, 1.0, 2.0])

    def test_chunk_seed_state_resets_between_chunks(self):
        first_chunk = gen_asym_data.GenerationChunkTask(
            chunk_index=0,
            tasks=(self._task(0, 1, 1), self._task(1, 1, 2)),
        )
        second_chunk = gen_asym_data.GenerationChunkTask(
            chunk_index=1,
            tasks=(self._task(2, 2, 1),),
        )
        received_seeds = []

        def fake_measure(task, measurement_plan, *, M, r_obs, initial_shadow_seed=None):
            del measurement_plan, M, r_obs
            received_seeds.append(initial_shadow_seed)
            return np.asarray([float(task.sample_index)], dtype=np.float64), f"seed-{task.sample_index}"

        with mock.patch.object(gen_asym_data, "_measure_generation_sample", side_effect=fake_measure):
            gen_asym_data._compute_generation_chunk_value_matrix(
                first_chunk,
                measurement_plan=(),
                quantity_count=1,
                M=1.0,
                r_obs=30.0,
            )
            gen_asym_data._compute_generation_chunk_value_matrix(
                second_chunk,
                measurement_plan=(),
                quantity_count=1,
                M=1.0,
                r_obs=30.0,
            )

        self.assertEqual(received_seeds, [None, "seed-0", None])


if __name__ == "__main__":
    unittest.main()
