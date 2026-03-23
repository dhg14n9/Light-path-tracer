import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
from scripts import gen_asym_data


class GenerationBenchmarkOptionTests(unittest.TestCase):
    def test_live_benchmark_field_hidden_until_benchmark_enabled(self):
        state = gen_asym_data.default_state()
        general_keys = [spec["key"] for spec in gen_asym_data.page_field_specs("general", state)]
        self.assertNotIn("live_benchmark", general_keys)

        state["benchmark"] = True
        general_keys = [spec["key"] for spec in gen_asym_data.page_field_specs("general", state)]
        self.assertIn("live_benchmark", general_keys)

    def test_turning_benchmark_off_resets_live_benchmark(self):
        state = gen_asym_data.default_state()
        state["benchmark"] = True
        state["live_benchmark"] = True

        message = gen_asym_data.apply_bool_change(state, "benchmark", "Benchmark")

        self.assertFalse(state["benchmark"])
        self.assertFalse(state["live_benchmark"])
        self.assertIn("Live Benchmark reset to Off", message)

    def test_parse_state_disables_live_benchmark_when_benchmark_is_off(self):
        state = gen_asym_data.default_state()
        state["live_benchmark"] = True

        settings, err = gen_asym_data.parse_state(state)

        self.assertIsNone(err)
        assert settings is not None
        self.assertFalse(settings.benchmark)
        self.assertFalse(settings.live_benchmark)

    def test_build_settings_document_includes_benchmarking_config(self):
        state = gen_asym_data.default_state()
        state["benchmark"] = True
        state["live_benchmark"] = True

        settings, err = gen_asym_data.parse_state(state)

        self.assertIsNone(err)
        assert settings is not None
        document = gen_asym_data.build_settings_document(settings)
        benchmarking = document["generation"]["benchmarking"]
        self.assertEqual(
            benchmarking,
            {
                "enabled": True,
                "live_progress": True,
            },
        )


class GenerationBenchmarkSummaryTests(unittest.TestCase):
    def test_generation_benchmark_summary_lines_include_expected_metrics(self):
        summary = gen_asym_data._generation_benchmark_summary_dict(
            settings=gen_asym_data.GenerationSettings(
                run_root=gen_asym_data.DEFAULT_RUN_ROOT,
                M=1.0,
                r_obs=100.0,
                generation_mode="spin_only",
                debug=False,
                benchmark=True,
                live_benchmark=True,
                worker_count=2,
                spin_sweep=gen_asym_data.SweepRange(start=0.0, end=0.1, step=0.1),
                adaptive_edge_steps=False,
                adaptive_spin_edge_abs_threshold=0.9,
                adaptive_spin_edge_step_scale=0.2,
                adaptive_inclination_edge_steps=False,
                adaptive_inclination_edge_polar_band_deg=10.0,
                adaptive_inclination_edge_step_scale=0.2,
                fixed_theta_obs_deg=90.0,
                theta_obs_sweep=None,
                asymmetry_selection_mode="selected",
                asymmetry_measurements=("right_left_tangent_ratio",),
                sampling=gen_asym_data.SamplingConfig(
                    profile="normal",
                    advanced_tuning=False,
                    circle_fit="global",
                    n_bracket_samples=65,
                    tol=1e-8,
                    max_iter=64,
                    n_theta_samples=181,
                    n_refine_samples=17,
                    refine_levels=4,
                    n_boundary_samples=361,
                ),
            ),
            total_points=20,
            quantity_count=3,
            measurement_count=2,
            worker_count=2,
            total_chunks=5,
            planning_seconds=0.1,
            hdf5_open_seconds=0.2,
            generation_loop_seconds=2.0,
            chunk_compute_seconds_sum=3.5,
            write_seconds=0.5,
            finalize_seconds=0.1,
            total_seconds=2.9,
            chunk_compute_count=5,
            max_chunk_compute_seconds=0.9,
            write_chunk_count=5,
            max_chunk_write_seconds=0.2,
        )

        lines = gen_asym_data.generation_benchmark_summary_lines(summary)
        text = "\n".join(lines)

        self.assertIn("Generation benchmark", text)
        self.assertIn("overall_throughput", text)
        self.assertIn("parallel_eff_est", text)
        self.assertIn("mean_chunk_compute", text)

    def test_generate_run_data_returns_and_persists_benchmark_summary(self):
        class FakeWriter:
            def __init__(self, output_path, settings_document, quantity_names, *, planned_row_count=None):
                del settings_document, planned_row_count
                self.output_path = output_path
                self.quantity_names = quantity_names
                self.row_count = 0
                self.status = "running"

            def append_numeric_rows(
                self,
                *,
                sample_indices,
                spins,
                inclination_degs,
                quantity_matrix,
                flush=True,
            ):
                del sample_indices, spins, inclination_degs, flush
                self.row_count += int(quantity_matrix.shape[0])
                return self.row_count

            def set_run_status(self, status):
                self.status = status

            def flush(self):
                return None

            def close(self):
                return None

        state = gen_asym_data.default_state()
        state["benchmark"] = True
        state["live_benchmark"] = True
        state["worker_count"] = "1"
        state["spin_start"] = "0.1"
        state["spin_end"] = "0.1"
        state["spin_step"] = "0.1"
        state["fixed_theta_obs_deg"] = "60"
        state["asymmetry_measurements"] = "right_left_tangent_ratio"
        settings, err = gen_asym_data.parse_state(state)
        self.assertIsNone(err)
        assert settings is not None

        with TemporaryDirectory() as tmpdir:
            settings = replace(settings, run_root=Path(tmpdir))
            saved_run = gen_asym_data.save_settings(settings)
            with mock.patch.object(gen_asym_data, "require_h5py", return_value=None), mock.patch.object(
                gen_asym_data,
                "StreamingHDF5Writer",
                FakeWriter,
            ), mock.patch.object(
                gen_asym_data,
                "_compute_generation_chunk_value_matrix",
                return_value=np.asarray([[1.0]], dtype=np.float64),
            ):
                result = gen_asym_data.generate_run_data(saved_run, settings)

            self.assertIsNotNone(result.benchmark_summary)
            assert result.benchmark_summary is not None
            self.assertTrue(result.benchmark_summary["enabled"])
            self.assertTrue(
                "benchmark" in saved_run.document["outputs"],
                msg="benchmark summary should be persisted to the run settings document",
            )
            benchmark_path = saved_run.run_dir / gen_asym_data.RUN_BENCHMARK_FILENAME
            self.assertTrue(benchmark_path.is_file())
            report_text = benchmark_path.read_text(encoding="ascii")
            self.assertIn("Status: completed", report_text)
            self.assertIn("Generation benchmark", report_text)
            self.assertEqual(
                saved_run.document["outputs"]["benchmark_file"],
                str(benchmark_path),
            )
            self.assertEqual(saved_run.document["outputs"]["status"], "completed")

    def test_failed_benchmark_run_writes_report_in_run_folder(self):
        class FakeWriter:
            def __init__(self, output_path, settings_document, quantity_names, *, planned_row_count=None):
                del settings_document, planned_row_count
                self.output_path = output_path
                self.quantity_names = quantity_names
                self.row_count = 0
                self.status = "running"

            def append_numeric_rows(
                self,
                *,
                sample_indices,
                spins,
                inclination_degs,
                quantity_matrix,
                flush=True,
            ):
                del sample_indices, spins, inclination_degs, quantity_matrix, flush
                return self.row_count

            def set_run_status(self, status):
                self.status = status

            def flush(self):
                return None

            def close(self):
                return None

        state = gen_asym_data.default_state()
        state["benchmark"] = True
        state["worker_count"] = "1"
        state["spin_start"] = "0.1"
        state["spin_end"] = "0.1"
        state["spin_step"] = "0.1"
        state["fixed_theta_obs_deg"] = "60"
        state["asymmetry_measurements"] = "right_left_tangent_ratio"
        settings, err = gen_asym_data.parse_state(state)
        self.assertIsNone(err)
        assert settings is not None

        with TemporaryDirectory() as tmpdir:
            settings = replace(settings, run_root=Path(tmpdir))
            saved_run = gen_asym_data.save_settings(settings)
            with mock.patch.object(gen_asym_data, "require_h5py", return_value=None), mock.patch.object(
                gen_asym_data,
                "StreamingHDF5Writer",
                FakeWriter,
            ), mock.patch.object(
                gen_asym_data,
                "_compute_generation_chunk_value_matrix",
                side_effect=RuntimeError("boom"),
            ):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    gen_asym_data.generate_run_data(saved_run, settings)

            benchmark_path = saved_run.run_dir / gen_asym_data.RUN_BENCHMARK_FILENAME
            self.assertTrue(benchmark_path.is_file())
            report_text = benchmark_path.read_text(encoding="ascii")
            self.assertIn("Status: failed", report_text)
            self.assertIn("Error: boom", report_text)
            self.assertIn("Generation benchmark", report_text)
            self.assertEqual(saved_run.document["outputs"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
