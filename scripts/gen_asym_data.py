#!/usr/bin/env python3
"""Curses-based CLI for asymmetry-data generation settings."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import curses
import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
from light_path_tracer.asymmetries import (
    ASYMMETRY_CIRCLE_FIT_CHOICES,
    ASYMMETRY_PERFORMANCE_PROFILE_NAMES,
    AsymmetryMeasurements,
    ShadowSolveSeed,
    asymmetry_performance_profile_preset,
    filter_asymmetry_measurement_kwargs,
)
from light_path_tracer.metrics import Kerr, Schwarzschild


SCHEMA_VERSION = "2"
DEFAULT_RUN_ROOT = Path("data/asym_data")
RUN_SETTINGS_FILENAME = "settings.json"
RUN_DATA_FILENAME = "asymmetry_data.h5"
RUN_BENCHMARK_FILENAME = "benchmark_summary.txt"
RUN_DIRECTORY_PREFIX = "run_"
GENERATION_MODE_CHOICES = ("spin_only", "spin_and_inclination")
MEASUREMENT_ALL_TOKEN = "all"
EDGE_REFINEMENT_SPIN_ABS_THRESHOLD_DEFAULT = 0.9
EDGE_REFINEMENT_POLAR_BAND_DEG_DEFAULT = 10.0
EDGE_REFINEMENT_STEP_SCALE_DEFAULT = 0.2
GENERATION_WORKERS_ENV_VAR = "GEN_ASYM_WORKERS"
_DEFAULT_GENERATION_CHUNK_GROUPS_PER_WORKER = 4
_MIN_GENERATION_CHUNK_SIZE = 4
_MAX_GENERATION_CHUNK_SIZE = 64
_GENERATION_WORKER_CONTEXT: dict[str, Any] | None = None

GENERATION_MODE_LABELS = {
    "spin_only": "Spin Only",
    "spin_and_inclination": "Spin + Inclination",
}
ASYMMETRY_PROFILE_LABELS = {
    "quick": "Quick",
    "normal": "Normal",
    "accurate": "Accurate",
    "ultra_accurate": "Ultra Accurate",
}
ASYMMETRY_CIRCLE_FIT_LABELS = {
    "global": "Global",
    "cardinal": "Cardinal",
}
CHOICE_VALUES: dict[str, tuple[str, ...]] = {
    "generation_mode": GENERATION_MODE_CHOICES,
    "asymmetry_profile": ASYMMETRY_PERFORMANCE_PROFILE_NAMES,
    "asymmetry_circle_fit": ASYMMETRY_CIRCLE_FIT_CHOICES,
}

FIELD_SPECS: list[dict[str, str]] = [
    {
        "key": "M",
        "label": "BH Mass",
        "kind": "float",
        "description": "Black-hole mass in geometric units. This sets the internal scale for later generation.",
    },
    {
        "key": "r_obs",
        "label": "Observer Radius",
        "kind": "float",
        "description": "Observer distance in units of M.",
    },
    {
        "key": "generation_mode",
        "label": "Generation Mode",
        "kind": "choice",
        "description": "Choose whether the later generator sweeps only spin, or sweeps both spin and observer inclination.",
    },
    {
        "key": "debug",
        "label": "Debug",
        "kind": "bool",
        "description": "Show expanded stage-by-stage progress details while generating data, including chunk ranges, sampling settings, and write progress.",
    },
    {
        "key": "benchmark",
        "label": "Benchmark",
        "kind": "bool",
        "description": "Collect and save a timing summary for the generation run, including stage runtimes, chunk timings, and throughput estimates.",
    },
    {
        "key": "live_benchmark",
        "label": "Live Benchmark",
        "kind": "bool",
        "description": "Benchmark mode only. Show rolling elapsed-time, throughput, ETA, and chunk-timing details on the live progress screen while the run is still executing.",
    },
    {
        "key": "worker_count",
        "label": "Workers",
        "kind": "text",
        "description": "Enter 'auto' to use the default worker selection for this machine, or enter a positive integer to override the number of generation workers for this run.",
    },
    {
        "key": "spin_start",
        "label": "Spin Start",
        "kind": "float",
        "description": "Starting value of the dimensionless spin sweep a/M.",
    },
    {
        "key": "spin_end",
        "label": "Spin End",
        "kind": "float",
        "description": "Ending value of the dimensionless spin sweep a/M.",
    },
    {
        "key": "spin_step",
        "label": "Spin Step",
        "kind": "float",
        "description": "Positive step size for the spin sweep.",
    },
    {
        "key": "adaptive_edge_steps",
        "label": "Adaptive Spin Edge Steps",
        "kind": "bool",
        "description": "When enabled, generation shrinks the spin sweep step near extremal |a| values.",
    },
    {
        "key": "adaptive_spin_edge_abs_threshold",
        "label": "Spin Edge Threshold",
        "kind": "float",
        "description": "Start using smaller spin steps when |a| reaches or exceeds this value.",
    },
    {
        "key": "adaptive_spin_edge_step_scale",
        "label": "Spin Edge Step Scale",
        "kind": "float",
        "description": "Multiply the spin step by this factor inside the adaptive spin-edge region.",
    },
    {
        "key": "adaptive_inclination_edge_steps",
        "label": "Adaptive Inclination Edge Steps",
        "kind": "bool",
        "description": "When enabled, generation shrinks the inclination sweep step near 0 or 180 degrees.",
    },
    {
        "key": "adaptive_inclination_edge_polar_band_deg",
        "label": "Inclination Polar Band",
        "kind": "float",
        "description": "Start using smaller inclination steps when theta_obs is within this many degrees of 0 or 180.",
    },
    {
        "key": "adaptive_inclination_edge_step_scale",
        "label": "Inclination Edge Step Scale",
        "kind": "float",
        "description": "Multiply the inclination step by this factor inside the adaptive inclination-edge region.",
    },
    {
        "key": "fixed_theta_obs_deg",
        "label": "Inclination",
        "kind": "float",
        "description": "Fixed observer inclination in degrees for spin-only mode.",
    },
    {
        "key": "theta_obs_start_deg",
        "label": "Inclination Start",
        "kind": "float",
        "description": "Starting observer inclination in degrees for spin-plus-inclination mode.",
    },
    {
        "key": "theta_obs_end_deg",
        "label": "Inclination End",
        "kind": "float",
        "description": "Ending observer inclination in degrees for spin-plus-inclination mode.",
    },
    {
        "key": "theta_obs_step_deg",
        "label": "Inclination Step",
        "kind": "float",
        "description": "Positive step size for the observer-inclination sweep.",
    },
    {
        "key": "asymmetry_measurements",
        "label": "Asymmetry Set",
        "kind": "text",
        "description": "Type 'all' or enter a comma-separated list of measurement names or indices, such as '1,3' or 'circularity_metrics,x_span_over_y_span'.",
    },
    {
        "key": "asymmetry_profile",
        "label": "Asymmetry Profile",
        "kind": "choice",
        "description": "Preset for the asymmetry sampling settings. Changing this updates the hidden numeric tuning values.",
    },
    {
        "key": "asymmetry_advanced_tuning",
        "label": "Advanced Tuning",
        "kind": "bool",
        "description": "Turn this on to reveal and edit the individual asymmetry tuning values.",
    },
    {
        "key": "asymmetry_circle_fit",
        "label": "Circle Fit",
        "kind": "choice",
        "description": "Choose how circularity metrics fit their reference circle.",
    },
    {
        "key": "asymmetry_n_bracket_samples",
        "label": "Bracket Samples",
        "kind": "int",
        "description": "Number of initial samples used to bracket alpha_crit.",
    },
    {
        "key": "asymmetry_tol",
        "label": "Alpha Tolerance",
        "kind": "float",
        "description": "Tolerance for the alpha_crit solve in radians.",
    },
    {
        "key": "asymmetry_max_iter",
        "label": "Max Iterations",
        "kind": "int",
        "description": "Maximum number of root-finding iterations for alpha_crit.",
    },
    {
        "key": "asymmetry_n_theta_samples",
        "label": "Theta Samples",
        "kind": "int",
        "description": "Angular samples used when searching for boundary extrema.",
    },
    {
        "key": "asymmetry_n_refine_samples",
        "label": "Refine Samples",
        "kind": "int",
        "description": "Samples per extremum-refinement level.",
    },
    {
        "key": "asymmetry_refine_levels",
        "label": "Refine Levels",
        "kind": "int",
        "description": "Number of boundary-extremum refinement rounds.",
    },
    {
        "key": "asymmetry_n_boundary_samples",
        "label": "Boundary Samples",
        "kind": "int",
        "description": "Samples used when tracing the full shadow boundary for circularity metrics.",
    },
]

ACTION_SPECS: list[dict[str, str]] = [
    {
        "key": "run",
        "label": "Run",
        "description": "Validate the current fields, create the next data/asym_data/run_<n>/ folder, generate the requested asymmetry grid, and stream each completed row into the HDF5 data file for this run.",
    },
    {
        "key": "reset",
        "label": "Reset",
        "description": "Restore every field to its default value.",
    },
    {
        "key": "quit",
        "label": "Quit",
        "description": "Exit the CLI without saving again.",
    },
]

PAGE_SPECS: list[dict[str, str]] = [
    {
        "key": "general",
        "label": "General",
        "description": "Choose the global black-hole and observer settings. Each run will be saved under data/asym_data/run_<n>/.",
    },
    {
        "key": "sweep",
        "label": "Sweep",
        "description": "Set the spin sweep and, if enabled, the observer-inclination sweep.",
    },
    {
        "key": "asymmetry",
        "label": "Asymmetry",
        "description": "Choose which asymmetry outputs to include and which high-level profile to use.",
    },
    {
        "key": "tuning",
        "label": "Tuning",
        "description": "Edit the detailed asymmetry tuning values when advanced tuning is turned on.",
    },
    {
        "key": "run",
        "label": "Run",
        "description": "Review the current configuration and launch the full data-generation pipeline.",
    },
]

GENERAL_PAGE_KEYS = ("M", "r_obs", "generation_mode", "debug", "benchmark", "worker_count")
SPIN_SWEEP_BASE_PAGE_KEYS = (
    "spin_start",
    "spin_end",
    "spin_step",
)
SPIN_EDGE_CONTROL_FIELDS = (
    "adaptive_edge_steps",
)
SPIN_EDGE_REFINEMENT_CONFIG_FIELDS = (
    "adaptive_spin_edge_abs_threshold",
    "adaptive_spin_edge_step_scale",
)
SPIN_ONLY_PAGE_KEYS = ("fixed_theta_obs_deg",)
SPIN_AND_INCLINATION_PAGE_KEYS = (
    "theta_obs_start_deg",
    "theta_obs_end_deg",
    "theta_obs_step_deg",
)
INCLINATION_EDGE_CONTROL_FIELDS = (
    "adaptive_inclination_edge_steps",
)
INCLINATION_EDGE_REFINEMENT_CONFIG_FIELDS = (
    "adaptive_inclination_edge_polar_band_deg",
    "adaptive_inclination_edge_step_scale",
)
ASYMMETRY_PAGE_KEYS = (
    "asymmetry_measurements",
    "asymmetry_profile",
    "asymmetry_advanced_tuning",
    "asymmetry_circle_fit",
)
ADVANCED_TUNING_PAGE_KEYS = (
    "asymmetry_n_bracket_samples",
    "asymmetry_tol",
    "asymmetry_max_iter",
    "asymmetry_n_theta_samples",
    "asymmetry_n_refine_samples",
    "asymmetry_refine_levels",
    "asymmetry_n_boundary_samples",
)

SPIN_ONLY_FIELDS = {"fixed_theta_obs_deg"}
SPIN_AND_INCLINATION_FIELDS = {
    "theta_obs_start_deg",
    "theta_obs_end_deg",
    "theta_obs_step_deg",
}
ADVANCED_TUNING_FIELDS = {
    "asymmetry_n_bracket_samples",
    "asymmetry_tol",
    "asymmetry_max_iter",
    "asymmetry_n_theta_samples",
    "asymmetry_n_refine_samples",
    "asymmetry_refine_levels",
    "asymmetry_n_boundary_samples",
}

FIELD_UNIT_SUFFIXES: dict[str, str] = {
    "M": "M",
    "r_obs": "M",
    "fixed_theta_obs_deg": "deg",
    "theta_obs_start_deg": "deg",
    "theta_obs_end_deg": "deg",
    "theta_obs_step_deg": "deg",
    "spin_start": "a/M",
    "spin_end": "a/M",
    "spin_step": "a/M",
    "adaptive_spin_edge_abs_threshold": "a/M",
    "adaptive_spin_edge_step_scale": "x",
    "adaptive_inclination_edge_polar_band_deg": "deg",
    "adaptive_inclination_edge_step_scale": "x",
    "asymmetry_tol": "rad",
}


@dataclass(frozen=True)
class SweepRange:
    """One inclusive parameter sweep described by start, end, and step."""

    start: float
    end: float
    step: float


@dataclass(frozen=True)
class SamplingConfig:
    """Resolved asymmetry sampling settings."""

    profile: str
    advanced_tuning: bool
    circle_fit: str
    n_bracket_samples: int
    tol: float
    max_iter: int
    n_theta_samples: int
    n_refine_samples: int
    refine_levels: int
    n_boundary_samples: int


@dataclass(frozen=True)
class GenerationSettings:
    """All settings needed for one asymmetry-data generation run."""

    run_root: Path
    M: float
    r_obs: float
    generation_mode: str
    debug: bool
    benchmark: bool
    live_benchmark: bool
    worker_count: int | None
    spin_sweep: SweepRange
    adaptive_edge_steps: bool
    adaptive_spin_edge_abs_threshold: float
    adaptive_spin_edge_step_scale: float
    adaptive_inclination_edge_steps: bool
    adaptive_inclination_edge_polar_band_deg: float
    adaptive_inclination_edge_step_scale: float
    fixed_theta_obs_deg: float | None
    theta_obs_sweep: SweepRange | None
    asymmetry_selection_mode: str
    asymmetry_measurements: tuple[str, ...]
    sampling: SamplingConfig


@dataclass(frozen=True)
class SavedRun:
    """Filesystem details for one saved run configuration."""

    run_index: int
    run_root: Path
    run_dir: Path
    settings_path: Path
    document: dict[str, object]


@dataclass(frozen=True)
class SpinLinePlan:
    """One spin sweep associated with a single observer inclination."""

    inclination_deg: float
    spin_values: tuple[float, ...]


@dataclass(frozen=True)
class GenerationPlan:
    """Resolved parameter grid for one run."""

    lines: tuple[SpinLinePlan, ...]
    total_points: int


@dataclass(frozen=True)
class GeneratedRun:
    """Artifacts written by one completed data-generation run."""

    saved_run: SavedRun
    data_path: Path
    total_points: int
    quantity_names: tuple[str, ...]
    benchmark_summary: dict[str, Any] | None = None


@dataclass(frozen=True)
class GenerationSampleTask:
    """One independent asymmetry sample to compute."""

    sample_index: int
    line_index: int
    total_lines: int
    spin_index: int
    total_spins_for_line: int
    spin: float
    inclination_deg: float


@dataclass(frozen=True)
class GenerationMeasurementSpec:
    """One resolved asymmetry measurement execution spec."""

    name: str
    kwargs: tuple[tuple[str, Any], ...]
    estimated_work_units: float
    output_names: tuple[str, ...]


@dataclass(frozen=True)
class GenerationChunkTask:
    """One worker batch of contiguous generation samples."""

    chunk_index: int
    tasks: tuple[GenerationSampleTask, ...]


@dataclass(frozen=True)
class GenerationChunkResult:
    """Computed numeric row data for one worker batch."""

    chunk_index: int
    tasks: tuple[GenerationSampleTask, ...]
    value_matrix: np.ndarray
    compute_seconds: float = 0.0


def default_state() -> dict[str, Any]:
    preset = asymmetry_performance_profile_preset("normal")
    return {
        "M": "1",
        "r_obs": "100",
        "generation_mode": "spin_only",
        "debug": False,
        "benchmark": False,
        "live_benchmark": False,
        "worker_count": "auto",
        "spin_start": "0",
        "spin_end": "0.99",
        "spin_step": "0.05",
        "adaptive_edge_steps": False,
        "adaptive_spin_edge_abs_threshold": f"{EDGE_REFINEMENT_SPIN_ABS_THRESHOLD_DEFAULT:g}",
        "adaptive_spin_edge_step_scale": f"{EDGE_REFINEMENT_STEP_SCALE_DEFAULT:g}",
        "fixed_theta_obs_deg": "90",
        "theta_obs_start_deg": "30",
        "theta_obs_end_deg": "150",
        "theta_obs_step_deg": "15",
        "adaptive_inclination_edge_steps": False,
        "adaptive_inclination_edge_polar_band_deg": f"{EDGE_REFINEMENT_POLAR_BAND_DEG_DEFAULT:g}",
        "adaptive_inclination_edge_step_scale": f"{EDGE_REFINEMENT_STEP_SCALE_DEFAULT:g}",
        "asymmetry_measurements": MEASUREMENT_ALL_TOKEN,
        "asymmetry_profile": "normal",
        "asymmetry_advanced_tuning": False,
        "asymmetry_circle_fit": "global",
        "asymmetry_n_bracket_samples": f"{int(preset['n_bracket_samples']):d}",
        "asymmetry_tol": f"{float(preset['tol']):g}",
        "asymmetry_max_iter": f"{int(preset['max_iter']):d}",
        "asymmetry_n_theta_samples": f"{int(preset['n_theta_samples']):d}",
        "asymmetry_n_refine_samples": f"{int(preset['n_refine_samples']):d}",
        "asymmetry_refine_levels": f"{int(preset['refine_levels']):d}",
        "asymmetry_n_boundary_samples": f"{int(preset['n_boundary_samples']):d}",
    }


def toggle_choice(key: str, value: str) -> str:
    options = CHOICE_VALUES.get(key)
    if options is None:
        return value
    try:
        index = options.index(value)
    except ValueError:
        return options[0]
    return options[(index + 1) % len(options)]


def apply_asymmetry_profile_to_state(state: dict[str, Any], profile: str) -> None:
    preset = asymmetry_performance_profile_preset(profile)
    state["asymmetry_n_bracket_samples"] = f"{int(preset['n_bracket_samples']):d}"
    state["asymmetry_tol"] = f"{float(preset['tol']):g}"
    state["asymmetry_max_iter"] = f"{int(preset['max_iter']):d}"
    state["asymmetry_n_theta_samples"] = f"{int(preset['n_theta_samples']):d}"
    state["asymmetry_n_refine_samples"] = f"{int(preset['n_refine_samples']):d}"
    state["asymmetry_refine_levels"] = f"{int(preset['refine_levels']):d}"
    state["asymmetry_n_boundary_samples"] = f"{int(preset['n_boundary_samples']):d}"


def apply_choice_change(state: dict[str, Any], key: str, label: str) -> str:
    next_value = toggle_choice(key, str(state[key]))
    state[key] = next_value
    if key == "asymmetry_profile":
        apply_asymmetry_profile_to_state(state, next_value)
        return f"{label} set to {format_choice_value(key, next_value)} and tuning reset to the profile preset."
    return f"{label} set to {format_choice_value(key, next_value)}."


def apply_bool_change(state: dict[str, Any], key: str, label: str) -> str:
    next_value = not bool(state[key])
    state[key] = next_value
    if key == "benchmark" and not next_value:
        state["live_benchmark"] = False
        return f"{label} set to Off and Live Benchmark reset to Off."
    if key == "asymmetry_advanced_tuning" and not next_value:
        apply_asymmetry_profile_to_state(state, str(state.get("asymmetry_profile", "normal")))
        return f"{label} set to Off and numeric tuning reset to the selected profile."
    return f"{label} set to {'On' if next_value else 'Off'}."


def choice_label_map(key: str) -> dict[str, str]:
    if key == "generation_mode":
        return GENERATION_MODE_LABELS
    if key == "asymmetry_profile":
        return ASYMMETRY_PROFILE_LABELS
    if key == "asymmetry_circle_fit":
        return ASYMMETRY_CIRCLE_FIT_LABELS
    return {}


def format_choice_value(key: str, value: str) -> str:
    return choice_label_map(key).get(value, value)


def append_unit(value: Any, unit: str | None) -> str:
    text = str(value).strip()
    if not text or unit is None:
        return text
    if text.endswith(f" {unit}"):
        return text
    return f"{text} {unit}"


def format_field_value(key: str, value: Any) -> str:
    if key == "asymmetry_measurements":
        return str(value)
    if key == "worker_count" and not str(value).strip():
        return "auto"
    return append_unit(value, FIELD_UNIT_SUFFIXES.get(key))


def get_visible_field_specs(state: dict[str, Any]) -> list[dict[str, str]]:
    mode = str(state.get("generation_mode", "spin_only")).strip().lower()
    advanced_tuning = bool(state.get("asymmetry_advanced_tuning", False))
    adaptive_spin_edges = bool(state.get("adaptive_edge_steps", False))
    adaptive_inclination_edges = bool(state.get("adaptive_inclination_edge_steps", False))
    benchmark_enabled = bool(state.get("benchmark", False))
    visible: list[dict[str, str]] = []
    for spec in FIELD_SPECS:
        key = spec["key"]
        if key == "live_benchmark" and not benchmark_enabled:
            continue
        if mode == "spin_only" and key in SPIN_AND_INCLINATION_FIELDS:
            continue
        if mode == "spin_and_inclination" and key in SPIN_ONLY_FIELDS:
            continue
        if not adaptive_spin_edges and key in set(SPIN_EDGE_REFINEMENT_CONFIG_FIELDS):
            continue
        if (
            mode != "spin_and_inclination" or not adaptive_inclination_edges
        ) and key in set(INCLINATION_EDGE_REFINEMENT_CONFIG_FIELDS):
            continue
        if not advanced_tuning and key in ADVANCED_TUNING_FIELDS:
            continue
        visible.append(spec)
    return visible


def _normalize_profile_name(raw_value: str) -> str:
    normalized = str(raw_value).strip().replace("-", "_").lower()
    if normalized not in ASYMMETRY_PERFORMANCE_PROFILE_NAMES:
        available = ", ".join(
            name.replace("_", "-") for name in ASYMMETRY_PERFORMANCE_PROFILE_NAMES
        )
        raise ValueError(
            f"Unknown asymmetry profile {raw_value!r}. Available profiles: {available}."
        )
    return normalized


def _validate_positive(label: str, value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label} must be > 0.")
    return float(value)


def _parse_worker_count(raw_value: Any) -> tuple[int | None, str | None]:
    normalized = str(raw_value).strip().lower()
    if not normalized or normalized == "auto":
        return None, None
    try:
        worker_count = int(normalized)
    except ValueError:
        return None, "Workers must be 'auto' or an integer >= 1."
    if worker_count < 1:
        return None, "Workers must be 'auto' or an integer >= 1."
    return int(worker_count), None


def _format_worker_count_setting(worker_count: int | None) -> str:
    if worker_count is None:
        return "Auto"
    return str(int(worker_count))


def _validate_closed_interval(
    label: str,
    value: float,
    lower: float,
    upper: float,
) -> float:
    if not math.isfinite(value) or value < lower or value > upper:
        raise ValueError(f"{label} must be in [{lower:g}, {upper:g}].")
    return float(value)


def _validate_inclination(label: str, value: float) -> float:
    if not math.isfinite(value) or value < 0.0 or value > 180.0:
        raise ValueError(f"{label} must be in [0, 180] degrees.")
    return float(value)


def _validate_sweep(
    label: str,
    start: float,
    end: float,
    step: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    unit_suffix: str = "",
) -> SweepRange:
    values = {
        "start": float(start),
        "end": float(end),
        "step": float(step),
    }
    for name, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"{label} {name} must be finite.")
    if values["step"] <= 0.0:
        raise ValueError(f"{label} step must be > 0.")
    if values["end"] < values["start"]:
        raise ValueError(f"{label} end must be greater than or equal to start.")
    if min_value is not None and (
        values["start"] < min_value or values["end"] < min_value
    ):
        raise ValueError(
            f"{label} start and end must be >= {min_value:g}{unit_suffix}."
        )
    if max_value is not None and (
        values["start"] > max_value or values["end"] > max_value
    ):
        raise ValueError(
            f"{label} start and end must be <= {max_value:g}{unit_suffix}."
        )
    return SweepRange(
        start=values["start"],
        end=values["end"],
        step=values["step"],
    )


def _sweep_point_count(sweep: SweepRange) -> int:
    ratio = (sweep.end - sweep.start) / sweep.step
    tolerance = 1e-12 * max(1.0, abs(ratio))
    return int(math.floor(ratio + tolerance)) + 1


def _sweep_includes_nonzero(sweep: SweepRange) -> bool:
    return _sweep_point_count(sweep) > 1 or not math.isclose(
        sweep.start,
        0.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def _sweep_hits_value(sweep: SweepRange, target: float) -> bool:
    tolerance = 1e-12 * max(1.0, abs(sweep.start), abs(sweep.end), abs(target))
    if target < sweep.start - tolerance or target > sweep.end + tolerance:
        return False
    index = (target - sweep.start) / sweep.step
    return math.isclose(index, round(index), rel_tol=0.0, abs_tol=1e-9)


def spin_mass_units_to_kerr_a(M: float, spin_over_mass: float) -> float:
    return float(M) * float(spin_over_mass)


def build_metric_for_spin(M: float, spin_over_mass: float) -> Schwarzschild | Kerr:
    if math.isclose(float(spin_over_mass), 0.0, rel_tol=0.0, abs_tol=1e-12):
        return Schwarzschild(M=float(M))
    return Kerr(M=float(M), a=spin_mass_units_to_kerr_a(M, spin_over_mass))


def asymmetry_measurement_kwargs(settings: GenerationSettings) -> dict[str, Any]:
    return {
        "circle_fit": settings.sampling.circle_fit,
        "n_bracket_samples": settings.sampling.n_bracket_samples,
        "tol": settings.sampling.tol,
        "max_iter": settings.sampling.max_iter,
        "n_theta_samples": settings.sampling.n_theta_samples,
        "n_refine_samples": settings.sampling.n_refine_samples,
        "refine_levels": settings.sampling.refine_levels,
        "n_boundary_samples": settings.sampling.n_boundary_samples,
    }


def _measurement_spec_kwargs(spec: GenerationMeasurementSpec) -> dict[str, Any]:
    return dict(spec.kwargs)


def _runtime_measurement_plan(
    measurement_specs: tuple[GenerationMeasurementSpec, ...],
) -> tuple[tuple[str, dict[str, Any]], ...]:
    return tuple(
        (spec.name, _measurement_spec_kwargs(spec))
        for spec in measurement_specs
    )


def _resolve_generation_measurement_plan(
    settings: GenerationSettings,
) -> tuple[tuple[GenerationMeasurementSpec, ...], tuple[str, ...]]:
    common_kwargs = asymmetry_measurement_kwargs(settings)
    measurement_specs: list[GenerationMeasurementSpec] = []
    quantity_names: list[str] = []

    for measurement_name in tuple(settings.asymmetry_measurements):
        measurement_kwargs = filter_asymmetry_measurement_kwargs(
            measurement_name,
            common_kwargs,
        )
        estimated_work_units = max(
            float(
                AsymmetryMeasurements.estimate_measurement_work_units(
                    measurement_name,
                    **measurement_kwargs,
                )
            ),
            1.0,
        )
        output_names = AsymmetryMeasurements.measurement_output_names(measurement_name)
        measurement_specs.append(
            GenerationMeasurementSpec(
                name=str(measurement_name),
                kwargs=tuple(measurement_kwargs.items()),
                estimated_work_units=estimated_work_units,
                output_names=tuple(output_names),
            )
        )
        quantity_names.extend(output_names)

    return tuple(measurement_specs), tuple(quantity_names)


def _normalize_generated_coordinate(value: float) -> float:
    if math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1e-12):
        return 0.0
    if math.isclose(value, 180.0, rel_tol=0.0, abs_tol=1e-12):
        return 180.0
    if math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-12):
        return 1.0
    if math.isclose(value, -1.0, rel_tol=0.0, abs_tol=1e-12):
        return -1.0
    return float(value)


def _simple_sweep_values(sweep: SweepRange) -> tuple[float, ...]:
    count = _sweep_point_count(sweep)
    values: list[float] = []
    for index in range(count):
        value = sweep.start + index * sweep.step
        if value > sweep.end or math.isclose(
            value,
            sweep.end,
            rel_tol=0.0,
            abs_tol=1e-12 * max(1.0, abs(sweep.end)),
        ):
            value = sweep.end
        values.append(_normalize_generated_coordinate(float(value)))
    return tuple(values)


def _generate_adaptive_sweep_values(
    sweep: SweepRange,
    *,
    step_scale: float,
    boundaries: tuple[float, ...],
    should_refine: Callable[[float], bool],
) -> tuple[float, ...]:
    tolerance = 1e-12 * max(1.0, abs(sweep.start), abs(sweep.end), abs(sweep.step))
    current = float(sweep.start)
    end = float(sweep.end)
    base_step = float(sweep.step)
    refined_step = base_step * float(step_scale)
    values = [_normalize_generated_coordinate(current)]

    if refined_step <= 0.0:
        raise ValueError("Adaptive step scale must produce a positive refined step.")

    transition_points = tuple(
        sorted(
            boundary
            for boundary in boundaries
            if sweep.start + tolerance < boundary < sweep.end - tolerance
        )
    )

    while current < end - tolerance:
        local_step = refined_step if should_refine(current) else base_step
        next_value = current + local_step

        for boundary in transition_points:
            if current + tolerance < boundary < next_value - tolerance:
                next_value = boundary
                break

        if next_value >= end - tolerance:
            next_value = end

        if next_value <= current + tolerance:
            next_value = min(end, current + max(local_step, 10.0 * tolerance))

        current = _normalize_generated_coordinate(float(next_value))
        if math.isclose(current, values[-1], rel_tol=0.0, abs_tol=tolerance):
            break
        values.append(current)

    if not math.isclose(values[-1], end, rel_tol=0.0, abs_tol=tolerance):
        values.append(_normalize_generated_coordinate(end))

    deduped: list[float] = []
    for value in values:
        if deduped and math.isclose(value, deduped[-1], rel_tol=0.0, abs_tol=tolerance):
            continue
        deduped.append(_normalize_generated_coordinate(value))
    return tuple(deduped)


def _polar_distance_deg(theta_deg: float) -> float:
    theta_value = float(theta_deg)
    return min(abs(theta_value), abs(180.0 - theta_value))


def _should_refine_for_inclination_deg(theta_deg: float, settings: GenerationSettings) -> bool:
    if not settings.adaptive_inclination_edge_steps:
        return False
    tolerance = 1e-12 * max(1.0, abs(theta_deg))
    return _polar_distance_deg(theta_deg) <= (
        float(settings.adaptive_inclination_edge_polar_band_deg) + tolerance
    )


def _resolve_inclination_values(settings: GenerationSettings) -> tuple[float, ...]:
    if settings.theta_obs_sweep is None:
        assert settings.fixed_theta_obs_deg is not None
        return (_normalize_generated_coordinate(float(settings.fixed_theta_obs_deg)),)

    sweep = settings.theta_obs_sweep
    if not settings.adaptive_inclination_edge_steps:
        return _simple_sweep_values(sweep)

    band = float(settings.adaptive_inclination_edge_polar_band_deg)
    boundaries = tuple(
        boundary
        for boundary in (band, 180.0 - band)
        if 0.0 < boundary < 180.0
    )
    return _generate_adaptive_sweep_values(
        sweep,
        step_scale=settings.adaptive_inclination_edge_step_scale,
        boundaries=boundaries,
        should_refine=lambda theta_deg: _should_refine_for_inclination_deg(theta_deg, settings),
    )


def _resolve_spin_values(
    settings: GenerationSettings,
) -> tuple[float, ...]:
    if not settings.adaptive_edge_steps:
        return _simple_sweep_values(settings.spin_sweep)

    threshold = float(settings.adaptive_spin_edge_abs_threshold)
    boundaries = tuple(
        boundary
        for boundary in (-threshold, threshold)
        if -1.0 < boundary < 1.0
    )

    def should_refine(spin_value: float) -> bool:
        tolerance = 1e-12 * max(1.0, abs(spin_value))
        return abs(float(spin_value)) >= threshold - tolerance

    return _generate_adaptive_sweep_values(
        settings.spin_sweep,
        step_scale=settings.adaptive_spin_edge_step_scale,
        boundaries=boundaries,
        should_refine=should_refine,
    )


def resolve_generation_plan(settings: GenerationSettings) -> GenerationPlan:
    lines: list[SpinLinePlan] = []
    total_points = 0
    spin_values = _resolve_spin_values(settings)
    for inclination_deg in _resolve_inclination_values(settings):
        lines.append(
            SpinLinePlan(
                inclination_deg=float(inclination_deg),
                spin_values=spin_values,
            )
        )
        total_points += len(spin_values)
    return GenerationPlan(lines=tuple(lines), total_points=int(total_points))


def _resolve_generation_worker_count(
    total_points: int,
    requested_worker_count: int | None = None,
) -> int:
    total = int(max(0, total_points))
    if total <= 1:
        return 1

    if requested_worker_count is not None:
        requested = int(requested_worker_count)
        if requested < 1:
            raise ValueError("Worker count must be >= 1 when specified.")
        return max(1, min(total, int(requested)))

    raw_value = os.environ.get(GENERATION_WORKERS_ENV_VAR)
    if raw_value is None or not raw_value.strip():
        requested = os.cpu_count() or 1
    else:
        try:
            requested = int(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"{GENERATION_WORKERS_ENV_VAR} must be an integer >= 1 when set."
            ) from exc
        if requested < 1:
            raise ValueError(f"{GENERATION_WORKERS_ENV_VAR} must be >= 1 when set.")

    return max(1, min(total, int(requested)))


def _generation_sample_label(task: GenerationSampleTask) -> str:
    return f"spin={task.spin:g}, theta_obs={task.inclination_deg:g} deg"


def _build_generation_sample_tasks(plan: GenerationPlan) -> tuple[GenerationSampleTask, ...]:
    tasks: list[GenerationSampleTask] = []
    sample_index = 0
    total_lines = max(1, len(plan.lines))
    for line_index, line in enumerate(plan.lines, start=1):
        total_spins_for_line = max(1, len(line.spin_values))
        for spin_index, spin in enumerate(line.spin_values, start=1):
            tasks.append(
                GenerationSampleTask(
                    sample_index=sample_index,
                    line_index=line_index,
                    total_lines=total_lines,
                    spin_index=spin_index,
                    total_spins_for_line=total_spins_for_line,
                    spin=float(spin),
                    inclination_deg=float(line.inclination_deg),
                )
            )
            sample_index += 1
    return tuple(tasks)


def _resolve_generation_chunk_size(total_points: int, worker_count: int) -> int:
    total = int(max(0, total_points))
    workers = max(1, int(worker_count))
    if total <= 0:
        return _MIN_GENERATION_CHUNK_SIZE
    target = int(math.ceil(total / max(1, workers * _DEFAULT_GENERATION_CHUNK_GROUPS_PER_WORKER)))
    return max(
        _MIN_GENERATION_CHUNK_SIZE,
        min(_MAX_GENERATION_CHUNK_SIZE, max(1, target)),
    )


def _build_generation_chunk_tasks(
    tasks: tuple[GenerationSampleTask, ...],
    worker_count: int,
) -> tuple[GenerationChunkTask, ...]:
    if not tasks:
        return ()
    chunk_size = _resolve_generation_chunk_size(len(tasks), worker_count)
    chunk_tasks: list[GenerationChunkTask] = []
    for chunk_index, start in enumerate(range(0, len(tasks), chunk_size)):
        chunk_tasks.append(
            GenerationChunkTask(
                chunk_index=chunk_index,
                tasks=tasks[start : start + chunk_size],
            )
        )
    return tuple(chunk_tasks)


def _init_generation_worker(
    M: float,
    r_obs: float,
    measurement_specs: tuple[GenerationMeasurementSpec, ...],
) -> None:
    global _GENERATION_WORKER_CONTEXT
    _GENERATION_WORKER_CONTEXT = {
        "M": float(M),
        "r_obs": float(r_obs),
        "measurement_specs": tuple(measurement_specs),
        "measurement_plan": _runtime_measurement_plan(tuple(measurement_specs)),
    }


def _generation_worker_context() -> dict[str, Any]:
    if _GENERATION_WORKER_CONTEXT is None:
        raise RuntimeError("Generation worker context was not initialized.")
    return _GENERATION_WORKER_CONTEXT


def _measure_generation_sample(
    task: GenerationSampleTask,
    measurement_plan: tuple[tuple[str, dict[str, Any]], ...],
    *,
    M: float,
    r_obs: float,
    initial_shadow_seed: ShadowSolveSeed | None = None,
) -> tuple[np.ndarray, ShadowSolveSeed | None]:
    theta_obs_rad = math.radians(task.inclination_deg)
    metric = build_metric_for_spin(M, task.spin)
    measurements = AsymmetryMeasurements(
        metric,
        r_obs,
        theta_obs_rad,
        initial_shadow_seed=initial_shadow_seed,
    )
    return (
        measurements.measure_flat_values(measurement_plan),
        measurements.export_shadow_solve_seed(),
    )


def _compute_generation_chunk_value_matrix(
    chunk_task: GenerationChunkTask,
    measurement_plan: tuple[tuple[str, dict[str, Any]], ...],
    quantity_count: int,
    *,
    M: float,
    r_obs: float,
) -> np.ndarray:
    """Return one chunk's measurement matrix using a shared runtime plan."""
    value_matrix = np.empty((len(chunk_task.tasks), int(quantity_count)), dtype=np.float64)
    previous_shadow_seed: ShadowSolveSeed | None = None

    for row_index, task in enumerate(chunk_task.tasks):
        row_values, previous_shadow_seed = _measure_generation_sample(
            task,
            measurement_plan,
            M=float(M),
            r_obs=float(r_obs),
            initial_shadow_seed=previous_shadow_seed,
        )
        value_matrix[row_index, :] = row_values

    return value_matrix


def _compute_generation_chunk(
    chunk_task: GenerationChunkTask,
) -> GenerationChunkResult:
    context = _generation_worker_context()
    measurement_specs = context["measurement_specs"]
    measurement_plan = context["measurement_plan"]
    quantity_count = int(sum(len(spec.output_names) for spec in measurement_specs))
    compute_start = perf_counter()
    value_matrix = _compute_generation_chunk_value_matrix(
        chunk_task,
        measurement_plan,
        quantity_count,
        M=float(context["M"]),
        r_obs=float(context["r_obs"]),
    )
    compute_seconds = perf_counter() - compute_start

    return GenerationChunkResult(
        chunk_index=int(chunk_task.chunk_index),
        tasks=tuple(chunk_task.tasks),
        value_matrix=value_matrix,
        compute_seconds=float(compute_seconds),
    )


def _generation_chunk_result_arrays(
    chunk_result: GenerationChunkResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return one chunk result as column arrays for batched HDF5 writes."""
    row_count = len(chunk_result.tasks)
    sample_indices = np.fromiter(
        (int(task.sample_index) for task in chunk_result.tasks),
        dtype=np.int64,
        count=row_count,
    )
    spins = np.fromiter(
        (float(task.spin) for task in chunk_result.tasks),
        dtype=np.float64,
        count=row_count,
    )
    inclination_degs = np.fromiter(
        (float(task.inclination_deg) for task in chunk_result.tasks),
        dtype=np.float64,
        count=row_count,
    )
    quantity_matrix = np.ascontiguousarray(chunk_result.value_matrix, dtype=np.float64)
    return sample_indices, spins, inclination_degs, quantity_matrix

def _validate_sampling_config(sampling: SamplingConfig) -> SamplingConfig:
    if sampling.n_bracket_samples < 2:
        raise ValueError("Asymmetry bracket samples must be >= 2.")
    if sampling.tol <= 0.0:
        raise ValueError("Asymmetry alpha tolerance must be > 0.")
    if sampling.max_iter <= 0:
        raise ValueError("Asymmetry max iterations must be > 0.")
    if sampling.n_theta_samples < 8:
        raise ValueError("Asymmetry theta samples must be >= 8.")
    if sampling.n_refine_samples < 3:
        raise ValueError("Asymmetry refine samples must be >= 3.")
    if sampling.refine_levels < 0:
        raise ValueError("Asymmetry refine levels must be >= 0.")
    if sampling.n_boundary_samples < 8:
        raise ValueError("Asymmetry boundary samples must be >= 8.")
    return sampling


def _resolve_measurement_tokens(raw_value: str) -> tuple[str, tuple[str, ...]]:
    measurement_names = AsymmetryMeasurements.measurement_names()
    by_index = {str(index): name for index, name in enumerate(measurement_names, start=1)}
    by_name = {name.lower(): name for name in measurement_names}

    tokens = str(raw_value).replace(",", " ").split()
    if not tokens:
        tokens = [MEASUREMENT_ALL_TOKEN]

    resolved: list[str] = []
    for token in tokens:
        normalized = token.strip().replace("-", "_").lower()
        if not normalized:
            continue
        if normalized == MEASUREMENT_ALL_TOKEN:
            if len(tokens) > 1:
                raise ValueError(
                    "Use either 'all' or a list of specific measurements, not both."
                )
            return "all", measurement_names
        if normalized in by_index:
            resolved.append(by_index[normalized])
            continue
        if normalized in by_name:
            resolved.append(by_name[normalized])
            continue
        available = ", ".join(
            f"{index}:{name}"
            for index, name in enumerate(measurement_names, start=1)
        )
        raise ValueError(
            f"Unknown asymmetry measurement {token!r}. Available: {available}, or 'all'."
        )

    deduped = tuple(dict.fromkeys(resolved))
    if not deduped:
        return "all", measurement_names
    return "selected", deduped


def parse_state(state: dict[str, Any]) -> tuple[GenerationSettings | None, str | None]:
    def parse_float(key: str, label: str) -> tuple[float | None, str | None]:
        raw = str(state[key]).strip()
        try:
            return float(raw), None
        except ValueError:
            return None, f"{label} must be a number."

    def parse_int(key: str, label: str) -> tuple[int | None, str | None]:
        raw = str(state[key]).strip()
        try:
            return int(raw), None
        except ValueError:
            return None, f"{label} must be an integer."

    generation_mode = str(state["generation_mode"]).strip().lower()
    if generation_mode not in GENERATION_MODE_CHOICES:
        return None, "Generation mode must be either 'spin_only' or 'spin_and_inclination'."

    asymmetry_profile = str(state["asymmetry_profile"]).strip().lower()
    if asymmetry_profile not in ASYMMETRY_PERFORMANCE_PROFILE_NAMES:
        return None, "Asymmetry profile must be one of: quick, normal, accurate, ultra_accurate."

    asymmetry_circle_fit = str(state["asymmetry_circle_fit"]).strip().lower()
    if asymmetry_circle_fit not in ASYMMETRY_CIRCLE_FIT_CHOICES:
        return None, "Asymmetry circle fit must be either 'global' or 'cardinal'."

    m, err = parse_float("M", "BH mass")
    if err:
        return None, err
    r_obs, err = parse_float("r_obs", "Observer radius")
    if err:
        return None, err
    spin_start, err = parse_float("spin_start", "Spin start")
    if err:
        return None, err
    spin_end, err = parse_float("spin_end", "Spin end")
    if err:
        return None, err
    spin_step, err = parse_float("spin_step", "Spin step")
    if err:
        return None, err
    adaptive_spin_edge_abs_threshold, err = parse_float(
        "adaptive_spin_edge_abs_threshold",
        "Spin edge threshold",
    )
    if err:
        return None, err
    adaptive_spin_edge_step_scale, err = parse_float(
        "adaptive_spin_edge_step_scale",
        "Spin edge step scale",
    )
    if err:
        return None, err
    adaptive_inclination_edge_polar_band_deg, err = parse_float(
        "adaptive_inclination_edge_polar_band_deg",
        "Inclination polar band",
    )
    if err:
        return None, err
    adaptive_inclination_edge_step_scale, err = parse_float(
        "adaptive_inclination_edge_step_scale",
        "Inclination edge step scale",
    )
    if err:
        return None, err
    fixed_theta_obs_deg, err = parse_float("fixed_theta_obs_deg", "Observer inclination")
    if err:
        return None, err
    theta_obs_start_deg, err = parse_float("theta_obs_start_deg", "Inclination start")
    if err:
        return None, err
    theta_obs_end_deg, err = parse_float("theta_obs_end_deg", "Inclination end")
    if err:
        return None, err
    theta_obs_step_deg, err = parse_float("theta_obs_step_deg", "Inclination step")
    if err:
        return None, err
    worker_count, err = _parse_worker_count(state.get("worker_count", "auto"))
    if err:
        return None, err

    assert m is not None and r_obs is not None
    assert spin_start is not None and spin_end is not None and spin_step is not None
    assert adaptive_spin_edge_abs_threshold is not None
    assert adaptive_spin_edge_step_scale is not None
    assert adaptive_inclination_edge_polar_band_deg is not None
    assert adaptive_inclination_edge_step_scale is not None
    assert fixed_theta_obs_deg is not None
    assert theta_obs_start_deg is not None and theta_obs_end_deg is not None
    assert theta_obs_step_deg is not None

    adaptive_edge_steps = bool(state.get("adaptive_edge_steps", False))
    adaptive_inclination_edge_steps = bool(state.get("adaptive_inclination_edge_steps", False))
    debug = bool(state.get("debug", False))
    benchmark = bool(state.get("benchmark", False))
    live_benchmark = benchmark and bool(state.get("live_benchmark", False))

    try:
        m = _validate_positive("BH mass", m)
        r_obs = _validate_positive("Observer radius", r_obs)
        spin_sweep = _validate_sweep(
            "Spin sweep",
            spin_start,
            spin_end,
            spin_step,
            min_value=-1.0,
            max_value=1.0,
        )
        adaptive_spin_edge_abs_threshold = _validate_closed_interval(
            "Spin edge threshold",
            adaptive_spin_edge_abs_threshold,
            0.0,
            1.0,
        )
        adaptive_spin_edge_step_scale = _validate_closed_interval(
            "Spin edge step scale",
            adaptive_spin_edge_step_scale,
            0.0,
            1.0,
        )
        if adaptive_edge_steps and math.isclose(
            adaptive_spin_edge_step_scale,
            0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Spin edge step scale must be > 0 when adaptive spin edge steps are enabled."
            )
        adaptive_inclination_edge_polar_band_deg = _validate_closed_interval(
            "Inclination polar band",
            adaptive_inclination_edge_polar_band_deg,
            0.0,
            90.0,
        )
        adaptive_inclination_edge_step_scale = _validate_closed_interval(
            "Inclination edge step scale",
            adaptive_inclination_edge_step_scale,
            0.0,
            1.0,
        )
        if adaptive_inclination_edge_steps and math.isclose(
            adaptive_inclination_edge_step_scale,
            0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Inclination edge step scale must be > 0 when adaptive inclination edge steps are enabled."
            )
    except ValueError as exc:
        return None, str(exc)

    try:
        selection_mode, asymmetry_measurements = _resolve_measurement_tokens(
            str(state["asymmetry_measurements"])
        )
    except ValueError as exc:
        return None, str(exc)

    asymmetry_advanced_tuning = bool(state["asymmetry_advanced_tuning"])
    if asymmetry_advanced_tuning:
        n_bracket_samples, err = parse_int("asymmetry_n_bracket_samples", "Bracket samples")
        if err:
            return None, err
        asymmetry_tol, err = parse_float("asymmetry_tol", "Alpha tolerance")
        if err:
            return None, err
        asymmetry_max_iter, err = parse_int("asymmetry_max_iter", "Max iterations")
        if err:
            return None, err
        asymmetry_n_theta_samples, err = parse_int("asymmetry_n_theta_samples", "Theta samples")
        if err:
            return None, err
        asymmetry_n_refine_samples, err = parse_int(
            "asymmetry_n_refine_samples",
            "Refine samples",
        )
        if err:
            return None, err
        asymmetry_refine_levels, err = parse_int("asymmetry_refine_levels", "Refine levels")
        if err:
            return None, err
        asymmetry_n_boundary_samples, err = parse_int(
            "asymmetry_n_boundary_samples",
            "Boundary samples",
        )
        if err:
            return None, err
    else:
        preset = asymmetry_performance_profile_preset(asymmetry_profile)
        n_bracket_samples = int(preset["n_bracket_samples"])
        asymmetry_tol = float(preset["tol"])
        asymmetry_max_iter = int(preset["max_iter"])
        asymmetry_n_theta_samples = int(preset["n_theta_samples"])
        asymmetry_n_refine_samples = int(preset["n_refine_samples"])
        asymmetry_refine_levels = int(preset["refine_levels"])
        asymmetry_n_boundary_samples = int(preset["n_boundary_samples"])

    assert n_bracket_samples is not None and asymmetry_tol is not None
    assert asymmetry_max_iter is not None and asymmetry_n_theta_samples is not None
    assert asymmetry_n_refine_samples is not None and asymmetry_refine_levels is not None
    assert asymmetry_n_boundary_samples is not None

    try:
        sampling = _validate_sampling_config(
            SamplingConfig(
                profile=asymmetry_profile,
                advanced_tuning=asymmetry_advanced_tuning,
                circle_fit=asymmetry_circle_fit,
                n_bracket_samples=int(n_bracket_samples),
                tol=float(asymmetry_tol),
                max_iter=int(asymmetry_max_iter),
                n_theta_samples=int(asymmetry_n_theta_samples),
                n_refine_samples=int(asymmetry_n_refine_samples),
                refine_levels=int(asymmetry_refine_levels),
                n_boundary_samples=int(asymmetry_n_boundary_samples),
            )
        )
    except ValueError as exc:
        return None, str(exc)

    fixed_theta_value: float | None = None
    theta_sweep_value: SweepRange | None = None
    resolved_adaptive_inclination_edge_steps = adaptive_inclination_edge_steps
    try:
        if generation_mode == "spin_only":
            fixed_theta_value = _validate_inclination(
                "Observer inclination",
                fixed_theta_obs_deg,
            )
            if _sweep_includes_nonzero(spin_sweep) and math.isclose(
                math.sin(math.radians(fixed_theta_value)),
                0.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "Observer inclination cannot be exactly 0 or 180 degrees when the spin sweep includes nonzero values."
                )
            resolved_adaptive_inclination_edge_steps = False
        else:
            theta_sweep_value = _validate_sweep(
                "Observer inclination sweep",
                theta_obs_start_deg,
                theta_obs_end_deg,
                theta_obs_step_deg,
                min_value=0.0,
                max_value=180.0,
                unit_suffix=" deg",
            )
            if _sweep_includes_nonzero(spin_sweep) and (
                _sweep_hits_value(theta_sweep_value, 0.0)
                or _sweep_hits_value(theta_sweep_value, 180.0)
            ):
                raise ValueError(
                    "Observer-inclination sweeps cannot include 0 or 180 degrees when the spin sweep includes nonzero values."
                )
    except ValueError as exc:
        return None, str(exc)

    return GenerationSettings(
        run_root=DEFAULT_RUN_ROOT,
        M=m,
        r_obs=r_obs,
        generation_mode=generation_mode,
        debug=debug,
        benchmark=benchmark,
        live_benchmark=live_benchmark,
        worker_count=worker_count,
        spin_sweep=spin_sweep,
        adaptive_edge_steps=adaptive_edge_steps,
        adaptive_spin_edge_abs_threshold=adaptive_spin_edge_abs_threshold,
        adaptive_spin_edge_step_scale=adaptive_spin_edge_step_scale,
        adaptive_inclination_edge_steps=resolved_adaptive_inclination_edge_steps,
        adaptive_inclination_edge_polar_band_deg=adaptive_inclination_edge_polar_band_deg,
        adaptive_inclination_edge_step_scale=adaptive_inclination_edge_step_scale,
        fixed_theta_obs_deg=fixed_theta_value,
        theta_obs_sweep=theta_sweep_value,
        asymmetry_selection_mode=selection_mode,
        asymmetry_measurements=asymmetry_measurements,
        sampling=sampling,
    ), None


def _sweep_to_dict(sweep: SweepRange) -> dict[str, float | int]:
    return {
        "start": float(sweep.start),
        "end": float(sweep.end),
        "step": float(sweep.step),
        "planned_count": int(_sweep_point_count(sweep)),
    }


def build_settings_document(settings: GenerationSettings) -> dict[str, object]:
    plan = resolve_generation_plan(settings)
    inclination_payload: dict[str, object]
    if settings.theta_obs_sweep is None:
        inclination_payload = {
            "mode": "fixed",
            "fixed_deg": float(settings.fixed_theta_obs_deg),
            "sweep": None,
            "planned_count": 1,
        }
    else:
        inclination_values = [line.inclination_deg for line in plan.lines]
        inclination_payload = {
            "mode": "sweep",
            "fixed_deg": None,
            "sweep": _sweep_to_dict(settings.theta_obs_sweep),
            "planned_count": len(inclination_values),
            "resolved_values_deg": inclination_values,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "generator": "scripts/gen_asym_data.py",
        "run_storage": {
            "root_dir": str(settings.run_root),
        },
        "black_hole": {
            "M": float(settings.M),
            "r_obs": float(settings.r_obs),
            "r_obs_units": "M",
        },
        "generation": {
            "mode": settings.generation_mode,
            "debug": bool(settings.debug),
            "benchmarking": {
                "enabled": bool(settings.benchmark),
                "live_progress": bool(settings.live_benchmark),
            },
            "workers": {
                "selection": "auto" if settings.worker_count is None else "manual",
                "requested_count": (
                    None if settings.worker_count is None else int(settings.worker_count)
                ),
                "environment_variable": GENERATION_WORKERS_ENV_VAR,
            },
            "spin_sweep": _sweep_to_dict(settings.spin_sweep),
            "adaptive_edge_steps": {
                "spin": {
                    "enabled": bool(settings.adaptive_edge_steps),
                    "abs_threshold": float(settings.adaptive_spin_edge_abs_threshold),
                    "step_scale": float(settings.adaptive_spin_edge_step_scale),
                },
                "observer_inclination": {
                    "enabled": bool(settings.adaptive_inclination_edge_steps),
                    "polar_band_deg": float(settings.adaptive_inclination_edge_polar_band_deg),
                    "step_scale": float(settings.adaptive_inclination_edge_step_scale),
                },
            },
            "observer_inclination": inclination_payload,
            "planned_total_points": int(plan.total_points),
        },
        "asymmetry_measurements": {
            "selection_mode": settings.asymmetry_selection_mode,
            "names": list(settings.asymmetry_measurements),
        },
        "sampling": {
            "profile": settings.sampling.profile,
            "advanced_tuning": bool(settings.sampling.advanced_tuning),
            "circle_fit": settings.sampling.circle_fit,
            "n_bracket_samples": int(settings.sampling.n_bracket_samples),
            "tol": float(settings.sampling.tol),
            "max_iter": int(settings.sampling.max_iter),
            "n_theta_samples": int(settings.sampling.n_theta_samples),
            "n_refine_samples": int(settings.sampling.n_refine_samples),
            "refine_levels": int(settings.sampling.refine_levels),
            "n_boundary_samples": int(settings.sampling.n_boundary_samples),
        },
    }


def parse_run_directory_index(name: str) -> int | None:
    if not str(name).startswith(RUN_DIRECTORY_PREFIX):
        return None
    suffix = str(name)[len(RUN_DIRECTORY_PREFIX) :]
    if not suffix.isdigit():
        return None
    return int(suffix)


def next_run_index(run_root: Path) -> int:
    if not run_root.is_dir():
        return 1
    max_index = 0
    for child in run_root.iterdir():
        if not child.is_dir():
            continue
        index = parse_run_directory_index(child.name)
        if index is not None:
            max_index = max(max_index, index)
    return max_index + 1


def run_directory_for_index(run_root: Path, run_index: int) -> Path:
    return run_root / f"{RUN_DIRECTORY_PREFIX}{run_index}"


def write_json_document(path: Path, document: dict[str, object]) -> None:
    with path.open("w", encoding="ascii") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def write_text_lines(path: Path, lines: list[str]) -> None:
    with path.open("w", encoding="ascii") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def benchmark_summary_path(run_dir: Path) -> Path:
    return run_dir / RUN_BENCHMARK_FILENAME


def _benchmarking_enabled_for_document(document: dict[str, object]) -> bool:
    generation = document.get("generation", {})
    if not isinstance(generation, dict):
        return False
    benchmarking = generation.get("benchmarking", {})
    if not isinstance(benchmarking, dict):
        return False
    return bool(benchmarking.get("enabled", False))


def _saved_benchmark_summary(document: dict[str, object]) -> dict[str, object] | None:
    outputs = document.get("outputs", {})
    if not isinstance(outputs, dict):
        return None
    benchmark_summary = outputs.get("benchmark")
    if not isinstance(benchmark_summary, dict):
        return None
    return dict(benchmark_summary)


def write_benchmark_summary_report(
    saved_run: SavedRun,
    *,
    status: str,
    benchmark_summary: dict[str, object] | None = None,
    error: str | None = None,
) -> None:
    report_lines = [
        f"Run #{saved_run.run_index}",
        f"Status: {status}",
        f"Run folder: {saved_run.run_dir}",
        f"Settings file: {saved_run.settings_path.name}",
        f"Data file: {RUN_DATA_FILENAME}",
    ]
    if error:
        report_lines.append(f"Error: {error}")
    report_lines.append("")
    if benchmark_summary is None:
        report_lines.append("Benchmark summary unavailable.")
    else:
        report_lines.extend(generation_benchmark_summary_lines(benchmark_summary))
    write_text_lines(benchmark_summary_path(saved_run.run_dir), report_lines)


def update_saved_run_document(
    saved_run: SavedRun,
    *,
    status: str,
    data_path: Path | None = None,
    completed_points: int | None = None,
    quantity_names: tuple[str, ...] | None = None,
    benchmark_summary: dict[str, object] | None = None,
    error: str | None = None,
) -> None:
    document = saved_run.document
    run_storage = document.setdefault("run_storage", {})
    outputs = document.setdefault("outputs", {})
    outputs["status"] = str(status)
    benchmark_file_path: Path | None = None
    if _benchmarking_enabled_for_document(document):
        benchmark_file_path = benchmark_summary_path(saved_run.run_dir)
        run_storage["benchmark_file"] = str(benchmark_file_path)
        outputs["benchmark_file"] = str(benchmark_file_path)
    if data_path is not None:
        run_storage["data_file"] = str(data_path)
        outputs["data_file"] = str(data_path)
    if completed_points is not None:
        outputs["completed_points"] = int(completed_points)
    if quantity_names is not None:
        outputs["quantity_names"] = list(quantity_names)
    effective_benchmark_summary = _saved_benchmark_summary(document)
    if benchmark_summary is not None:
        effective_benchmark_summary = dict(benchmark_summary)
        outputs["benchmark"] = dict(effective_benchmark_summary)
    if error:
        outputs["error"] = str(error)
    else:
        outputs.pop("error", None)
    write_json_document(saved_run.settings_path, document)
    if benchmark_file_path is not None and str(status) in {"completed", "failed"}:
        write_benchmark_summary_report(
            saved_run,
            status=str(status),
            benchmark_summary=effective_benchmark_summary,
            error=error,
        )


def save_settings(settings: GenerationSettings) -> SavedRun:
    run_root = settings.run_root
    run_root.mkdir(parents=True, exist_ok=True)
    run_index = next_run_index(run_root)
    run_dir = run_directory_for_index(run_root, run_index)
    while run_dir.exists():
        run_index += 1
        run_dir = run_directory_for_index(run_root, run_index)
    run_dir.mkdir(parents=True, exist_ok=False)

    settings_path = run_dir / RUN_SETTINGS_FILENAME
    document = build_settings_document(settings)
    document["run_storage"]["run_index"] = int(run_index)
    document["run_storage"]["run_dir"] = str(run_dir)
    document["run_storage"]["settings_file"] = str(settings_path)
    document["run_storage"]["data_file"] = str(run_dir / RUN_DATA_FILENAME)
    document["outputs"] = {
        "status": "pending",
        "data_file": str(run_dir / RUN_DATA_FILENAME),
        "completed_points": 0,
        "quantity_names": [],
    }
    if settings.benchmark:
        benchmark_path = benchmark_summary_path(run_dir)
        document["run_storage"]["benchmark_file"] = str(benchmark_path)
        document["outputs"]["benchmark_file"] = str(benchmark_path)
    write_json_document(settings_path, document)
    return SavedRun(
        run_index=run_index,
        run_root=run_root,
        run_dir=run_dir,
        settings_path=settings_path,
        document=document,
    )


def require_h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "h5py is required to write the generated HDF5 file. Install the project "
            "dependencies, then run the generator again."
        ) from exc
    return h5py


def _format_elapsed_seconds(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(float(seconds)):
        return "n/a"
    value = max(0.0, float(seconds))
    if value < 1.0:
        return f"{value:.3f} s"
    if value < 60.0:
        return f"{value:.2f} s"
    minutes = int(value // 60.0)
    remainder = value - 60.0 * minutes
    return f"{minutes:d}m {remainder:04.1f}s"


def _format_points_rate(points: int, seconds: float | None) -> str:
    if seconds is None or not math.isfinite(float(seconds)) or float(seconds) <= 0.0:
        return "n/a"
    rate = float(points) / float(seconds)
    if rate >= 1e6:
        return f"{rate / 1e6:.2f} Mpt/s"
    if rate >= 1e3:
        return f"{rate / 1e3:.2f} kpt/s"
    return f"{rate:.2f} pt/s"


def _generation_benchmark_summary_dict(
    *,
    settings: GenerationSettings,
    total_points: int,
    quantity_count: int,
    measurement_count: int,
    worker_count: int,
    total_chunks: int,
    planning_seconds: float,
    hdf5_open_seconds: float,
    generation_loop_seconds: float,
    chunk_compute_seconds_sum: float,
    write_seconds: float,
    finalize_seconds: float,
    total_seconds: float,
    chunk_compute_count: int,
    max_chunk_compute_seconds: float,
    write_chunk_count: int,
    max_chunk_write_seconds: float,
) -> dict[str, object]:
    mean_chunk_compute_seconds = (
        float(chunk_compute_seconds_sum) / float(chunk_compute_count)
        if chunk_compute_count > 0
        else 0.0
    )
    mean_chunk_write_seconds = (
        float(write_seconds) / float(write_chunk_count)
        if write_chunk_count > 0
        else 0.0
    )
    parallel_efficiency = (
        float(chunk_compute_seconds_sum) / float(generation_loop_seconds)
        if generation_loop_seconds > 0.0
        else 0.0
    )
    return {
        "enabled": bool(settings.benchmark),
        "live_progress": bool(settings.live_benchmark),
        "workers": int(worker_count),
        "chunks": int(total_chunks),
        "points": int(total_points),
        "measurements": int(measurement_count),
        "quantities": int(quantity_count),
        "timings_seconds": {
            "planning": float(planning_seconds),
            "open_hdf5": float(hdf5_open_seconds),
            "generation_loop": float(generation_loop_seconds),
            "chunk_compute_sum": float(chunk_compute_seconds_sum),
            "write_rows": float(write_seconds),
            "finalize": float(finalize_seconds),
            "total": float(total_seconds),
        },
        "chunk_timings_seconds": {
            "count": int(chunk_compute_count),
            "mean_compute": float(mean_chunk_compute_seconds),
            "max_compute": float(max_chunk_compute_seconds),
            "write_count": int(write_chunk_count),
            "mean_write": float(mean_chunk_write_seconds),
            "max_write": float(max_chunk_write_seconds),
        },
        "throughput": {
            "overall_points_per_second": (
                float(total_points) / float(total_seconds) if total_seconds > 0.0 else 0.0
            ),
            "generation_loop_points_per_second": (
                float(total_points) / float(generation_loop_seconds)
                if generation_loop_seconds > 0.0
                else 0.0
            ),
            "write_points_per_second": (
                float(total_points) / float(write_seconds) if write_seconds > 0.0 else 0.0
            ),
            "parallel_efficiency_estimate": float(parallel_efficiency),
        },
    }


def generation_benchmark_summary_lines(benchmark_summary: dict[str, object]) -> list[str]:
    timings = dict(benchmark_summary.get("timings_seconds", {}))
    chunk_timings = dict(benchmark_summary.get("chunk_timings_seconds", {}))
    throughput = dict(benchmark_summary.get("throughput", {}))
    total_points = int(benchmark_summary.get("points", 0))
    total_seconds = float(timings.get("total", 0.0))
    return [
        "Generation benchmark",
        f"  workers: {int(benchmark_summary.get('workers', 0))}",
        f"  chunks: {int(benchmark_summary.get('chunks', 0))}",
        f"  points: {total_points}",
        f"  measurements: {int(benchmark_summary.get('measurements', 0))}",
        f"  quantities: {int(benchmark_summary.get('quantities', 0))}",
        f"  {'planning':<24}{float(timings.get('planning', 0.0)):>10.3f} s",
        f"  {'open_hdf5':<24}{float(timings.get('open_hdf5', 0.0)):>10.3f} s",
        f"  {'generation_loop':<24}{float(timings.get('generation_loop', 0.0)):>10.3f} s",
        f"  {'chunk_compute_sum':<24}{float(timings.get('chunk_compute_sum', 0.0)):>10.3f} s",
        f"  {'write_rows':<24}{float(timings.get('write_rows', 0.0)):>10.3f} s",
        f"  {'finalize':<24}{float(timings.get('finalize', 0.0)):>10.3f} s",
        f"  {'total':<24}{total_seconds:>10.3f} s",
        f"  {'overall_throughput':<24}{_format_points_rate(total_points, total_seconds):>10}",
        (
            f"  {'loop_throughput':<24}"
            f"{_format_points_rate(total_points, float(timings.get('generation_loop', 0.0))):>10}"
        ),
        (
            f"  {'write_throughput':<24}"
            f"{_format_points_rate(total_points, float(timings.get('write_rows', 0.0))):>10}"
        ),
        (
            f"  {'parallel_eff_est':<24}"
            f"{float(throughput.get('parallel_efficiency_estimate', 0.0)):>10.2f}x"
        ),
        (
            f"  {'mean_chunk_compute':<24}"
            f"{float(chunk_timings.get('mean_compute', 0.0)):>10.3f} s"
        ),
        (
            f"  {'max_chunk_compute':<24}"
            f"{float(chunk_timings.get('max_compute', 0.0)):>10.3f} s"
        ),
        (
            f"  {'mean_chunk_write':<24}"
            f"{float(chunk_timings.get('mean_write', 0.0)):>10.3f} s"
        ),
        (
            f"  {'max_chunk_write':<24}"
            f"{float(chunk_timings.get('max_write', 0.0)):>10.3f} s"
        ),
    ]


class StreamingHDF5Writer:
    """Append generated sample rows to an HDF5 file."""

    def __init__(
        self,
        output_path: Path,
        settings_document: dict[str, object],
        quantity_names: tuple[str, ...],
        *,
        planned_row_count: int | None = None,
    ) -> None:
        if not quantity_names:
            raise RuntimeError("Need at least one quantity name before opening the HDF5 writer.")

        h5py = require_h5py()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.output_path = output_path
        self.quantity_names = tuple(quantity_names)
        self._handle = h5py.File(output_path, "w", libver="latest")
        self._handle.attrs["schema_version"] = SCHEMA_VERSION
        self._handle.attrs["generator"] = "scripts/gen_asym_data.py"
        self._handle.attrs["run_status"] = "running"
        self._handle.attrs["row_count"] = 0
        self._handle.attrs["quantity_count"] = int(len(self.quantity_names))
        self._handle.attrs["quantity_names_json"] = json.dumps(list(self.quantity_names))
        self._handle.create_dataset(
            "settings_json",
            data=np.bytes_(json.dumps(settings_document, sort_keys=True)),
        )

        group = self._handle.create_group("data")
        chunk_size = int(
            min(
                1024,
                max(
                    16,
                    int(planned_row_count) if planned_row_count is not None else 256,
                ),
            )
        )
        dataset_kwargs = {
            "shape": (0,),
            "maxshape": (None,),
            "chunks": (chunk_size,),
        }
        self._datasets: dict[str, Any] = {
            "sample_index": group.create_dataset("sample_index", dtype=np.int64, **dataset_kwargs),
            "spin": group.create_dataset("spin", dtype=np.float64, **dataset_kwargs),
            "inclination_deg": group.create_dataset(
                "inclination_deg",
                dtype=np.float64,
                **dataset_kwargs,
            ),
        }
        for quantity_name in self.quantity_names:
            self._datasets[quantity_name] = group.create_dataset(
                quantity_name,
                dtype=np.float64,
                **dataset_kwargs,
            )
        self._quantity_datasets = tuple(
            self._datasets[quantity_name]
            for quantity_name in self.quantity_names
        )
        self.row_count = 0
        self.flush()
        self._enable_live_reading()

    def _resize_datasets(self, row_count: int) -> None:
        for dataset in self._datasets.values():
            dataset.resize((int(row_count),))

    def _enable_live_reading(self) -> None:
        try:
            self._handle.swmr_mode = True
        except Exception as exc:
            raise RuntimeError(
                "Could not enable live HDF5 reading for this run. Install a build of h5py/HDF5 "
                "with SWMR support before using render_graph.py alongside the generator."
            ) from exc

    def append_numeric_rows(
        self,
        *,
        sample_indices: np.ndarray,
        spins: np.ndarray,
        inclination_degs: np.ndarray,
        quantity_matrix: np.ndarray,
        flush: bool = True,
    ) -> int:
        """Append one contiguous batch of numeric rows."""
        sample_indices = np.ascontiguousarray(
            np.asarray(sample_indices, dtype=np.int64).reshape(-1)
        )
        spins = np.ascontiguousarray(np.asarray(spins, dtype=np.float64).reshape(-1))
        inclination_degs = np.ascontiguousarray(
            np.asarray(inclination_degs, dtype=np.float64).reshape(-1)
        )
        quantity_matrix = np.ascontiguousarray(np.asarray(quantity_matrix, dtype=np.float64))
        if quantity_matrix.ndim == 1:
            quantity_matrix = quantity_matrix.reshape(1, -1)

        row_count = int(sample_indices.shape[0])
        if row_count == 0:
            return int(self.row_count)
        if spins.shape[0] != row_count or inclination_degs.shape[0] != row_count:
            raise RuntimeError("Row metadata arrays must all have the same length.")
        if quantity_matrix.shape != (row_count, len(self.quantity_names)):
            raise RuntimeError(
                "Quantity matrix shape does not match the configured HDF5 quantity schema."
            )

        row_index = int(self.row_count)
        next_size = row_index + row_count
        self._resize_datasets(next_size)
        row_slice = slice(row_index, next_size)

        self._datasets["sample_index"][row_slice] = sample_indices
        self._datasets["spin"][row_slice] = spins
        self._datasets["inclination_deg"][row_slice] = inclination_degs
        for index, dataset in enumerate(self._quantity_datasets):
            dataset[row_slice] = quantity_matrix[:, index]

        self.row_count = next_size
        self._handle.attrs["row_count"] = int(self.row_count)
        if flush:
            self.flush()
        return row_index

    def append_numeric_row(
        self,
        *,
        sample_index: int,
        spin: float,
        inclination_deg: float,
        quantity_values: np.ndarray,
    ) -> int:
        quantity_values = np.ascontiguousarray(
            np.asarray(quantity_values, dtype=np.float64).reshape(-1)
        )
        if quantity_values.shape[0] != len(self.quantity_names):
            raise RuntimeError(
                "Quantity vector length does not match the configured HDF5 quantity schema."
            )
        return self.append_numeric_rows(
            sample_indices=np.asarray([int(sample_index)], dtype=np.int64),
            spins=np.asarray([float(spin)], dtype=np.float64),
            inclination_degs=np.asarray([float(inclination_deg)], dtype=np.float64),
            quantity_matrix=quantity_values.reshape(1, -1),
        )

    def append_row(self, row: dict[str, float]) -> int:
        quantity_values = np.asarray(
            [float(row[quantity_name]) for quantity_name in self.quantity_names],
            dtype=np.float64,
        )
        return self.append_numeric_row(
            sample_index=int(row.get("sample_index", self.row_count)),
            spin=float(row["spin"]),
            inclination_deg=float(row["inclination_deg"]),
            quantity_values=quantity_values,
        )

    def set_run_status(self, status: str) -> None:
        self._handle.attrs["run_status"] = str(status)
        self.flush()

    def flush(self) -> None:
        self._handle.flush()

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is None:
            return
        try:
            handle.flush()
        finally:
            handle.close()
            self._handle = None


def generate_run_data(
    saved_run: SavedRun,
    settings: GenerationSettings,
    *,
    progress_callback: Callable[[str, float, list[str]], None] | None = None,
) -> GeneratedRun:
    def emit_progress(message: str, fraction: float, details: list[str]) -> None:
        if progress_callback is None:
            return
        progress_callback(
            message,
            max(0.0, min(1.0, float(fraction))),
            list(details),
        )

    benchmark_enabled = bool(settings.benchmark)
    live_benchmark_enabled = benchmark_enabled and bool(settings.live_benchmark)
    total_start = perf_counter() if benchmark_enabled else None
    require_h5py()
    plan = resolve_generation_plan(settings)
    measurement_specs, quantity_names = _resolve_generation_measurement_plan(settings)
    measurement_names = tuple(spec.name for spec in measurement_specs)
    measurement_plan = _runtime_measurement_plan(measurement_specs)
    quantity_count = len(quantity_names)
    point_work_units = max(
        1.0,
        float(sum(spec.estimated_work_units for spec in measurement_specs)),
    )
    total_measurement_work = max(
        1.0,
        float(point_work_units * plan.total_points),
    )
    r_obs = float(settings.r_obs) * float(settings.M)
    tasks = _build_generation_sample_tasks(plan)
    worker_count = _resolve_generation_worker_count(
        plan.total_points,
        settings.worker_count,
    )
    chunk_tasks = _build_generation_chunk_tasks(tasks, worker_count)
    hdf5_writer: StreamingHDF5Writer | None = None
    completed_points = 0
    completed_compute_points = 0
    completed_work = 0.0
    data_path = saved_run.run_dir / RUN_DATA_FILENAME
    debug_enabled = bool(settings.debug)
    planning_seconds = 0.0
    hdf5_open_seconds = 0.0
    generation_loop_seconds = 0.0
    write_seconds = 0.0
    finalize_seconds = 0.0
    chunk_compute_seconds_sum = 0.0
    chunk_compute_count = 0
    max_chunk_compute_seconds = 0.0
    write_chunk_count = 0
    max_chunk_write_seconds = 0.0
    last_chunk_compute_seconds: float | None = None
    last_chunk_write_seconds: float | None = None
    sampling_summary_line = (
        "Sampling: "
        f"profile={settings.sampling.profile}, "
        f"circle_fit={settings.sampling.circle_fit}, "
        f"bracket={settings.sampling.n_bracket_samples}, "
        f"tol={settings.sampling.tol:g}, "
        f"max_iter={settings.sampling.max_iter}, "
        f"theta={settings.sampling.n_theta_samples}, "
        f"refine_samples={settings.sampling.n_refine_samples}, "
        f"refine_levels={settings.sampling.refine_levels}, "
        f"boundary={settings.sampling.n_boundary_samples}"
    )

    def work_fraction() -> float:
        return 0.05 + 0.87 * min(1.0, completed_work / total_measurement_work)

    def live_benchmark_lines() -> list[str]:
        if not live_benchmark_enabled or total_start is None:
            return []
        elapsed = max(perf_counter() - total_start, 0.0)
        lines = [
            "Benchmark: On (live)",
            f"Elapsed: {_format_elapsed_seconds(elapsed)}",
            f"Computed throughput: {_format_points_rate(completed_compute_points, elapsed)}",
            f"Written throughput: {_format_points_rate(completed_points, elapsed)}",
        ]
        if completed_points > 0 and elapsed > 0.0:
            remaining_points = max(0, plan.total_points - completed_points)
            eta_seconds = float(remaining_points) / max(float(completed_points) / elapsed, 1e-12)
            lines.append(f"ETA: {_format_elapsed_seconds(eta_seconds)}")
        if chunk_compute_count > 0:
            lines.append(
                "Chunk compute: "
                f"last={_format_elapsed_seconds(last_chunk_compute_seconds)}, "
                f"mean={_format_elapsed_seconds(chunk_compute_seconds_sum / chunk_compute_count)}, "
                f"max={_format_elapsed_seconds(max_chunk_compute_seconds)}"
            )
        if write_chunk_count > 0:
            lines.append(
                "Chunk write: "
                f"last={_format_elapsed_seconds(last_chunk_write_seconds)}, "
                f"mean={_format_elapsed_seconds(write_seconds / write_chunk_count)}, "
                f"max={_format_elapsed_seconds(max_chunk_write_seconds)}"
            )
        if worker_count > 1 and generation_loop_seconds > 0.0:
            lines.append(
                "Parallel efficiency est: "
                f"{(chunk_compute_seconds_sum / generation_loop_seconds):.2f}x"
            )
        return lines

    def with_debug_lines(details: list[str], *debug_lines: str) -> list[str]:
        enriched_details = list(details)
        if live_benchmark_enabled:
            enriched_details.extend(live_benchmark_lines())
        if not debug_enabled:
            return enriched_details
        return enriched_details + [f"Debug mode: On"] + [line for line in debug_lines if line]

    def write_chunk_result(
        chunk_result: GenerationChunkResult,
        *,
        include_worker_count: bool,
    ) -> None:
        nonlocal completed_points
        nonlocal write_seconds
        nonlocal write_chunk_count
        nonlocal max_chunk_write_seconds
        nonlocal last_chunk_write_seconds
        assert hdf5_writer is not None

        sample_indices, spins, inclination_degs, quantity_matrix = _generation_chunk_result_arrays(
            chunk_result
        )
        first_task = chunk_result.tasks[0]
        last_task = chunk_result.tasks[-1]
        write_start = perf_counter() if benchmark_enabled else None
        hdf5_writer.append_numeric_rows(
            sample_indices=sample_indices,
            spins=spins,
            inclination_degs=inclination_degs,
            quantity_matrix=quantity_matrix,
        )
        if write_start is not None:
            last_chunk_write_seconds = perf_counter() - write_start
            write_seconds += float(last_chunk_write_seconds)
            write_chunk_count += 1
            max_chunk_write_seconds = max(
                float(max_chunk_write_seconds),
                float(last_chunk_write_seconds),
            )
        completed_points += len(chunk_result.tasks)
        latest_task = chunk_result.tasks[-1]

        update_saved_run_document(
            saved_run,
            status="running",
            data_path=data_path,
            completed_points=completed_points,
            quantity_names=quantity_names,
        )

        detail_lines = [f"Run folder: {saved_run.run_dir}"]
        if include_worker_count:
            detail_lines.append(f"Workers: {worker_count}")
        detail_lines.extend(
            [
                f"Completed points: {completed_points}/{plan.total_points}",
                (
                    f"Latest chunk: line {latest_task.line_index}/{latest_task.total_lines} | "
                    f"spin point {latest_task.spin_index}/{latest_task.total_spins_for_line}"
                ),
                f"Latest sample: {_generation_sample_label(latest_task)}",
                f"Chunk rows written: {len(chunk_result.tasks)}",
                f"Quantities: {len(quantity_names)}",
                f"Latest rows written to {data_path.name}",
            ]
        )
        emit_progress(
            "Completed chunk and wrote rows",
            work_fraction(),
            with_debug_lines(
                detail_lines,
                f"Chunk index: {chunk_result.chunk_index + 1}/{max(1, len(chunk_tasks))}",
                (
                    f"Chunk sample indices: {first_task.sample_index}"
                    f" -> {last_task.sample_index}"
                ),
                f"Quantity columns: {', '.join(quantity_names)}",
            ),
        )

    update_saved_run_document(
        saved_run,
        status="running",
        data_path=data_path,
        completed_points=0,
        quantity_names=quantity_names,
    )
    if total_start is not None:
        planning_seconds = perf_counter() - total_start

    emit_progress(
        "Preparing generation plan",
        0.02,
        with_debug_lines(
            [
                f"Run folder: {saved_run.run_dir}",
                f"Inclination lines: {len(plan.lines)}",
                f"Total points: {plan.total_points}",
                f"Measurements: {', '.join(measurement_names)}",
                f"Workers: {worker_count}",
                f"Chunks: {max(1, len(chunk_tasks))}",
            ],
            sampling_summary_line,
            f"Quantity columns: {', '.join(quantity_names)}",
            f"Estimated work units per point: {point_work_units:g}",
            f"Estimated total work units: {total_measurement_work:g}",
        ),
    )

    try:
        hdf5_open_start = perf_counter() if benchmark_enabled else None
        hdf5_writer = StreamingHDF5Writer(
            data_path,
            saved_run.document,
            quantity_names,
            planned_row_count=plan.total_points,
        )
        if hdf5_open_start is not None:
            hdf5_open_seconds = perf_counter() - hdf5_open_start
        generation_loop_start = perf_counter() if benchmark_enabled else None

        if worker_count <= 1:
            for chunk_task in chunk_tasks:
                first_task = chunk_task.tasks[0]
                last_task = chunk_task.tasks[-1]
                emit_progress(
                    "Generating asymmetry data",
                    work_fraction(),
                    with_debug_lines(
                        [
                            f"Run folder: {saved_run.run_dir}",
                            f"Completed points: {completed_points}/{plan.total_points}",
                            (
                                f"Current chunk: line {first_task.line_index}/{first_task.total_lines} | "
                                f"spin point {first_task.spin_index}/{first_task.total_spins_for_line}"
                            ),
                            (
                                f"Chunk range: {_generation_sample_label(first_task)}"
                                f" -> {_generation_sample_label(last_task)}"
                            ),
                            f"Chunk rows: {len(chunk_task.tasks)}",
                            f"Quantities: {len(quantity_names)}",
                        ],
                        f"Chunk index: {chunk_task.chunk_index + 1}/{max(1, len(chunk_tasks))}",
                        (
                        f"Chunk sample indices: {first_task.sample_index}"
                        f" -> {last_task.sample_index}"
                    ),
                    sampling_summary_line,
                ),
            )
                chunk_compute_start = perf_counter() if benchmark_enabled else None
                chunk_result = GenerationChunkResult(
                    chunk_index=int(chunk_task.chunk_index),
                    tasks=tuple(chunk_task.tasks),
                    value_matrix=_compute_generation_chunk_value_matrix(
                        chunk_task,
                        measurement_plan,
                        quantity_count,
                        M=float(settings.M),
                        r_obs=r_obs,
                    ),
                    compute_seconds=(
                        perf_counter() - chunk_compute_start
                        if chunk_compute_start is not None
                        else 0.0
                    ),
                )
                completed_compute_points += len(chunk_task.tasks)
                if benchmark_enabled:
                    chunk_compute_seconds_sum += float(chunk_result.compute_seconds)
                    chunk_compute_count += 1
                    max_chunk_compute_seconds = max(
                        float(max_chunk_compute_seconds),
                        float(chunk_result.compute_seconds),
                    )
                    last_chunk_compute_seconds = float(chunk_result.compute_seconds)
                completed_work = min(
                    total_measurement_work,
                    completed_work + len(chunk_task.tasks) * point_work_units,
                )
                write_chunk_result(
                    chunk_result,
                    include_worker_count=False,
                )
        else:
            emit_progress(
                "Launching parallel workers",
                0.03,
                with_debug_lines(
                    [
                        f"Run folder: {saved_run.run_dir}",
                        f"Workers: {worker_count}",
                        f"Total points: {plan.total_points}",
                        f"Chunks: {len(chunk_tasks)}",
                        f"Output file: {data_path.name}",
                    ],
                    sampling_summary_line,
                    f"Quantity columns: {', '.join(quantity_names)}",
                ),
            )

            pending_results: dict[int, GenerationChunkResult] = {}
            next_write_chunk_index = 0

            with ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_init_generation_worker,
                initargs=(float(settings.M), r_obs, measurement_specs),
            ) as executor:
                future_to_task = {
                    executor.submit(
                        _compute_generation_chunk,
                        chunk_task,
                    ): chunk_task
                    for chunk_task in chunk_tasks
                }

                for future in as_completed(future_to_task):
                    chunk_task = future_to_task[future]
                    result = future.result()
                    pending_results[result.chunk_index] = result
                    completed_compute_points += len(result.tasks)
                    if benchmark_enabled:
                        chunk_compute_seconds_sum += float(result.compute_seconds)
                        chunk_compute_count += 1
                        max_chunk_compute_seconds = max(
                            float(max_chunk_compute_seconds),
                            float(result.compute_seconds),
                        )
                        last_chunk_compute_seconds = float(result.compute_seconds)
                    completed_work = min(
                        total_measurement_work,
                        completed_compute_points * point_work_units,
                    )
                    latest_task = result.tasks[-1]

                    emit_progress(
                        "Computing asymmetry samples in parallel",
                        work_fraction(),
                        with_debug_lines(
                            [
                                f"Run folder: {saved_run.run_dir}",
                                f"Workers: {worker_count}",
                                f"Computed samples: {completed_compute_points}/{plan.total_points}",
                                (
                                    f"Completed chunk: line {latest_task.line_index}/{latest_task.total_lines} | "
                                    f"spin point {latest_task.spin_index}/{latest_task.total_spins_for_line}"
                                ),
                                f"Latest completed sample: {_generation_sample_label(latest_task)}",
                                f"Chunk rows: {len(chunk_task.tasks)}",
                            ],
                            f"Chunk index: {result.chunk_index + 1}/{max(1, len(chunk_tasks))}",
                            (
                                f"Chunk sample indices: {result.tasks[0].sample_index}"
                                f" -> {result.tasks[-1].sample_index}"
                            ),
                            f"Pending completed chunks waiting to write: {len(pending_results)}",
                            f"Next write chunk index: {next_write_chunk_index + 1}",
                        ),
                    )

                    while next_write_chunk_index in pending_results:
                        ready_result = pending_results.pop(next_write_chunk_index)
                        write_chunk_result(
                            ready_result,
                            include_worker_count=True,
                        )
                        next_write_chunk_index += 1
        if generation_loop_start is not None:
            generation_loop_seconds = perf_counter() - generation_loop_start

        emit_progress(
            "Finalizing data files",
            0.95,
            with_debug_lines(
                [
                    f"Run folder: {saved_run.run_dir}",
                    f"Output file: {data_path.name}",
                    f"Completed points: {completed_points}/{plan.total_points}",
                    f"Quantities: {len(quantity_names)}",
                ],
                f"Final HDF5 flush pending",
                f"Settings file: {saved_run.settings_path.name}",
            ),
        )
        finalize_start = perf_counter() if benchmark_enabled else None
        if hdf5_writer is not None:
            hdf5_writer.set_run_status("completed")
            hdf5_writer.flush()
        if finalize_start is not None:
            finalize_seconds = perf_counter() - finalize_start

        benchmark_summary = (
            _generation_benchmark_summary_dict(
                settings=settings,
                total_points=completed_points,
                quantity_count=len(quantity_names),
                measurement_count=len(measurement_names),
                worker_count=worker_count,
                total_chunks=len(chunk_tasks),
                planning_seconds=planning_seconds,
                hdf5_open_seconds=hdf5_open_seconds,
                generation_loop_seconds=generation_loop_seconds,
                chunk_compute_seconds_sum=chunk_compute_seconds_sum,
                write_seconds=write_seconds,
                finalize_seconds=finalize_seconds,
                total_seconds=(
                    perf_counter() - total_start if total_start is not None else 0.0
                ),
                chunk_compute_count=chunk_compute_count,
                max_chunk_compute_seconds=max_chunk_compute_seconds,
                write_chunk_count=write_chunk_count,
                max_chunk_write_seconds=max_chunk_write_seconds,
            )
            if benchmark_enabled
            else None
        )

        update_saved_run_document(
            saved_run,
            status="completed",
            data_path=data_path,
            completed_points=completed_points,
            quantity_names=quantity_names,
            benchmark_summary=benchmark_summary,
        )
        emit_progress(
            "Generation complete",
            1.0,
            with_debug_lines(
                [
                    f"Run folder: {saved_run.run_dir}",
                    f"HDF5 file: {data_path.name}",
                    f"Completed points: {completed_points}/{plan.total_points}",
                    f"Quantities: {len(quantity_names)}",
                ],
                sampling_summary_line,
            )
            + (
                generation_benchmark_summary_lines(benchmark_summary)
                if benchmark_summary is not None
                else []
            ),
        )
        return GeneratedRun(
            saved_run=saved_run,
            data_path=data_path,
            total_points=completed_points,
            quantity_names=quantity_names,
            benchmark_summary=benchmark_summary,
        )
    except Exception as exc:
        if hdf5_writer is not None:
            try:
                hdf5_writer.set_run_status("failed")
            except Exception:
                pass
        partial_benchmark_summary = (
            _generation_benchmark_summary_dict(
                settings=settings,
                total_points=completed_points,
                quantity_count=len(quantity_names),
                measurement_count=len(measurement_names),
                worker_count=worker_count,
                total_chunks=len(chunk_tasks),
                planning_seconds=planning_seconds,
                hdf5_open_seconds=hdf5_open_seconds,
                generation_loop_seconds=generation_loop_seconds,
                chunk_compute_seconds_sum=chunk_compute_seconds_sum,
                write_seconds=write_seconds,
                finalize_seconds=finalize_seconds,
                total_seconds=(
                    perf_counter() - total_start if total_start is not None else 0.0
                ),
                chunk_compute_count=chunk_compute_count,
                max_chunk_compute_seconds=max_chunk_compute_seconds,
                write_chunk_count=write_chunk_count,
                max_chunk_write_seconds=max_chunk_write_seconds,
            )
            if benchmark_enabled
            else None
        )
        update_saved_run_document(
            saved_run,
            status="failed",
            data_path=data_path,
            completed_points=completed_points,
            quantity_names=quantity_names,
            benchmark_summary=partial_benchmark_summary,
            error=str(exc),
        )
        raise
    finally:
        if hdf5_writer is not None:
            hdf5_writer.close()


def make_progress_bar(percent: float, width: int = 24) -> str:
    clamped = max(0.0, min(100.0, percent))
    filled = int(round((clamped / 100.0) * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def is_enter_key(ch: Any) -> bool:
    if isinstance(ch, str):
        return ch in ("\n", "\r")
    return ch in (10, 13, curses.KEY_ENTER)


def field_constraint_text(key: str) -> str:
    constraints = {
        "M": "Must be greater than 0.",
        "r_obs": "Must be greater than 0.",
        "generation_mode": "Allowed values: spin_only or spin_and_inclination.",
        "debug": "Turn this on when you want more detailed generation-stage progress and chunk diagnostics.",
        "benchmark": "Turn this on to collect and save timing summaries for the run.",
        "live_benchmark": "Benchmark mode only. Shows rolling throughput, ETA, and chunk timings while the run is still executing.",
        "worker_count": "Use 'auto' or enter an integer >= 1. Explicit values override the environment/default worker selection.",
        "spin_start": "Dimensionless Kerr spin a/M must remain in [-1, 1].",
        "spin_end": "Dimensionless Kerr spin a/M must remain in [-1, 1].",
        "spin_step": "Must be positive.",
        "adaptive_edge_steps": (
            "Turn this on to reveal the spin-edge threshold and spin-edge step scale."
        ),
        "adaptive_spin_edge_abs_threshold": "Must stay in [0, 1]. Spin refinement begins once |a| reaches this value.",
        "adaptive_spin_edge_step_scale": "Must stay in (0, 1] when enabled. Smaller values mean finer spin refinement.",
        "fixed_theta_obs_deg": "Must be in [0, 180] degrees and avoid exact poles if spin includes nonzero values.",
        "theta_obs_start_deg": "Must be in [0, 180] degrees.",
        "theta_obs_end_deg": "Must be in [0, 180] degrees.",
        "theta_obs_step_deg": "Must be positive.",
        "adaptive_inclination_edge_steps": (
            "Turn this on to reveal the inclination polar band and inclination-edge step scale."
        ),
        "adaptive_inclination_edge_polar_band_deg": "Must stay in [0, 90]. Inclination refinement applies within this many degrees of 0 or 180.",
        "adaptive_inclination_edge_step_scale": "Must stay in (0, 1] when enabled. Smaller values mean finer inclination refinement.",
        "asymmetry_measurements": "Use 'all' or comma-separated measurement names or indices.",
        "asymmetry_profile": "Allowed values: quick, normal, accurate, ultra_accurate.",
        "asymmetry_advanced_tuning": "Off means the selected profile controls the hidden numeric tuning values.",
        "asymmetry_circle_fit": "Allowed values: global or cardinal.",
        "asymmetry_n_bracket_samples": "Must be >= 2.",
        "asymmetry_tol": "Must be > 0.",
        "asymmetry_max_iter": "Must be > 0.",
        "asymmetry_n_theta_samples": "Must be >= 8.",
        "asymmetry_n_refine_samples": "Must be >= 3.",
        "asymmetry_refine_levels": "Must be >= 0.",
        "asymmetry_n_boundary_samples": "Must be >= 8.",
    }
    return constraints.get(key, "")


def format_display_value(spec: dict[str, str], values: dict[str, Any]) -> str:
    key = spec["key"]
    kind = spec["kind"]
    if kind == "bool":
        return "On" if values[key] else "Off"
    if kind == "choice":
        return format_choice_value(key, str(values[key]))
    return format_field_value(key, values[key])


def sweep_page_field_columns(
    state: dict[str, Any],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    spec_by_key = {spec["key"]: spec for spec in FIELD_SPECS}
    generation_mode = str(state.get("generation_mode", "spin_only")).strip().lower()

    left_keys = SPIN_SWEEP_BASE_PAGE_KEYS + (
        SPIN_ONLY_PAGE_KEYS
        if generation_mode == "spin_only"
        else SPIN_AND_INCLINATION_PAGE_KEYS
    )
    right_keys = SPIN_EDGE_CONTROL_FIELDS + (
        SPIN_EDGE_REFINEMENT_CONFIG_FIELDS
        if bool(state.get("adaptive_edge_steps", False))
        else ()
    )
    if generation_mode == "spin_and_inclination":
        right_keys += INCLINATION_EDGE_CONTROL_FIELDS + (
            INCLINATION_EDGE_REFINEMENT_CONFIG_FIELDS
            if bool(state.get("adaptive_inclination_edge_steps", False))
            else ()
        )

    return (
        [spec_by_key[key] for key in left_keys],
        [spec_by_key[key] for key in right_keys],
    )


def page_field_specs(page_key: str, state: dict[str, Any]) -> list[dict[str, str]]:
    spec_by_key = {spec["key"]: spec for spec in FIELD_SPECS}
    if page_key == "general":
        keys = (
            GENERAL_PAGE_KEYS[:-1]
            + (("live_benchmark",) if bool(state.get("benchmark", False)) else ())
            + GENERAL_PAGE_KEYS[-1:]
        )
    elif page_key == "sweep":
        left_items, right_items = sweep_page_field_columns(state)
        return left_items + right_items
    elif page_key == "asymmetry":
        keys = ASYMMETRY_PAGE_KEYS
    elif page_key == "tuning":
        if not bool(state.get("asymmetry_advanced_tuning", False)):
            return []
        keys = ADVANCED_TUNING_PAGE_KEYS
    else:
        return []
    return [spec_by_key[key] for key in keys]


def page_header_line(page_index: int) -> str:
    labels: list[str] = []
    for index, page in enumerate(PAGE_SPECS):
        label = f" {page['label']} "
        if index == page_index:
            labels.append(f"[{label}]")
        else:
            labels.append(f" {label} ")
    return " ".join(labels)


def build_run_review_lines(state: dict[str, Any]) -> list[str]:
    settings, err = parse_state(state)
    if err:
        return [
            f"Validation issue: {err}",
            "",
            "Fix the settings on the earlier pages before running.",
        ]

    assert settings is not None
    document = build_settings_document(settings)
    preview_run_index = next_run_index(settings.run_root)
    preview_run_dir = run_directory_for_index(settings.run_root, preview_run_index)
    inclination = document["generation"]["observer_inclination"]
    review_lines = [
        f"Run root: {settings.run_root}",
        f"Next run folder: {preview_run_dir}",
        f"Settings file: {preview_run_dir / RUN_SETTINGS_FILENAME}",
        f"Data file: {preview_run_dir / RUN_DATA_FILENAME}",
        *(
            [f"Benchmark file: {benchmark_summary_path(preview_run_dir)}"]
            if settings.benchmark
            else []
        ),
        f"Mode: {format_choice_value('generation_mode', settings.generation_mode)}",
        f"Debug progress: {'On' if settings.debug else 'Off'}",
        (
            "Benchmarking: On"
            + (" (live progress: On)" if settings.live_benchmark else " (live progress: Off)")
            if settings.benchmark
            else "Benchmarking: Off"
        ),
        f"Workers: {_format_worker_count_setting(settings.worker_count)}",
        (
            "Spin sweep: "
            f"{settings.spin_sweep.start:g} -> {settings.spin_sweep.end:g} "
            f"(step {settings.spin_sweep.step:g})"
        ),
    ]
    if settings.adaptive_edge_steps:
        review_lines.append(
            "Adaptive spin edge steps: On "
            f"(|a| >= {settings.adaptive_spin_edge_abs_threshold:g}, "
            f"step scale {settings.adaptive_spin_edge_step_scale:g}x)"
        )
    else:
        review_lines.append("Adaptive spin edge steps: Off")
    if inclination["mode"] == "fixed":
        review_lines.append(
            f"Observer inclination: fixed at {inclination['fixed_deg']:g} deg"
        )
    else:
        inclination_sweep = inclination["sweep"]
        review_lines.append(
            "Observer inclination: "
            f"{inclination_sweep['start']:g} -> {inclination_sweep['end']:g} deg "
            f"(step {inclination_sweep['step']:g} deg)"
        )
        if settings.adaptive_inclination_edge_steps:
            review_lines.append(
                "Adaptive inclination edge steps: On "
                f"(theta within {settings.adaptive_inclination_edge_polar_band_deg:g} deg of 0/180, "
                f"step scale {settings.adaptive_inclination_edge_step_scale:g}x)"
            )
        else:
            review_lines.append("Adaptive inclination edge steps: Off")
    review_lines.append(
        "Measurements: "
        + ", ".join(document["asymmetry_measurements"]["names"])
    )
    review_lines.append(
        f"Profile: {format_choice_value('asymmetry_profile', settings.sampling.profile)}"
    )
    review_lines.append(
        f"Planned grid points: {document['generation']['planned_total_points']}"
    )
    review_lines.append("")
    review_lines.append(
        "Run will generate the selected asymmetry quantities and stream each completed row into the HDF5 file immediately in that run folder."
    )
    return review_lines


def draw_progress_screen(
    stdscr: Any,
    title: str,
    message: str,
    percent: float,
    detail_lines: list[str] | None = None,
) -> None:
    stdscr.erase()
    h, w = stdscr.getmaxyx()
    bar_width = max(20, min(60, w - 12))
    title_y = max(1, (h // 2) - 4)
    message_y = title_y + 2
    bar_y = message_y + 2
    percent_y = bar_y + 2

    title_x = max(0, (w - len(title)) // 2)
    message_x = max(0, (w - len(message)) // 2)
    bar_x = max(2, (w - bar_width) // 2)

    stdscr.addnstr(title_y, title_x, title, w - 1, curses.A_BOLD)
    stdscr.addnstr(message_y, message_x, message, w - 1)
    stdscr.addnstr(bar_y, bar_x, make_progress_bar(percent, width=bar_width), bar_width + 2)
    percent_text = f"{percent:6.2f}%"
    stdscr.addnstr(percent_y, max(0, (w - len(percent_text)) // 2), percent_text, len(percent_text))
    if detail_lines:
        detail_row = percent_y + 2
        for detail in detail_lines[: max(0, h - detail_row - 1)]:
            if detail_row >= h - 1:
                break
            wrapped_lines = textwrap.wrap(detail, width=max(20, w - 8)) or [""]
            for line in wrapped_lines:
                if detail_row >= h - 1:
                    break
                stdscr.addnstr(detail_row, 4, line, w - 8)
                detail_row += 1
    stdscr.refresh()


def prompt_input(stdscr: Any, label: str, initial: str) -> str | None:
    h, w = stdscr.getmaxyx()
    prompt = f"{label}: "
    text = list(initial)
    pos = len(text)

    try:
        curses.curs_set(1)
    except curses.error:
        pass

    try:
        while True:
            stdscr.move(h - 1, 0)
            stdscr.clrtoeol()
            line = prompt + "".join(text)
            stdscr.addnstr(h - 1, 0, line, w - 1)
            cursor_x = min(len(prompt) + pos, w - 1)
            stdscr.move(h - 1, cursor_x)
            stdscr.refresh()

            ch = stdscr.get_wch()

            if is_enter_key(ch):
                return "".join(text).strip()

            if isinstance(ch, str):
                if ch == "\x1b":
                    return None
                if ch in ("\b", "\x7f", "\x08"):
                    if pos > 0:
                        del text[pos - 1]
                        pos -= 1
                    continue
                if ch.isprintable():
                    text.insert(pos, ch)
                    pos += 1
                continue

            if ch in (curses.KEY_BACKSPACE,):
                if pos > 0:
                    del text[pos - 1]
                    pos -= 1
            elif ch == curses.KEY_LEFT:
                pos = max(0, pos - 1)
            elif ch == curses.KEY_RIGHT:
                pos = min(len(text), pos + 1)
            elif ch == curses.KEY_DC:
                if pos < len(text):
                    del text[pos]
            elif ch == curses.KEY_HOME:
                pos = 0
            elif ch == curses.KEY_END:
                pos = len(text)
    finally:
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        stdscr.move(h - 1, 0)
        stdscr.clrtoeol()


def draw_page(
    stdscr: Any,
    state: dict[str, Any],
    defaults: dict[str, Any],
    page_index: int,
    cursor: int,
    status: str,
    last_result: str,
    run_review_lines: list[str] | None = None,
) -> None:
    stdscr.erase()
    h, w = stdscr.getmaxyx()

    if h < 24 or w < 72:
        msg = "Terminal too small (min: 72x24). Resize and try again."
        stdscr.addnstr(0, 0, msg, max(0, w - 1))
        stdscr.refresh()
        return

    current_page = PAGE_SPECS[page_index]
    stdscr.addnstr(0, 2, "Asymmetry Data CLI", w - 4, curses.A_BOLD)
    stdscr.addnstr(
        1,
        2,
        "Left/Right or h/l: page | Up/Down or j/k: move | Enter: edit/cycle/run | Space: toggle/cycle | q: quit",
        w - 4,
    )
    stdscr.addnstr(2, 2, page_header_line(page_index), w - 4)

    row = 4
    stdscr.addnstr(row, 2, current_page["label"], w - 4, curses.A_BOLD)
    row += 1
    for line in textwrap.wrap(current_page["description"], width=max(20, w - 4)):
        stdscr.addnstr(row, 2, line, w - 4)
        row += 1
    row += 1

    if current_page["key"] == "run":
        stdscr.addnstr(row, 2, "Review", w - 4, curses.A_BOLD)
        row += 1
        review_lines = build_run_review_lines(state) if run_review_lines is None else run_review_lines

        for line in review_lines:
            if row >= h - 8:
                break
            stdscr.addnstr(row, 2, line, w - 4)
            row += 1

        row = max(row + 1, h - 7)
        stdscr.hline(row - 1, 2, curses.ACS_HLINE, max(1, w - 4))
        stdscr.addnstr(row, 2, "Actions", w - 4, curses.A_BOLD)
        row += 1
        for index, action in enumerate(ACTION_SPECS):
            attr = curses.A_REVERSE if index == cursor else curses.A_NORMAL
            stdscr.addnstr(row, 4, action["label"], w - 8, attr)
            row += 1
    else:
        items = page_field_specs(current_page["key"], state)
        if not items:
            message = "Advanced tuning is currently Off. Turn it on on the Asymmetry page to edit these values."
            for line in textwrap.wrap(message, width=max(20, w - 4)):
                stdscr.addnstr(row, 2, line, w - 4)
                row += 1
        else:
            if current_page["key"] == "sweep":
                left_items, right_items = sweep_page_field_columns(state)
                selected = items[min(cursor, len(items) - 1)]
                left_width = max(28, (w - 8) // 2)
                right_x = min(w - 24, 2 + left_width + 4)
                right_width = max(20, w - right_x - 2)

                stdscr.addnstr(row, 2, "Normal Settings", left_width, curses.A_BOLD)
                stdscr.addnstr(row, right_x, "Adaptive Settings", right_width, curses.A_BOLD)
                row += 1
                list_top = row

                for index, spec in enumerate(left_items):
                    attr = (
                        curses.A_REVERSE
                        if spec["key"] == selected["key"]
                        else curses.A_NORMAL
                    )
                    label = spec["label"]
                    value = format_display_value(spec, state)
                    line = f"{label:<22} {value}"
                    stdscr.addnstr(list_top + index, 2, line, left_width, attr)

                for index, spec in enumerate(right_items):
                    attr = (
                        curses.A_REVERSE
                        if spec["key"] == selected["key"]
                        else curses.A_NORMAL
                    )
                    label = spec["label"]
                    value = format_display_value(spec, state)
                    line = f"{label:<22} {value}"
                    stdscr.addnstr(list_top + index, right_x, line, right_width, attr)

                row = list_top + max(len(left_items), len(right_items))
            else:
                for index, spec in enumerate(items):
                    attr = curses.A_REVERSE if index == cursor else curses.A_NORMAL
                    label = spec["label"]
                    value = format_display_value(spec, state)
                    line = f"{label:<22} {value}"
                    stdscr.addnstr(row, 2, line, w - 4, attr)
                    row += 1

            selected = items[min(cursor, len(items) - 1)]
            detail_row = max(row + 1, h // 2)
            stdscr.hline(detail_row - 1, 2, curses.ACS_HLINE, max(1, w - 4))
            stdscr.addnstr(detail_row, 2, selected["label"], w - 4, curses.A_BOLD)
            detail_row += 1
            default_text = format_display_value(selected, defaults)
            detail_lines = [
                f"Current value: {format_display_value(selected, state)}",
                f"Default value: {default_text}",
                f"Checks: {field_constraint_text(selected['key'])}",
                "",
                selected["description"],
            ]
            for block in detail_lines:
                if detail_row >= h - 3:
                    break
                if not block:
                    detail_row += 1
                    continue
                for line in textwrap.wrap(block, width=max(20, w - 4)):
                    if detail_row >= h - 3:
                        break
                    stdscr.addnstr(detail_row, 2, line, w - 4)
                    detail_row += 1

    if last_result:
        stdscr.hline(h - 3, 0, "-", w)
        stdscr.addnstr(h - 2, 2, last_result, w - 4)
    stdscr.hline(h - 1, 0, "-", w)
    stdscr.addnstr(h - 1, 2, status, w - 4)
    stdscr.refresh()


def run_save_flow(
    stdscr: Any,
    state: dict[str, Any],
) -> tuple[str, str]:
    settings, err = parse_state(state)
    if err:
        return f"Validation error: {err}", ""
    assert settings is not None

    title = "Run"
    draw_progress_screen(
        stdscr,
        title,
        "Validating settings",
        5.0,
        [
            f"Mode: {format_choice_value('generation_mode', settings.generation_mode)}",
            f"Debug progress: {'On' if settings.debug else 'Off'}",
            (
                "Benchmarking: On"
                + (" (live)" if settings.live_benchmark else "")
                if settings.benchmark
                else "Benchmarking: Off"
            ),
            f"Workers: {_format_worker_count_setting(settings.worker_count)}",
            f"Measurements: {', '.join(settings.asymmetry_measurements)}",
            f"Run root: {settings.run_root}",
        ],
    )
    curses.napms(90)

    try:
        saved_run = save_settings(settings)
    except Exception as exc:  # pragma: no cover - user-facing save path
        curses.napms(120)
        return f"Run failed: {exc}", ""

    draw_progress_screen(
        stdscr,
        title,
        "Created run folder",
        10.0,
        [
            f"Run folder: {saved_run.run_dir}",
            f"Settings file: {saved_run.settings_path.name}",
            f"Planned points: {saved_run.document['generation']['planned_total_points']}",
            *(
                [f"Benchmark file: {RUN_BENCHMARK_FILENAME}"]
                if settings.benchmark
                else []
            ),
            f"Debug progress: {'On' if settings.debug else 'Off'}",
            (
                "Benchmarking: On"
                + (" (live)" if settings.live_benchmark else "")
                if settings.benchmark
                else "Benchmarking: Off"
            ),
            f"Workers: {_format_worker_count_setting(settings.worker_count)}",
        ],
    )
    curses.napms(120)

    try:
        generated_run = generate_run_data(
            saved_run,
            settings,
            progress_callback=lambda message, fraction, details: draw_progress_screen(
                stdscr,
                title,
                message,
                10.0 + 90.0 * fraction,
                details,
            ),
        )
    except Exception as exc:  # pragma: no cover - user-facing runtime path
        update_saved_run_document(saved_run, status="failed", error=str(exc))
        curses.napms(120)
        return (
            f"Run #{saved_run.run_index} failed after creating {saved_run.run_dir}: {exc}",
            "",
        )

    result = (
        f"Completed run #{saved_run.run_index} in {saved_run.run_dir}. "
        f"Wrote {generated_run.data_path.name}."
    )
    summary = (
        f"Run #{saved_run.run_index} | "
        f"Generated {generated_run.total_points} points | "
        f"{len(generated_run.quantity_names)} quantities"
    )
    if generated_run.benchmark_summary is not None:
        total_seconds = float(
            dict(generated_run.benchmark_summary.get("timings_seconds", {})).get("total", 0.0)
        )
        result = (
            f"Completed run #{saved_run.run_index} in {saved_run.run_dir}. "
            f"Wrote {generated_run.data_path.name}. "
            f"Saved {RUN_BENCHMARK_FILENAME}. "
            f"Total {_format_elapsed_seconds(total_seconds)} "
            f"at {_format_points_rate(generated_run.total_points, total_seconds)}."
        )
        summary = (
            f"Run #{saved_run.run_index} | "
            f"Generated {generated_run.total_points} points | "
            f"{len(generated_run.quantity_names)} quantities | "
            f"{_format_elapsed_seconds(total_seconds)} total | "
            f"{_format_points_rate(generated_run.total_points, total_seconds)}"
        )
    return result, summary


def run_form(stdscr: Any, seed_state: dict[str, Any]) -> None:
    try:
        curses.curs_set(0)
    except curses.error:
        pass

    stdscr.keypad(True)
    state = dict(seed_state)
    defaults = default_state()
    page_index = 0
    cursor_positions = {page["key"]: 0 for page in PAGE_SPECS}
    status = "Edit the settings page by page, then go to Run."
    last_result = ""
    state_revision = 0
    cached_run_review_revision = -1
    cached_run_review_lines: list[str] = []

    def current_page_key() -> str:
        return PAGE_SPECS[page_index]["key"]

    def current_items() -> list[dict[str, str]]:
        if current_page_key() == "run":
            return []
        return page_field_specs(current_page_key(), state)

    def current_action_count() -> int:
        return len(ACTION_SPECS) if current_page_key() == "run" else len(current_items())

    def clamp_cursor() -> int:
        key = current_page_key()
        count = current_action_count()
        if count <= 0:
            cursor_positions[key] = 0
            return 0
        cursor_positions[key] = max(0, min(cursor_positions[key], count - 1))
        return cursor_positions[key]

    def mark_state_changed() -> None:
        nonlocal state_revision
        state_revision += 1

    def current_run_review_lines() -> list[str]:
        nonlocal cached_run_review_revision, cached_run_review_lines
        if cached_run_review_revision != state_revision:
            cached_run_review_lines = build_run_review_lines(state)
            cached_run_review_revision = state_revision
        return cached_run_review_lines

    def redraw() -> None:
        run_review_lines = current_run_review_lines() if current_page_key() == "run" else None
        draw_page(
            stdscr,
            state,
            defaults,
            page_index,
            clamp_cursor(),
            status,
            last_result,
            run_review_lines=run_review_lines,
        )

    while True:
        redraw()

        try:
            ch = stdscr.get_wch()
        except curses.error:
            continue

        if isinstance(ch, str) and ch.lower() == "q":
            return
        if ch in (curses.KEY_LEFT,) or ch == "h":
            page_index = (page_index - 1) % len(PAGE_SPECS)
            continue
        if ch in (curses.KEY_RIGHT, 9) or ch == "l":
            page_index = (page_index + 1) % len(PAGE_SPECS)
            continue
        count = current_action_count()
        if ch in (curses.KEY_UP,) or ch == "k":
            if count > 0:
                key = current_page_key()
                cursor_positions[key] = (clamp_cursor() - 1) % count
            continue
        if ch in (curses.KEY_DOWN,) or ch == "j":
            if count > 0:
                key = current_page_key()
                cursor_positions[key] = (clamp_cursor() + 1) % count
            continue
        if ch == " " and current_page_key() != "run":
            items = current_items()
            if items:
                spec = items[clamp_cursor()]
                key = spec["key"]
                if spec["kind"] == "bool":
                    status = apply_bool_change(state, key, spec["label"])
                    mark_state_changed()
                elif spec["kind"] == "choice":
                    status = apply_choice_change(state, key, spec["label"])
                    mark_state_changed()
            continue
        if not is_enter_key(ch):
            continue

        if current_page_key() != "run":
            items = current_items()
            if not items:
                status = "Turn on Advanced Tuning on the Asymmetry page to edit these values."
                continue
            spec = items[clamp_cursor()]
            key = spec["key"]
            kind = spec["kind"]
            if kind == "bool":
                status = apply_bool_change(state, key, spec["label"])
                mark_state_changed()
                continue
            if kind == "choice":
                status = apply_choice_change(state, key, spec["label"])
                mark_state_changed()
                continue

            new_value = prompt_input(stdscr, spec["label"], str(state[key]))
            if new_value is None:
                status = "Edit cancelled."
            else:
                state[key] = new_value
                status = f"Updated {spec['label']}."
                mark_state_changed()
            continue

        action_spec = ACTION_SPECS[clamp_cursor()]
        action_key = action_spec["key"]
        if action_key == "quit":
            return
        if action_key == "reset":
            state = dict(defaults)
            page_index = 0
            status = "Defaults restored."
            last_result = ""
            mark_state_changed()
            continue

        status, last_result = run_save_flow(stdscr, state)
        mark_state_changed()


def main() -> int:
    curses.wrapper(run_form, default_state())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
