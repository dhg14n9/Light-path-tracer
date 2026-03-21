#!/usr/bin/env python3
"""Interactive CLI wrapper for image_lens.py and stripes_lens.py.

This script provides a terminal form UI with editable value boxes and
checkbox toggles, plus a non-interactive mode for automation.
"""

from __future__ import annotations

import argparse
import curses
import os
import resource
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import matplotlib.image as mpimg
import numpy as np

from asymmetries import (
    ASYMMETRY_CIRCLE_FIT_CHOICES,
    ASYMMETRY_PERFORMANCE_PROFILE_NAMES,
    AsymmetryMeasurements,
    asymmetry_performance_profile_preset,
    filter_asymmetry_measurement_kwargs,
)
from image_lens import (
    _psi_to_cam_projection,
    build_alpha_lookup,
    precompute_final_alpha_lookup as precompute_final_alpha_lookup_image,
    precompute_final_alpha_lookup_2d as precompute_final_alpha_lookup_2d_image,
    render_lensed_image as render_lensed_input_image,
)
from metrics import Kerr, Schwarzschild
from solid_angle import kerr_shadow_solid_angle, schwarzschild_shadow_solid_angle
from stripes_lens import (
    DEFAULT_IMAGE_DIMENSION as DEFAULT_STRIPES_IMAGE_DIMENSION,
    DEFAULT_N_X,
    DEFAULT_N_Y,
    precompute_final_alpha_lookup as precompute_final_alpha_lookup_stripes,
    precompute_final_alpha_lookup_2d as precompute_final_alpha_lookup_2d_stripes,
    render_lensed_image as render_lensed_stripes_image,
)


@dataclass
class AppConfig:
    operation_mode: str = "lensing"
    source_mode: str = "image"
    shadow_metric: str = "kerr"
    solid_angle_profile: str = "normal"
    solid_angle_advanced_tuning: bool = False
    asymmetry_measurement: str = "all"
    asymmetry_profile: str = "normal"
    asymmetry_advanced_tuning: bool = False
    asymmetry_circle_fit: str = "global"
    input_image: str = "image.jpg"
    output_image: str = "lensed_image.png"
    width: int = DEFAULT_STRIPES_IMAGE_DIMENSION[1]
    height: int = DEFAULT_STRIPES_IMAGE_DIMENSION[0]
    n_x: int = DEFAULT_N_X
    n_y: int = DEFAULT_N_Y
    color_visualization: bool = False
    M: float = 1.0
    a: float = 0.0
    r_obs: float = 100.0
    theta_obs_deg: float = 90.0
    psi_y: float = 0.0
    psi_x: float = 0.0
    fov_v: float = 40.0
    solid_angle_base_n_alpha: int = 48
    solid_angle_base_n_theta: int = 96
    solid_angle_refine_levels: int = 4
    solid_angle_edge_samples: int = 4
    solid_angle_chunk: int = 50_000
    asymmetry_n_bracket_samples: int = 65
    asymmetry_tol: float = 1e-8
    asymmetry_max_iter: int = 64
    asymmetry_n_theta_samples: int = 181
    asymmetry_n_refine_samples: int = 17
    asymmetry_refine_levels: int = 4
    asymmetry_n_boundary_samples: int = 361
    debug: bool = False
    benchmark: bool = False
    wrap_outside_background_plane: bool = False


FIELD_SPECS: list[dict[str, str]] = [
    {
        "key": "operation_mode",
        "label": "Mode",
        "kind": "choice",
        "description": "Choose whether to render a lensed image, compute the black-hole shadow solid angle directly, or print registered asymmetry measurements for the current observer setup.",
    },
    {
        "key": "source_mode",
        "label": "Background Mode",
        "kind": "choice",
        "description": "Choose which background the black hole lenses. Image mode samples pixels from an input file, while stripes mode uses the procedural spherical stripe pattern.",
    },
    {
        "key": "shadow_metric",
        "label": "Shadow Metric",
        "kind": "choice",
        "description": "Choose which shadow solid-angle function to use. Schwarzschild uses the analytic circular-cap formula, while Kerr uses the adaptive numerical sky integral.",
    },
    {
        "key": "solid_angle_profile",
        "label": "SA Profile",
        "kind": "choice",
        "description": "Kerr shadow mode only. Load a preset for the adaptive Kerr solid-angle integrator. Quick is fastest and roughest, Normal is the default balance, Accurate spends more work on the shadow boundary, and Ultra Accurate pushes the integration settings further. You can still edit the individual solid-angle settings afterward.",
    },
    {
        "key": "solid_angle_advanced_tuning",
        "label": "SA Advanced",
        "kind": "bool",
        "description": "Shadow mode / Kerr only. Turn this on to reveal the individual solid-angle integration knobs. When off, the selected solid-angle profile alone controls those numeric settings.",
    },
    {
        "key": "asymmetry_measurement",
        "label": "Asymmetry Measure",
        "kind": "choice",
        "description": "Asymmetry mode only. Choose which registered asymmetry measurement to compute. 'All' prints every currently registered asymmetry measurement for the selected metric and observer.",
    },
    {
        "key": "asymmetry_profile",
        "label": "Asymmetry Profile",
        "kind": "choice",
        "description": "Asymmetry mode only. Load a preset for the asymmetry ray-sampling settings. Quick is fastest and roughest, Normal matches the current defaults, Accurate increases the search density, and Ultra Accurate pushes the sampling and tolerances further. You can still edit the individual asymmetry settings afterward.",
    },
    {
        "key": "asymmetry_advanced_tuning",
        "label": "Advanced Tuning",
        "kind": "bool",
        "description": "Asymmetry mode only. Turn this on to reveal the individual asymmetry performance knobs. When off, the selected asymmetry profile alone controls those numeric settings.",
    },
    {
        "key": "asymmetry_circle_fit",
        "label": "Circle Algorithm",
        "kind": "choice",
        "description": "Asymmetry mode only. Choose how the circularity measurement fits its reference circle. Global Least Squares uses many boundary samples, while Cardinal Points fits only the top, bottom, left, and right extrema.",
    },
    {
        "key": "input_image",
        "label": "Background Image",
        "kind": "text",
        "description": "Path to the source image used in image mode. The file must exist when you run in image mode, and this setting is ignored entirely in stripes mode.",
    },
    {
        "key": "output_image",
        "label": "Output Image",
        "kind": "text",
        "description": "Path where the rendered result will be written. Use a filename extension Matplotlib can save, such as .png.",
    },
    {
        "key": "width",
        "label": "Width",
        "kind": "int",
        "description": "Output width in pixels for stripes mode. Larger values produce more detail but increase tracing time and memory use.",
    },
    {
        "key": "height",
        "label": "Height",
        "kind": "int",
        "description": "Output height in pixels for stripes mode. Larger values produce more detail but increase tracing time and memory use.",
    },
    {
        "key": "n_x",
        "label": "Vertical Stripes",
        "kind": "int",
        "description": "Number of vertical stripe sectors wrapped around the background sphere. Higher values create finer longitudinal striping.",
    },
    {
        "key": "n_y",
        "label": "Horizontal Stripes",
        "kind": "int",
        "description": "Number of horizontal stripe sectors wrapped around the background sphere. Higher values create finer latitudinal striping.",
    },
    {
        "key": "color_visualization",
        "label": "Color Visualization",
        "kind": "bool",
        "description": "Stripes mode only. Replace the normal black-and-white stripe shading with the debugging visualization: stripe hits stay black and other escaped rays are colored by their traced final-theta quadrant.",
    },
    {
        "key": "M",
        "label": "BH Mass",
        "kind": "float",
        "description": "Black-hole mass in geometric units. This sets the internal scale of the metric and therefore rescales distances and the physical Kerr spin parameter.",
    },
    {
        "key": "a",
        "label": "BH Spin",
        "kind": "float",
        "description": "Dimensionless black-hole spin a/M. Use 0 for Schwarzschild, positive values for one spin orientation, and negative values for the opposite orientation.",
    },
    {
        "key": "r_obs",
        "label": "Observer Distance",
        "kind": "float",
        "description": "Observer distance from the black hole, measured in units of M. Larger values place the camera farther away and usually reduce the apparent distortion scale.",
    },
    {
        "key": "theta_obs_deg",
        "label": "Observer Inclination",
        "kind": "float",
        "description": "Observer polar angle in degrees, measured from the black-hole spin axis. Use 90 for the equatorial view; values near 0 or 180 look close to the poles.",
    },
    {
        "key": "psi_y",
        "label": "Vertical Offset",
        "kind": "float",
        "description": "Vertical screen offset of the black hole in degrees. Positive values move it upward on screen and negative values move it downward.",
    },
    {
        "key": "psi_x",
        "label": "Horizontal Offset",
        "kind": "float",
        "description": "Horizontal screen offset of the black hole in degrees. Positive values move it to the right and negative values move it to the left.",
    },
    {
        "key": "fov_v",
        "label": "Vertical FOV",
        "kind": "float",
        "description": "Vertical field of view in degrees. Larger values show more of the sky at once, while smaller values zoom in; the horizontal FOV is derived from this and the aspect ratio.",
    },
    {
        "key": "solid_angle_base_n_alpha",
        "label": "SA Base Alpha",
        "kind": "int",
        "description": "Kerr shadow mode only. Number of coarse cells in polar angle alpha before adaptive refinement begins.",
    },
    {
        "key": "solid_angle_base_n_theta",
        "label": "SA Base Theta",
        "kind": "int",
        "description": "Kerr shadow mode only. Number of coarse cells in azimuthal sky angle theta before adaptive refinement begins.",
    },
    {
        "key": "solid_angle_refine_levels",
        "label": "SA Refine Levels",
        "kind": "int",
        "description": "Kerr shadow mode only. Number of adaptive refinement rounds applied to cells that straddle the shadow boundary.",
    },
    {
        "key": "solid_angle_edge_samples",
        "label": "SA Edge Samples",
        "kind": "int",
        "description": "Kerr shadow mode only. Terminal subgrid resolution per side for the final mixed edge cells.",
    },
    {
        "key": "solid_angle_chunk",
        "label": "SA Chunk Size",
        "kind": "int",
        "description": "Kerr shadow mode only. Number of sample rays to trace per batch during solid-angle integration.",
    },
    {
        "key": "asymmetry_n_bracket_samples",
        "label": "Asym Bracket Samples",
        "kind": "int",
        "description": "Asymmetry mode only. Number of coarse alpha samples used to bracket the shadow edge at one screen azimuth before bisection refinement begins.",
    },
    {
        "key": "asymmetry_tol",
        "label": "Asym Alpha Tol",
        "kind": "float",
        "description": "Asymmetry mode only. Bisection stopping tolerance for alpha_crit(theta). Smaller values improve edge precision but increase tracing work.",
    },
    {
        "key": "asymmetry_max_iter",
        "label": "Asym Max Iter",
        "kind": "int",
        "description": "Asymmetry mode only. Maximum number of bisection iterations allowed while refining alpha_crit(theta).",
    },
    {
        "key": "asymmetry_n_theta_samples",
        "label": "Asym Theta Samples",
        "kind": "int",
        "description": "Asymmetry mode only. Number of coarse screen-azimuth samples used to locate left, right, top, and bottom shadow extrema.",
    },
    {
        "key": "asymmetry_n_refine_samples",
        "label": "Asym Refine Samples",
        "kind": "int",
        "description": "Asymmetry mode only. Number of samples per local refinement pass while honing in on a shadow extremum.",
    },
    {
        "key": "asymmetry_refine_levels",
        "label": "Asym Refine Levels",
        "kind": "int",
        "description": "Asymmetry mode only. Number of refinement rounds applied after the coarse extremum search.",
    },
    {
        "key": "asymmetry_n_boundary_samples",
        "label": "Asym Boundary Samples",
        "kind": "int",
        "description": "Asymmetry mode only. Number of full-boundary samples used by the circularity measurement and global circle fit.",
    },
    {
        "key": "debug",
        "label": "Debug",
        "kind": "bool",
        "description": "Show extra logging about the chosen metric, screen placement, tracing progress, and render behavior. Useful when diagnosing a setup.",
    },
    {
        "key": "benchmark",
        "label": "Benchmark",
        "kind": "bool",
        "description": "Print a timing summary after the render finishes, including per-stage runtimes and throughput estimates.",
    },
    {
        "key": "wrap_outside_background_plane",
        "label": "Wrap Outside Plane",
        "kind": "bool",
        "description": "Image mode only. When a ray misses or lands behind the image plane, wrap it back onto the image with modulo indexing instead of using the normal miss fallback colors.",
    },
]

ACTION_SPECS: list[dict[str, str]] = [
    {
        "key": "run",
        "label": "Run",
        "description": "Validate the current settings and run the selected mode. Lensing mode renders an image, shadow solid angle mode computes the shadow area on the observer sky, and asymmetry mode prints the selected asymmetry measurements.",
    },
    {
        "key": "reset",
        "label": "Reset",
        "description": "Restore every form field to its built-in default value so you can start from a clean baseline again.",
    },
    {
        "key": "quit",
        "label": "Quit",
        "description": "Leave the interactive CLI immediately and return to the terminal without starting a new render.",
    },
]

STAGE_SPECS: list[tuple[str, str, float]] = [
    ("prepare_source", "Prepare source", 0.05),
    ("build_lookup", "Build lookup", 0.05),
    ("precompute", "Trace rays", 0.75),
    ("render", "Render image", 0.10),
    ("save_image", "Save image", 0.05),
]
STAGE_LABELS = {key: label for key, label, _ in STAGE_SPECS}
OPERATION_MODE_LABELS = {
    "lensing": "Lensing",
    "shadow_solid_angle": "Shadow Solid Angle",
    "asymmetry_measurements": "Asymmetry Measurements",
}
SOURCE_MODE_LABELS = {
    "image": "Image",
    "stripes": "Stripes",
}
SHADOW_METRIC_LABELS = {
    "kerr": "Kerr",
    "schwarzschild": "Schwarzschild",
}
SOLID_ANGLE_PROFILE_LABELS = {
    "quick": "Quick",
    "normal": "Normal",
    "accurate": "Accurate",
    "ultra_accurate": "Ultra Accurate",
}
ASYMMETRY_PROFILE_LABELS = {
    "quick": "Quick",
    "normal": "Normal",
    "accurate": "Accurate",
    "ultra_accurate": "Ultra Accurate",
}
ASYMMETRY_CIRCLE_FIT_LABELS = {
    "global": "Global Least Squares",
    "cardinal": "Cardinal Points",
}
ASYMMETRY_MEASUREMENT_VALUES = ("all",) + AsymmetryMeasurements.measurement_names()
ASYMMETRY_MEASUREMENT_LABELS = {
    "all": "All",
}
CHOICE_VALUES: dict[str, tuple[str, ...]] = {
    "operation_mode": ("lensing", "shadow_solid_angle", "asymmetry_measurements"),
    "source_mode": ("image", "stripes"),
    "shadow_metric": ("kerr", "schwarzschild"),
    "solid_angle_profile": ("quick", "normal", "accurate", "ultra_accurate"),
    "asymmetry_measurement": ASYMMETRY_MEASUREMENT_VALUES,
    "asymmetry_profile": ASYMMETRY_PERFORMANCE_PROFILE_NAMES,
    "asymmetry_circle_fit": ASYMMETRY_CIRCLE_FIT_CHOICES,
}
KERR_SOLID_ANGLE_PROFILE_PRESETS: dict[str, dict[str, int]] = {
    "quick": {
        "solid_angle_base_n_alpha": 24,
        "solid_angle_base_n_theta": 48,
        "solid_angle_refine_levels": 2,
        "solid_angle_edge_samples": 2,
        "solid_angle_chunk": 20_000,
    },
    "normal": {
        "solid_angle_base_n_alpha": 48,
        "solid_angle_base_n_theta": 96,
        "solid_angle_refine_levels": 4,
        "solid_angle_edge_samples": 4,
        "solid_angle_chunk": 50_000,
    },
    "accurate": {
        "solid_angle_base_n_alpha": 96,
        "solid_angle_base_n_theta": 192,
        "solid_angle_refine_levels": 5,
        "solid_angle_edge_samples": 6,
        "solid_angle_chunk": 50_000,
    },
    "ultra_accurate": {
        "solid_angle_base_n_alpha": 160,
        "solid_angle_base_n_theta": 320,
        "solid_angle_refine_levels": 6,
        "solid_angle_edge_samples": 8,
        "solid_angle_chunk": 50_000,
    },
}
LENSING_ONLY_FIELDS = {
    "source_mode",
    "input_image",
    "output_image",
    "width",
    "height",
    "n_x",
    "n_y",
    "color_visualization",
    "psi_y",
    "psi_x",
    "fov_v",
    "wrap_outside_background_plane",
}
SHADOW_ONLY_FIELDS = {
    "shadow_metric",
    "solid_angle_profile",
    "solid_angle_advanced_tuning",
    "solid_angle_base_n_alpha",
    "solid_angle_base_n_theta",
    "solid_angle_refine_levels",
    "solid_angle_edge_samples",
    "solid_angle_chunk",
}
SHADOW_ADVANCED_TUNING_FIELDS = {
    "solid_angle_base_n_alpha",
    "solid_angle_base_n_theta",
    "solid_angle_refine_levels",
    "solid_angle_edge_samples",
    "solid_angle_chunk",
}
ASYMMETRY_ONLY_FIELDS = {
    "asymmetry_measurement",
    "asymmetry_profile",
    "asymmetry_advanced_tuning",
    "asymmetry_circle_fit",
    "asymmetry_n_bracket_samples",
    "asymmetry_tol",
    "asymmetry_max_iter",
    "asymmetry_n_theta_samples",
    "asymmetry_n_refine_samples",
    "asymmetry_refine_levels",
    "asymmetry_n_boundary_samples",
}
ASYMMETRY_ADVANCED_TUNING_FIELDS = {
    "asymmetry_n_bracket_samples",
    "asymmetry_tol",
    "asymmetry_max_iter",
    "asymmetry_n_theta_samples",
    "asymmetry_n_refine_samples",
    "asymmetry_refine_levels",
    "asymmetry_n_boundary_samples",
}
SHADOW_KERR_ONLY_FIELDS = {
    "solid_angle_profile",
    "solid_angle_advanced_tuning",
    "a",
    "theta_obs_deg",
    "solid_angle_base_n_alpha",
    "solid_angle_base_n_theta",
    "solid_angle_refine_levels",
    "solid_angle_edge_samples",
    "solid_angle_chunk",
}
IMAGE_MODE_ONLY_FIELDS = {
    "input_image",
    "wrap_outside_background_plane",
}
STRIPES_MODE_ONLY_FIELDS = {
    "width",
    "height",
    "n_x",
    "n_y",
    "color_visualization",
}


def toggle_choice(key: str, value: str) -> str:
    options = CHOICE_VALUES.get(key)
    if options is None:
        return value
    try:
        idx = options.index(value)
    except ValueError:
        return options[0]
    return options[(idx + 1) % len(options)]


def solid_angle_profile_preset(profile: str) -> dict[str, int]:
    preset = KERR_SOLID_ANGLE_PROFILE_PRESETS.get(profile)
    if preset is None:
        preset = KERR_SOLID_ANGLE_PROFILE_PRESETS["normal"]
    return dict(preset)


def apply_solid_angle_profile_to_state(state: dict[str, Any], profile: str) -> None:
    for key, value in solid_angle_profile_preset(profile).items():
        state[key] = f"{value:d}"


def apply_asymmetry_profile_to_state(state: dict[str, Any], profile: str) -> None:
    for key, value in asymmetry_performance_profile_preset(profile).items():
        state[f"asymmetry_{key}"] = f"{value:g}" if isinstance(value, float) else f"{value:d}"


def apply_choice_change(state: dict[str, Any], key: str, label: str | None = None) -> str:
    next_value = toggle_choice(key, str(state[key]))
    state[key] = next_value
    name = label if label is not None else key
    if key == "solid_angle_profile":
        apply_solid_angle_profile_to_state(state, next_value)
        return (
            f"{name} set to {format_choice_value(key, next_value)} "
            "and Kerr solid-angle settings updated."
        )
    if key == "asymmetry_profile":
        apply_asymmetry_profile_to_state(state, next_value)
        return (
            f"{name} set to {format_choice_value(key, next_value)} "
            "and asymmetry settings updated."
        )
    return f"{name} set to {format_choice_value(key, next_value)}"


def apply_bool_change(state: dict[str, Any], key: str, label: str | None = None) -> str:
    next_value = not bool(state[key])
    state[key] = next_value
    name = label if label is not None else key
    if key == "solid_angle_advanced_tuning" and not next_value:
        apply_solid_angle_profile_to_state(state, str(state.get("solid_angle_profile", "normal")))
        return f"{name} set to Off and solid-angle numeric settings reset to the profile preset"
    if key == "asymmetry_advanced_tuning" and not next_value:
        apply_asymmetry_profile_to_state(state, str(state.get("asymmetry_profile", "normal")))
        return f"{name} set to Off and asymmetry numeric settings reset to the profile preset"
    return f"{name} set to {next_value}"


def choice_label_map(key: str) -> dict[str, str]:
    if key == "operation_mode":
        return OPERATION_MODE_LABELS
    if key == "source_mode":
        return SOURCE_MODE_LABELS
    if key == "shadow_metric":
        return SHADOW_METRIC_LABELS
    if key == "solid_angle_profile":
        return SOLID_ANGLE_PROFILE_LABELS
    if key == "asymmetry_measurement":
        return ASYMMETRY_MEASUREMENT_LABELS
    if key == "asymmetry_profile":
        return ASYMMETRY_PROFILE_LABELS
    if key == "asymmetry_circle_fit":
        return ASYMMETRY_CIRCLE_FIT_LABELS
    return {}


def format_choice_value(key: str, value: str) -> str:
    return choice_label_map(key).get(value, value)


FIELD_UNIT_SUFFIXES: dict[str, str] = {
    "width": "px",
    "height": "px",
    "n_x": "stripes",
    "n_y": "stripes",
    "M": "M",
    "a": "a/M",
    "r_obs": "M",
    "theta_obs_deg": "deg",
    "psi_y": "deg",
    "psi_x": "deg",
    "fov_v": "deg",
    "solid_angle_base_n_alpha": "alpha bins",
    "solid_angle_base_n_theta": "theta bins",
    "solid_angle_refine_levels": "levels",
    "solid_angle_edge_samples": "samples/side",
    "solid_angle_chunk": "rays/chunk",
    "asymmetry_n_bracket_samples": "samples",
    "asymmetry_tol": "rad",
    "asymmetry_max_iter": "iters",
    "asymmetry_n_theta_samples": "theta bins",
    "asymmetry_n_refine_samples": "samples/level",
    "asymmetry_refine_levels": "levels",
    "asymmetry_n_boundary_samples": "samples",
}


def append_unit(value: Any, unit: str | None) -> str:
    text = str(value).strip()
    if not text or unit is None:
        return text
    if text == unit or text.endswith(f" {unit}"):
        return text
    return f"{text} {unit}"


def format_field_value(key: str, value: Any) -> str:
    return append_unit(value, FIELD_UNIT_SUFFIXES.get(key))


def format_ray_rate(ray_count: int, elapsed_seconds: float) -> str:
    rate = 0.0 if elapsed_seconds <= 0.0 else float(ray_count) / float(elapsed_seconds)
    if rate >= 1e6:
        return f"{rate / 1e6:.2f} Mray/s"
    if rate >= 1e3:
        return f"{rate / 1e3:.2f} kray/s"
    return f"{rate:.2f} ray/s"


def format_resolution_value(width: int, height: int) -> str:
    return f"{append_unit(width, 'px')} x {append_unit(height, 'px')}"


def get_visible_field_specs(state: dict[str, Any]) -> list[dict[str, str]]:
    operation_mode = str(state.get("operation_mode", "lensing")).strip().lower()
    source_mode = str(state.get("source_mode", "image")).strip().lower()
    shadow_metric = str(state.get("shadow_metric", "kerr")).strip().lower()
    solid_angle_advanced_tuning = bool(state.get("solid_angle_advanced_tuning", False))
    asymmetry_advanced_tuning = bool(state.get("asymmetry_advanced_tuning", False))
    visible: list[dict[str, str]] = []
    for spec in FIELD_SPECS:
        key = spec["key"]
        if operation_mode == "lensing":
            if key in SHADOW_ONLY_FIELDS or key in ASYMMETRY_ONLY_FIELDS:
                continue
            if source_mode == "image" and key in STRIPES_MODE_ONLY_FIELDS:
                continue
            if source_mode == "stripes" and key in IMAGE_MODE_ONLY_FIELDS:
                continue
        elif operation_mode == "shadow_solid_angle":
            if key in LENSING_ONLY_FIELDS or key in ASYMMETRY_ONLY_FIELDS:
                continue
            if shadow_metric == "schwarzschild" and key in SHADOW_KERR_ONLY_FIELDS:
                continue
            if not solid_angle_advanced_tuning and key in SHADOW_ADVANCED_TUNING_FIELDS:
                continue
        else:
            if key in LENSING_ONLY_FIELDS:
                continue
            if key in SHADOW_ONLY_FIELDS:
                continue
            if not asymmetry_advanced_tuning and key in ASYMMETRY_ADVANCED_TUNING_FIELDS:
                continue
        visible.append(spec)
    return visible


def make_progress_bar(percent: float, width: int = 30) -> str:
    clamped = max(0.0, min(100.0, percent))
    filled = int(round((clamped / 100.0) * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


class ProcessUsageSampler:
    def __init__(self) -> None:
        self._clock_ticks = os.sysconf("SC_CLK_TCK")
        self._page_size = os.sysconf("SC_PAGE_SIZE")
        self._last_wall = perf_counter()
        self._last_cpu = self._read_cpu_seconds()
        self.cpu_percent = 0.0
        self.ram_mb = self._read_rss_mb()

    def _read_cpu_seconds(self) -> float:
        stat_path = Path("/proc/self/stat")
        if stat_path.is_file():
            parts = stat_path.read_text(encoding="utf-8").split()
            if len(parts) > 14:
                utime = float(parts[13])
                stime = float(parts[14])
                return (utime + stime) / float(self._clock_ticks)
        return float(os.times().user + os.times().system)

    def _read_rss_mb(self) -> float:
        statm_path = Path("/proc/self/statm")
        if statm_path.is_file():
            parts = statm_path.read_text(encoding="utf-8").split()
            if len(parts) > 1:
                resident_pages = int(parts[1])
                return resident_pages * self._page_size / (1024.0 * 1024.0)
        # Linux fallback (max RSS may not be current RSS but is better than none)
        ru_maxrss_kb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return ru_maxrss_kb / 1024.0

    def sample(self) -> tuple[float, float]:
        now_wall = perf_counter()
        now_cpu = self._read_cpu_seconds()
        wall_dt = max(now_wall - self._last_wall, 1e-9)
        cpu_dt = max(now_cpu - self._last_cpu, 0.0)
        self.cpu_percent = 100.0 * cpu_dt / wall_dt
        self.ram_mb = self._read_rss_mb()
        self._last_wall = now_wall
        self._last_cpu = now_cpu
        return self.cpu_percent, self.ram_mb


def config_to_state(config: AppConfig) -> dict[str, Any]:
    return {
        "operation_mode": config.operation_mode,
        "source_mode": config.source_mode,
        "shadow_metric": config.shadow_metric,
        "solid_angle_profile": config.solid_angle_profile,
        "solid_angle_advanced_tuning": bool(config.solid_angle_advanced_tuning),
        "asymmetry_measurement": config.asymmetry_measurement,
        "asymmetry_profile": config.asymmetry_profile,
        "asymmetry_advanced_tuning": bool(config.asymmetry_advanced_tuning),
        "asymmetry_circle_fit": config.asymmetry_circle_fit,
        "input_image": config.input_image,
        "output_image": config.output_image,
        "width": f"{config.width:d}",
        "height": f"{config.height:d}",
        "n_x": f"{config.n_x:d}",
        "n_y": f"{config.n_y:d}",
        "color_visualization": bool(config.color_visualization),
        "M": f"{config.M:g}",
        "a": f"{config.a:g}",
        "r_obs": f"{config.r_obs:g}",
        "theta_obs_deg": f"{config.theta_obs_deg:g}",
        "psi_y": f"{config.psi_y:g}",
        "psi_x": f"{config.psi_x:g}",
        "fov_v": f"{config.fov_v:g}",
        "solid_angle_base_n_alpha": f"{config.solid_angle_base_n_alpha:d}",
        "solid_angle_base_n_theta": f"{config.solid_angle_base_n_theta:d}",
        "solid_angle_refine_levels": f"{config.solid_angle_refine_levels:d}",
        "solid_angle_edge_samples": f"{config.solid_angle_edge_samples:d}",
        "solid_angle_chunk": f"{config.solid_angle_chunk:d}",
        "asymmetry_n_bracket_samples": f"{config.asymmetry_n_bracket_samples:d}",
        "asymmetry_tol": f"{config.asymmetry_tol:g}",
        "asymmetry_max_iter": f"{config.asymmetry_max_iter:d}",
        "asymmetry_n_theta_samples": f"{config.asymmetry_n_theta_samples:d}",
        "asymmetry_n_refine_samples": f"{config.asymmetry_n_refine_samples:d}",
        "asymmetry_refine_levels": f"{config.asymmetry_refine_levels:d}",
        "asymmetry_n_boundary_samples": f"{config.asymmetry_n_boundary_samples:d}",
        "debug": bool(config.debug),
        "benchmark": bool(config.benchmark),
        "wrap_outside_background_plane": bool(config.wrap_outside_background_plane),
    }


def parse_state(
    state: dict[str, Any],
    require_source_assets: bool = True,
) -> tuple[AppConfig | None, str | None]:
    def parse_float(key: str, name: str) -> tuple[float | None, str | None]:
        raw = str(state[key]).strip()
        try:
            return float(raw), None
        except ValueError:
            return None, f"{name} must be a number"

    def parse_int(key: str, name: str) -> tuple[int | None, str | None]:
        raw = str(state[key]).strip()
        try:
            return int(raw), None
        except ValueError:
            return None, f"{name} must be an integer"

    operation_mode = str(state["operation_mode"]).strip().lower()
    if operation_mode not in {"lensing", "shadow_solid_angle", "asymmetry_measurements"}:
        return None, (
            "Mode must be one of 'lensing', 'shadow_solid_angle', "
            "or 'asymmetry_measurements'"
        )

    source_mode = str(state["source_mode"]).strip().lower()
    if source_mode not in {"image", "stripes"}:
        return None, "Source mode must be either 'image' or 'stripes'"
    shadow_metric = str(state["shadow_metric"]).strip().lower()
    if shadow_metric not in {"kerr", "schwarzschild"}:
        return None, "Shadow metric must be either 'kerr' or 'schwarzschild'"
    solid_angle_profile = str(state["solid_angle_profile"]).strip().lower()
    if solid_angle_profile not in {"quick", "normal", "accurate", "ultra_accurate"}:
        return None, (
            "Solid-angle profile must be one of 'quick', 'normal', "
            "'accurate', or 'ultra_accurate'"
        )
    solid_angle_advanced_tuning = bool(state["solid_angle_advanced_tuning"])
    asymmetry_measurement = str(state["asymmetry_measurement"]).strip().lower()
    if asymmetry_measurement not in ASYMMETRY_MEASUREMENT_VALUES:
        return None, (
            "Asymmetry measurement must be 'all' or one of: "
            + ", ".join(AsymmetryMeasurements.measurement_names())
        )
    asymmetry_profile = str(state["asymmetry_profile"]).strip().lower()
    if asymmetry_profile not in ASYMMETRY_PERFORMANCE_PROFILE_NAMES:
        return None, (
            "Asymmetry profile must be one of 'quick', 'normal', "
            "'accurate', or 'ultra_accurate'"
        )
    asymmetry_advanced_tuning = bool(state["asymmetry_advanced_tuning"])
    asymmetry_circle_fit = str(state["asymmetry_circle_fit"]).strip().lower()
    if asymmetry_circle_fit not in ASYMMETRY_CIRCLE_FIT_CHOICES:
        return None, (
            "Asymmetry circle algorithm must be either 'global' or 'cardinal'"
        )

    m, err = parse_float("M", "BH mass")
    if err:
        return None, err
    a, err = parse_float("a", "BH spin")
    if err:
        return None, err
    r_obs, err = parse_float("r_obs", "Observer distance")
    if err:
        return None, err
    theta_obs_deg, err = parse_float("theta_obs_deg", "Observer inclination")
    if err:
        return None, err
    psi_y, err = parse_float("psi_y", "Vertical offset")
    if err:
        return None, err
    psi_x, err = parse_float("psi_x", "Horizontal offset")
    if err:
        return None, err
    fov_v, err = parse_float("fov_v", "Vertical FOV")
    if err:
        return None, err
    width, err = parse_int("width", "Output width")
    if err:
        return None, err
    height, err = parse_int("height", "Output height")
    if err:
        return None, err
    n_x, err = parse_int("n_x", "Vertical stripe count")
    if err:
        return None, err
    n_y, err = parse_int("n_y", "Horizontal stripe count")
    if err:
        return None, err
    solid_angle_base_n_alpha: int | None = None
    solid_angle_base_n_theta: int | None = None
    solid_angle_refine_levels: int | None = None
    solid_angle_edge_samples: int | None = None
    solid_angle_chunk: int | None = None
    if operation_mode == "shadow_solid_angle" and shadow_metric == "kerr" and solid_angle_advanced_tuning:
        solid_angle_base_n_alpha, err = parse_int(
            "solid_angle_base_n_alpha", "Solid-angle base alpha"
        )
        if err:
            return None, err
        solid_angle_base_n_theta, err = parse_int(
            "solid_angle_base_n_theta", "Solid-angle base theta"
        )
        if err:
            return None, err
        solid_angle_refine_levels, err = parse_int(
            "solid_angle_refine_levels", "Solid-angle refine levels"
        )
        if err:
            return None, err
        solid_angle_edge_samples, err = parse_int(
            "solid_angle_edge_samples", "Solid-angle edge samples"
        )
        if err:
            return None, err
        solid_angle_chunk, err = parse_int(
            "solid_angle_chunk", "Solid-angle chunk size"
        )
        if err:
            return None, err
    else:
        solid_angle_preset = solid_angle_profile_preset(solid_angle_profile)
        solid_angle_base_n_alpha = int(solid_angle_preset["solid_angle_base_n_alpha"])
        solid_angle_base_n_theta = int(solid_angle_preset["solid_angle_base_n_theta"])
        solid_angle_refine_levels = int(solid_angle_preset["solid_angle_refine_levels"])
        solid_angle_edge_samples = int(solid_angle_preset["solid_angle_edge_samples"])
        solid_angle_chunk = int(solid_angle_preset["solid_angle_chunk"])
    asymmetry_n_bracket_samples: int | None = None
    asymmetry_tol: float | None = None
    asymmetry_max_iter: int | None = None
    asymmetry_n_theta_samples: int | None = None
    asymmetry_n_refine_samples: int | None = None
    asymmetry_refine_levels: int | None = None
    asymmetry_n_boundary_samples: int | None = None
    if operation_mode == "asymmetry_measurements" and asymmetry_advanced_tuning:
        asymmetry_n_bracket_samples, err = parse_int(
            "asymmetry_n_bracket_samples", "Asymmetry bracket samples"
        )
        if err:
            return None, err
        asymmetry_tol, err = parse_float(
            "asymmetry_tol", "Asymmetry alpha tolerance"
        )
        if err:
            return None, err
        asymmetry_max_iter, err = parse_int(
            "asymmetry_max_iter", "Asymmetry max iterations"
        )
        if err:
            return None, err
        asymmetry_n_theta_samples, err = parse_int(
            "asymmetry_n_theta_samples", "Asymmetry theta samples"
        )
        if err:
            return None, err
        asymmetry_n_refine_samples, err = parse_int(
            "asymmetry_n_refine_samples", "Asymmetry refine samples"
        )
        if err:
            return None, err
        asymmetry_refine_levels, err = parse_int(
            "asymmetry_refine_levels", "Asymmetry refine levels"
        )
        if err:
            return None, err
        asymmetry_n_boundary_samples, err = parse_int(
            "asymmetry_n_boundary_samples", "Asymmetry boundary samples"
        )
        if err:
            return None, err
    else:
        asymmetry_preset = asymmetry_performance_profile_preset(asymmetry_profile)
        asymmetry_n_bracket_samples = int(asymmetry_preset["n_bracket_samples"])
        asymmetry_tol = float(asymmetry_preset["tol"])
        asymmetry_max_iter = int(asymmetry_preset["max_iter"])
        asymmetry_n_theta_samples = int(asymmetry_preset["n_theta_samples"])
        asymmetry_n_refine_samples = int(asymmetry_preset["n_refine_samples"])
        asymmetry_refine_levels = int(asymmetry_preset["refine_levels"])
        asymmetry_n_boundary_samples = int(asymmetry_preset["n_boundary_samples"])

    assert m is not None and a is not None and r_obs is not None
    assert theta_obs_deg is not None
    assert psi_y is not None and psi_x is not None and fov_v is not None
    assert width is not None and height is not None
    assert n_x is not None and n_y is not None
    assert solid_angle_base_n_alpha is not None and solid_angle_base_n_theta is not None
    assert solid_angle_refine_levels is not None
    assert solid_angle_edge_samples is not None and solid_angle_chunk is not None
    assert asymmetry_n_bracket_samples is not None and asymmetry_tol is not None
    assert asymmetry_max_iter is not None and asymmetry_n_theta_samples is not None
    assert asymmetry_n_refine_samples is not None and asymmetry_refine_levels is not None
    assert asymmetry_n_boundary_samples is not None

    if m <= 0:
        return None, "BH mass must be > 0"
    if r_obs <= 0:
        return None, "Observer distance must be > 0"
    if operation_mode == "lensing":
        if abs(a) > 1.0:
            return None, "Dimensionless BH spin a/M must be between -1 and 1"
        if theta_obs_deg < 0 or theta_obs_deg > 180:
            return None, "Observer inclination must be in [0, 180] degrees"
        if not np.isclose(a, 0.0) and np.isclose(np.sin(np.radians(theta_obs_deg)), 0.0):
            return None, (
                "For Kerr, observer inclination must avoid exact 0 or 180 degrees "
                "with the current tracer"
            )
        if fov_v <= 0 or fov_v >= 179:
            return None, "Vertical FOV must be in (0, 179) degrees"
        if width <= 0 or height <= 0:
            return None, "Output width and height must both be > 0"
        if n_x <= 0 or n_y <= 0:
            return None, "Stripe counts must both be > 0"
    elif operation_mode == "shadow_solid_angle":
        if shadow_metric == "kerr":
            if abs(a) > 1.0:
                return None, "Dimensionless Kerr spin a/M must be between -1 and 1"
            if theta_obs_deg < 0 or theta_obs_deg > 180:
                return None, "Observer inclination must be in [0, 180] degrees"
            if np.isclose(np.sin(np.radians(theta_obs_deg)), 0.0):
                return None, (
                    "For Kerr, observer inclination must avoid exact 0 or 180 degrees "
                    "with the current tracer"
                )
            if solid_angle_base_n_alpha <= 0 or solid_angle_base_n_theta <= 0:
                return None, "Solid-angle base grid sizes must be > 0"
            if solid_angle_refine_levels < 0:
                return None, "Solid-angle refine levels must be >= 0"
            if solid_angle_edge_samples <= 0:
                return None, "Solid-angle edge samples must be > 0"
            if solid_angle_chunk <= 0:
                return None, "Solid-angle chunk size must be > 0"
    else:
        if abs(a) > 1.0:
            return None, "Dimensionless BH spin a/M must be between -1 and 1"
        if theta_obs_deg < 0 or theta_obs_deg > 180:
            return None, "Observer inclination must be in [0, 180] degrees"
        if not np.isclose(a, 0.0) and np.isclose(np.sin(np.radians(theta_obs_deg)), 0.0):
            return None, (
                "For Kerr, observer inclination must avoid exact 0 or 180 degrees "
                "with the current tracer"
            )
        if asymmetry_n_bracket_samples < 2:
            return None, "Asymmetry bracket samples must be >= 2"
        if asymmetry_tol <= 0:
            return None, "Asymmetry alpha tolerance must be > 0"
        if asymmetry_max_iter <= 0:
            return None, "Asymmetry max iterations must be > 0"
        if asymmetry_n_theta_samples < 8:
            return None, "Asymmetry theta samples must be >= 8"
        if asymmetry_n_refine_samples < 3:
            return None, "Asymmetry refine samples must be >= 3"
        if asymmetry_refine_levels < 0:
            return None, "Asymmetry refine levels must be >= 0"
        if asymmetry_n_boundary_samples < 8:
            return None, "Asymmetry boundary samples must be >= 8"

    input_image = str(state["input_image"]).strip()
    output_image = str(state["output_image"]).strip()

    if operation_mode == "lensing" and not output_image:
        return None, "Output image path cannot be empty"

    if operation_mode == "lensing" and require_source_assets and source_mode == "image":
        if not input_image:
            return None, "Background image path cannot be empty in image mode"
        input_path = Path(input_image).expanduser()
        if not input_path.is_file():
            return None, f"Background image not found: {input_path}"

    return AppConfig(
        operation_mode=operation_mode,
        source_mode=source_mode,
        shadow_metric=shadow_metric,
        solid_angle_profile=solid_angle_profile,
        solid_angle_advanced_tuning=solid_angle_advanced_tuning,
        asymmetry_measurement=asymmetry_measurement,
        asymmetry_profile=asymmetry_profile,
        asymmetry_advanced_tuning=asymmetry_advanced_tuning,
        asymmetry_circle_fit=asymmetry_circle_fit,
        input_image=input_image,
        output_image=output_image,
        width=width,
        height=height,
        n_x=n_x,
        n_y=n_y,
        color_visualization=bool(state["color_visualization"]),
        M=m,
        a=a,
        r_obs=r_obs,
        theta_obs_deg=theta_obs_deg,
        psi_y=psi_y,
        psi_x=psi_x,
        fov_v=fov_v,
        solid_angle_base_n_alpha=solid_angle_base_n_alpha,
        solid_angle_base_n_theta=solid_angle_base_n_theta,
        solid_angle_refine_levels=solid_angle_refine_levels,
        solid_angle_edge_samples=solid_angle_edge_samples,
        solid_angle_chunk=solid_angle_chunk,
        asymmetry_n_bracket_samples=asymmetry_n_bracket_samples,
        asymmetry_tol=asymmetry_tol,
        asymmetry_max_iter=asymmetry_max_iter,
        asymmetry_n_theta_samples=asymmetry_n_theta_samples,
        asymmetry_n_refine_samples=asymmetry_n_refine_samples,
        asymmetry_refine_levels=asymmetry_refine_levels,
        asymmetry_n_boundary_samples=asymmetry_n_boundary_samples,
        debug=bool(state["debug"]),
        benchmark=bool(state["benchmark"]),
        wrap_outside_background_plane=bool(state["wrap_outside_background_plane"]),
    ), None


def is_enter_key(ch: Any) -> bool:
    if isinstance(ch, str):
        return ch in ("\n", "\r")
    return ch in (10, 13, curses.KEY_ENTER)


def draw_form(
    stdscr: Any,
    state: dict[str, Any],
    defaults: dict[str, Any],
    visible_fields: list[dict[str, str]],
    cursor: int,
    status: str,
    logs: list[str],
    run_info: dict[str, Any],
) -> None:
    stdscr.erase()
    h, w = stdscr.getmaxyx()

    if h < 26 or w < 72:
        msg = "Terminal too small (min: 72x26). Resize and try again."
        stdscr.addnstr(0, 0, msg, max(0, w - 1))
        stdscr.refresh()
        return

    title = "Lens CLI Wrapper"
    help_line = "Arrows or j/k: move | Enter: edit/cycle | Space: toggle/cycle | q: quit"
    stdscr.addnstr(0, 2, title, w - 4, curses.A_BOLD)
    stdscr.addnstr(1, 2, help_line, w - 4)

    def format_display_value(spec: dict[str, str], values: dict[str, Any]) -> str:
        key = spec["key"]
        kind = spec["kind"]
        if kind == "bool":
            return "On" if values[key] else "Off"
        if kind == "choice":
            return format_choice_value(key, str(values[key]))
        return format_field_value(key, values[key])

    def truncate_for_column(text: str, width: int) -> str:
        if width <= 0:
            return ""
        if len(text) <= width:
            return text
        if width <= 3:
            return "." * width
        return text[: width - 3] + "..."

    def field_type_label(kind: str) -> str:
        return {
            "choice": "Choice",
            "text": "Text",
            "int": "Integer",
            "float": "Float",
            "bool": "Toggle",
        }.get(kind, kind.title())

    def field_scope_label(key: str) -> str:
        if key in SHADOW_KERR_ONLY_FIELDS:
            return "Shadow mode / Kerr only"
        if key in ASYMMETRY_ONLY_FIELDS:
            return "Asymmetry mode only"
        if key in LENSING_ONLY_FIELDS:
            return "Lensing mode only"
        if key in SHADOW_ONLY_FIELDS:
            return "Shadow mode only"
        if key in IMAGE_MODE_ONLY_FIELDS:
            return "Lensing / image background only"
        if key in STRIPES_MODE_ONLY_FIELDS:
            return "Lensing / stripes background only"
        return "All modes"

    def field_constraint_text(key: str) -> str:
        constraints = {
            "operation_mode": "Allowed values: lensing, shadow_solid_angle, or asymmetry_measurements.",
            "source_mode": "Allowed values: image or stripes.",
            "shadow_metric": "Allowed values: kerr or schwarzschild.",
            "solid_angle_profile": "Allowed values: quick, normal, accurate, or ultra_accurate. Selecting one loads the corresponding Kerr integration preset.",
            "solid_angle_advanced_tuning": "Shadow mode / Kerr only. Off means the profile controls the hidden numeric integration settings; on reveals those settings for manual overrides.",
            "asymmetry_measurement": "Allowed values: all or any registered asymmetry measurement method.",
            "asymmetry_profile": "Allowed values: quick, normal, accurate, or ultra_accurate. Selecting one loads the corresponding asymmetry integration preset.",
            "asymmetry_advanced_tuning": "Asymmetry mode only. Off means the profile controls the hidden numeric tuning settings; on reveals those settings for manual overrides.",
            "asymmetry_circle_fit": "Asymmetry mode only. Allowed values: global or cardinal. This only affects the circularity measurement.",
            "input_image": "Must point to an existing file when Mode is image.",
            "output_image": "Lensing mode only. Cannot be empty. Existing files may be overwritten.",
            "width": "Enter an integer greater than 0.",
            "height": "Enter an integer greater than 0.",
            "n_x": "Enter an integer greater than 0.",
            "n_y": "Enter an integer greater than 0.",
            "color_visualization": "Only changes the stripes renderer. Off keeps the normal stripe shading.",
            "M": "Must be greater than 0.",
            "a": "Enter the dimensionless spin a/M in the range [-1, 1].",
            "r_obs": "Must be greater than 0 and is interpreted in units of M.",
            "theta_obs_deg": "Measured from the spin axis in degrees. Use 90 for equatorial viewing.",
            "psi_y": "Measured in degrees on the screen. Positive is up.",
            "psi_x": "Measured in degrees on the screen. Positive is right.",
            "fov_v": "Enter a value strictly between 0 and 179 degrees.",
            "solid_angle_base_n_alpha": "Kerr shadow mode only. Enter an integer greater than 0.",
            "solid_angle_base_n_theta": "Kerr shadow mode only. Enter an integer greater than 0.",
            "solid_angle_refine_levels": "Kerr shadow mode only. Enter an integer greater than or equal to 0.",
            "solid_angle_edge_samples": "Kerr shadow mode only. Enter an integer greater than 0.",
            "solid_angle_chunk": "Kerr shadow mode only. Enter an integer greater than 0.",
            "asymmetry_n_bracket_samples": "Asymmetry mode only. Enter an integer greater than or equal to 2.",
            "asymmetry_tol": "Asymmetry mode only. Enter a positive number.",
            "asymmetry_max_iter": "Asymmetry mode only. Enter an integer greater than 0.",
            "asymmetry_n_theta_samples": "Asymmetry mode only. Enter an integer greater than or equal to 8.",
            "asymmetry_n_refine_samples": "Asymmetry mode only. Enter an integer greater than or equal to 3.",
            "asymmetry_refine_levels": "Asymmetry mode only. Enter an integer greater than or equal to 0.",
            "asymmetry_n_boundary_samples": "Asymmetry mode only. Enter an integer greater than or equal to 8.",
            "debug": "Useful when you want metric setup and progress logs.",
            "benchmark": "Adds a timing summary after the render completes.",
            "wrap_outside_background_plane": "Ignored in stripes mode.",
        }
        return constraints.get(key, "")

    min_name_w = 10
    min_value_w = 8
    max_value_w = 20  # Fits the longest choice label, "Shadow Solid Angle".
    min_right_w = 24
    left_x = 2
    left_divider_gap = 2
    right_divider_gap = 2
    max_label_len = max(
        [len("Name")]
        + [len(spec["label"]) for spec in visible_fields]
        + [len(action["label"]) for action in ACTION_SPECS]
    )
    max_value_len = max(
        [len("Value")]
        + [len(format_display_value(spec, state)) for spec in visible_fields]
    )
    desired_name_w = max(min_name_w, max_label_len)
    desired_value_w = min(max_value_w, max(min_value_w, max_value_len))
    max_left_w = max(
        min_name_w + 2 + min_value_w,
        w - min_right_w - (left_x + left_divider_gap + right_divider_gap + 3),
    )
    left_w = min(desired_name_w + 2 + desired_value_w, max_left_w)
    name_w = min(desired_name_w, max(min_name_w, left_w - 2 - min_value_w))
    value_w = min(desired_value_w, max(min_value_w, left_w - 2 - name_w))
    left_w = name_w + 2 + value_w
    divider_x = left_x + left_w + left_divider_gap
    right_x = divider_x + 1 + right_divider_gap
    right_w = max(16, w - right_x - 2)
    value_x = left_x + name_w + 2

    if cursor < len(visible_fields):
        selected = visible_fields[cursor]
        detail_title = selected["label"]
        detail_default = format_display_value(selected, defaults)
        detail_type = field_type_label(selected["kind"])
        detail_scope = field_scope_label(selected["key"])
        detail_constraints = field_constraint_text(selected["key"])
        detail_description = selected["description"]
        if selected["kind"] in {"bool", "choice"}:
            detail_control = "Press Enter or Space to cycle this option."
        else:
            detail_control = "Press Enter to edit this value."
    else:
        selected_action = ACTION_SPECS[cursor - len(visible_fields)]
        detail_title = selected_action["label"]
        detail_default = ""
        detail_type = "Action"
        detail_scope = "Interactive form"
        detail_constraints = ""
        detail_description = selected_action["description"]
        detail_control = "Press Enter to activate this action."

    row = 3
    stdscr.addnstr(row, left_x, "Options", left_w, curses.A_BOLD)
    stdscr.addnstr(row, right_x, "Details", right_w, curses.A_BOLD)
    row += 1
    stdscr.addnstr(row, left_x, "Name", name_w, curses.A_DIM)
    stdscr.addnstr(row, value_x, "Value", value_w, curses.A_DIM)
    row += 1

    for idx, spec in enumerate(visible_fields):
        attr = curses.A_REVERSE if idx == cursor else curses.A_NORMAL
        stdscr.addnstr(row, left_x, truncate_for_column(spec["label"], name_w), name_w, attr)
        stdscr.addnstr(
            row,
            value_x,
            truncate_for_column(format_display_value(spec, state), value_w),
            value_w,
            attr,
        )
        row += 1

    row += 1
    stdscr.addnstr(row, left_x, "Actions", left_w, curses.A_BOLD)
    row += 1

    action_start = len(visible_fields)
    for i, action in enumerate(ACTION_SPECS):
        idx = action_start + i
        attr = curses.A_REVERSE if idx == cursor else curses.A_NORMAL
        stdscr.addnstr(row, left_x, truncate_for_column(action["label"], name_w), name_w, attr)
        row += 1

    detail_bottom = row
    stdscr.vline(3, divider_x, curses.ACS_VLINE, max(1, detail_bottom - 3))

    detail_row = 5
    for line in textwrap.wrap(f"Name: {detail_title}", width=max(12, right_w)):
        if detail_row >= detail_bottom:
            break
        stdscr.addnstr(detail_row, right_x, line, right_w, curses.A_BOLD)
        detail_row += 1

    if detail_default and detail_row < detail_bottom:
        detail_row += 1
        for line in textwrap.wrap(f"Default value: {detail_default}", width=max(12, right_w)):
            if detail_row >= detail_bottom:
                break
            stdscr.addnstr(detail_row, right_x, line, right_w)
            detail_row += 1

    if detail_row < detail_bottom:
        detail_row += 1
    for line in textwrap.wrap(f"Type: {detail_type}", width=max(12, right_w)):
        if detail_row >= detail_bottom:
            break
        stdscr.addnstr(detail_row, right_x, line, right_w)
        detail_row += 1

    if detail_row < detail_bottom:
        detail_row += 1
    for line in textwrap.wrap(f"Applies to: {detail_scope}", width=max(12, right_w)):
        if detail_row >= detail_bottom:
            break
        stdscr.addnstr(detail_row, right_x, line, right_w)
        detail_row += 1

    if detail_constraints and detail_row < detail_bottom:
        detail_row += 1
        for line in textwrap.wrap(f"Checks: {detail_constraints}", width=max(12, right_w)):
            if detail_row >= detail_bottom:
                break
            stdscr.addnstr(detail_row, right_x, line, right_w)
            detail_row += 1

    if detail_row < detail_bottom:
        detail_row += 1
    for line in textwrap.wrap(f"Details: {detail_description}", width=max(12, right_w)):
        if detail_row >= detail_bottom:
            break
        stdscr.addnstr(detail_row, right_x, line, right_w)
        detail_row += 1

    if detail_row < detail_bottom:
        detail_row += 1
    for line in textwrap.wrap(detail_control, width=max(12, right_w)):
        if detail_row >= detail_bottom:
            break
        stdscr.addnstr(detail_row, right_x, line, right_w, curses.A_DIM)
        detail_row += 1

    row += 1
    stdscr.addnstr(row, 2, "Output", w - 4, curses.A_BOLD)
    row += 1

    stage_label = str(run_info.get("stage_label", "Idle"))
    stage_percent = float(run_info.get("stage_percent", 0.0))
    overall_percent = float(run_info.get("overall_percent", 0.0))
    cpu_percent = float(run_info.get("cpu_percent", 0.0))
    ram_mb = float(run_info.get("ram_mb", 0.0))

    progress_line = (
        f"Progress {make_progress_bar(overall_percent)} {overall_percent:6.2f}% "
        f"| Stage: {stage_label} {stage_percent:6.2f}%"
    )
    usage_line = f"Usage: CPU {cpu_percent:6.1f}% | RAM {ram_mb:8.1f} MB"
    stdscr.addnstr(row, 2, progress_line, w - 4)
    row += 1
    stdscr.addnstr(row, 2, usage_line, w - 4)
    row += 1

    max_log_lines = max(1, h - row - 3)
    start = max(0, len(logs) - max_log_lines)
    visible_logs = logs[start:]
    for i, line in enumerate(visible_logs):
        stdscr.addnstr(row + i, 2, line, w - 4)

    stdscr.hline(h - 2, 0, "-", w)
    stdscr.addnstr(h - 1, 2, status, w - 4)
    stdscr.refresh()


def prompt_input(stdscr: Any, label: str, initial: str) -> str | None:
    h, w = stdscr.getmaxyx()
    prompt = f"{label}: "
    text = list(initial)
    pos = len(text)

    curses.curs_set(1)
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
        curses.curs_set(0)
        stdscr.move(h - 1, 0)
        stdscr.clrtoeol()


def run_form(stdscr: Any, base_config: AppConfig) -> None:
    try:
        curses.curs_set(0)
    except curses.error:
        pass

    stdscr.keypad(True)
    state = config_to_state(base_config)
    defaults = config_to_state(AppConfig())
    logs: list[str] = [
        "Ready.",
    ]
    run_info: dict[str, Any] = {
        "stage_label": "Idle",
        "stage_percent": 0.0,
        "overall_percent": 0.0,
        "cpu_percent": 0.0,
        "ram_mb": 0.0,
    }

    cursor = 0
    status = "Edit values and choose Run."

    def redraw() -> None:
        visible_fields = get_visible_field_specs(state)
        draw_form(stdscr, state, defaults, visible_fields, cursor, status, logs, run_info)

    def append_log(message: str) -> None:
        parts = message.splitlines() or [""]
        for part in parts:
            logs.append(part)
        if len(logs) > 800:
            del logs[: len(logs) - 800]
        redraw()

    while True:
        visible_fields = get_visible_field_specs(state)
        total_rows = len(visible_fields) + len(ACTION_SPECS)
        cursor = max(0, min(cursor, total_rows - 1))
        redraw()

        try:
            ch = stdscr.get_wch()
        except curses.error:
            continue

        if isinstance(ch, str) and ch.lower() == "q":
            return

        if ch in (curses.KEY_UP,) or ch == "k":
            cursor = (cursor - 1) % total_rows
            continue

        if ch in (curses.KEY_DOWN,) or ch == "j":
            cursor = (cursor + 1) % total_rows
            continue

        if ch == " ":
            if cursor < len(visible_fields):
                spec = visible_fields[cursor]
                key = spec["key"]
                if spec["kind"] == "bool":
                    status = apply_bool_change(state, key, spec["label"])
                elif spec["kind"] == "choice":
                    status = apply_choice_change(state, key, spec["label"])
            continue

        if not is_enter_key(ch):
            continue

        if cursor < len(visible_fields):
            spec = visible_fields[cursor]
            key = spec["key"]
            kind = spec["kind"]

            if kind == "bool":
                status = apply_bool_change(state, key, spec["label"])
                continue
            if kind == "choice":
                status = apply_choice_change(state, key, spec["label"])
                continue

            new_value = prompt_input(stdscr, spec["label"], str(state[key]))
            if new_value is None:
                status = "Edit cancelled."
            else:
                state[key] = new_value
                status = f"Updated {spec['label']}."
            continue

        action_idx = cursor - len(visible_fields)
        action_spec = ACTION_SPECS[action_idx]
        action_key = action_spec["key"]
        action_label = action_spec["label"]

        if action_key == "quit":
            return

        if action_key == "reset":
            state = dict(defaults)
            status = "Defaults restored."
            append_log("Values reset to defaults.")
            continue

        if action_key == "run":
            config, err = parse_state(
                state,
                require_source_assets=(
                    str(state.get("operation_mode", "lensing")).strip().lower() == "lensing"
                ),
            )
            if err:
                status = f"Validation error: {err}"
                continue
            logs.clear()
            status = f"Running: {action_label}..."
            append_log(f"Starting run in '{OPERATION_MODE_LABELS[config.operation_mode]}' mode")
            if config.operation_mode == "lensing":
                append_log(f"Background mode: {SOURCE_MODE_LABELS[config.source_mode]}")
                if config.source_mode == "image":
                    append_log(f"Using input '{config.input_image}'")
                else:
                    append_log(
                        f"Using stripes at {format_resolution_value(config.width, config.height)} "
                        f"(n_x={append_unit(config.n_x, 'stripes')}, "
                        f"n_y={append_unit(config.n_y, 'stripes')})"
                    )
                    if config.color_visualization:
                        append_log(
                            "Color visualization will render stripe hits in black "
                            "and color the remaining escaped rays by final theta "
                            "quadrant."
                        )
                append_log(f"Output will be saved to '{config.output_image}'")
                run_info["stage_label"] = "Preparing"
            elif config.operation_mode == "shadow_solid_angle":
                append_log(
                    f"Using {SHADOW_METRIC_LABELS[config.shadow_metric]} "
                    f"with M={append_unit(config.M, 'M')}, "
                    f"r_obs={append_unit(config.r_obs, 'M')}"
                )
                if config.shadow_metric == "kerr":
                    if config.solid_angle_advanced_tuning:
                        append_log(
                            f"profile={format_choice_value('solid_angle_profile', config.solid_angle_profile)}, "
                            f"spin={append_unit(config.a, 'a/M')}, "
                            f"theta_obs={append_unit(config.theta_obs_deg, 'deg')}, "
                            f"grid={format_field_value('solid_angle_base_n_alpha', config.solid_angle_base_n_alpha)} x "
                            f"{format_field_value('solid_angle_base_n_theta', config.solid_angle_base_n_theta)}, "
                            f"refine={format_field_value('solid_angle_refine_levels', config.solid_angle_refine_levels)}, "
                            f"edge={format_field_value('solid_angle_edge_samples', config.solid_angle_edge_samples)}, "
                            f"chunk={format_field_value('solid_angle_chunk', config.solid_angle_chunk)}"
                        )
                    else:
                        append_log(
                            f"profile={format_choice_value('solid_angle_profile', config.solid_angle_profile)}, "
                            f"spin={append_unit(config.a, 'a/M')}, "
                            f"theta_obs={append_unit(config.theta_obs_deg, 'deg')}, "
                            "advanced numeric tuning hidden"
                        )
                run_info["stage_label"] = "Compute solid angle"
            else:
                append_log(
                    f"Using metric parameters M={append_unit(config.M, 'M')}, "
                    f"spin={append_unit(config.a, 'a/M')}, "
                    f"r_obs={append_unit(config.r_obs, 'M')}, "
                    f"theta_obs={append_unit(config.theta_obs_deg, 'deg')}"
                )
                append_log(
                    "Measurement selection: "
                    f"{format_choice_value('asymmetry_measurement', config.asymmetry_measurement)}"
                )
                if config.asymmetry_advanced_tuning:
                    append_log(
                        f"profile={format_choice_value('asymmetry_profile', config.asymmetry_profile)}, "
                        f"circle_fit={format_choice_value('asymmetry_circle_fit', config.asymmetry_circle_fit)}, "
                        f"bracket={format_field_value('asymmetry_n_bracket_samples', config.asymmetry_n_bracket_samples)}, "
                        f"tol={format_field_value('asymmetry_tol', config.asymmetry_tol)}, "
                        f"max_iter={format_field_value('asymmetry_max_iter', config.asymmetry_max_iter)}, "
                        f"theta={format_field_value('asymmetry_n_theta_samples', config.asymmetry_n_theta_samples)}, "
                        f"refine_samples={format_field_value('asymmetry_n_refine_samples', config.asymmetry_n_refine_samples)}, "
                        f"levels={format_field_value('asymmetry_refine_levels', config.asymmetry_refine_levels)}, "
                        f"boundary={format_field_value('asymmetry_n_boundary_samples', config.asymmetry_n_boundary_samples)}"
                    )
                else:
                    append_log(
                        f"profile={format_choice_value('asymmetry_profile', config.asymmetry_profile)}, "
                        f"circle_fit={format_choice_value('asymmetry_circle_fit', config.asymmetry_circle_fit)}, "
                        "advanced numeric tuning hidden"
                    )
                run_info["stage_label"] = "Measure asymmetry"
            run_info["stage_percent"] = 0.0
            run_info["overall_percent"] = 0.0
            usage = ProcessUsageSampler()
            cpu_percent, ram_mb = usage.sample()
            run_info["cpu_percent"] = cpu_percent
            run_info["ram_mb"] = ram_mb
            last_draw_time = perf_counter()
            last_draw_progress = -1.0

            def update_progress(stage_label: str, stage_percent: float, overall_percent: float) -> None:
                nonlocal last_draw_time, last_draw_progress
                run_info["stage_label"] = stage_label
                run_info["stage_percent"] = stage_percent
                run_info["overall_percent"] = overall_percent
                sampled_cpu, sampled_ram = usage.sample()
                run_info["cpu_percent"] = sampled_cpu
                run_info["ram_mb"] = sampled_ram
                now = perf_counter()
                must_draw = (
                    abs(overall_percent - last_draw_progress) >= 0.2
                    or (now - last_draw_time) >= 0.15
                    or overall_percent <= 0.0
                    or overall_percent >= 100.0
                )
                if must_draw:
                    redraw()
                    last_draw_time = now
                    last_draw_progress = overall_percent

            try:
                if config.operation_mode == "lensing":
                    run_lensing(
                        config,
                        log=append_log,
                        show_progress=False,
                        metric_debug=False,
                        progress=update_progress,
                    )
                elif config.operation_mode == "shadow_solid_angle":
                    run_shadow_solid_angle(config, log=append_log, progress=update_progress)
                else:
                    run_asymmetry_measurements(config, log=append_log, progress=update_progress)
            except Exception as exc:  # pragma: no cover - user-facing error path
                append_log(f"Error: {exc}")
                status = f"Run failed: {exc}"
                continue
            run_info["stage_label"] = "Done"
            run_info["stage_percent"] = 100.0
            run_info["overall_percent"] = 100.0
            cpu_percent, ram_mb = usage.sample()
            run_info["cpu_percent"] = cpu_percent
            run_info["ram_mb"] = ram_mb
            if config.operation_mode == "lensing":
                status = "Generation complete. You can edit and run again."
            elif config.operation_mode == "shadow_solid_angle":
                status = "Solid-angle computation complete."
            else:
                status = "Asymmetry measurements complete."


def benchmark_summary_lines(
    image_dimension: tuple[int, int],
    alpha_crit: float,
    total_rays: int,
    traced_rays: int,
    timings: dict[str, float],
) -> list[str]:
    height, width = image_dimension
    pixel_count = width * height
    render_time = max(timings.get("render", 0.0), 1e-12)
    total_time = max(timings.get("total", 0.0), 1e-12)
    resolution = format_resolution_value(width, height)
    pixel_count_text = append_unit(f"{pixel_count:,}", "px")
    total_rays_text = append_unit(f"{total_rays:,}", "rays")
    traced_rays_text = append_unit(f"{traced_rays:,}", "rays")
    return [
        "Benchmark summary",
        f"  resolution: {resolution} ({pixel_count_text})",
        f"  alpha_crit: {alpha_crit:.6f} rad",
        f"  total rays: {total_rays_text}",
        f"  traced rays: {traced_rays_text}",
        f"  {'prepare_source':<26}{timings.get('prepare_source', 0.0):>10.3f} s",
        f"  {'build_lookup':<26}{timings.get('build_lookup', 0.0):>10.3f} s",
        f"  {'precompute':<26}{timings.get('precompute', 0.0):>10.3f} s",
        f"  {'render':<26}{timings.get('render', 0.0):>10.3f} s",
        f"  {'save_image':<26}{timings.get('save_image', 0.0):>10.3f} s",
        f"  {'total':<26}{timings.get('total', 0.0):>10.3f} s",
        f"  {'render_throughput':<26}{(pixel_count / render_time) / 1e6:>10.2f} MPix/s",
        f"  {'overall_throughput':<26}{(pixel_count / total_time) / 1e6:>10.2f} MPix/s",
    ]


def shadow_benchmark_summary_lines(
    config: AppConfig,
    stats: dict[str, float | int],
) -> list[str]:
    total_time = max(float(stats.get("total_time", 0.0)), 1e-12)
    if config.shadow_metric == "schwarzschild":
        analytic_time = float(stats.get("analytic_time", total_time))
        return [
            "Shadow solid-angle benchmark",
            "  metric: Schwarzschild",
            f"  {'analytic':<26}{analytic_time:>10.6f} s",
            f"  {'total':<26}{total_time:>10.6f} s",
        ]

    trace_time = max(
        float(stats.get("stencil_trace_time", 0.0))
        + float(stats.get("terminal_trace_time", 0.0)),
        1e-12,
    )
    total_rays = int(stats.get("total_rays", 0))
    base_cells_text = append_unit(f"{int(stats.get('base_cells', 0)):,}", "cells")
    levels_text = (
        f"{append_unit(int(stats.get('levels_processed', 0)), 'levels')} / "
        f"{append_unit(int(stats.get('levels_requested', 0)), 'levels requested')}"
    )
    evaluated_cells_text = append_unit(f"{int(stats.get('cells_evaluated', 0)):,}", "cells")
    refined_cells_text = append_unit(f"{int(stats.get('refined_cells', 0)):,}", "cells")
    terminal_mixed_text = append_unit(f"{int(stats.get('terminal_mixed_cells', 0)):,}", "cells")
    captured_cells_text = append_unit(f"{int(stats.get('captured_full_cells', 0)):,}", "cells")
    escaped_cells_text = append_unit(f"{int(stats.get('escaped_full_cells', 0)):,}", "cells")
    mixed_cells_text = append_unit(f"{int(stats.get('mixed_cells', 0)):,}", "cells")
    stencil_rays_text = append_unit(f"{int(stats.get('stencil_rays', 0)):,}", "rays")
    terminal_rays_text = append_unit(f"{int(stats.get('terminal_rays', 0)):,}", "rays")
    retry_rays_text = append_unit(f"{int(stats.get('retry_rays', 0)):,}", "rays")
    total_rays_text = append_unit(f"{total_rays:,}", "rays")
    return [
        "Shadow solid-angle benchmark",
        f"  metric: Kerr ({format_choice_value('solid_angle_profile', config.solid_angle_profile)})",
        f"  base cells: {base_cells_text}",
        f"  levels: {levels_text}",
        f"  evaluated cells: {evaluated_cells_text}",
        f"  refined cells: {refined_cells_text}",
        f"  terminal mixed cells: {terminal_mixed_text}",
        f"  full captured cells: {captured_cells_text}",
        f"  full escaped cells: {escaped_cells_text}",
        f"  mixed cells seen: {mixed_cells_text}",
        f"  stencil rays: {stencil_rays_text}",
        f"  terminal rays: {terminal_rays_text}",
        f"  retry rays: {retry_rays_text}",
        f"  total rays: {total_rays_text}",
        f"  {'setup':<26}{float(stats.get('setup_time', 0.0)):>10.3f} s",
        f"  {'trace_stencil':<26}{float(stats.get('stencil_trace_time', 0.0)):>10.3f} s",
        f"  {'classify':<26}{float(stats.get('classification_time', 0.0)):>10.3f} s",
        f"  {'full_area':<26}{float(stats.get('full_area_time', 0.0)):>10.3f} s",
        f"  {'subdivide':<26}{float(stats.get('subdivide_time', 0.0)):>10.3f} s",
        f"  {'terminal_setup':<26}{float(stats.get('terminal_setup_time', 0.0)):>10.3f} s",
        f"  {'trace_terminal':<26}{float(stats.get('terminal_trace_time', 0.0)):>10.3f} s",
        f"  {'terminal_area':<26}{float(stats.get('terminal_area_time', 0.0)):>10.3f} s",
        f"  {'total':<26}{total_time:>10.3f} s",
        f"  {'trace_throughput':<26}{(total_rays / trace_time) / 1e6:>10.2f} Mray/s",
        f"  {'overall_throughput':<26}{(total_rays / total_time) / 1e6:>10.2f} Mray/s",
    ]


def asymmetry_benchmark_summary_lines(
    metric: Any,
    measurement_names: tuple[str, ...],
    timings: dict[str, float],
    measurement_stats: dict[str, dict[str, Any]],
) -> list[str]:
    total_time = max(timings.get("total", 0.0), 1e-12)
    total_trace_rays = sum(int(stats.get("trace_ray_calls", 0)) for stats in measurement_stats.values())
    total_trace_outcomes = sum(
        int(stats.get("trace_outcome_calls", 0)) for stats in measurement_stats.values()
    )
    total_invalid_results = sum(
        int(stats.get("invalid_trace_results", 0)) for stats in measurement_stats.values()
    )
    total_alpha_crit_calls = sum(
        int(stats.get("alpha_crit_calls", 0)) for stats in measurement_stats.values()
    )
    total_boundary_requests = sum(
        int(stats.get("boundary_point_requests", 0)) for stats in measurement_stats.values()
    )
    total_boundary_samples = sum(
        int(stats.get("boundary_samples_requested", 0)) for stats in measurement_stats.values()
    )
    total_cache_hits = sum(int(stats.get("point_cache_hits", 0)) for stats in measurement_stats.values())
    total_cache_misses = sum(
        int(stats.get("point_cache_misses", 0)) for stats in measurement_stats.values()
    )
    total_trace_time_raw = sum(float(stats.get("trace_time", 0.0)) for stats in measurement_stats.values())
    lines = [
        "Asymmetry benchmark",
        f"  metric: {type(metric).__name__}",
        f"  measurements: {len(measurement_names)}",
        f"  trace rays: {append_unit(f'{total_trace_rays:,}', 'rays')}",
        f"  trace outcomes: {append_unit(f'{total_trace_outcomes:,}', 'outcomes')}",
        f"  invalid results: {append_unit(f'{total_invalid_results:,}', 'results')}",
        f"  alpha_crit calls: {append_unit(f'{total_alpha_crit_calls:,}', 'calls')}",
        f"  boundary requests: {append_unit(f'{total_boundary_requests:,}', 'requests')}",
        f"  boundary samples: {append_unit(f'{total_boundary_samples:,}', 'samples')}",
        f"  cache hits: {append_unit(f'{total_cache_hits:,}', 'hits')}",
        f"  cache misses: {append_unit(f'{total_cache_misses:,}', 'misses')}",
    ]
    for name in measurement_names:
        stats = measurement_stats.get(name, {})
        lines.append(
            f"  {name:<26}{timings.get(name, 0.0):>10.3f} s"
            f" | {int(stats.get('trace_ray_calls', 0)):>8,} rays"
            f" | {int(stats.get('alpha_crit_calls', 0)):>6,} alpha_crit"
        )
    lines.append(f"  {'trace_time':<26}{total_trace_time_raw:>10.3f} s")
    lines.append(f"  {'total':<26}{timings.get('total', 0.0):>10.3f} s")
    lines.append(
        f"  {'trace_throughput':<26}{format_ray_rate(total_trace_rays, total_trace_time_raw):>10}"
    )
    lines.append(
        f"  {'measurements_per_second':<26}{(len(measurement_names) / total_time):>10.2f}"
    )
    return lines


def overall_progress_percent(stage_key: str, stage_fraction: float) -> float:
    clamped_stage = max(0.0, min(1.0, stage_fraction))
    total = 0.0
    for key, _, weight in STAGE_SPECS:
        if key == stage_key:
            total += weight * clamped_stage
            break
        total += weight
    return max(0.0, min(100.0, total * 100.0))


def spin_mass_units_to_kerr_a(M: float, spin_over_mass: float) -> float:
    return float(M) * float(spin_over_mass)


def metric_spin_over_mass(metric: Any) -> float:
    metric_mass = float(getattr(metric, "M", 0.0))
    if np.isclose(metric_mass, 0.0):
        return 0.0
    return float(getattr(metric, "a", 0.0)) / metric_mass


def build_observer_metric(config: AppConfig) -> Schwarzschild | Kerr:
    if np.isclose(config.a, 0.0):
        return Schwarzschild(M=config.M)
    return Kerr(M=config.M, a=spin_mass_units_to_kerr_a(config.M, config.a))


def selected_asymmetry_measurement_names(config: AppConfig) -> tuple[str, ...]:
    if config.asymmetry_measurement == "all":
        return AsymmetryMeasurements.measurement_names()
    return (config.asymmetry_measurement,)


def asymmetry_measurement_kwargs(config: AppConfig) -> dict[str, Any]:
    return {
        "circle_fit": config.asymmetry_circle_fit,
        "n_bracket_samples": config.asymmetry_n_bracket_samples,
        "tol": config.asymmetry_tol,
        "max_iter": config.asymmetry_max_iter,
        "n_theta_samples": config.asymmetry_n_theta_samples,
        "n_refine_samples": config.asymmetry_n_refine_samples,
        "refine_levels": config.asymmetry_refine_levels,
        "n_boundary_samples": config.asymmetry_n_boundary_samples,
    }


def emit_asymmetry_measurement_value(
    emit: Callable[[str], None],
    name: str,
    value: Any,
) -> None:
    if isinstance(value, dict):
        emit(f"{name}:")
        for key, subvalue in value.items():
            if isinstance(subvalue, (float, int, np.floating, np.integer)):
                emit(f"  {key:<20} = {float(subvalue):.10f}")
            else:
                emit(f"  {key:<20} = {subvalue}")
        return

    if isinstance(value, (float, int, np.floating, np.integer)):
        emit(f"{name:<22} = {float(value):.10f}")
        return

    emit(f"{name:<22} = {value}")


def run_asymmetry_measurements(
    config: AppConfig,
    log: Callable[[str], None] | None = None,
    progress: Callable[[str, float, float], None] | None = None,
    show_progress: bool | None = None,
) -> None:
    def emit(message: str) -> None:
        if log is None:
            print(message)
        else:
            log(message)

    if show_progress is None:
        show_progress = config.debug and progress is None and log is None

    last_progress_time = perf_counter()
    last_progress_overall = -1.0

    def emit_progress(stage_label: str, stage_fraction: float, overall_fraction: float) -> None:
        nonlocal last_progress_time, last_progress_overall
        stage_fraction = max(0.0, min(1.0, stage_fraction))
        overall_fraction = max(0.0, min(1.0, overall_fraction))
        stage_percent = 100.0 * stage_fraction
        overall_percent = 100.0 * overall_fraction
        if progress is not None:
            progress(stage_label, stage_percent, overall_percent)
            return
        if not show_progress:
            return
        now = perf_counter()
        if (
            overall_fraction < 1.0
            and abs(overall_fraction - last_progress_overall) < 0.08
            and (now - last_progress_time) < 0.5
        ):
            return
        print(
            f"[asymmetry] {make_progress_bar(overall_percent, width=24)} "
            f"{overall_percent:6.2f}% total | {stage_label:<28} {stage_percent:6.2f}% stage"
        )
        last_progress_time = now
        last_progress_overall = overall_fraction

    metric = build_observer_metric(config)
    r_obs = config.r_obs * metric.M
    theta_obs = np.radians(config.theta_obs_deg)
    measurements = AsymmetryMeasurements(metric, r_obs, theta_obs)
    measurement_names = selected_asymmetry_measurement_names(config)
    common_kwargs = asymmetry_measurement_kwargs(config)
    measurement_plan: list[tuple[str, dict[str, Any], float]] = []
    for name in measurement_names:
        measurement_kwargs = filter_asymmetry_measurement_kwargs(name, common_kwargs)
        estimated_work_units = float(
            AsymmetryMeasurements.estimate_measurement_work_units(name, **measurement_kwargs)
        )
        measurement_plan.append((name, measurement_kwargs, max(estimated_work_units, 1.0)))
    total_estimated_work = max(
        1.0,
        float(sum(estimated_work_units for _, _, estimated_work_units in measurement_plan)),
    )

    timings: dict[str, float] = {}
    measurement_stats: dict[str, dict[str, Any]] = {}
    total_start = perf_counter() if config.benchmark else None

    if config.debug:
        metric_mass = append_unit(f"{metric.M:g}", "M")
        metric_spin = append_unit(f"{metric_spin_over_mass(metric):g}", "a/M")
        metric_a = append_unit(f"{getattr(metric, 'a', 0.0):g}", "M")
        emit(
            f"Metric: {type(metric).__name__} "
            f"(M={metric_mass}, spin={metric_spin}, a={metric_a})"
        )

    emit("Asymmetry measurements")
    emit(f"M                     = {append_unit(config.M, 'M')}")
    emit(f"spin                  = {append_unit(config.a, 'a/M')}")
    emit(f"a                     = {append_unit(getattr(metric, 'a', 0.0), 'M')}")
    emit(f"r_obs                 = {append_unit(r_obs, 'M')}")
    emit(f"theta_obs             = {config.theta_obs_deg:.6f} deg")
    emit(f"profile               = {format_choice_value('asymmetry_profile', config.asymmetry_profile)}")
    emit(
        f"advanced tuning       = "
        f"{'On' if config.asymmetry_advanced_tuning else 'Off'}"
    )
    emit(
        f"circle algorithm      = "
        f"{format_choice_value('asymmetry_circle_fit', config.asymmetry_circle_fit)}"
    )
    if config.asymmetry_advanced_tuning:
        emit(
            "sampling              = "
            f"bracket={config.asymmetry_n_bracket_samples}, "
            f"tol={config.asymmetry_tol:g} rad, "
            f"max_iter={config.asymmetry_max_iter}, "
            f"theta={config.asymmetry_n_theta_samples}, "
            f"refine_samples={config.asymmetry_n_refine_samples}, "
            f"refine_levels={config.asymmetry_refine_levels}, "
            f"boundary={config.asymmetry_n_boundary_samples}"
        )
    else:
        emit("sampling              = controlled by asymmetry profile preset")
    emit("selected measurements = " + ", ".join(measurement_names))

    emit_progress("Measure asymmetry", 0.0, 0.0)

    completed_work = 0.0
    monitor_enabled = config.benchmark or progress is not None or show_progress
    for name, measurement_kwargs, estimated_work_units in measurement_plan:
        stage_label = name.replace("_", " ")
        if monitor_enabled:
            def measurement_progress(done_units: float, total_units: float, *, offset=completed_work, estimate=estimated_work_units, label=stage_label) -> None:
                measurement_total = max(float(total_units), float(estimate), 1e-12)
                bounded_done = max(0.0, min(float(done_units), measurement_total))
                emit_progress(
                    label,
                    bounded_done / measurement_total,
                    min(1.0, (offset + bounded_done) / total_estimated_work),
                )

            measurements.begin_measurement_run(
                name,
                estimated_work_units=estimated_work_units,
                progress_callback=measurement_progress,
            )
        start = perf_counter() if config.benchmark else None
        try:
            value = measurements.measure(name, **measurement_kwargs)
        except Exception:
            if monitor_enabled:
                measurement_stats[name] = measurements.finish_measurement_run(completed=False)
            raise
        if monitor_enabled:
            measurement_stats[name] = measurements.finish_measurement_run(completed=True)
        if config.benchmark and start is not None:
            timings[name] = perf_counter() - start
        emit_asymmetry_measurement_value(emit, name, value)
        completed_work += estimated_work_units

    if config.benchmark and total_start is not None:
        timings["total"] = perf_counter() - total_start
        for line in asymmetry_benchmark_summary_lines(
            metric,
            measurement_names,
            timings,
            measurement_stats,
        ):
            emit(line)
    emit_progress("Asymmetry complete", 1.0, 1.0)


def run_shadow_solid_angle(
    config: AppConfig,
    log: Callable[[str], None] | None = None,
    progress: Callable[[str, float, float], None] | None = None,
    show_progress: bool | None = None,
) -> None:
    def emit(message: str) -> None:
        if log is None:
            print(message)
        else:
            log(message)

    if show_progress is None:
        show_progress = config.debug and progress is None and log is None

    last_progress_time = perf_counter()
    last_progress_overall = -1.0

    def emit_progress(stage_label: str, stage_fraction: float, overall_fraction: float) -> None:
        nonlocal last_progress_time, last_progress_overall
        stage_percent = max(0.0, min(100.0, 100.0 * stage_fraction))
        overall_percent = max(0.0, min(100.0, 100.0 * overall_fraction))
        if progress is not None:
            progress(stage_label, stage_percent, overall_percent)
            return
        if not show_progress:
            return
        now = perf_counter()
        if (
            overall_fraction < 1.0
            and abs(overall_fraction - last_progress_overall) < 0.02
            and (now - last_progress_time) < 0.25
        ):
            return
        print(
            f"[solid-angle] {stage_label:<24} "
            f"{overall_percent:6.2f}% total | {stage_percent:6.2f}% stage"
        )
        last_progress_time = now
        last_progress_overall = overall_fraction

    r_obs = config.r_obs * config.M
    theta_obs = np.radians(config.theta_obs_deg)
    metric_a = spin_mass_units_to_kerr_a(config.M, config.a)
    stats: dict[str, float | int]

    if config.shadow_metric == "schwarzschild":
        analytic_start = perf_counter()
        emit_progress("Analytic solution", 0.0, 0.0)
        omega, alpha = schwarzschild_shadow_solid_angle(r_obs, M=config.M)
        fraction = omega / (4.0 * np.pi)
        stats = {
            "analytic_time": perf_counter() - analytic_start,
            "total_time": perf_counter() - analytic_start,
        }
        emit_progress("Analytic solution", 1.0, 1.0)
        emit("Schwarzschild black-hole shadow")
        emit(f"M                     = {append_unit(config.M, 'M')}")
        emit(f"r_obs                 = {append_unit(r_obs, 'M')}")
        emit(f"shadow angular radius = {np.degrees(alpha):.6f} deg")
        emit(f"shadow solid angle    = {omega:.10f} sr")
        emit(f"fraction of full sky  = {fraction:.10%}")
    else:
        omega, fraction, stats = kerr_shadow_solid_angle(
            r_obs,
            metric_a,
            M=config.M,
            theta_obs=theta_obs,
            base_n_alpha=config.solid_angle_base_n_alpha,
            base_n_theta=config.solid_angle_base_n_theta,
            refine_levels=config.solid_angle_refine_levels,
            edge_samples=config.solid_angle_edge_samples,
            chunk=config.solid_angle_chunk,
            progress_callback=emit_progress,
            return_stats=True,
        )
        emit("Kerr black-hole shadow")
        emit(f"M                     = {append_unit(config.M, 'M')}")
        emit(f"spin                  = {append_unit(config.a, 'a/M')}")
        emit(f"a                     = {append_unit(metric_a, 'M')}")
        emit(f"r_obs                 = {append_unit(r_obs, 'M')}")
        emit(f"theta_obs             = {config.theta_obs_deg:.6f} deg")
        emit(
            "integration profile   = "
            f"{format_choice_value('solid_angle_profile', config.solid_angle_profile)}"
        )
        emit(
            f"advanced tuning       = "
            f"{'On' if config.solid_angle_advanced_tuning else 'Off'}"
        )
        emit(f"shadow solid angle    = {omega:.10f} sr")
        emit(f"fraction of full sky  = {fraction:.10%}")
        if config.solid_angle_advanced_tuning:
            emit(
                "integration settings  = "
                f"{format_field_value('solid_angle_base_n_alpha', config.solid_angle_base_n_alpha)} x "
                f"{format_field_value('solid_angle_base_n_theta', config.solid_angle_base_n_theta)}, "
                f"refine={format_field_value('solid_angle_refine_levels', config.solid_angle_refine_levels)}, "
                f"edge={format_field_value('solid_angle_edge_samples', config.solid_angle_edge_samples)}, "
                f"chunk={format_field_value('solid_angle_chunk', config.solid_angle_chunk)}"
            )
        else:
            emit("integration settings  = controlled by solid-angle profile preset")

    if config.benchmark:
        for line in shadow_benchmark_summary_lines(config, stats):
            emit(line)


def run_lensing(
    config: AppConfig,
    log: Callable[[str], None] | None = None,
    show_progress: bool | None = None,
    metric_debug: bool | None = None,
    progress: Callable[[str, float, float], None] | None = None,
) -> None:
    def emit(message: str) -> None:
        if log is None:
            print(message)
        else:
            log(message)

    if show_progress is None:
        show_progress = config.debug and log is None
    if metric_debug is None:
        metric_debug = config.debug and log is None

    def emit_progress(stage_key: str, stage_fraction: float) -> None:
        if progress is None:
            return
        stage_pct = max(0.0, min(100.0, stage_fraction * 100.0))
        progress(STAGE_LABELS[stage_key], stage_pct, overall_progress_percent(stage_key, stage_fraction))

    metric = build_observer_metric(config)

    timings: dict[str, float] = {}
    total_start = perf_counter() if config.benchmark else None

    def bench_start() -> float | None:
        return perf_counter() if config.benchmark else None

    def bench_stop(key: str, start_time: float | None) -> None:
        if config.benchmark and start_time is not None:
            timings[key] = perf_counter() - start_time

    if config.debug:
        metric_mass = append_unit(f"{metric.M:g}", "M")
        metric_spin = append_unit(f"{metric_spin_over_mass(metric):g}", "a/M")
        metric_a = append_unit(f"{getattr(metric, 'a', 0.0):g}", "M")
        emit(
            f"Metric: {type(metric).__name__} "
            f"(M={metric_mass}, spin={metric_spin}, a={metric_a})"
        )

    emit_progress("prepare_source", 0.0)
    stage_start = bench_start()
    img: np.ndarray | None = None
    if config.source_mode == "image":
        image_path = Path(config.input_image).expanduser()
        img = mpimg.imread(str(image_path))
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        height, width = img.shape[:2]
    else:
        height, width = config.height, config.width
    bench_stop("prepare_source", stage_start)
    emit_progress("prepare_source", 1.0)

    if config.debug:
        if config.source_mode == "image":
            emit(f"Image: {format_resolution_value(width, height)}")
        else:
            emit(f"Stripes output: {format_resolution_value(width, height)}")

    r_obs = config.r_obs * metric.M
    theta_obs = np.radians(config.theta_obs_deg)
    alpha_crit = metric.alpha_crit(r_obs, theta_obs)

    vertical_fov = np.radians(config.fov_v)
    horizontal_fov = 2.0 * np.arctan(np.tan(vertical_fov / 2.0) * width / height)
    fov = (horizontal_fov, vertical_fov)
    psi = (np.radians(config.psi_y), np.radians(config.psi_x))

    if config.debug:
        psi_y, psi_x = psi
        bh_y_cam, bh_x_cam, bh_in_front = _psi_to_cam_projection(psi)
        bh_in_fov = (
            bh_in_front
            and abs(bh_y_cam) <= np.tan(vertical_fov / 2.0)
            and abs(bh_x_cam) <= np.tan(horizontal_fov / 2.0)
        )
        bh_status = "behind observer" if not bh_in_front else ("inside FOV" if bh_in_fov else "outside FOV")

        emit(f"r_obs = {r_obs:.1f} M, alpha_crit = {np.degrees(alpha_crit):.4f} deg")
        emit(f"theta_obs = {config.theta_obs_deg:.4f} deg")
        emit(
            "BH screen offset: "
            f"psi_y={np.degrees(psi_y):.4f} deg, "
            f"psi_x={np.degrees(psi_x):.4f} deg "
            f"({bh_status})"
        )
        if config.source_mode == "stripes":
            emit(
                f"Stripe counts: n_x={append_unit(config.n_x, 'stripes')}, "
                f"n_y={append_unit(config.n_y, 'stripes')}"
            )
            if config.color_visualization:
                emit(
                    "Color visualization enabled; rendering stripe hits in "
                    "black and coloring the remaining escaped rays by final "
                    "theta quadrant."
                )
            if config.wrap_outside_background_plane:
                emit("Wrap outside background plane is ignored in stripes mode.")
        elif config.wrap_outside_background_plane:
            emit(
                "Wrap outside background plane enabled; "
                "winding-color and magenta miss fallbacks disabled."
            )

    if metric.is_spherically_symmetric:
        if config.debug:
            emit("Building per-pixel alpha lookup...")
        emit_progress("build_lookup", 0.0)
        stage_start = bench_start()
        alpha_lookup = build_alpha_lookup((height, width), fov, psi=psi)
        bench_stop("build_lookup", stage_start)
        emit_progress("build_lookup", 1.0)

        emit_progress("precompute", 0.0)
        stage_start = bench_start()
        if config.source_mode == "stripes":
            (final_alpha_lookup, winding_lookup,
             total_rays, traced_rays,
             source_direction_lookup) = precompute_final_alpha_lookup_stripes(
                alpha_lookup,
                alpha_crit,
                r_obs,
                metric,
                fov=fov,
                psi=psi,
                return_direction_lookup=config.color_visualization,
                show_progress=show_progress,
                progress_callback=lambda done, total: emit_progress(
                    "precompute",
                    0.0 if total <= 0 else (done / total),
                ),
            )
        else:
            final_alpha_lookup, winding_lookup, total_rays, traced_rays = precompute_final_alpha_lookup_image(
                alpha_lookup,
                alpha_crit,
                r_obs,
                metric,
                show_progress=show_progress,
                progress_callback=lambda done, total: emit_progress(
                    "precompute",
                    0.0 if total <= 0 else (done / total),
                ),
            )
            source_direction_lookup = None
        bench_stop("precompute", stage_start)
        emit_progress("precompute", 1.0)
    else:
        if config.debug:
            emit("Building per-pixel (alpha, theta) lookup...")
        emit_progress("build_lookup", 0.0)
        stage_start = bench_start()
        alpha_lookup = build_alpha_lookup((height, width), fov, psi=psi)
        bench_stop("build_lookup", stage_start)
        emit_progress("build_lookup", 1.0)

        emit_progress("precompute", 0.0)
        stage_start = bench_start()
        if config.source_mode == "stripes":
            (final_alpha_lookup, winding_lookup,
             total_rays, traced_rays,
             source_direction_lookup) = precompute_final_alpha_lookup_2d_stripes(
                alpha_lookup,
                fov,
                alpha_crit,
                r_obs,
                metric,
                theta_obs=theta_obs,
                psi=psi,
                return_direction_lookup=config.color_visualization,
                show_progress=show_progress,
                debug=metric_debug,
                progress_callback=lambda done, total: emit_progress(
                    "precompute",
                    0.0 if total <= 0 else (done / total),
                ),
            )
        else:
            final_alpha_lookup, winding_lookup, total_rays, traced_rays = precompute_final_alpha_lookup_2d_image(
                alpha_lookup,
                fov,
                alpha_crit,
                r_obs,
                metric,
                theta_obs=theta_obs,
                psi=psi,
                show_progress=show_progress,
                debug=metric_debug,
                progress_callback=lambda done, total: emit_progress(
                    "precompute",
                    0.0 if total <= 0 else (done / total),
                ),
            )
            source_direction_lookup = None
        bench_stop("precompute", stage_start)
        emit_progress("precompute", 1.0)

    if config.debug:
        emit(
            f"Traced rays: {append_unit(f'{traced_rays:,}', 'rays')} / "
            f"{append_unit(f'{total_rays:,}', 'rays')}"
        )

    emit_progress("render", 0.0)
    stage_start = bench_start()
    if config.source_mode == "image":
        assert img is not None
        lensed_image = render_lensed_input_image(
            img,
            alpha_lookup,
            final_alpha_lookup,
            winding_lookup,
            alpha_crit,
            fov,
            render_loop_around=False,
            wrap_outside_background_plane=config.wrap_outside_background_plane,
            psi=psi,
        )
    else:
        lensed_image = render_lensed_stripes_image(
            final_alpha_lookup,
            fov,
            config.n_x,
            config.n_y,
            color_visualization=config.color_visualization,
            source_direction_lookup=source_direction_lookup,
            psi=psi,
        )
    bench_stop("render", stage_start)
    emit_progress("render", 1.0)

    output_path = Path(config.output_image).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    emit_progress("save_image", 0.0)
    stage_start = bench_start()
    mpimg.imsave(str(output_path), lensed_image)
    bench_stop("save_image", stage_start)
    emit_progress("save_image", 1.0)

    if config.benchmark and total_start is not None:
        timings["total"] = perf_counter() - total_start
        for line in benchmark_summary_lines((height, width), alpha_crit, total_rays, traced_rays, timings):
            emit(line)

    emit(f"Saved lensed image to: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive wrapper around lens rendering, shadow solid-angle, "
            "and asymmetry-measurement tools"
        )
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Run directly from CLI arguments without the interactive form",
    )
    parser.add_argument(
        "--mode",
        choices=("lensing", "shadow-solid-angle", "asymmetry-measurements"),
        default="lensing",
        help=(
            "Top-level mode: render a lensing image, compute shadow solid angle, "
            "or print asymmetry measurements"
        ),
    )
    parser.add_argument(
        "--source-mode",
        choices=("image", "stripes"),
        default="image",
        help="Background source mode",
    )
    parser.add_argument(
        "--shadow-metric",
        choices=("kerr", "schwarzschild"),
        default="kerr",
        help="Metric used in shadow solid-angle mode",
    )
    parser.add_argument(
        "--solid-angle-profile",
        choices=("quick", "normal", "accurate", "ultra-accurate"),
        default="normal",
        help="Preset for Kerr shadow solid-angle integration; explicit solid-angle knobs override it",
    )
    parser.add_argument(
        "--solid-angle-advanced-tuning",
        action="store_true",
        help="Enable the individual solid-angle integration knobs instead of using only the selected profile",
    )
    parser.add_argument(
        "--asymmetry-measurement",
        choices=ASYMMETRY_MEASUREMENT_VALUES,
        default="all",
        help="Asymmetry mode only. Choose one measurement or 'all'.",
    )
    parser.add_argument(
        "--asymmetry-profile",
        choices=("quick", "normal", "accurate", "ultra-accurate"),
        default="normal",
        help="Preset for asymmetry sampling and refinement; explicit asymmetry knobs override it",
    )
    parser.add_argument(
        "--asymmetry-advanced-tuning",
        action="store_true",
        help="Enable the individual asymmetry performance knobs instead of using only the selected profile",
    )
    parser.add_argument(
        "--asymmetry-circle-fit",
        choices=ASYMMETRY_CIRCLE_FIT_CHOICES,
        default="global",
        help="Circle-fit algorithm used by the circularity measurement",
    )
    parser.add_argument("--input-image", default="image.jpg", help="Background image path")
    parser.add_argument("--output-image", default="lensed_image.png", help="Output image path")
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_STRIPES_IMAGE_DIMENSION[1],
        help="Output width in pixels for stripes mode",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_STRIPES_IMAGE_DIMENSION[0],
        help="Output height in pixels for stripes mode",
    )
    parser.add_argument("--n-x", type=int, default=DEFAULT_N_X, help="Vertical stripe count")
    parser.add_argument("--n-y", type=int, default=DEFAULT_N_Y, help="Horizontal stripe count")
    parser.add_argument(
        "--color-visualization",
        action="store_true",
        help=("Stripes mode only: render stripe hits in black and color the "
              "remaining escaped rays by final theta quadrant"),
    )
    parser.add_argument("--M", type=float, default=1.0, help="BH mass")
    parser.add_argument("--a", type=float, default=0.0, help="Dimensionless BH spin a/M (-1 <= a/M <= 1)")
    parser.add_argument("--r-obs", type=float, default=100.0, help="Observer distance in units of M")
    parser.add_argument(
        "--theta-obs-deg",
        type=float,
        default=90.0,
        help="Observer inclination from the spin axis in deg",
    )
    parser.add_argument("--psi-y", type=float, default=0.0, help="BH vertical offset in deg")
    parser.add_argument("--psi-x", type=float, default=0.0, help="BH horizontal offset in deg")
    parser.add_argument("--fov-v", type=float, default=40.0, help="Vertical field of view in deg")
    parser.add_argument("--debug", action="store_true", help="Enable debug logs and progress bars")
    parser.add_argument("--benchmark", action="store_true", help="Enable benchmark timing summary")
    parser.add_argument(
        "--solid-angle-only",
        action="store_true",
        help="Compatibility alias for --mode shadow-solid-angle",
    )
    parser.add_argument(
        "--solid-angle-base-n-alpha",
        type=int,
        default=None,
        help="Base alpha grid for Kerr shadow solid-angle integration",
    )
    parser.add_argument(
        "--solid-angle-base-n-theta",
        type=int,
        default=None,
        help="Base theta grid for Kerr shadow solid-angle integration",
    )
    parser.add_argument(
        "--solid-angle-refine-levels",
        type=int,
        default=None,
        help="Adaptive refinement depth for Kerr shadow solid-angle integration",
    )
    parser.add_argument(
        "--solid-angle-edge-samples",
        type=int,
        default=None,
        help="Terminal edge subgrid size for Kerr shadow solid-angle integration",
    )
    parser.add_argument(
        "--solid-angle-chunk",
        type=int,
        default=None,
        help="Batch size for Kerr shadow solid-angle ray tracing",
    )
    parser.add_argument(
        "--asymmetry-n-bracket-samples",
        type=int,
        default=None,
        help="Coarse alpha sample count used to bracket alpha_crit(theta)",
    )
    parser.add_argument(
        "--asymmetry-tol",
        type=float,
        default=None,
        help="Bisection stopping tolerance for alpha_crit(theta)",
    )
    parser.add_argument(
        "--asymmetry-max-iter",
        type=int,
        default=None,
        help="Maximum bisection iterations used while refining alpha_crit(theta)",
    )
    parser.add_argument(
        "--asymmetry-n-theta-samples",
        type=int,
        default=None,
        help="Coarse theta sample count used to search for shadow extrema",
    )
    parser.add_argument(
        "--asymmetry-n-refine-samples",
        type=int,
        default=None,
        help="Samples per extremum-refinement pass in asymmetry mode",
    )
    parser.add_argument(
        "--asymmetry-refine-levels",
        type=int,
        default=None,
        help="Number of extremum-refinement rounds in asymmetry mode",
    )
    parser.add_argument(
        "--asymmetry-n-boundary-samples",
        type=int,
        default=None,
        help="Full-boundary sample count used by circularity measurements",
    )
    parser.add_argument(
        "--wrap-outside-background-plane",
        action="store_true",
        help=(
            "Image mode only: wrap rays that miss or fall behind the "
            "background plane back onto the image with modulo indexing "
            "and disable the winding-color and magenta miss fallbacks"
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    solid_angle_profile = args.solid_angle_profile.replace("-", "_")
    solid_angle_preset = solid_angle_profile_preset(solid_angle_profile)
    solid_angle_advanced_tuning = bool(
        args.solid_angle_advanced_tuning
        or args.solid_angle_base_n_alpha is not None
        or args.solid_angle_base_n_theta is not None
        or args.solid_angle_refine_levels is not None
        or args.solid_angle_edge_samples is not None
        or args.solid_angle_chunk is not None
    )
    asymmetry_profile = args.asymmetry_profile.replace("-", "_")
    asymmetry_preset = asymmetry_performance_profile_preset(asymmetry_profile)
    asymmetry_advanced_tuning = bool(
        args.asymmetry_advanced_tuning
        or args.asymmetry_n_bracket_samples is not None
        or args.asymmetry_tol is not None
        or args.asymmetry_max_iter is not None
        or args.asymmetry_n_theta_samples is not None
        or args.asymmetry_n_refine_samples is not None
        or args.asymmetry_refine_levels is not None
        or args.asymmetry_n_boundary_samples is not None
    )

    seed_config = AppConfig(
        operation_mode=(
            "shadow_solid_angle" if (args.solid_angle_only or args.mode == "shadow-solid-angle")
            else "asymmetry_measurements" if args.mode == "asymmetry-measurements"
            else "lensing"
        ),
        source_mode=args.source_mode,
        shadow_metric=args.shadow_metric,
        solid_angle_profile=solid_angle_profile,
        solid_angle_advanced_tuning=solid_angle_advanced_tuning,
        asymmetry_measurement=args.asymmetry_measurement,
        asymmetry_profile=asymmetry_profile,
        asymmetry_advanced_tuning=asymmetry_advanced_tuning,
        asymmetry_circle_fit=args.asymmetry_circle_fit,
        input_image=args.input_image,
        output_image=args.output_image,
        width=args.width,
        height=args.height,
        n_x=args.n_x,
        n_y=args.n_y,
        color_visualization=args.color_visualization,
        M=args.M,
        a=args.a,
        r_obs=args.r_obs,
        theta_obs_deg=args.theta_obs_deg,
        psi_y=args.psi_y,
        psi_x=args.psi_x,
        fov_v=args.fov_v,
        solid_angle_base_n_alpha=(
            args.solid_angle_base_n_alpha
            if args.solid_angle_base_n_alpha is not None
            else solid_angle_preset["solid_angle_base_n_alpha"]
        ),
        solid_angle_base_n_theta=(
            args.solid_angle_base_n_theta
            if args.solid_angle_base_n_theta is not None
            else solid_angle_preset["solid_angle_base_n_theta"]
        ),
        solid_angle_refine_levels=(
            args.solid_angle_refine_levels
            if args.solid_angle_refine_levels is not None
            else solid_angle_preset["solid_angle_refine_levels"]
        ),
        solid_angle_edge_samples=(
            args.solid_angle_edge_samples
            if args.solid_angle_edge_samples is not None
            else solid_angle_preset["solid_angle_edge_samples"]
        ),
        solid_angle_chunk=(
            args.solid_angle_chunk
            if args.solid_angle_chunk is not None
            else solid_angle_preset["solid_angle_chunk"]
        ),
        asymmetry_n_bracket_samples=(
            args.asymmetry_n_bracket_samples
            if args.asymmetry_n_bracket_samples is not None
            else asymmetry_preset["n_bracket_samples"]
        ),
        asymmetry_tol=(
            args.asymmetry_tol
            if args.asymmetry_tol is not None
            else asymmetry_preset["tol"]
        ),
        asymmetry_max_iter=(
            args.asymmetry_max_iter
            if args.asymmetry_max_iter is not None
            else asymmetry_preset["max_iter"]
        ),
        asymmetry_n_theta_samples=(
            args.asymmetry_n_theta_samples
            if args.asymmetry_n_theta_samples is not None
            else asymmetry_preset["n_theta_samples"]
        ),
        asymmetry_n_refine_samples=(
            args.asymmetry_n_refine_samples
            if args.asymmetry_n_refine_samples is not None
            else asymmetry_preset["n_refine_samples"]
        ),
        asymmetry_refine_levels=(
            args.asymmetry_refine_levels
            if args.asymmetry_refine_levels is not None
            else asymmetry_preset["refine_levels"]
        ),
        asymmetry_n_boundary_samples=(
            args.asymmetry_n_boundary_samples
            if args.asymmetry_n_boundary_samples is not None
            else asymmetry_preset["n_boundary_samples"]
        ),
        debug=args.debug,
        benchmark=args.benchmark,
        wrap_outside_background_plane=args.wrap_outside_background_plane,
    )

    if args.no_ui:
        state = config_to_state(seed_config)
        config, err = parse_state(
            state,
            require_source_assets=(seed_config.operation_mode == "lensing"),
        )
        if err:
            print(f"Validation error: {err}")
            return 2
        assert config is not None
        try:
            if config.operation_mode == "lensing":
                run_lensing(config)
            elif config.operation_mode == "shadow_solid_angle":
                run_shadow_solid_angle(config)
            else:
                run_asymmetry_measurements(config)
        except Exception as exc:  # pragma: no cover - user-facing error path
            print(f"Error: {exc}")
            if config.debug:
                raise
            return 1
        return 0

    try:
        curses.wrapper(run_form, seed_config)
    except KeyboardInterrupt:
        print("Cancelled.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
