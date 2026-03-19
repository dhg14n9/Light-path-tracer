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
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import matplotlib.image as mpimg
import numpy as np

from image_lens import (
    _psi_to_cam_projection,
    build_alpha_lookup,
    precompute_final_alpha_lookup as precompute_final_alpha_lookup_image,
    precompute_final_alpha_lookup_2d as precompute_final_alpha_lookup_2d_image,
    render_lensed_image as render_lensed_input_image,
)
from metrics import Kerr, Schwarzschild
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
    source_mode: str = "image"
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
    psi_y: float = 0.0
    psi_x: float = 0.0
    fov_v: float = 40.0
    debug: bool = False
    benchmark: bool = False
    wrap_outside_background_plane: bool = False


FIELD_SPECS: list[dict[str, str]] = [
    {"key": "source_mode", "label": "Background mode", "kind": "choice"},
    {"key": "input_image", "label": "Background image", "kind": "text"},
    {"key": "output_image", "label": "Output image", "kind": "text"},
    {"key": "width", "label": "Output width (stripes)", "kind": "int"},
    {"key": "height", "label": "Output height (stripes)", "kind": "int"},
    {"key": "n_x", "label": "Vertical stripe count", "kind": "int"},
    {"key": "n_y", "label": "Horizontal stripe count", "kind": "int"},
    {"key": "color_visualization", "label": "Color visualization", "kind": "bool"},
    {"key": "M", "label": "BH mass (M)", "kind": "float"},
    {"key": "a", "label": "BH spin (a)", "kind": "float"},
    {"key": "r_obs", "label": "Observer distance (in M)", "kind": "float"},
    {"key": "psi_y", "label": "BH vertical offset (deg)", "kind": "float"},
    {"key": "psi_x", "label": "BH horizontal offset (deg)", "kind": "float"},
    {"key": "fov_v", "label": "Vertical FOV (deg)", "kind": "float"},
    {"key": "debug", "label": "Debug logs + progress", "kind": "bool"},
    {"key": "benchmark", "label": "Benchmark summary", "kind": "bool"},
    {
        "key": "wrap_outside_background_plane",
        "label": "Wrap outside background plane (image)",
        "kind": "bool",
    },
]

ACTION_SPECS: list[tuple[str, str]] = [
    ("run", "Run lensing"),
    ("reset", "Reset defaults"),
    ("quit", "Quit"),
]

STAGE_SPECS: list[tuple[str, str, float]] = [
    ("prepare_source", "Prepare source", 0.05),
    ("build_lookup", "Build lookup", 0.05),
    ("precompute", "Trace rays", 0.75),
    ("render", "Render image", 0.10),
    ("save_image", "Save image", 0.05),
]
STAGE_LABELS = {key: label for key, label, _ in STAGE_SPECS}
SOURCE_MODE_LABELS = {
    "image": "Normal",
    "stripes": "Stripes",
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


def toggle_source_mode(value: str) -> str:
    return "stripes" if value == "image" else "image"


def get_visible_field_specs(state: dict[str, Any]) -> list[dict[str, str]]:
    source_mode = str(state.get("source_mode", "image")).strip().lower()
    visible: list[dict[str, str]] = []
    for spec in FIELD_SPECS:
        key = spec["key"]
        if source_mode == "image" and key in STRIPES_MODE_ONLY_FIELDS:
            continue
        if source_mode == "stripes" and key in IMAGE_MODE_ONLY_FIELDS:
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
        "source_mode": config.source_mode,
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
        "psi_y": f"{config.psi_y:g}",
        "psi_x": f"{config.psi_x:g}",
        "fov_v": f"{config.fov_v:g}",
        "debug": bool(config.debug),
        "benchmark": bool(config.benchmark),
        "wrap_outside_background_plane": bool(config.wrap_outside_background_plane),
    }


def parse_state(state: dict[str, Any]) -> tuple[AppConfig | None, str | None]:
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

    source_mode = str(state["source_mode"]).strip().lower()
    if source_mode not in {"image", "stripes"}:
        return None, "Source mode must be either 'image' or 'stripes'"

    m, err = parse_float("M", "BH mass")
    if err:
        return None, err
    a, err = parse_float("a", "BH spin")
    if err:
        return None, err
    r_obs, err = parse_float("r_obs", "Observer distance")
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

    assert m is not None and a is not None and r_obs is not None
    assert psi_y is not None and psi_x is not None and fov_v is not None
    assert width is not None and height is not None
    assert n_x is not None and n_y is not None

    if m <= 0:
        return None, "BH mass must be > 0"
    if abs(a) > m:
        return None, "For Kerr metric, |a| must be <= M"
    if r_obs <= 0:
        return None, "Observer distance must be > 0"
    if fov_v <= 0 or fov_v >= 179:
        return None, "Vertical FOV must be in (0, 179) degrees"
    if width <= 0 or height <= 0:
        return None, "Output width and height must both be > 0"
    if n_x <= 0 or n_y <= 0:
        return None, "Stripe counts must both be > 0"

    input_image = str(state["input_image"]).strip()
    output_image = str(state["output_image"]).strip()

    if not output_image:
        return None, "Output image path cannot be empty"

    if source_mode == "image":
        if not input_image:
            return None, "Background image path cannot be empty in image mode"
        input_path = Path(input_image).expanduser()
        if not input_path.is_file():
            return None, f"Background image not found: {input_path}"

    return AppConfig(
        source_mode=source_mode,
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
        psi_y=psi_y,
        psi_x=psi_x,
        fov_v=fov_v,
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
    help_line = "Arrows or j/k: move | Enter: edit/toggle | Space: toggle | q: quit"
    stdscr.addnstr(0, 2, title, w - 4, curses.A_BOLD)
    stdscr.addnstr(1, 2, help_line, w - 4)

    row = 3
    for idx, spec in enumerate(visible_fields):
        key = spec["key"]
        label = spec["label"]
        kind = spec["kind"]

        if kind == "bool":
            val_text = "[x]" if state[key] else "[ ]"
        elif kind == "choice" and key == "source_mode":
            mode_label = SOURCE_MODE_LABELS.get(str(state[key]), str(state[key]))
            val_text = f"[ {mode_label} ]"
        else:
            val_text = f"[ {state[key]} ]"

        line = f"{label:<30} {val_text}"
        attr = curses.A_REVERSE if idx == cursor else curses.A_NORMAL
        stdscr.addnstr(row, 2, line, w - 4, attr)
        row += 1

    row += 1
    stdscr.addnstr(row, 2, "Actions", w - 4, curses.A_BOLD)
    row += 1

    action_start = len(visible_fields)
    for i, (_, label) in enumerate(ACTION_SPECS):
        idx = action_start + i
        line = f"[ {label} ]"
        attr = curses.A_REVERSE if idx == cursor else curses.A_NORMAL
        stdscr.addnstr(row, 2, line, w - 4, attr)
        row += 1

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
    status = "Edit values and choose Run lensing."

    def redraw() -> None:
        visible_fields = get_visible_field_specs(state)
        draw_form(stdscr, state, visible_fields, cursor, status, logs, run_info)

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
                    state[key] = not state[key]
                    status = f"{spec['label']} set to {state[key]}"
                elif spec["kind"] == "choice" and key == "source_mode":
                    state[key] = toggle_source_mode(str(state[key]))
                    status = f"{spec['label']} set to {SOURCE_MODE_LABELS[state[key]]}"
            continue

        if not is_enter_key(ch):
            continue

        if cursor < len(visible_fields):
            spec = visible_fields[cursor]
            key = spec["key"]
            kind = spec["kind"]

            if kind == "bool":
                state[key] = not state[key]
                status = f"{spec['label']} set to {state[key]}"
                continue
            if kind == "choice" and key == "source_mode":
                state[key] = toggle_source_mode(str(state[key]))
                status = f"{spec['label']} set to {SOURCE_MODE_LABELS[state[key]]}"
                continue

            new_value = prompt_input(stdscr, spec["label"], str(state[key]))
            if new_value is None:
                status = "Edit cancelled."
            else:
                state[key] = new_value
                status = f"Updated {spec['label']}."
            continue

        action_idx = cursor - len(visible_fields)
        action_key, action_label = ACTION_SPECS[action_idx]

        if action_key == "quit":
            return

        if action_key == "reset":
            state = dict(defaults)
            status = "Defaults restored."
            append_log("Values reset to defaults.")
            continue

        if action_key == "run":
            config, err = parse_state(state)
            if err:
                status = f"Validation error: {err}"
                continue
            logs.clear()
            status = f"Running: {action_label}..."
            append_log(f"Starting run in '{config.source_mode}' mode")
            if config.source_mode == "image":
                append_log(f"Using input '{config.input_image}'")
            else:
                append_log(
                    f"Using stripes at {config.width}x{config.height} "
                    f"(n_x={config.n_x}, n_y={config.n_y})"
                )
                if config.color_visualization:
                    append_log(
                        "Color visualization will render stripe hits in black "
                        "and color the remaining escaped rays by final theta "
                        "quadrant."
                    )
            append_log(f"Output will be saved to '{config.output_image}'")
            run_info["stage_label"] = "Preparing"
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
                run_lensing(
                    config,
                    log=append_log,
                    show_progress=False,
                    metric_debug=False,
                    progress=update_progress,
                )
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
            status = "Generation complete. You can edit and run again."


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
    return [
        "Benchmark summary",
        f"  resolution: {width}x{height} ({pixel_count:,} pixels)",
        f"  alpha_crit: {alpha_crit:.6f} rad",
        f"  total rays: {total_rays:,}",
        f"  traced rays: {traced_rays:,}",
        f"  {'prepare_source':<26}{timings.get('prepare_source', 0.0):>10.3f} s",
        f"  {'build_lookup':<26}{timings.get('build_lookup', 0.0):>10.3f} s",
        f"  {'precompute':<26}{timings.get('precompute', 0.0):>10.3f} s",
        f"  {'render':<26}{timings.get('render', 0.0):>10.3f} s",
        f"  {'save_image':<26}{timings.get('save_image', 0.0):>10.3f} s",
        f"  {'total':<26}{timings.get('total', 0.0):>10.3f} s",
        f"  {'render_throughput':<26}{(pixel_count / render_time) / 1e6:>10.2f} MPix/s",
        f"  {'overall_throughput':<26}{(pixel_count / total_time) / 1e6:>10.2f} MPix/s",
    ]


def overall_progress_percent(stage_key: str, stage_fraction: float) -> float:
    clamped_stage = max(0.0, min(1.0, stage_fraction))
    total = 0.0
    for key, _, weight in STAGE_SPECS:
        if key == stage_key:
            total += weight * clamped_stage
            break
        total += weight
    return max(0.0, min(100.0, total * 100.0))


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

    metric = Schwarzschild(M=config.M) if np.isclose(config.a, 0.0) else Kerr(M=config.M, a=config.a)

    timings: dict[str, float] = {}
    total_start = perf_counter() if config.benchmark else None

    def bench_start() -> float | None:
        return perf_counter() if config.benchmark else None

    def bench_stop(key: str, start_time: float | None) -> None:
        if config.benchmark and start_time is not None:
            timings[key] = perf_counter() - start_time

    if config.debug:
        emit(f"Metric: {type(metric).__name__} (M={metric.M}, a={getattr(metric, 'a', 0.0)})")

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
            emit(f"Image: {width}x{height}")
        else:
            emit(f"Stripes output: {width}x{height}")

    r_obs = config.r_obs * metric.M
    alpha_crit = metric.alpha_crit(r_obs)

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
        emit(
            "BH screen offset: "
            f"psi_y={np.degrees(psi_y):.4f} deg, "
            f"psi_x={np.degrees(psi_x):.4f} deg "
            f"({bh_status})"
        )
        if config.source_mode == "stripes":
            emit(f"Stripe counts: n_x={config.n_x}, n_y={config.n_y}")
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
        emit(f"Traced rays: {traced_rays:,}/{total_rays:,}")

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
        description="Interactive wrapper around image_lens.py with stripes mode"
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Run directly from CLI arguments without the interactive form",
    )
    parser.add_argument(
        "--source-mode",
        choices=("image", "stripes"),
        default="image",
        help="Background source mode",
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
    parser.add_argument("--a", type=float, default=0.0, help="BH spin (|a| <= M)")
    parser.add_argument("--r-obs", type=float, default=100.0, help="Observer distance in units of M")
    parser.add_argument("--psi-y", type=float, default=0.0, help="BH vertical offset in deg")
    parser.add_argument("--psi-x", type=float, default=0.0, help="BH horizontal offset in deg")
    parser.add_argument("--fov-v", type=float, default=40.0, help="Vertical field of view in deg")
    parser.add_argument("--debug", action="store_true", help="Enable debug logs and progress bars")
    parser.add_argument("--benchmark", action="store_true", help="Enable benchmark timing summary")
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

    seed_config = AppConfig(
        source_mode=args.source_mode,
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
        psi_y=args.psi_y,
        psi_x=args.psi_x,
        fov_v=args.fov_v,
        debug=args.debug,
        benchmark=args.benchmark,
        wrap_outside_background_plane=args.wrap_outside_background_plane,
    )

    if args.no_ui:
        state = config_to_state(seed_config)
        config, err = parse_state(state)
        if err:
            print(f"Validation error: {err}")
            return 2
        assert config is not None
        try:
            run_lensing(config)
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
