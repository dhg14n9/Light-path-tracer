#!/usr/bin/env python3
"""Inspect one asymmetry-data run and render Plotly graphs for its quantities."""

from __future__ import annotations

import argparse
import json
import re
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


RUN_ROOT = Path("data/asym_data")
LEGACY_RUN_ROOT = Path("asym_data")
RUN_DIRECTORY_PATTERN = re.compile(r"run_(\d+)$")
SETTINGS_FILENAME = "settings.json"
DATA_FILENAME = "asymmetry_data.h5"


@dataclass(frozen=True)
class RunPaths:
    """Resolved filesystem paths for one saved asymmetry-data run."""

    run_dir: Path
    settings_path: Path
    data_path: Path
    run_index: int | None


@dataclass(frozen=True)
class RunData:
    """Loaded metadata and numeric columns for one saved asymmetry-data run."""

    paths: RunPaths
    settings: dict[str, Any]
    generation_mode: str
    quantity_names: tuple[str, ...]
    spins: np.ndarray
    inclination_degs: np.ndarray
    quantity_columns: dict[str, np.ndarray]


def require_h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "h5py is required to inspect saved asymmetry runs. Install the project "
            "dependencies, then run this script again."
        ) from exc
    return h5py


def require_plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "plotly is required to generate interactive graphs. Install the project "
            "dependencies, then run this script again."
        ) from exc
    return go


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect one asymmetry-data run and optionally write Plotly HTML graphs."
        )
    )
    parser.add_argument(
        "run",
        help=(
            "Run index such as '3' or a run directory such as 'data/asym_data/run_3'."
        ),
    )
    parser.add_argument(
        "--save-html",
        nargs="?",
        const="",
        metavar="PATH",
        help=(
            "Write Plotly HTML output. If PATH is omitted, files are written into the "
            "chosen run directory. If PATH is a directory, one HTML file per quantity "
            "is written there. If PATH ends in .html, it may only be used for a single "
            "selected quantity."
        ),
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open generated HTML files in the default browser after writing them.",
    )
    parser.add_argument(
        "--quantities",
        help=(
            "Comma-separated subset of quantity names to plot. Defaults to all saved "
            "quantities for the run."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        run_data = load_run_data(args.run)
        selected_quantities = resolve_selected_quantities(
            run_data.quantity_names,
            args.quantities,
        )
        if args.save_html is None and not args.open:
            print_run_summary(run_data, selected_quantities)
            print(
                "\nNo graph output requested. Use --save-html to write Plotly HTML "
                "files and add --open to launch them in your browser."
            )
            return 0

        figures = build_figures(run_data, selected_quantities)
        output_paths = write_figures(run_data, figures, args.save_html, args.open)
        print_run_summary(run_data, selected_quantities)
        print("\nWrote Plotly HTML output:")
        for output_path in output_paths:
            print(f"  {output_path}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def load_run_data(run_selector: str) -> RunData:
    paths = resolve_run_paths(run_selector)
    settings = load_settings_document(paths.settings_path)
    generation = require_mapping(settings, "generation", context="settings")
    outputs = require_mapping(settings, "outputs", context="settings")

    generation_mode = str(generation.get("mode", "")).strip()
    if generation_mode not in {"spin_only", "spin_and_inclination"}:
        raise RuntimeError(
            f"Unsupported or missing generation mode {generation_mode!r} in "
            f"{paths.settings_path}."
        )

    quantity_names = tuple(str(name) for name in outputs.get("quantity_names", ()))
    if not quantity_names:
        raise RuntimeError(
            f"No quantity names were saved in {paths.settings_path}. "
            "The run may be incomplete."
        )

    spins, inclination_degs, quantity_columns = load_hdf5_columns(paths, quantity_names)
    return RunData(
        paths=paths,
        settings=settings,
        generation_mode=generation_mode,
        quantity_names=quantity_names,
        spins=spins,
        inclination_degs=inclination_degs,
        quantity_columns=quantity_columns,
    )


def resolve_run_paths(run_selector: str) -> RunPaths:
    raw_value = str(run_selector).strip()
    if not raw_value:
        raise RuntimeError("Run selector must not be empty.")

    candidate = Path(raw_value)
    if raw_value.isdigit():
        run_index = int(raw_value)
        run_dir = RUN_ROOT / f"run_{run_index}"
    else:
        run_dir = _resolve_run_selector_path(candidate)
        run_index = parse_run_index_from_path(run_dir)

    run_dir = run_dir.expanduser().resolve()
    settings_path = run_dir / SETTINGS_FILENAME
    data_path = run_dir / DATA_FILENAME

    if not run_dir.is_dir():
        raise RuntimeError(f"Run directory does not exist: {run_dir}")
    if not settings_path.is_file():
        raise RuntimeError(f"Missing settings file: {settings_path}")
    if not data_path.is_file():
        raise RuntimeError(f"Missing HDF5 data file: {data_path}")

    return RunPaths(
        run_dir=run_dir,
        settings_path=settings_path,
        data_path=data_path,
        run_index=run_index,
    )


def _resolve_run_selector_path(candidate: Path) -> Path:
    if candidate.exists():
        return candidate

    if not candidate.is_absolute():
        parts = candidate.parts
        if parts and parts[0] == LEGACY_RUN_ROOT.name:
            remapped = RUN_ROOT.joinpath(*parts[1:])
            if remapped.exists():
                return remapped

    return candidate


def parse_run_index_from_path(path: Path) -> int | None:
    match = RUN_DIRECTORY_PATTERN.search(path.name)
    if match is None:
        return None
    return int(match.group(1))


def load_settings_document(settings_path: Path) -> dict[str, Any]:
    with settings_path.open("r", encoding="ascii") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise RuntimeError(f"Expected {settings_path} to contain a JSON object.")
    return document


def load_hdf5_columns(
    paths: RunPaths,
    quantity_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    h5py = require_h5py()
    with h5py.File(paths.data_path, "r") as handle:
        if "data" not in handle:
            raise RuntimeError(f"Missing 'data' group in {paths.data_path}.")
        data_group = handle["data"]

        spins = read_required_dataset(data_group, "spin", paths.data_path)
        inclination_degs = read_required_dataset(
            data_group,
            "inclination_deg",
            paths.data_path,
        )
        quantity_columns = {
            quantity_name: read_required_dataset(data_group, quantity_name, paths.data_path)
            for quantity_name in quantity_names
        }

    row_count = int(spins.shape[0])
    if inclination_degs.shape[0] != row_count:
        raise RuntimeError("Saved spin and inclination columns have different lengths.")

    for quantity_name, values in quantity_columns.items():
        if values.shape[0] != row_count:
            raise RuntimeError(
                f"Quantity column {quantity_name!r} has {values.shape[0]} rows, "
                f"expected {row_count}."
            )

    return spins, inclination_degs, quantity_columns


def read_required_dataset(group: Any, name: str, data_path: Path) -> np.ndarray:
    if name not in group:
        raise RuntimeError(f"Missing dataset 'data/{name}' in {data_path}.")
    values = np.asarray(group[name][...], dtype=np.float64).reshape(-1)
    return values


def resolve_selected_quantities(
    available_quantities: tuple[str, ...],
    raw_value: str | None,
) -> tuple[str, ...]:
    if raw_value is None:
        return available_quantities

    requested = tuple(
        quantity.strip()
        for quantity in str(raw_value).split(",")
        if quantity.strip()
    )
    if not requested:
        raise RuntimeError("--quantities was provided but no valid quantity names were parsed.")

    available = set(available_quantities)
    missing = [quantity for quantity in requested if quantity not in available]
    if missing:
        available_text = ", ".join(available_quantities)
        missing_text = ", ".join(missing)
        raise RuntimeError(
            f"Unknown quantity name(s): {missing_text}. Available quantities: {available_text}."
        )
    return requested


def build_figures(run_data: RunData, selected_quantities: tuple[str, ...]) -> dict[str, Any]:
    go = require_plotly()
    title_prefix = format_run_label(run_data.paths)

    figures: dict[str, Any] = {}
    for quantity_name in selected_quantities:
        values = run_data.quantity_columns[quantity_name]
        if run_data.generation_mode == "spin_only":
            order = np.argsort(run_data.spins, kind="stable")
            x_values = run_data.spins[order]
            y_values = values[order]
            figure = go.Figure(
                data=[
                    go.Scatter(
                        x=x_values,
                        y=y_values,
                        mode="lines+markers",
                        name=quantity_name,
                    )
                ]
            )
            figure.update_layout(
                title=f"{title_prefix}: {quantity_name}",
                xaxis_title="Spin (a/M)",
                yaxis_title=quantity_name,
                template="plotly_white",
            )
            figure.add_annotation(
                text=spin_only_annotation(run_data.settings),
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.0,
                y=1.08,
                xanchor="left",
            )
        else:
            figure = go.Figure(
                data=[
                    go.Scatter3d(
                        x=run_data.spins,
                        y=run_data.inclination_degs,
                        z=values,
                        mode="markers",
                        name=quantity_name,
                        marker={
                            "size": 4,
                            "color": values,
                            "colorscale": "Viridis",
                            "opacity": 0.9,
                            "colorbar": {"title": quantity_name},
                        },
                    )
                ]
            )
            figure.update_layout(
                title=f"{title_prefix}: {quantity_name}",
                template="plotly_white",
                scene={
                    "xaxis_title": "Spin (a/M)",
                    "yaxis_title": "Inclination (deg)",
                    "zaxis_title": quantity_name,
                },
            )
        figures[quantity_name] = figure
    return figures


def spin_only_annotation(settings: dict[str, Any]) -> str:
    generation = require_mapping(settings, "generation", context="settings")
    inclination = require_mapping(
        generation,
        "observer_inclination",
        context="settings['generation']",
    )
    fixed_deg = inclination.get("fixed_deg")
    if fixed_deg is None:
        return "Spin-only run"
    return f"Fixed inclination: {float(fixed_deg):g} deg"


def format_run_label(paths: RunPaths) -> str:
    if paths.run_index is None:
        return paths.run_dir.name
    return f"Run {paths.run_index}"


def write_figures(
    run_data: RunData,
    figures: dict[str, Any],
    save_html_value: str | None,
    open_in_browser: bool,
) -> list[Path]:
    if save_html_value is None and open_in_browser:
        target_root = run_data.paths.run_dir
    elif save_html_value is None:
        raise RuntimeError("Internal error: write_figures called without an output target.")
    elif save_html_value == "":
        target_root = run_data.paths.run_dir
    else:
        target_root = Path(save_html_value).expanduser()

    output_paths = resolve_output_paths(run_data.paths, tuple(figures), target_root)
    for quantity_name, output_path in output_paths.items():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figures[quantity_name].write_html(
            str(output_path),
            include_plotlyjs=True,
            full_html=True,
        )
        if open_in_browser:
            webbrowser.open(output_path.resolve().as_uri())
    return [output_paths[quantity_name] for quantity_name in figures]


def resolve_output_paths(
    paths: RunPaths,
    quantity_names: tuple[str, ...],
    target: Path,
) -> dict[str, Path]:
    is_explicit_html_file = target.suffix.lower() == ".html"
    if is_explicit_html_file:
        if len(quantity_names) != 1:
            raise RuntimeError(
                "A single .html output path can only be used when plotting one quantity. "
                "Use --quantities to select one quantity or pass a directory instead."
            )
        return {quantity_names[0]: target.resolve()}

    output_dir = target.resolve()
    prefix = paths.run_dir.name
    return {
        quantity_name: output_dir / f"{prefix}_{slugify(quantity_name)}.html"
        for quantity_name in quantity_names
    }


def slugify(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    normalized = normalized.strip("._-")
    return normalized or "quantity"


def print_run_summary(run_data: RunData, selected_quantities: tuple[str, ...]) -> None:
    generation = require_mapping(run_data.settings, "generation", context="settings")
    spin_sweep = require_mapping(
        generation,
        "spin_sweep",
        context="settings['generation']",
    )
    outputs = require_mapping(run_data.settings, "outputs", context="settings")
    status = outputs.get("status", "unknown")
    completed_points = outputs.get("completed_points", run_data.spins.shape[0])

    print(f"Run directory: {run_data.paths.run_dir}")
    if run_data.paths.run_index is not None:
        print(f"Run index: {run_data.paths.run_index}")
    print(f"Generation mode: {run_data.generation_mode}")
    print(
        "Spin sweep: "
        f"{float(spin_sweep['start']):g} -> {float(spin_sweep['end']):g} "
        f"(step {float(spin_sweep['step']):g})"
    )

    if run_data.generation_mode == "spin_only":
        print(spin_only_annotation(run_data.settings))
    else:
        inclination = require_mapping(
            generation,
            "observer_inclination",
            context="settings['generation']",
        )
        sweep = require_mapping(
            inclination,
            "sweep",
            context="settings['generation']['observer_inclination']",
        )
        print(
            "Inclination sweep: "
            f"{float(sweep['start']):g} -> {float(sweep['end']):g} "
            f"(step {float(sweep['step']):g})"
        )

    print(f"Run status: {status}")
    print(f"Completed points: {int(completed_points)}")
    print(f"Loaded rows: {run_data.spins.shape[0]}")
    print(f"Available quantities: {', '.join(run_data.quantity_names)}")
    print(f"Selected quantities: {', '.join(selected_quantities)}")


def require_mapping(
    parent: dict[str, Any],
    key: str,
    *,
    context: str,
) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected {context}['{key}'] to be a JSON object.")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
