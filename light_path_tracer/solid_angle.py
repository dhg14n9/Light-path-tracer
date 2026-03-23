import argparse
from time import perf_counter

import numpy as np

from .metrics import Kerr

try:
    from numba import njit
except Exception:  # pragma: no cover - optional dependency
    def njit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(func):
            return func
        return decorator


def schwarzschild_shadow_solid_angle(r_obs, M=1.0):
    """
    Solid angle of the Schwarzschild black-hole shadow for a static observer.

    Returns
    -------
    omega : float
        Shadow solid angle in steradians.
    alpha : float
        Angular radius of the shadow in radians.
    """
    r_s = 2.0 * M
    r_photon = 3.0 * M
    b_crit = 3.0 * np.sqrt(3.0) * M

    if r_obs <= r_s:
        raise ValueError(f"Need r_obs > {r_s:.6g} for a static observer.")

    sin_alpha = b_crit * np.sqrt(1.0 - r_s / r_obs) / r_obs
    alpha = np.arcsin(np.clip(sin_alpha, 0.0, 1.0))

    # Inside the photon sphere the shadow covers more than half the sky.
    if r_obs < r_photon:
        alpha = np.pi - alpha

    omega = 2.0 * np.pi * (1.0 - np.cos(alpha))
    return omega, alpha


@njit(cache=True)
def _build_cell_stencil(alpha_lo, alpha_hi, theta_lo, theta_hi):
    n_cells = alpha_lo.shape[0]
    n_samples = 9
    alphas = np.empty(n_cells * n_samples, dtype=np.float64)
    thetas = np.empty(n_cells * n_samples, dtype=np.float64)

    for i in range(n_cells):
        alo = alpha_lo[i]
        ahi = alpha_hi[i]
        tlo = theta_lo[i]
        thi = theta_hi[i]
        amid = 0.5 * (alo + ahi)
        tmid = 0.5 * (tlo + thi)
        base = i * n_samples

        alphas[base + 0] = amid
        thetas[base + 0] = tmid

        alphas[base + 1] = alo
        thetas[base + 1] = tmid

        alphas[base + 2] = ahi
        thetas[base + 2] = tmid

        alphas[base + 3] = amid
        thetas[base + 3] = tlo

        alphas[base + 4] = amid
        thetas[base + 4] = thi

        alphas[base + 5] = alo
        thetas[base + 5] = tlo

        alphas[base + 6] = alo
        thetas[base + 6] = thi

        alphas[base + 7] = ahi
        thetas[base + 7] = tlo

        alphas[base + 8] = ahi
        thetas[base + 8] = thi

    return alphas, thetas


@njit(cache=True)
def _classify_cells(sample_statuses):
    n_cells = sample_statuses.shape[0]
    classes = np.empty(n_cells, dtype=np.int8)

    for i in range(n_cells):
        has_captured = False
        has_escaped = False
        invalid = False
        for j in range(sample_statuses.shape[1]):
            status = sample_statuses[i, j]
            if status == -1:
                has_captured = True
            elif status == 1 or status == 2:
                has_escaped = True
            else:
                invalid = True
                break

        if invalid:
            classes[i] = -2
            continue
        if has_captured and has_escaped:
            classes[i] = 0
        elif has_captured:
            classes[i] = -1
        else:
            classes[i] = 1

    return classes


@njit(cache=True)
def _captured_full_area(alpha_lo, alpha_hi, theta_lo, theta_hi, cell_classes):
    area = 0.0
    for i in range(cell_classes.shape[0]):
        if cell_classes[i] == -1:
            area += ((theta_hi[i] - theta_lo[i])
                     * (np.cos(alpha_lo[i]) - np.cos(alpha_hi[i])))
    return area


@njit(cache=True)
def _subdivide_mixed_cells(alpha_lo, alpha_hi, theta_lo, theta_hi, cell_classes):
    mixed_count = 0
    for i in range(cell_classes.shape[0]):
        if cell_classes[i] == 0:
            mixed_count += 1

    child_count = mixed_count * 4
    child_alpha_lo = np.empty(child_count, dtype=np.float64)
    child_alpha_hi = np.empty(child_count, dtype=np.float64)
    child_theta_lo = np.empty(child_count, dtype=np.float64)
    child_theta_hi = np.empty(child_count, dtype=np.float64)

    out_idx = 0
    for i in range(cell_classes.shape[0]):
        if cell_classes[i] != 0:
            continue

        amid = 0.5 * (alpha_lo[i] + alpha_hi[i])
        tmid = 0.5 * (theta_lo[i] + theta_hi[i])

        child_alpha_lo[out_idx] = alpha_lo[i]
        child_alpha_hi[out_idx] = amid
        child_theta_lo[out_idx] = theta_lo[i]
        child_theta_hi[out_idx] = tmid
        out_idx += 1

        child_alpha_lo[out_idx] = alpha_lo[i]
        child_alpha_hi[out_idx] = amid
        child_theta_lo[out_idx] = tmid
        child_theta_hi[out_idx] = theta_hi[i]
        out_idx += 1

        child_alpha_lo[out_idx] = amid
        child_alpha_hi[out_idx] = alpha_hi[i]
        child_theta_lo[out_idx] = theta_lo[i]
        child_theta_hi[out_idx] = tmid
        out_idx += 1

        child_alpha_lo[out_idx] = amid
        child_alpha_hi[out_idx] = alpha_hi[i]
        child_theta_lo[out_idx] = tmid
        child_theta_hi[out_idx] = theta_hi[i]
        out_idx += 1

    return child_alpha_lo, child_alpha_hi, child_theta_lo, child_theta_hi


@njit(cache=True)
def _build_terminal_subgrid(alpha_lo, alpha_hi, theta_lo, theta_hi, n_side):
    n_cells = alpha_lo.shape[0]
    n_sub = n_cells * n_side * n_side

    sub_alpha_lo = np.empty(n_sub, dtype=np.float64)
    sub_alpha_hi = np.empty(n_sub, dtype=np.float64)
    sub_theta_lo = np.empty(n_sub, dtype=np.float64)
    sub_theta_hi = np.empty(n_sub, dtype=np.float64)
    sample_alpha = np.empty(n_sub, dtype=np.float64)
    sample_theta = np.empty(n_sub, dtype=np.float64)

    out_idx = 0
    for i in range(n_cells):
        d_alpha = (alpha_hi[i] - alpha_lo[i]) / n_side
        d_theta = (theta_hi[i] - theta_lo[i]) / n_side

        for ia in range(n_side):
            alo = alpha_lo[i] + ia * d_alpha
            ahi = alo + d_alpha
            amid = 0.5 * (alo + ahi)
            for it in range(n_side):
                tlo = theta_lo[i] + it * d_theta
                thi = tlo + d_theta
                tmid = 0.5 * (tlo + thi)

                sub_alpha_lo[out_idx] = alo
                sub_alpha_hi[out_idx] = ahi
                sub_theta_lo[out_idx] = tlo
                sub_theta_hi[out_idx] = thi
                sample_alpha[out_idx] = amid
                sample_theta[out_idx] = tmid
                out_idx += 1

    return (sub_alpha_lo, sub_alpha_hi,
            sub_theta_lo, sub_theta_hi,
            sample_alpha, sample_theta)


@njit(cache=True)
def _captured_terminal_area(sub_alpha_lo, sub_alpha_hi,
                            sub_theta_lo, sub_theta_hi,
                            sub_status):
    area = 0.0
    for i in range(sub_status.shape[0]):
        if sub_status[i] == -1:
            area += ((sub_theta_hi[i] - sub_theta_lo[i])
                     * (np.cos(sub_alpha_lo[i]) - np.cos(sub_alpha_hi[i])))
    return area


def _kerr_static_limit_radius(M, a, theta_obs):
    radicand = max(M * M - (a * np.cos(theta_obs)) ** 2, 0.0)
    return M + np.sqrt(radicand)


def _trace_status_points(metric, r_obs, alphas, thetas, theta_obs, chunk,
                         progress_callback=None, progress_label="Trace rays",
                         progress_base=0.0, progress_span=1.0):
    statuses = np.empty(alphas.shape[0], dtype=np.int8)
    total = alphas.shape[0]

    def emit_progress(stage_fraction):
        if progress_callback is None:
            return
        clamped_stage = min(max(float(stage_fraction), 0.0), 1.0)
        overall = progress_base + progress_span * clamped_stage
        progress_callback(
            progress_label,
            clamped_stage,
            min(max(float(overall), 0.0), 1.0),
        )

    if total == 0:
        emit_progress(1.0)
        return statuses, 0

    emit_progress(0.0)

    for start in range(0, total, chunk):
        end = min(start + chunk, alphas.shape[0])
        axis_refines = np.zeros(end - start, dtype=np.bool_)
        metric.trace_rays_batch_status(
            r_obs, alphas[start:end], thetas[start:end],
            theta_obs, axis_refines, statuses[start:end],
        )
        emit_progress(end / total)

    invalid = statuses == 0
    retry_count = 0
    if np.any(invalid):
        retry_idx = np.flatnonzero(invalid)
        retry_count = int(retry_idx.size)
        retry_status = np.empty(retry_idx.size, dtype=np.int8)
        axis_refines = np.ones(retry_idx.size, dtype=np.bool_)
        metric.trace_rays_batch_status(
            r_obs, alphas[retry_idx], thetas[retry_idx],
            theta_obs, axis_refines, retry_status,
        )
        statuses[retry_idx] = retry_status

    emit_progress(1.0)

    if np.any(statuses == 0):
        raise RuntimeError("Kerr solid-angle integration hit unresolved invalid rays.")

    return statuses, retry_count


def kerr_shadow_solid_angle(r_obs, a, M=1.0, theta_obs=np.pi / 2,
                            base_n_alpha=48, base_n_theta=96,
                            refine_levels=4, edge_samples=4,
                            chunk=50_000,
                            progress_callback=None,
                            return_stats=False):
    """
    Numerically integrate the Kerr shadow solid angle over the observer sky.

    Optional callbacks
    ------------------
    progress_callback(stage_label, stage_fraction, overall_fraction)
        Receives progress updates with fractions in [0, 1].
    """
    total_start = perf_counter()
    stats = {
        "base_cells": int(base_n_alpha * base_n_theta),
        "levels_requested": int(refine_levels + 1),
        "levels_processed": 0,
        "cells_evaluated": 0,
        "captured_full_cells": 0,
        "escaped_full_cells": 0,
        "mixed_cells": 0,
        "refined_cells": 0,
        "terminal_mixed_cells": 0,
        "stencil_rays": 0,
        "terminal_rays": 0,
        "retry_rays": 0,
        "setup_time": 0.0,
        "stencil_trace_time": 0.0,
        "classification_time": 0.0,
        "full_area_time": 0.0,
        "subdivide_time": 0.0,
        "terminal_setup_time": 0.0,
        "terminal_trace_time": 0.0,
        "terminal_area_time": 0.0,
        "total_time": 0.0,
    }

    def emit_progress(stage_label, stage_fraction, overall_fraction):
        if progress_callback is None:
            return
        progress_callback(
            stage_label,
            min(max(float(stage_fraction), 0.0), 1.0),
            min(max(float(overall_fraction), 0.0), 1.0),
        )

    if abs(a) > M:
        raise ValueError(f"Need |a| <= M, got a={a:.6g}, M={M:.6g}.")
    if base_n_alpha <= 0 or base_n_theta <= 0:
        raise ValueError("Need positive base grid dimensions.")
    if refine_levels < 0 or edge_samples <= 0:
        raise ValueError("Need non-negative refine_levels and positive edge_samples.")

    static_limit = _kerr_static_limit_radius(M, a, theta_obs)
    if r_obs <= static_limit:
        raise ValueError(
            "Need r_obs outside the Kerr static limit surface: "
            f"r_obs={r_obs:.6g}, r_static={static_limit:.6g}."
        )

    setup_start = perf_counter()
    emit_progress("Prepare integrator", 0.0, 0.0)
    metric = Kerr(M=M, a=a)

    alpha_edges = np.linspace(0.0, np.pi, base_n_alpha + 1, dtype=np.float64)
    theta_edges = np.linspace(-np.pi, np.pi, base_n_theta + 1, dtype=np.float64)

    alpha_lo = np.repeat(alpha_edges[:-1], base_n_theta)
    alpha_hi = np.repeat(alpha_edges[1:], base_n_theta)
    theta_lo = np.tile(theta_edges[:-1], base_n_alpha)
    theta_hi = np.tile(theta_edges[1:], base_n_alpha)
    stats["setup_time"] = perf_counter() - setup_start

    omega = 0.0
    emit_progress("Prepare integrator", 1.0, 0.05)
    total_levels = max(refine_levels + 1, 1)
    level_span = 0.85 / total_levels

    for level in range(refine_levels + 1):
        if alpha_lo.size == 0:
            break

        level_idx = level + 1
        level_base = 0.05 + level * level_span
        level_label = f"Trace level {level_idx}/{total_levels}"
        sample_alpha, sample_theta = _build_cell_stencil(
            alpha_lo, alpha_hi, theta_lo, theta_hi
        )
        stats["levels_processed"] += 1
        stats["cells_evaluated"] += int(alpha_lo.size)
        stats["stencil_rays"] += int(sample_alpha.size)

        trace_start = perf_counter()
        sample_status, retry_count = _trace_status_points(
            metric, r_obs, sample_alpha, sample_theta, theta_obs, chunk,
            progress_callback=progress_callback,
            progress_label=level_label,
            progress_base=level_base,
            progress_span=(level_span if level < refine_levels else level_span * 0.7),
        )
        stats["stencil_trace_time"] += perf_counter() - trace_start
        stats["retry_rays"] += retry_count

        classify_start = perf_counter()
        sample_status = sample_status.reshape(alpha_lo.shape[0], 9)
        cell_classes = _classify_cells(sample_status)
        stats["classification_time"] += perf_counter() - classify_start

        if np.any(cell_classes == -2):
            raise RuntimeError("Cell classification encountered invalid sample status.")

        stats["captured_full_cells"] += int(np.count_nonzero(cell_classes == -1))
        stats["escaped_full_cells"] += int(np.count_nonzero(cell_classes == 1))
        mixed_count = int(np.count_nonzero(cell_classes == 0))
        stats["mixed_cells"] += mixed_count

        area_start = perf_counter()
        omega += _captured_full_area(
            alpha_lo, alpha_hi, theta_lo, theta_hi, cell_classes
        )
        stats["full_area_time"] += perf_counter() - area_start

        mixed_mask = cell_classes == 0
        if not np.any(mixed_mask):
            emit_progress(level_label, 1.0, level_base + level_span)
            break

        if level == refine_levels:
            mixed_alpha_lo = alpha_lo[mixed_mask]
            mixed_alpha_hi = alpha_hi[mixed_mask]
            mixed_theta_lo = theta_lo[mixed_mask]
            mixed_theta_hi = theta_hi[mixed_mask]
            stats["terminal_mixed_cells"] += mixed_count

            terminal_setup_start = perf_counter()
            (sub_alpha_lo, sub_alpha_hi,
             sub_theta_lo, sub_theta_hi,
             sub_sample_alpha, sub_sample_theta) = _build_terminal_subgrid(
                mixed_alpha_lo, mixed_alpha_hi,
                mixed_theta_lo, mixed_theta_hi,
                edge_samples,
            )
            stats["terminal_setup_time"] += perf_counter() - terminal_setup_start
            stats["terminal_rays"] += int(sub_sample_alpha.size)

            terminal_trace_start = perf_counter()
            sub_status, retry_count = _trace_status_points(
                metric, r_obs, sub_sample_alpha, sub_sample_theta, theta_obs, chunk,
                progress_callback=progress_callback,
                progress_label="Trace edge subgrid",
                progress_base=level_base + level_span * 0.7,
                progress_span=level_span * 0.3,
            )
            stats["terminal_trace_time"] += perf_counter() - terminal_trace_start
            stats["retry_rays"] += retry_count

            terminal_area_start = perf_counter()
            omega += _captured_terminal_area(
                sub_alpha_lo, sub_alpha_hi,
                sub_theta_lo, sub_theta_hi,
                sub_status,
            )
            stats["terminal_area_time"] += perf_counter() - terminal_area_start
            break

        stats["refined_cells"] += mixed_count
        subdivide_start = perf_counter()
        alpha_lo, alpha_hi, theta_lo, theta_hi = _subdivide_mixed_cells(
            alpha_lo, alpha_hi, theta_lo, theta_hi, cell_classes
        )
        stats["subdivide_time"] += perf_counter() - subdivide_start
        emit_progress(f"Refine level {level_idx}/{total_levels}", 1.0, level_base + level_span)

    fraction = omega / (4.0 * np.pi)
    stats["total_rays"] = stats["stencil_rays"] + stats["terminal_rays"]
    stats["total_time"] = perf_counter() - total_start
    emit_progress("Finalize", 1.0, 1.0)
    if return_stats:
        return omega, fraction, stats
    return omega, fraction


def main():
    parser = argparse.ArgumentParser(
        description="Compute the solid angle of a black-hole shadow."
    )
    parser.add_argument("--M", type=float, default=1.0)
    parser.add_argument("--a", type=float, default=0.0)
    parser.add_argument("--r-obs", type=float, default=50.0)
    parser.add_argument("--theta-obs-deg", type=float, default=90.0)
    parser.add_argument("--base-n-alpha", type=int, default=48)
    parser.add_argument("--base-n-theta", type=int, default=96)
    parser.add_argument("--refine-levels", type=int, default=4)
    parser.add_argument("--edge-samples", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=50_000)
    args = parser.parse_args()

    theta_obs = np.radians(args.theta_obs_deg)

    if np.isclose(args.a, 0.0):
        omega, alpha = schwarzschild_shadow_solid_angle(args.r_obs, M=args.M)
        print("Schwarzschild black-hole shadow")
        print(f"M                     = {args.M}")
        print(f"r_obs                 = {args.r_obs}")
        print(f"shadow angular radius = {np.degrees(alpha):.6f} deg")
        print(f"shadow solid angle    = {omega:.10f} sr")
        print(f"fraction of full sky  = {omega / (4.0 * np.pi):.10%}")
        return

    omega, fraction = kerr_shadow_solid_angle(
        args.r_obs,
        args.a,
        M=args.M,
        theta_obs=theta_obs,
        base_n_alpha=args.base_n_alpha,
        base_n_theta=args.base_n_theta,
        refine_levels=args.refine_levels,
        edge_samples=args.edge_samples,
        chunk=args.chunk,
    )

    print("Kerr black-hole shadow")
    print(f"M                     = {args.M}")
    print(f"a                     = {args.a}")
    print(f"r_obs                 = {args.r_obs}")
    print(f"theta_obs             = {args.theta_obs_deg:.6f} deg")
    print(f"shadow solid angle    = {omega:.10f} sr")
    print(f"fraction of full sky  = {fraction:.10%}")


if __name__ == "__main__":
    main()
