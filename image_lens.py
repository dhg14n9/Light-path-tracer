# Every set of coordinate is written as (y, x)
# FOV pairs: (horizontal, vertical)

import numpy as np
import matplotlib.image as mpimg
from tqdm import tqdm
from time import perf_counter

from metrics import Schwarzschild, Kerr


WINDING_DTYPE = np.uint16
WINDING_MAX = np.iinfo(WINDING_DTYPE).max
Y_AXIS_REFINE_FRAC = 0.07


# ============================================================================
# Pixel <-> angle conversions
# ============================================================================

def _psi_to_bh_direction(psi):
    """Convert psi=(pitch_up, yaw_right) [rad] to a BH direction in camera coords."""
    psi_y, psi_x = psi
    sin_pitch = np.sin(psi_y)
    cos_pitch = np.cos(psi_y)
    sin_yaw = np.sin(psi_x)
    cos_yaw = np.cos(psi_x)

    # Camera axes: +x right, +y down, +z forward.
    # psi_y > 0 means BH moves upward, hence negative y component.
    return np.array([
        sin_yaw * cos_pitch,
        -sin_pitch,
        cos_yaw * cos_pitch,
    ], dtype=np.float64)


def _psi_frame(psi):
    """Return (d, e_x, e_y, in_front) for BH direction and local screen basis."""
    d = _psi_to_bh_direction(psi)
    in_front = bool(d[2] > 1e-12)

    cam_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    cam_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    # Tangent basis around the BH direction. e_x/e_y align with image axes at psi=0.
    e_x = cam_x - np.dot(cam_x, d) * d
    e_x_norm = np.linalg.norm(e_x)
    if e_x_norm < 1e-12:
        e_x = cam_y - np.dot(cam_y, d) * d
        e_x_norm = np.linalg.norm(e_x)
    e_x /= max(e_x_norm, 1e-12)

    e_y = cam_y - np.dot(cam_y, d) * d - np.dot(cam_y, e_x) * e_x
    e_y_norm = np.linalg.norm(e_y)
    if e_y_norm < 1e-12:
        e_y = np.cross(d, e_x)
        e_y_norm = np.linalg.norm(e_y)
    e_y /= max(e_y_norm, 1e-12)

    return d, e_x, e_y, in_front


def _psi_to_cam_projection(psi):
    """Project BH direction onto the pinhole camera plane; returns (y_cam, x_cam, in_front)."""
    d, _, _, in_front = _psi_frame(psi)
    if not in_front:
        return (np.nan, np.nan, False)
    return (float(d[1] / d[2]), float(d[0] / d[2]), True)


def pixel_to_angles(pixel, image_dimension, fov, psi=(0.0, 0.0)):
    height, width = image_dimension
    horizontal_fov, vertical_fov = fov

    x = pixel[1] - width / 2
    y = pixel[0] - height / 2

    fx = (width / 2) / np.tan(horizontal_fov / 2)
    fy = (height / 2) / np.tan(vertical_fov / 2)

    x_cam = x / fx
    y_cam = y / fy

    d, e_x, e_y, _ = _psi_frame(psi)

    ray = np.array([x_cam, y_cam, 1.0], dtype=np.float64)
    ray /= np.linalg.norm(ray)

    cos_alpha = np.clip(np.dot(ray, d), -1.0, 1.0)
    alpha = float(np.arccos(cos_alpha))
    theta = float(np.arctan2(np.dot(ray, e_x), np.dot(ray, e_y)))
    return (alpha, theta)


def angles_to_pixel(angles, image_dimension, fov, clip=False, psi=(0.0, 0.0)):
    alpha, theta = angles
    height, width = image_dimension
    horizontal_fov, vertical_fov = fov

    fx = (width / 2) / np.tan(horizontal_fov / 2)
    fy = (height / 2) / np.tan(vertical_fov / 2)

    d, e_x, e_y, _ = _psi_frame(psi)

    ray = (np.cos(alpha) * d
           + np.sin(alpha) * (np.sin(theta) * e_x + np.cos(theta) * e_y))
    if ray[2] <= 1e-12:
        if not clip:
            return (-1, -1)
        return (0, 0)

    x_cam = ray[0] / ray[2]
    y_cam = ray[1] / ray[2]

    x = x_cam * fx
    y = y_cam * fy

    px = int(np.rint(x + width / 2))
    py = int(np.rint(y + height / 2))

    if clip:
        px = int(np.clip(px, 0, width - 1))
        py = int(np.clip(py, 0, height - 1))

    return (py, px)


# ============================================================================
# Alpha lookup (1D, for spherically symmetric metrics)
# ============================================================================

def build_alpha_lookup(image_dimension, fov, decimals=None, psi=(0.0, 0.0)):
    """Build a per-pixel alpha lookup table (vectorized)."""
    height, width = image_dimension
    horizontal_fov, vertical_fov = fov

    fx = (width / 2) / np.tan(horizontal_fov / 2)
    fy = (height / 2) / np.tan(vertical_fov / 2)

    x_cam = (np.arange(width) - width / 2) / fx
    y_cam = (np.arange(height) - height / 2) / fy
    d, _, _, _ = _psi_frame(psi)

    denom = np.sqrt(1.0 + x_cam[None, :]**2 + y_cam[:, None]**2)
    cos_alpha = ((x_cam[None, :] * d[0])
                 + (y_cam[:, None] * d[1])
                 + d[2]) / denom
    alpha = np.arccos(np.clip(cos_alpha, -1.0, 1.0))
    if decimals is not None:
        alpha = np.round(alpha, decimals)
    return alpha.astype(np.float32)


def precompute_final_alpha_lookup(alpha_lookup, alpha_crit, r_obs, metric,
                                  show_progress=False,
                                  progress_callback=None):
    """Trace one ray per pixel (1D alpha-only, spherically symmetric)."""
    alpha_flat = alpha_lookup.ravel().astype(np.float64)
    n = alpha_flat.size

    final_alpha_flat = np.full(n, np.nan, dtype=np.float64)
    winding_flat = np.zeros(n, dtype=np.int64)

    if n == 0:
        if progress_callback is not None:
            progress_callback(0, 0)
        return (np.full(alpha_lookup.shape, np.nan, dtype=np.float32),
                np.zeros(alpha_lookup.shape, dtype=WINDING_DTYPE),
                n, 0)

    chunk = 50_000
    for start in tqdm(range(0, n, chunk), desc="Tracing per-pixel rays",
                      unit="chunk", disable=not show_progress):
        end = min(start + chunk, n)
        metric.trace_rays_batch(
            r_obs, alpha_flat[start:end],
            final_alpha_flat[start:end], winding_flat[start:end])
        if progress_callback is not None:
            progress_callback(end, n)

    fa_out = final_alpha_flat.astype(np.float32).reshape(alpha_lookup.shape)
    w_out = np.clip(winding_flat, 0, WINDING_MAX).astype(WINDING_DTYPE).reshape(alpha_lookup.shape)
    return fa_out, w_out, n, n


# ============================================================================
# Alpha+theta lookup (2D, for non-spherically-symmetric metrics like Kerr)
# ============================================================================

def precompute_final_alpha_lookup_2d(
    alpha_lookup, fov, alpha_crit, r_obs, metric,
    theta_obs=np.pi / 2, psi=(0.0, 0.0),
    show_progress=False, debug=False,
    progress_callback=None,
):
    """Trace one ray per pixel for non-spherically-symmetric metrics."""
    shape = alpha_lookup.shape
    height, width = shape

    # Per-pixel screen theta
    hfov, vfov = fov
    fx = (width / 2) / np.tan(hfov / 2)
    fy = (height / 2) / np.tan(vfov / 2)
    x_cam = (np.arange(width) - width / 2) / fx
    y_cam = (np.arange(height) - height / 2) / fy
    d, e_x, e_y, _ = _psi_frame(psi)

    denom = np.sqrt(1.0 + x_cam[None, :]**2 + y_cam[:, None]**2)
    vx = x_cam[None, :] / denom
    vy = y_cam[:, None] / denom
    vz = 1.0 / denom
    theta_pixel = np.arctan2(
        vx * e_x[0] + vy * e_x[1] + vz * e_x[2],
        vx * e_y[0] + vy * e_y[1] + vz * e_y[2],
    )

    _, bh_x_cam, bh_proj_front = _psi_to_cam_projection(psi)
    if bh_proj_front:
        x_rel = x_cam - bh_x_cam
        x_cam_abs_max = max(float(np.max(np.abs(x_rel))), 1e-12)
        axis_refine_cols = np.abs(x_rel) <= (Y_AXIS_REFINE_FRAC * x_cam_abs_max)
    else:
        axis_refine_cols = np.zeros_like(x_cam, dtype=bool)

    use_tb_symmetry = (np.isclose(theta_obs, np.pi / 2)
                       and np.isclose(psi[0], 0.0))
    trace_rows = (height + 1) // 2 if use_tb_symmetry else height

    alpha_trace = alpha_lookup[:trace_rows, :]
    theta_trace = theta_pixel[:trace_rows, :]
    axis_refine_trace = np.broadcast_to(axis_refine_cols[None, :],
                                        (trace_rows, width))

    alpha_trace_flat = alpha_trace.ravel()
    theta_trace_flat = theta_trace.ravel()
    axis_refine_trace_flat = axis_refine_trace.ravel()
    valid_indices = np.arange(alpha_trace_flat.size, dtype=np.intp)

    final_alpha_trace_flat = np.full(alpha_trace_flat.shape, np.nan,
                                     dtype=np.float32)
    winding_trace_flat = np.zeros(alpha_trace_flat.shape, dtype=WINDING_DTYPE)

    if debug:
        if use_tb_symmetry:
            print(f"  tracing {valid_indices.size:,} rays with top/bottom symmetry "
                  f"({alpha_lookup.size:,} pixels total)")
        else:
            print(f"  tracing {valid_indices.size:,} rays "
                  f"({alpha_lookup.size:,} pixels total)")

    if valid_indices.size:
        alpha_f64 = alpha_trace_flat.astype(np.float64)
        theta_f64 = theta_trace_flat.astype(np.float64)
        axis_f64 = axis_refine_trace_flat.astype(np.bool_)

        fa_buf = np.full(alpha_f64.size, np.nan, dtype=np.float64)
        w_buf = np.zeros(alpha_f64.size, dtype=np.int64)

        chunk = 50_000
        for start in tqdm(range(0, alpha_f64.size, chunk),
                          desc="Tracing per-pixel rays", unit="chunk",
                          disable=not show_progress):
            end = min(start + chunk, alpha_f64.size)
            metric.trace_rays_batch(
                r_obs, alpha_f64[start:end], theta_f64[start:end],
                theta_obs, axis_f64[start:end],
                fa_buf[start:end], w_buf[start:end])
            if progress_callback is not None:
                progress_callback(end, alpha_f64.size)

        final_alpha_trace_flat[:] = fa_buf.astype(np.float32)
        winding_trace_flat[:] = np.clip(w_buf, 0, WINDING_MAX).astype(WINDING_DTYPE)
    elif progress_callback is not None:
        progress_callback(0, 0)

    final_alpha_out = np.full(shape, np.nan, dtype=np.float32)
    winding_out = np.zeros(shape, dtype=WINDING_DTYPE)

    final_alpha_trace = final_alpha_trace_flat.reshape((trace_rows, width))
    winding_trace = winding_trace_flat.reshape((trace_rows, width))

    final_alpha_out[:trace_rows, :] = final_alpha_trace
    winding_out[:trace_rows, :] = winding_trace

    if use_tb_symmetry:
        top_half = height // 2
        if top_half > 0:
            final_alpha_out[height - top_half:, :] = final_alpha_out[:top_half, :][::-1, :]
            winding_out[height - top_half:, :] = winding_out[:top_half, :][::-1, :]

    return (final_alpha_out,
            winding_out,
            int(alpha_lookup.size), int(valid_indices.size))


# ============================================================================
# Rendering
# ============================================================================

WINDING_COLORS = np.array([
    [0.0, 0.2, 1.0],   # blue
    [0.0, 0.7, 1.0],   # sky blue
    [0.0, 1.0, 0.4],   # green
    [1.0, 1.0, 0.0],   # yellow
    [1.0, 0.4, 0.0],   # orange
], dtype=np.float32)


def render_lensed_image(source_image, alpha_lookup, final_alpha_lookup,
                        winding_lookup, alpha_crit, fov,
                        render_loop_around=False, psi=(0.0, 0.0)):
    """Render the output image using precomputed alpha lookup tables."""
    height, width = source_image.shape[:2]
    horizontal_fov, vertical_fov = fov
    lensed = np.zeros_like(source_image)

    fx = (width / 2) / np.tan(horizontal_fov / 2)
    fy = (height / 2) / np.tan(vertical_fov / 2)
    x_cam = (np.arange(width) - width / 2) / fx
    y_cam = (np.arange(height) - height / 2) / fy
    d, e_x, e_y, _ = _psi_frame(psi)

    denom = np.sqrt(1.0 + x_cam[None, :]**2 + y_cam[:, None]**2)
    vx = x_cam[None, :] / denom
    vy = y_cam[:, None] / denom
    vz = 1.0 / denom
    theta_lookup = np.arctan2(
        vx * e_x[0] + vy * e_x[1] + vz * e_x[2],
        vx * e_y[0] + vy * e_y[1] + vz * e_y[2],
    )

    valid = np.isfinite(final_alpha_lookup)

    # Winding: escaped rays that actually orbited the BH (n_half_orbits > 0)
    if winding_lookup is not None:
        winding = valid & (winding_lookup > 0)
    else:
        winding = valid & (final_alpha_lookup > np.pi / 2)
    if np.any(winding):
        if winding_lookup is not None:
            idx = np.clip(winding_lookup[winding], 0, len(WINDING_COLORS) - 1)
        else:
            idx = np.zeros(np.count_nonzero(winding), dtype=np.intp)

        if source_image.ndim == 2:
            luma = np.array([0.299, 0.587, 0.114], dtype=np.float32)
            lensed[winding] = (WINDING_COLORS @ luma)[idx]
        else:
            lensed[winding] = WINDING_COLORS[idx]

    # Escaped rays (not winding)
    escaped = valid & ~winding
    n_escaped = np.count_nonzero(escaped)

    if n_escaped > 0:
        fa = final_alpha_lookup[escaped].astype(np.float64)
        th = theta_lookup[escaped]

        sin_fa = np.sin(fa)
        cos_fa = np.cos(fa)
        sin_th = np.sin(th)
        cos_th = np.cos(th)
        src_vx = (cos_fa * d[0]
                  + sin_fa * (sin_th * e_x[0] + cos_th * e_y[0]))
        src_vy = (cos_fa * d[1]
                  + sin_fa * (sin_th * e_x[1] + cos_th * e_y[1]))
        src_vz = (cos_fa * d[2]
                  + sin_fa * (sin_th * e_x[2] + cos_th * e_y[2]))

        if render_loop_around:
            # Keep legacy "wrap" behavior only for the front-facing branch.
            src_x_cam = np.zeros_like(src_vx)
            src_y_cam = np.zeros_like(src_vy)
            front = src_vz > 1e-12
            src_x_cam[front] = src_vx[front] / src_vz[front]
            src_y_cam[front] = src_vy[front] / src_vz[front]
            src_x = np.rint(src_x_cam * fx + width / 2).astype(np.intp)
            src_y = np.rint(src_y_cam * fy + height / 2).astype(np.intp)
            src_y %= height
            src_x %= width
            lensed[escaped] = source_image[src_y, src_x]
        else:
            front = src_vz > 1e-12
            src_x_cam = np.empty_like(src_vx)
            src_y_cam = np.empty_like(src_vy)
            src_x_cam[front] = src_vx[front] / src_vz[front]
            src_y_cam[front] = src_vy[front] / src_vz[front]
            src_x = np.full_like(src_vx, -1, dtype=np.intp)
            src_y = np.full_like(src_vy, -1, dtype=np.intp)
            src_x[front] = np.rint(src_x_cam[front] * fx + width / 2).astype(np.intp)
            src_y[front] = np.rint(src_y_cam[front] * fy + height / 2).astype(np.intp)

            in_bounds = (front
                         & (src_y >= 0) & (src_y < height)
                         & (src_x >= 0) & (src_x < width))

            if source_image.ndim == 3:
                ch = source_image.shape[2]
                magenta = np.zeros(ch, dtype=source_image.dtype)
                magenta[0] = 1.0
                if ch > 2:
                    magenta[2] = 1.0
            else:
                magenta = source_image.dtype.type(1.0)

            result = np.empty(
                (n_escaped,) + source_image.shape[2:],
                dtype=source_image.dtype)
            result[:] = magenta
            result[in_bounds] = source_image[src_y[in_bounds], src_x[in_bounds]]
            lensed[escaped] = result

    return lensed


# ============================================================================
# Debug / Benchmark
# ============================================================================

def _debug_log(enabled, message):
    if enabled:
        print(message)


def _bench_start(enabled):
    if enabled:
        return perf_counter()
    return None


def _bench_stop(enabled, timings, key, start_time):
    if enabled and start_time is not None:
        timings[key] = perf_counter() - start_time


def print_benchmark_summary(image_dimension, alpha_crit, total_rays,
                            traced_rays, timings):
    height, width = image_dimension
    pixel_count = width * height
    render_time = max(timings.get("render", 0.0), 1e-12)
    total_time = max(timings.get("total", 0.0), 1e-12)

    print("\nBenchmark summary")
    print(f"  resolution: {width}x{height} ({pixel_count:,} pixels)")
    print(f"  alpha_crit: {alpha_crit:.6f} rad")
    print(f"  total rays: {total_rays:,}")
    print(f"  traced rays: {traced_rays:,}")
    print(f"  {'load_image':<26}{timings.get('load_image', 0.0):>10.3f} s")
    print(f"  {'build_lookup':<26}{timings.get('build_lookup', 0.0):>10.3f} s")
    print(f"  {'precompute':<26}{timings.get('precompute', 0.0):>10.3f} s")
    print(f"  {'render':<26}{timings.get('render', 0.0):>10.3f} s")
    print(f"  {'save_image':<26}{timings.get('save_image', 0.0):>10.3f} s")
    print(f"  {'total':<26}{timings.get('total', 0.0):>10.3f} s")
    print(f"  {'render_throughput':<26}"
          f"{(pixel_count / render_time) / 1e6:>10.2f} MPix/s")
    print(f"  {'overall_throughput':<26}"
          f"{(pixel_count / total_time) / 1e6:>10.2f} MPix/s")


# ============================================================================
# Main
# ============================================================================

def main(metric=None, M=1.0, a=0.0, r_obs_mult=100.0,
         psi=(0.0, 0.0), vertical_fov_deg=40.0,
         debug=False, benchmark=False):
    if metric is None:
        if a == 0:
            metric = Schwarzschild(M=M)
        else:
            metric = Kerr(M=M, a=a)

    _debug_log(debug, f"Metric: {type(metric).__name__} "
                      f"(M={metric.M}, a={getattr(metric, 'a', 0)})")

    timings = {}
    total_start = _bench_start(benchmark)

    # Load image
    stage_start = _bench_start(benchmark)
    img = mpimg.imread('image.jpg')
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    _bench_stop(benchmark, timings, "load_image", stage_start)

    height, width = img.shape[:2]
    _debug_log(debug, f"Image: {width}x{height}")

    # Observer properties
    r_obs = r_obs_mult * metric.M
    alpha_crit = metric.alpha_crit(r_obs)
    _debug_log(
        debug,
        f"r_obs = {r_obs:.1f} M, alpha_crit = {np.degrees(alpha_crit):.4f} deg",
    )

    vertical_fov = np.radians(vertical_fov_deg)
    horizontal_fov = 2 * np.arctan(np.tan(vertical_fov / 2) * width / height)
    fov = (horizontal_fov, vertical_fov)
    psi_y, psi_x = psi
    bh_y_cam, bh_x_cam, bh_in_front = _psi_to_cam_projection(psi)
    bh_in_fov = (bh_in_front
                 and abs(bh_y_cam) <= np.tan(vertical_fov / 2)
                 and abs(bh_x_cam) <= np.tan(horizontal_fov / 2))
    bh_pos_status = ("behind observer" if not bh_in_front
                     else ("inside FOV" if bh_in_fov else "outside FOV"))
    _debug_log(
        debug,
        "BH screen offset: "
        f"psi_y={np.degrees(psi_y):.4f} deg, "
        f"psi_x={np.degrees(psi_x):.4f} deg "
        f"({bh_pos_status})",
    )

    render_loop_around = False
    if metric.is_spherically_symmetric:
        _debug_log(debug, "Building per-pixel alpha lookup...")
        stage_start = _bench_start(benchmark)
        alpha_lookup = build_alpha_lookup((height, width), fov, psi=psi)
        _bench_stop(benchmark, timings, "build_lookup", stage_start)

        stage_start = _bench_start(benchmark)
        (final_alpha_lookup, winding_lookup,
         total_rays, traced_rays) = precompute_final_alpha_lookup(
            alpha_lookup, alpha_crit, r_obs, metric,
            show_progress=debug)
        _bench_stop(benchmark, timings, "precompute", stage_start)
    else:
        _debug_log(debug, "Building per-pixel (alpha, theta) lookup...")
        stage_start = _bench_start(benchmark)
        alpha_lookup = build_alpha_lookup((height, width), fov, psi=psi)
        _bench_stop(benchmark, timings, "build_lookup", stage_start)

        stage_start = _bench_start(benchmark)
        (final_alpha_lookup, winding_lookup,
         total_rays, traced_rays) = precompute_final_alpha_lookup_2d(
            alpha_lookup, fov, alpha_crit, r_obs, metric,
            psi=psi, show_progress=debug, debug=debug)
        _bench_stop(benchmark, timings, "precompute", stage_start)

    # Render
    stage_start = _bench_start(benchmark)
    lensed_image = render_lensed_image(
        img, alpha_lookup, final_alpha_lookup, winding_lookup,
        alpha_crit, fov, render_loop_around, psi=psi,
    )
    _bench_stop(benchmark, timings, "render", stage_start)

    # Save
    stage_start = _bench_start(benchmark)
    mpimg.imsave('lensed_image.png', lensed_image)
    _bench_stop(benchmark, timings, "save_image", stage_start)

    if benchmark and total_start is not None:
        timings["total"] = perf_counter() - total_start
        print_benchmark_summary(
            (height, width), alpha_crit, total_rays, traced_rays, timings)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=float, default=1.0, help="BH mass")
    parser.add_argument("--a", type=float, default=0.0,
                        help="BH spin (|a| <= M, 0 = Schwarzschild)")
    parser.add_argument("--r-obs", type=float, default=100.0,
                        help="Observer distance in units of M (default: 100)")
    parser.add_argument("--psi-y", type=float, default=0.0,
                        help="BH vertical offset in deg (+ = top, - = bottom)")
    parser.add_argument("--psi-x", type=float, default=0.0,
                        help="BH horizontal offset in deg (+ = right, - = left)")
    parser.add_argument("--fov-v", type=float, default=40.0,
                        help="Vertical field of view in deg")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logs and progress bars")
    parser.add_argument("--benchmark", action="store_true",
                        help="Enable benchmark timing summary")
    args = parser.parse_args()
    main(M=args.M, a=args.a, r_obs_mult=args.r_obs,
         psi=(np.radians(args.psi_y), np.radians(args.psi_x)),
         vertical_fov_deg=args.fov_v,
         debug=args.debug, benchmark=args.benchmark)
