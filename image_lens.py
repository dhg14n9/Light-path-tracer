# Every set of coordinate is written as (y, x)
# FOV pairs: (horizontal, vertical)

import numpy as np
import matplotlib.image as mpimg
from tqdm import tqdm
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor

from metrics import Schwarzschild, Kerr


WINDING_DTYPE = np.uint16
WINDING_MAX = np.iinfo(WINDING_DTYPE).max
Y_AXIS_REFINE_FRAC = 0.07


# ============================================================================
# Pixel <-> angle conversions
# ============================================================================

def _psi_to_cam_offset(psi):
    """Convert BH screen offset psi=(psi_y, psi_x) [rad] to camera-plane offset."""
    psi_y, psi_x = psi
    # Image y grows downward, but psi_y > 0 means "move BH upward".
    return (-np.tan(psi_y), np.tan(psi_x))


def pixel_to_angles(pixel, image_dimension, fov, psi=(0.0, 0.0)):
    height, width = image_dimension
    horizontal_fov, vertical_fov = fov

    x = pixel[1] - width / 2
    y = pixel[0] - height / 2

    fx = (width / 2) / np.tan(horizontal_fov / 2)
    fy = (height / 2) / np.tan(vertical_fov / 2)

    x_cam = x / fx
    y_cam = y / fy
    bh_y_cam, bh_x_cam = _psi_to_cam_offset(psi)
    x_cam -= bh_x_cam
    y_cam -= bh_y_cam

    alpha = np.arctan2(np.sqrt(x_cam**2 + y_cam**2), 1.0)
    theta = np.arctan2(x_cam, y_cam)
    return (alpha, theta)


def angles_to_pixel(angles, image_dimension, fov, clip=False, psi=(0.0, 0.0)):
    alpha, theta = angles
    height, width = image_dimension
    horizontal_fov, vertical_fov = fov

    fx = (width / 2) / np.tan(horizontal_fov / 2)
    fy = (height / 2) / np.tan(vertical_fov / 2)

    r_cam = np.tan(alpha)
    x_cam = r_cam * np.sin(theta)
    y_cam = r_cam * np.cos(theta)
    bh_y_cam, bh_x_cam = _psi_to_cam_offset(psi)
    x_cam += bh_x_cam
    y_cam += bh_y_cam

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
    bh_y_cam, bh_x_cam = _psi_to_cam_offset(psi)
    x_cam = x_cam - bh_x_cam
    y_cam = y_cam - bh_y_cam

    alpha = np.arctan2(np.hypot(x_cam[None, :], y_cam[:, None]), 1.0)
    if decimals is not None:
        alpha = np.round(alpha, decimals)
    return alpha.astype(np.float32)


# Global reference to metric for multiprocessing workers
_worker_metric = None


def _init_worker(metric_cls_name, metric_kwargs):
    """Initialize the metric in each worker process."""
    global _worker_metric
    from metrics import Schwarzschild, Kerr
    cls = {'Schwarzschild': Schwarzschild, 'Kerr': Kerr}[metric_cls_name]
    _worker_metric = cls(**metric_kwargs)


def _trace_single_alpha(args):
    """Trace one alpha bin (1D, spherically symmetric)."""
    idx, alpha, r_obs = args
    final_alpha, n_half_orbits, outcome = _worker_metric.trace_ray(
        r_obs, alpha)
    if outcome != 'escaped':
        return (idx, np.nan, 0)
    return (idx, final_alpha, n_half_orbits)


def _trace_single_alpha_theta(args):
    """Trace one (alpha, theta) bin (2D, for non-spherically symmetric)."""
    idx, alpha, screen_theta, r_obs, theta_obs, axis_refine = args
    try:
        final_alpha, n_half_orbits, outcome = _worker_metric.trace_ray(
            r_obs, alpha, theta=screen_theta, theta_obs=theta_obs,
            axis_refine=bool(axis_refine))
    except TypeError:
        # Fallback for metrics that do not expose axis_refine.
        final_alpha, n_half_orbits, outcome = _worker_metric.trace_ray(
            r_obs, alpha, theta=screen_theta, theta_obs=theta_obs)
    if outcome != 'escaped':
        return (idx, np.nan, 0)
    return (idx, final_alpha, n_half_orbits)


def precompute_final_alpha_lookup(
    alpha_lookup, alpha_crit, r_obs, metric,
    num_worker=None,
):
    """Trace one ray per pixel (1D alpha-only, spherically symmetric)."""
    alpha_flat = alpha_lookup.ravel()
    valid_indices = np.arange(alpha_flat.size, dtype=np.intp)

    final_alpha_flat = np.full(alpha_flat.shape, np.nan, dtype=np.float32)
    winding_flat = np.zeros(alpha_flat.shape, dtype=WINDING_DTYPE)

    if valid_indices.size == 0:
        return (final_alpha_flat.reshape(alpha_lookup.shape),
                winding_flat.reshape(alpha_lookup.shape),
                int(alpha_flat.size), 0)

    metric_kwargs = {'M': metric.M}
    if hasattr(metric, 'a'):
        metric_kwargs['a'] = metric.a

    with ProcessPoolExecutor(
        max_workers=num_worker,
        initializer=_init_worker,
        initargs=(type(metric).__name__, metric_kwargs),
    ) as executor:
        task_iter = (
            (int(idx), float(alpha_flat[idx]), r_obs)
            for idx in valid_indices
        )
        for idx, final_alpha, n_half_orbits in tqdm(
            executor.map(_trace_single_alpha, task_iter, chunksize=256),
            total=int(valid_indices.size),
            desc="Tracing per-pixel rays",
            unit="ray",
        ):
            final_alpha_flat[idx] = final_alpha
            winding_flat[idx] = min(int(n_half_orbits), WINDING_MAX)

    return (final_alpha_flat.reshape(alpha_lookup.shape),
            winding_flat.reshape(alpha_lookup.shape),
            int(alpha_flat.size), int(valid_indices.size))


# ============================================================================
# Alpha+theta lookup (2D, for non-spherically-symmetric metrics like Kerr)
# ============================================================================

def precompute_final_alpha_lookup_2d(
    alpha_lookup, fov, alpha_crit, r_obs, metric,
    theta_obs=np.pi / 2, num_worker=None, psi=(0.0, 0.0),
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
    bh_y_cam, bh_x_cam = _psi_to_cam_offset(psi)
    x_cam = x_cam - bh_x_cam
    y_cam = y_cam - bh_y_cam
    theta_pixel = np.arctan2(x_cam[None, :], y_cam[:, None])
    x_cam_abs_max = max(float(np.max(np.abs(x_cam))), 1e-12)
    axis_refine_cols = np.abs(x_cam) <= (Y_AXIS_REFINE_FRAC * x_cam_abs_max)

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

    if use_tb_symmetry:
        print(f"  tracing {valid_indices.size:,} rays with top/bottom symmetry "
              f"({alpha_lookup.size:,} pixels total)")
    else:
        print(f"  tracing {valid_indices.size:,} rays "
              f"({alpha_lookup.size:,} pixels total)")

    if valid_indices.size:
        metric_kwargs = {'M': metric.M}
        if hasattr(metric, 'a'):
            metric_kwargs['a'] = metric.a

        with ProcessPoolExecutor(
            max_workers=num_worker,
            initializer=_init_worker,
            initargs=(type(metric).__name__, metric_kwargs),
        ) as executor:
            task_iter = (
                (int(idx), float(alpha_trace_flat[idx]),
                 float(theta_trace_flat[idx]),
                 r_obs, theta_obs,
                 bool(axis_refine_trace_flat[idx]))
                for idx in valid_indices
            )
            for idx, final_alpha, n_half_orbits in tqdm(
                executor.map(_trace_single_alpha_theta, task_iter,
                             chunksize=256),
                total=int(valid_indices.size),
                desc="Tracing per-pixel rays",
                unit="ray",
            ):
                final_alpha_trace_flat[idx] = final_alpha
                winding_trace_flat[idx] = min(int(n_half_orbits), WINDING_MAX)

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
    bh_y_cam, bh_x_cam = _psi_to_cam_offset(psi)
    x_cam = x_cam - bh_x_cam
    y_cam = y_cam - bh_y_cam
    theta_lookup = np.arctan2(x_cam[None, :], y_cam[:, None])

    valid = alpha_lookup >= alpha_crit

    # Winding: escaped rays that turned past pi/2
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

    # Escaped and within FOV
    escaped = valid & ~np.isnan(final_alpha_lookup) & (
        final_alpha_lookup <= np.pi / 2)
    n_escaped = np.count_nonzero(escaped)

    if n_escaped > 0:
        fa = final_alpha_lookup[escaped].astype(np.float64)
        th = theta_lookup[escaped]

        r_cam = np.tan(fa)
        src_x_cam = r_cam * np.sin(th) + bh_x_cam
        src_y_cam = r_cam * np.cos(th) + bh_y_cam
        src_x = np.rint(src_x_cam * fx + width / 2).astype(np.intp)
        src_y = np.rint(src_y_cam * fy + height / 2).astype(np.intp)

        if render_loop_around:
            src_y %= height
            src_x %= width
            lensed[escaped] = source_image[src_y, src_x]
        else:
            in_bounds = ((src_y >= 0) & (src_y < height)
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
# Benchmark
# ============================================================================

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
         psi=(0.0, 0.0), vertical_fov_deg=40.0):
    if metric is None:
        if a == 0:
            metric = Schwarzschild(M=M)
        else:
            metric = Kerr(M=M, a=a)

    print(f"Metric: {type(metric).__name__} "
          f"(M={metric.M}, a={getattr(metric, 'a', 0)})")

    debug_benchmark = True
    timings = {}
    total_start = perf_counter()

    # Load image
    stage_start = perf_counter()
    img = mpimg.imread('image.jpg')
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    timings["load_image"] = perf_counter() - stage_start

    height, width = img.shape[:2]
    print(f"Image: {width}x{height}")

    # Observer properties
    r_obs = r_obs_mult * metric.M
    alpha_crit = metric.alpha_crit(r_obs)
    print(f"r_obs = {r_obs:.1f} M, alpha_crit = {np.degrees(alpha_crit):.4f} deg")

    vertical_fov = np.radians(vertical_fov_deg)
    horizontal_fov = 2 * np.arctan(np.tan(vertical_fov / 2) * width / height)
    fov = (horizontal_fov, vertical_fov)
    psi_y, psi_x = psi
    bh_in_fov = (abs(psi_y) <= vertical_fov / 2
                 and abs(psi_x) <= horizontal_fov / 2)
    print("BH screen offset: "
          f"psi_y={np.degrees(psi_y):.4f} deg, "
          f"psi_x={np.degrees(psi_x):.4f} deg "
          f"({'inside' if bh_in_fov else 'outside'} FOV)")

    render_loop_around = False
    if metric.is_spherically_symmetric:
        print("Building per-pixel alpha lookup...")
        stage_start = perf_counter()
        alpha_lookup = build_alpha_lookup((height, width), fov, psi=psi)
        timings["build_lookup"] = perf_counter() - stage_start

        stage_start = perf_counter()
        (final_alpha_lookup, winding_lookup,
         total_rays, traced_rays) = precompute_final_alpha_lookup(
            alpha_lookup, alpha_crit, r_obs, metric)
        timings["precompute"] = perf_counter() - stage_start
    else:
        print("Building per-pixel (alpha, theta) lookup...")
        stage_start = perf_counter()
        alpha_lookup = build_alpha_lookup((height, width), fov, psi=psi)
        timings["build_lookup"] = perf_counter() - stage_start

        stage_start = perf_counter()
        (final_alpha_lookup, winding_lookup,
         total_rays, traced_rays) = precompute_final_alpha_lookup_2d(
            alpha_lookup, fov, alpha_crit, r_obs, metric, psi=psi)
        timings["precompute"] = perf_counter() - stage_start

    # Render
    stage_start = perf_counter()
    lensed_image = render_lensed_image(
        img, alpha_lookup, final_alpha_lookup, winding_lookup,
        alpha_crit, fov, render_loop_around, psi=psi,
    )
    timings["render"] = perf_counter() - stage_start

    # Save
    stage_start = perf_counter()
    mpimg.imsave('lensed_image.png', lensed_image)
    timings["save_image"] = perf_counter() - stage_start
    timings["total"] = perf_counter() - total_start

    if debug_benchmark:
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
    args = parser.parse_args()
    main(M=args.M, a=args.a, r_obs_mult=args.r_obs,
         psi=(np.radians(args.psi_y), np.radians(args.psi_x)),
         vertical_fov_deg=args.fov_v)
