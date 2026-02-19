# Every set of coordinate is written as (y, x)
# FOV pairs: (horizontal, vertical)

import numpy as np
import matplotlib.image as mpimg
from tqdm import tqdm
import os
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor, as_completed

from metrics import Schwarzschild, Kerr


# ============================================================================
# Pixel <-> angle conversions
# ============================================================================

def pixel_to_angles(pixel, image_dimension, fov):
    height, width = image_dimension
    horizontal_fov, vertical_fov = fov

    x = pixel[1] - width / 2
    y = pixel[0] - height / 2

    fx = (width / 2) / np.tan(horizontal_fov / 2)
    fy = (height / 2) / np.tan(vertical_fov / 2)

    x_cam = x / fx
    y_cam = y / fy

    alpha = np.arctan2(np.sqrt(x_cam**2 + y_cam**2), 1.0)
    theta = np.arctan2(x_cam, y_cam)
    return (alpha, theta)


def angles_to_pixel(angles, image_dimension, fov, clip=False):
    alpha, theta = angles
    height, width = image_dimension
    horizontal_fov, vertical_fov = fov

    fx = (width / 2) / np.tan(horizontal_fov / 2)
    fy = (height / 2) / np.tan(vertical_fov / 2)

    r_cam = np.tan(alpha)
    x_cam = r_cam * np.sin(theta)
    y_cam = r_cam * np.cos(theta)

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

def build_alpha_lookup(image_dimension, fov, decimals=4):
    """Build a per-pixel alpha lookup table (vectorized)."""
    height, width = image_dimension
    horizontal_fov, vertical_fov = fov

    fx = (width / 2) / np.tan(horizontal_fov / 2)
    fy = (height / 2) / np.tan(vertical_fov / 2)

    x_cam = (np.arange(width) - width / 2) / fx
    y_cam = (np.arange(height) - height / 2) / fy

    alpha = np.arctan2(np.hypot(x_cam[None, :], y_cam[:, None]), 1.0)
    return np.round(alpha, decimals).astype(np.float32)


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
    idx, alpha, screen_theta, r_obs, theta_obs = args
    final_alpha, n_half_orbits, outcome = _worker_metric.trace_ray(
        r_obs, alpha, theta=screen_theta, theta_obs=theta_obs)
    if outcome != 'escaped':
        return (idx, np.nan, 0)
    return (idx, final_alpha, n_half_orbits)


def precompute_final_alpha_lookup(
    alpha_lookup, alpha_crit, r_obs, metric,
    num_worker=None,
):
    """Precompute per-pixel final alpha by tracing unique alpha bins (1D).

    For spherically symmetric metrics only.
    """
    unique_alpha, inverse_alpha_idx = np.unique(
        alpha_lookup, return_inverse=True)
    valid_unique_mask = unique_alpha >= alpha_crit
    valid_indices = np.flatnonzero(valid_unique_mask)

    final_alpha_for_unique = np.full(unique_alpha.shape, np.nan,
                                     dtype=np.float32)
    winding_for_unique = np.zeros(unique_alpha.shape, dtype=np.int16)

    tasks = [(idx, unique_alpha[idx], r_obs) for idx in valid_indices]

    if not tasks:
        flat = inverse_alpha_idx.reshape(alpha_lookup.shape)
        return (final_alpha_for_unique[flat],
                winding_for_unique[flat],
                int(unique_alpha.size), int(valid_indices.size))

    metric_kwargs = {'M': metric.M}
    if hasattr(metric, 'a'):
        metric_kwargs['a'] = metric.a

    with ProcessPoolExecutor(
        max_workers=num_worker,
        initializer=_init_worker,
        initargs=(type(metric).__name__, metric_kwargs),
    ) as executor:
        futures = [executor.submit(_trace_single_alpha, task)
                   for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Precomputing alpha lookup", unit="alpha"):
            idx, final_alpha, n_half_orbits = future.result()
            final_alpha_for_unique[idx] = final_alpha
            winding_for_unique[idx] = n_half_orbits

    flat = inverse_alpha_idx.reshape(alpha_lookup.shape)
    return (final_alpha_for_unique[flat],
            winding_for_unique[flat],
            int(unique_alpha.size), int(valid_indices.size))


# ============================================================================
# Alpha+theta lookup (2D, for non-spherically-symmetric metrics like Kerr)
# ============================================================================

def precompute_final_alpha_lookup_2d(
    alpha_lookup, fov, alpha_crit, r_obs, metric,
    n_theta_bins=32, theta_obs=np.pi / 2, num_worker=None,
):
    """Precompute per-pixel final alpha for non-spherically-symmetric metrics.

    Uses unique alpha bins (same as 1D) × coarse theta bins.
    Reflection symmetry theta -> -theta halves the work for equatorial
    observers.
    """
    shape = alpha_lookup.shape
    height, width = shape

    # Unique alphas (same strategy as 1D)
    unique_alpha, inv_alpha = np.unique(alpha_lookup, return_inverse=True)
    inv_alpha = inv_alpha.reshape(shape)
    n_alpha = len(unique_alpha)

    # Per-pixel screen theta
    hfov, vfov = fov
    fx = (width / 2) / np.tan(hfov / 2)
    fy = (height / 2) / np.tan(vfov / 2)
    x_cam = (np.arange(width) - width / 2) / fx
    y_cam = (np.arange(height) - height / 2) / fy
    theta_pixel = np.arctan2(x_cam[None, :], y_cam[:, None])

    # Theta binning (|theta| symmetry for equatorial observer)
    use_symmetry = np.isclose(theta_obs, np.pi / 2)
    if use_symmetry:
        theta_for_bin = np.abs(theta_pixel)
        theta_edges = np.linspace(0, np.pi, n_theta_bins + 1)
    else:
        theta_for_bin = theta_pixel
        theta_edges = np.linspace(-np.pi, np.pi, n_theta_bins + 1)

    theta_bin_idx = np.clip(
        np.digitize(theta_for_bin, theta_edges) - 1,
        0, n_theta_bins - 1)
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])

    # Valid alpha mask
    valid_alpha_mask = unique_alpha >= alpha_crit
    valid_alpha_indices = np.flatnonzero(valid_alpha_mask)

    # Result grid: (n_alpha, n_theta_bins)
    final_alpha_grid = np.full(
        (n_alpha, n_theta_bins), np.nan, dtype=np.float32)
    winding_grid = np.zeros((n_alpha, n_theta_bins), dtype=np.int16)

    # Build tasks for valid (alpha, theta_bin) pairs
    tasks = []
    for ai in valid_alpha_indices:
        for ti in range(n_theta_bins):
            grid_idx = int(ai) * n_theta_bins + ti
            tasks.append((grid_idx, float(unique_alpha[ai]),
                          float(theta_centers[ti]), r_obs, theta_obs))

    n_grid = n_alpha * n_theta_bins
    print(f"  {n_alpha} unique alphas x {n_theta_bins} theta bins "
          f"= {n_grid:,} grid points, {len(tasks):,} to trace")

    if tasks:
        metric_kwargs = {'M': metric.M}
        if hasattr(metric, 'a'):
            metric_kwargs['a'] = metric.a

        with ProcessPoolExecutor(
            max_workers=num_worker,
            initializer=_init_worker,
            initargs=(type(metric).__name__, metric_kwargs),
        ) as executor:
            futures = [executor.submit(_trace_single_alpha_theta, task)
                       for task in tasks]
            for future in tqdm(
                as_completed(futures), total=len(futures),
                desc="Tracing rays", unit="ray",
            ):
                grid_idx, final_alpha, n_half_orbits = future.result()
                ai = grid_idx // n_theta_bins
                ti = grid_idx % n_theta_bins
                final_alpha_grid[ai, ti] = final_alpha
                winding_grid[ai, ti] = n_half_orbits

    # Map grid → pixels
    final_alpha_out = final_alpha_grid[inv_alpha, theta_bin_idx]
    winding_out = winding_grid[inv_alpha, theta_bin_idx]
    return (final_alpha_out, winding_out, n_grid, len(tasks))


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
                        render_loop_around=False):
    """Render the output image using precomputed alpha lookup tables."""
    height, width = source_image.shape[:2]
    horizontal_fov, vertical_fov = fov
    lensed = np.zeros_like(source_image)

    fx = (width / 2) / np.tan(horizontal_fov / 2)
    fy = (height / 2) / np.tan(vertical_fov / 2)
    x_cam = (np.arange(width) - width / 2) / fx
    y_cam = (np.arange(height) - height / 2) / fy
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
        src_x = np.rint(r_cam * np.sin(th) * fx + width / 2).astype(np.intp)
        src_y = np.rint(r_cam * np.cos(th) * fy + height / 2).astype(np.intp)

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

def print_benchmark_summary(image_dimension, alpha_crit, unique_bins,
                            traced_bins, timings):
    height, width = image_dimension
    pixel_count = width * height
    render_time = max(timings.get("render", 0.0), 1e-12)
    total_time = max(timings.get("total", 0.0), 1e-12)

    print("\nBenchmark summary")
    print(f"  resolution: {width}x{height} ({pixel_count:,} pixels)")
    print(f"  alpha_crit: {alpha_crit:.6f} rad")
    print(f"  unique bins: {unique_bins:,}")
    print(f"  traced bins: {traced_bins:,}")
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
# Cache
# ============================================================================

CACHE_FILE = "lookup_cache.npz"


def _cache_params(height, width, fov, decimals, r_obs, alpha_crit,
                  metric_name, metric_extra=None):
    name_hash = float(hash(metric_name) % 2**32)
    base = [height, width, fov[0], fov[1], decimals, r_obs, alpha_crit,
            name_hash]
    if metric_extra is not None:
        base.extend(metric_extra)
    return np.array(base)


def load_lookup_cache(height, width, fov, decimals, r_obs, alpha_crit,
                      metric):
    if not os.path.isfile(CACHE_FILE):
        return None
    try:
        data = np.load(CACHE_FILE)
        metric_extra = _metric_extra(metric)
        expected = _cache_params(height, width, fov, decimals, r_obs,
                                 alpha_crit, type(metric).__name__,
                                 metric_extra)
        if not np.allclose(data["params"], expected):
            return None
        if "winding_lookup" not in data:
            return None
        return (
            data["alpha_lookup"],
            data["final_alpha_lookup"],
            data["winding_lookup"],
            int(data["unique_bins"]),
            int(data["traced_bins"]),
        )
    except Exception:
        return None


def _metric_extra(metric):
    """Extra float parameters to include in cache key."""
    extra = [metric.M]
    if hasattr(metric, 'a'):
        extra.append(metric.a)
    return extra


def save_lookup_cache(alpha_lookup, final_alpha_lookup, winding_lookup,
                      unique_bins, traced_bins,
                      height, width, fov, decimals, r_obs, alpha_crit,
                      metric):
    metric_extra = _metric_extra(metric)
    np.savez(
        CACHE_FILE,
        params=_cache_params(height, width, fov, decimals, r_obs, alpha_crit,
                             type(metric).__name__, metric_extra),
        alpha_lookup=alpha_lookup,
        final_alpha_lookup=final_alpha_lookup,
        winding_lookup=winding_lookup,
        unique_bins=np.array(unique_bins),
        traced_bins=np.array(traced_bins),
    )


# ============================================================================
# Main
# ============================================================================

def main(metric=None, M=1.0, a=0.0):
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
    r_obs = 100.0 * metric.M
    alpha_crit = metric.alpha_crit(r_obs)
    print(f"r_obs = {r_obs:.1f} M, alpha_crit = {np.degrees(alpha_crit):.4f} deg")

    vertical_fov_deg = 40.0
    vertical_fov = np.radians(vertical_fov_deg)
    horizontal_fov = 2 * np.arctan(np.tan(vertical_fov / 2) * width / height)
    fov = (horizontal_fov, vertical_fov)

    render_loop_around = False
    decimals = 4

    # Try cache
    cached = load_lookup_cache(height, width, fov, decimals, r_obs,
                               alpha_crit, metric)
    if cached is not None:
        (alpha_lookup, final_alpha_lookup, winding_lookup,
         unique_bins, traced_bins) = cached
        timings["build_lookup"] = 0.0
        timings["precompute"] = 0.0
        print("Loaded lookup tables from cache.")
    else:
        if metric.is_spherically_symmetric:
            # 1D alpha-only lookup (fast path)
            print("Building 1D alpha lookup...")
            stage_start = perf_counter()
            alpha_lookup = build_alpha_lookup(
                (height, width), fov, decimals=decimals)
            timings["build_lookup"] = perf_counter() - stage_start

            stage_start = perf_counter()
            (final_alpha_lookup, winding_lookup,
             unique_bins, traced_bins) = precompute_final_alpha_lookup(
                alpha_lookup, alpha_crit, r_obs, metric)
            timings["precompute"] = perf_counter() - stage_start
        else:
            # 2D: reuse 1D alpha quantization + coarse theta bins
            print("Building 2D (alpha, theta) lookup...")
            stage_start = perf_counter()
            alpha_lookup = build_alpha_lookup(
                (height, width), fov, decimals=decimals)
            timings["build_lookup"] = perf_counter() - stage_start

            stage_start = perf_counter()
            (final_alpha_lookup, winding_lookup,
             unique_bins, traced_bins) = precompute_final_alpha_lookup_2d(
                alpha_lookup, fov, alpha_crit, r_obs, metric,
                n_theta_bins=4)
            timings["precompute"] = perf_counter() - stage_start

        save_lookup_cache(
            alpha_lookup, final_alpha_lookup, winding_lookup,
            unique_bins, traced_bins,
            height, width, fov, decimals, r_obs, alpha_crit, metric,
        )
        print("Saved lookup tables to cache.")

    # Render
    stage_start = perf_counter()
    lensed_image = render_lensed_image(
        img, alpha_lookup, final_alpha_lookup, winding_lookup,
        alpha_crit, fov, render_loop_around,
    )
    timings["render"] = perf_counter() - stage_start

    # Save
    stage_start = perf_counter()
    mpimg.imsave('lensed_image.png', lensed_image)
    timings["save_image"] = perf_counter() - stage_start
    timings["total"] = perf_counter() - total_start

    if debug_benchmark:
        print_benchmark_summary(
            (height, width), alpha_crit, unique_bins, traced_bins, timings)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=float, default=1.0, help="BH mass")
    parser.add_argument("--a", type=float, default=0.0,
                        help="BH spin (|a| <= M, 0 = Schwarzschild)")
    args = parser.parse_args()
    main(M=args.M, a=args.a)
