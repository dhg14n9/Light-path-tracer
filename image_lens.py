# Every set of coordinate is written as (y, x)
# FOV pairs: (horizontal, vertical)

import numpy as np
import matplotlib.image as mpimg
from tqdm import tqdm
import os
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor, as_completed

from geodesic_tracer import (
    M, R_S, R_PHOTON, B_CRIT,
    trace_ray, trace_ray_orbit, viewing_angle_to_impact_parameter
)

def pixel_to_angles(
    pixel: tuple[int, int], 
    image_dimension: tuple[int, int], 
    fov: tuple[float, float]
) -> tuple[float, float]:
    """
    This function takes in the pixel coordinates, the image dimensions, and fov and
    returns the corresponding angles (alpha, theta) for the given pixel.
    
    Args:
        pixel (tuple[int, int]): The coordinate of the pixel in the image
        image_dimension (tuple[int, int]): Dimension of the image (height, width)
        fov (tuple[float, float]): Field of view in radians (horizontal, vertical)

    Returns:
        tuple[float, float]: The corresponding angles (alpha, theta) for the given pixel
    """
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


def angles_to_pixel(
    angles: tuple[float, float], 
    image_dimension: tuple[int, int], 
    fov: tuple[float, float], 
    clip: bool = False
) -> tuple[int, int]:
    """
    Inverse of ``pixel_to_angles``: convert (alpha, theta) back to pixel (y, x).

    Args:
        angles (tuple[float, float]): (alpha, theta) in radians.
        image_dimension (tuple[int, int]): Image dimension (height, width).
        fov (tuple[float, float]): Field of view in radians (horizontal, vertical).
        clip (bool): Clamp output pixel to image bounds if True.

    Returns:
        tuple[int, int]: Pixel coordinate (y, x).
    """
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
        # TODO: This clipping is not yet perfect (I think)
        px = int(np.clip(px, 0, width - 1))
        py = int(np.clip(py, 0, height - 1))
        pass

    return (py, px)

def get_final_ray_angle(r_obs, alpha):
    """
    Trace a ray and return (final_heading, phi_final).
    Assumes the photon escapes.
    """
    heading, phi_final, outcome = trace_ray_orbit(r_obs, alpha)
    if outcome != 'escaped':
        return np.nan, 0.0
    return heading, phi_final


def world_heading_to_viewing_angle(world_heading: float) -> float:
    """
    Convert a world-space heading angle (from +x axis) to camera viewing angle
    from the forward axis (-x toward the black hole center).
    Essentially fix stuff
    """
    return np.arccos(np.clip(-np.cos(world_heading), -1.0, 1.0))



#! As I implement the alpha first look up table, this function will become obsolete
def get_color_from_background(
    angles: tuple[float, float], 
    background_image: np.ndarray, 
    fov: tuple[float, float], 
    r_obs: float
) -> np.ndarray:
    """
    This function takes in the final angles (alpha, theta) of the photon after escaping and returns the corresponding color from the background image.

    Args:
        angles (tuple[float, float]): (alpha, theta) in radians (initial angles from the observer).
        background_image (np.ndarray): Background image array.
        fov (tuple[float, float]): Field of view in radians (horizontal, vertical).

    Returns:
        np.ndarray: Color of the pixel in the background image corresponding to the given angles.
    """
    
    # check if the rays will actually escape the black hole
    alpha, theta = angles
    b = viewing_angle_to_impact_parameter(alpha, r_obs)
    if b < B_CRIT:
        return np.array([0.0, 0.0, 0.0])

    final_heading, _ = get_final_ray_angle(r_obs, alpha)
    final_alpha = world_heading_to_viewing_angle(final_heading)
    final_pixel = angles_to_pixel((final_alpha, theta), background_image.shape[:2], fov)
    
    # for debugging purposes, returns a specific color if the ray escapes but goes out of bounds of the background image
    if final_pixel[0] < 0 or final_pixel[0] >= background_image.shape[0] or final_pixel[1] < 0 or final_pixel[1] >= background_image.shape[1]:
        return np.array([1.0, 0.0, 1.0])
    
    return background_image[final_pixel]


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

def _trace_single_alpha(args):
    """Trace one alpha bin, return (index, final_alpha, n_half_orbits)."""
    idx, alpha, r_obs = args
    final_heading, phi_final = get_final_ray_angle(r_obs, alpha)
    final_alpha = world_heading_to_viewing_angle(final_heading)
    n_half_orbits = int(abs(phi_final) // np.pi)
    return (idx, final_alpha, n_half_orbits)

def precompute_final_alpha_lookup(
    alpha_lookup: np.ndarray,
    alpha_crit: float,
    r_obs: float,
    num_worker: int = None, # type: ignore
) -> tuple[np.ndarray, int, int]:
    """
    Precompute per-pixel final alpha by tracing only unique alpha bins that can escape.

    Args:
        alpha_lookup (np.ndarray): Per-pixel alpha array.
        alpha_crit (float): Critical angle in radians.
        r_obs (float): Observer radius.

    Returns:
        tuple: (final_alpha_lookup, winding_lookup, unique_alpha_bins, traced_alpha_bins).
    """
    unique_alpha, inverse_alpha_idx = np.unique(alpha_lookup, return_inverse=True)
    valid_unique_mask = unique_alpha >= alpha_crit
    valid_indices = np.flatnonzero(valid_unique_mask)

    final_alpha_for_unique = np.full(unique_alpha.shape, np.nan, dtype=np.float32)
    winding_for_unique = np.zeros(unique_alpha.shape, dtype=np.int16)

    tasks = [(idx, unique_alpha[idx], r_obs) for idx in valid_indices]

    if not tasks:
        final_alpha_lookup = final_alpha_for_unique[inverse_alpha_idx].reshape(alpha_lookup.shape)
        winding_lookup = winding_for_unique[inverse_alpha_idx].reshape(alpha_lookup.shape)
        return final_alpha_lookup, winding_lookup, int(unique_alpha.size), int(valid_indices.size)

    with ProcessPoolExecutor(max_workers=num_worker) as executor:
        futures = [executor.submit(_trace_single_alpha, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Precomputing alpha lookup", unit="alpha"):
            idx, final_alpha, n_half_orbits = future.result()
            final_alpha_for_unique[idx] = final_alpha
            winding_for_unique[idx] = n_half_orbits

    final_alpha_lookup = final_alpha_for_unique[inverse_alpha_idx].reshape(alpha_lookup.shape)
    winding_lookup = winding_for_unique[inverse_alpha_idx].reshape(alpha_lookup.shape)
    return final_alpha_lookup, winding_lookup, int(unique_alpha.size), int(valid_indices.size)


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
    """Render the output image using precomputed alpha lookup tables (vectorized)."""
    height, width = source_image.shape[:2]
    horizontal_fov, vertical_fov = fov
    lensed = np.zeros_like(source_image)

    # Vectorized theta lookup
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
    escaped = valid & ~np.isnan(final_alpha_lookup) & (final_alpha_lookup <= np.pi / 2)
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

            # Build per-escaped-pixel result: magenta default, source for in-bounds
            if source_image.ndim == 3:
                ch = source_image.shape[2]
                magenta = np.zeros(ch, dtype=source_image.dtype)
                magenta[0] = 1.0
                if ch > 2:
                    magenta[2] = 1.0
            else:
                magenta = source_image.dtype.type(1.0)

            result = np.empty(
                (n_escaped,) + source_image.shape[2:], dtype=source_image.dtype)
            result[:] = magenta
            result[in_bounds] = source_image[src_y[in_bounds], src_x[in_bounds]]
            lensed[escaped] = result

    return lensed


def print_benchmark_summary(
    image_dimension: tuple[int, int],
    alpha_crit: float,
    unique_alpha_bins: int,
    traced_alpha_bins: int,
    timings: dict[str, float],
) -> None:
    """
    Print a compact benchmark summary for the current render.
    """
    height, width = image_dimension
    pixel_count = width * height
    render_time = max(timings.get("render", 0.0), 1e-12)
    total_time = max(timings.get("total", 0.0), 1e-12)

    print("\nBenchmark summary")
    print(f"  resolution: {width}x{height} ({pixel_count:,} pixels)")
    print(f"  alpha_crit: {alpha_crit:.6f} rad")
    print(f"  unique alpha bins: {unique_alpha_bins:,}")
    print(f"  traced alpha bins: {traced_alpha_bins:,}")
    print(f"  {'load_image':<26}{timings.get('load_image', 0.0):>10.3f} s")
    print(f"  {'build_alpha_lookup':<26}{timings.get('build_alpha_lookup', 0.0):>10.3f} s")
    print(f"  {'precompute_final_alpha':<26}{timings.get('precompute_final_alpha', 0.0):>10.3f} s")
    print(f"  {'render':<26}{timings.get('render', 0.0):>10.3f} s")
    print(f"  {'save_image':<26}{timings.get('save_image', 0.0):>10.3f} s")
    print(f"  {'total':<26}{timings.get('total', 0.0):>10.3f} s")
    print(f"  {'render_throughput':<26}{(pixel_count / render_time) / 1e6:>10.2f} MPix/s")
    print(f"  {'overall_throughput':<26}{(pixel_count / total_time) / 1e6:>10.2f} MPix/s")





CACHE_FILE = "lookup_cache.npz"


def _cache_params(height, width, fov, decimals, r_obs, alpha_crit):
    return np.array([height, width, fov[0], fov[1], decimals, r_obs, alpha_crit])


def load_lookup_cache(height, width, fov, decimals, r_obs, alpha_crit):
    if not os.path.isfile(CACHE_FILE):
        return None
    try:
        data = np.load(CACHE_FILE)
        expected = _cache_params(height, width, fov, decimals, r_obs, alpha_crit)
        if not np.allclose(data["params"], expected):
            return None
        if "winding_lookup" not in data:
            return None
        winding = data["winding_lookup"]
        return (
            data["alpha_lookup"],
            data["final_alpha_lookup"],
            winding,
            int(data["unique_alpha_bins"]),
            int(data["traced_alpha_bins"]),
        )
    except Exception:
        return None


def save_lookup_cache(
    alpha_lookup, final_alpha_lookup, winding_lookup,
    unique_alpha_bins, traced_alpha_bins,
    height, width, fov, decimals, r_obs, alpha_crit,
):
    np.savez(
        CACHE_FILE,
        params=_cache_params(height, width, fov, decimals, r_obs, alpha_crit),
        alpha_lookup=alpha_lookup,
        final_alpha_lookup=final_alpha_lookup,
        winding_lookup=winding_lookup,
        unique_alpha_bins=np.array(unique_alpha_bins),
        traced_alpha_bins=np.array(traced_alpha_bins),
    )


def main():
    debug_benchmark: bool = True
    timings: dict[str, float] = {}
    total_start = perf_counter()

    # load image
    stage_start = perf_counter()
    img = mpimg.imread('image.jpg')
    
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    timings["load_image"] = perf_counter() - stage_start
    
    height, width = img.shape[:2]
    
    # obesrver properties
    r_obs = 100.0 * M
    alpha_crit_arg = B_CRIT * np.sqrt(1 - R_S / r_obs) / r_obs
    alpha_crit = np.arcsin(np.clip(alpha_crit_arg, -1.0, 1.0))
    
    vertical_fov_deg = 40.0
    vertical_fov: float = np.radians(vertical_fov_deg)
    horizontal_fov = 2 * np.arctan(np.tan(vertical_fov / 2) * width / height)
    fov = (horizontal_fov, vertical_fov)
    
    render_loop_around: bool = False # poor naming, I don't really know how to call this variable. If True, when the ray lands outside the bounds of the background image, it will be tiled instead of being marked as out of bounds.
    
    decimals = 4
    cached = load_lookup_cache(height, width, fov, decimals, r_obs, alpha_crit)
    if cached is not None:
        alpha_lookup, final_alpha_lookup, winding_lookup, unique_alpha_bins, traced_alpha_bins = cached
        timings["build_alpha_lookup"] = 0.0
        timings["precompute_final_alpha"] = 0.0
        print("Loaded lookup tables from cache.")
    else:
        stage_start = perf_counter()
        alpha_lookup = build_alpha_lookup((height, width), fov, decimals=decimals)
        timings["build_alpha_lookup"] = perf_counter() - stage_start

        stage_start = perf_counter()
        final_alpha_lookup, winding_lookup, unique_alpha_bins, traced_alpha_bins = precompute_final_alpha_lookup(alpha_lookup, alpha_crit, r_obs)
        timings["precompute_final_alpha"] = perf_counter() - stage_start

        save_lookup_cache(
            alpha_lookup, final_alpha_lookup, winding_lookup,
            unique_alpha_bins, traced_alpha_bins,
            height, width, fov, decimals, r_obs, alpha_crit,
        )
        print("Saved lookup tables to cache.")
    
    stage_start = perf_counter()
    lensed_image = render_lensed_image(
        img,
        alpha_lookup,
        final_alpha_lookup,
        winding_lookup,
        alpha_crit,
        fov,
        render_loop_around,
    )
    timings["render"] = perf_counter() - stage_start
    
    
    """
    ! As I implement the alpha first look up table, this loop will become obsolete and will be replaced by a simple lookup from the precomputed table.
    for y in tqdm(range(height), desc="Rendering", unit="row"):
        for x in range(width):
            angles = pixel_to_angles((y, x), (height, width), fov)
            color = get_color_from_background(angles, img, fov, r_obs)
            lensed_image[y, x] = color
    """
    
    
    # save output image
    stage_start = perf_counter()
    mpimg.imsave('lensed_image.png', lensed_image)
    timings["save_image"] = perf_counter() - stage_start
    timings["total"] = perf_counter() - total_start

    if debug_benchmark:
        print_benchmark_summary(
            (height, width),
            alpha_crit,
            unique_alpha_bins,
            traced_alpha_bins,
            timings,
        )

if __name__ == "__main__":
    main()
