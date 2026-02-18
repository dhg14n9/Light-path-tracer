# Every set of coordinate is written as (y, x)
# FOV pairs: (horizontal, vertical)

import numpy as np
import matplotlib.image as mpimg
from tqdm import tqdm
import os
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

from geodesic_tracer import (
    M, R_S, R_PHOTON, B_CRIT,
    trace_ray, viewing_angle_to_impact_parameter
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
    solution, _ = trace_ray(r_obs, alpha)
    r = solution.y[1] # type: ignore
    phi = solution.y[2] # type: ignore

    x = r * np.cos(phi)
    y = r * np.sin(phi)

    dx = x[-1] - x[-2]
    dy = y[-1] - y[-2]

    return np.arctan2(dy, dx), phi[-1]


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


def _resolve_worker_count(num_worker: int = None) -> int: # type: ignore
    """
    Resolve worker count from user input or system CPU count.
    """
    if num_worker is None:
        return max(1, int(os.cpu_count() or 1))
    return max(1, int(num_worker))


_BUILD_HEIGHT = 0
_BUILD_WIDTH = 0
_BUILD_FOV: tuple[float, float] = (0.0, 0.0)
_BUILD_DECIMALS = 4


def _init_build_alpha_worker(
    height: int,
    width: int,
    fov: tuple[float, float],
    decimals: int,
) -> None:
    """
    Initialize build-alpha row workers.
    """
    global _BUILD_HEIGHT, _BUILD_WIDTH, _BUILD_FOV, _BUILD_DECIMALS
    _BUILD_HEIGHT = int(height)
    _BUILD_WIDTH = int(width)
    _BUILD_FOV = fov
    _BUILD_DECIMALS = int(decimals)


def _build_alpha_row_worker(y: int) -> tuple[int, np.ndarray]:
    """
    Build one row of the alpha lookup table.
    """
    row = np.empty(_BUILD_WIDTH, dtype=np.float32)
    for x in range(_BUILD_WIDTH):
        alpha = round(pixel_to_angles((y, x), (_BUILD_HEIGHT, _BUILD_WIDTH), _BUILD_FOV)[0], _BUILD_DECIMALS)
        row[x] = alpha
    return int(y), row


def build_alpha_lookup(
    image_dimension: tuple[int, int],
    fov: tuple[float, float],
    decimals: int = 4,
    num_worker: int = None, # type: ignore
) -> np.ndarray:
    """
    Build a per-pixel alpha lookup table with optional rounding.

    Args:
        image_dimension (tuple[int, int]): Image dimension (height, width).
        fov (tuple[float, float]): Field of view in radians (horizontal, vertical).
        decimals (int): Decimal precision for alpha binning.
        num_worker (int): Worker process count. Defaults to CPU core count.

    Returns:
        np.ndarray: Alpha lookup array with shape (height, width).
    """
    height, width = image_dimension
    alpha_lookup = np.empty((height, width), dtype=np.float32)
    tasks = list(range(height))
    if not tasks:
        return alpha_lookup

    worker_cores_used = min(_resolve_worker_count(num_worker), len(tasks))

    with ProcessPoolExecutor(
        max_workers=worker_cores_used,
        initializer=_init_build_alpha_worker,
        initargs=(height, width, fov, decimals),
    ) as executor:
        futures = [executor.submit(_build_alpha_row_worker, y) for y in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Building alpha lookup", unit="row"):
            y, row = future.result()
            alpha_lookup[y] = row
    return alpha_lookup

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
        tuple[np.ndarray, int, int]: (final alpha lookup, unique alpha bins, traced alpha bins).
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

_RENDER_SOURCE_IMAGE: Optional[np.ndarray] = None
_RENDER_ALPHA_LOOKUP: Optional[np.ndarray] = None
_RENDER_FINAL_ALPHA_LOOKUP: Optional[np.ndarray] = None
_RENDER_WINDING_LOOKUP: Optional[np.ndarray] = None
_RENDER_ALPHA_CRIT = 0.0
_RENDER_FOV: tuple[float, float] = (0.0, 0.0)
_RENDER_LOOP_AROUND = False


def _init_render_worker(
    source_image,
    alpha_lookup,
    final_alpha_lookup,
    winding_lookup,
    alpha_crit,
    fov,
    render_loop_around,
):
    """
    Initialize render row workers.
    """
    global _RENDER_SOURCE_IMAGE, _RENDER_ALPHA_LOOKUP, _RENDER_FINAL_ALPHA_LOOKUP
    global _RENDER_WINDING_LOOKUP, _RENDER_ALPHA_CRIT, _RENDER_FOV, _RENDER_LOOP_AROUND
    _RENDER_SOURCE_IMAGE = source_image
    _RENDER_ALPHA_LOOKUP = alpha_lookup
    _RENDER_FINAL_ALPHA_LOOKUP = final_alpha_lookup
    _RENDER_WINDING_LOOKUP = winding_lookup
    _RENDER_ALPHA_CRIT = float(alpha_crit)
    _RENDER_FOV = fov
    _RENDER_LOOP_AROUND = bool(render_loop_around)


def _render_row_worker(y: int) -> tuple[int, np.ndarray]:
    """
    Render one output row.
    """
    if _RENDER_SOURCE_IMAGE is None or _RENDER_ALPHA_LOOKUP is None or _RENDER_FINAL_ALPHA_LOOKUP is None:
        raise RuntimeError("Render worker not initialized")

    source_image = _RENDER_SOURCE_IMAGE
    alpha_lookup = _RENDER_ALPHA_LOOKUP
    final_alpha_lookup = _RENDER_FINAL_ALPHA_LOOKUP
    winding_lookup = _RENDER_WINDING_LOOKUP

    height, width = source_image.shape[:2]
    row = np.zeros_like(source_image[y])

    if row.ndim == 1:
        black = 0.0
        magenta = 1.0
    else:
        channel_count = row.shape[-1]
        black = np.zeros(channel_count, dtype=source_image.dtype)
        magenta = np.zeros(channel_count, dtype=source_image.dtype)
        if channel_count > 0:
            magenta[0] = 1.0
        if channel_count > 2:
            magenta[2] = 1.0

    for x in range(width):
        alpha = alpha_lookup[y, x]
        if alpha < _RENDER_ALPHA_CRIT:
            row[x] = black
            continue

        final_alpha = final_alpha_lookup[y, x]
        if final_alpha > np.pi / 2:
            n = winding_lookup[y, x] if winding_lookup is not None else 0
            idx = min(n, len(WINDING_COLORS) - 1)
            row[x] = WINDING_COLORS[idx]
            continue
        theta = pixel_to_angles((y, x), (height, width), _RENDER_FOV)[1]
        final_pixel = angles_to_pixel((final_alpha, theta), source_image.shape[:2], _RENDER_FOV)

        # for debugging purposes, returns a specific color if the ray escapes but goes out of bounds of the background image
        if not _RENDER_LOOP_AROUND:
            if (
                final_pixel[0] < 0
                or final_pixel[0] >= source_image.shape[0]
                or final_pixel[1] < 0
                or final_pixel[1] >= source_image.shape[1]
            ):
                row[x] = magenta
                continue
        else:
            # Tile the background image when the ray lands outside bounds.
            final_pixel = (
                final_pixel[0] % source_image.shape[0],
                final_pixel[1] % source_image.shape[1],
            )

        row[x] = source_image[final_pixel]

    return int(y), row


def render_lensed_image(
    source_image,
    alpha_lookup,
    final_alpha_lookup,
    winding_lookup,
    alpha_crit,
    fov,
    render_loop_around=False,
    num_worker=None,
):
    """
    Render the output image using precomputed alpha lookup tables.
    """
    height = source_image.shape[0]
    lensed_image = np.zeros_like(source_image)
    tasks = list(range(height))
    if not tasks:
        return lensed_image

    worker_cores_used = min(_resolve_worker_count(num_worker), len(tasks))

    with ProcessPoolExecutor(
        max_workers=worker_cores_used,
        initializer=_init_render_worker,
        initargs=(
            source_image,
            alpha_lookup,
            final_alpha_lookup,
            winding_lookup,
            alpha_crit,
            fov,
            render_loop_around,
        ),
    ) as executor:
        futures = [executor.submit(_render_row_worker, y) for y in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Rendering", unit="row"):
            y, row = future.result()
            lensed_image[y] = row

    return lensed_image


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
