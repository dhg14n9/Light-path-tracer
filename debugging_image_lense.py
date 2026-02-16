# Every set of coordinate is written as (y, x)
# FOV pairs: (horizontal, vertical)

import numpy as np
import matplotlib.image as mpimg
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import resource
import sys

from geodesic_tracer import (
    M, R_S, R_PHOTON, B_CRIT,
    trace_ray, viewing_angle_to_impact_parameter
)


def _read_proc_status_kib(field: str, pid: int | None = None) -> int | None:
    """
    Read memory values from /proc status files when available.
    Returns values in KiB.
    """
    status_path = "/proc/self/status" if pid is None else f"/proc/{pid}/status"
    try:
        with open(status_path, "r", encoding="utf-8") as status_file:
            for line in status_file:
                if not line.startswith(field):
                    continue
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    return int(parts[1])
    except OSError:
        return None
    return None


def get_current_ram_bytes() -> int:
    """
    Best-effort current RAM usage for the parent process.
    """
    vmrss_kib = _read_proc_status_kib("VmRSS:")
    if vmrss_kib is not None:
        return vmrss_kib * 1024

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(peak)
    return int(peak * 1024)


def get_peak_ram_bytes() -> int:
    """
    Best-effort peak RAM usage for the parent process.
    """
    vmhwm_kib = _read_proc_status_kib("VmHWM:")
    if vmhwm_kib is not None:
        return vmhwm_kib * 1024

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(peak)
    return int(peak * 1024)


def get_total_ram_bytes(extra_pids: list[int] | None = None) -> int:
    """
    Approximate current RAM for this process plus optional worker pids.
    """
    total_bytes = get_current_ram_bytes()
    if not extra_pids:
        return total_bytes

    seen_pids = {os.getpid()}
    for pid in extra_pids:
        if pid <= 0 or pid in seen_pids:
            continue
        seen_pids.add(pid)
        vmrss_kib = _read_proc_status_kib("VmRSS:", pid=pid)
        if vmrss_kib is not None:
            total_bytes += vmrss_kib * 1024
    return total_bytes


def format_bytes(num_bytes: int) -> str:
    """
    Format bytes as a compact IEC unit string.
    """
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(max(0, num_bytes))
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:,.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def resolve_worker_count(num_worker: int | None) -> int:
    """
    Resolve worker count with a lower bound of 1.
    """
    if num_worker is not None:
        return max(1, int(num_worker))

    process_cpu_count = getattr(os, "process_cpu_count", None)
    if callable(process_cpu_count):
        detected = process_cpu_count()
    else:
        detected = os.cpu_count()

    return max(1, int(detected or 1))


def _get_clock_ticks_per_second() -> int:
    """
    Return process clock ticks per second with a safe fallback.
    """
    try:
        return int(os.sysconf("SC_CLK_TCK"))
    except (AttributeError, OSError, ValueError):
        return 100


CLOCK_TICKS_PER_SECOND = _get_clock_ticks_per_second()


def _read_proc_cpu_ticks(pid: int | None = None) -> int | None:
    """
    Read utime+stime CPU ticks from /proc/<pid>/stat.
    """
    stat_path = "/proc/self/stat" if pid is None else f"/proc/{pid}/stat"
    try:
        with open(stat_path, "r", encoding="utf-8") as stat_file:
            line = stat_file.readline().strip()
    except OSError:
        return None

    close_idx = line.rfind(")")
    if close_idx == -1:
        return None
    fields = line[close_idx + 2 :].split()
    if len(fields) <= 12:
        return None

    try:
        utime_ticks = int(fields[11])
        stime_ticks = int(fields[12])
    except ValueError:
        return None
    return utime_ticks + stime_ticks


def get_total_cpu_ticks(extra_pids: list[int] | None = None) -> int | None:
    """
    Read total CPU ticks for this process and optional child pids.
    """
    total_ticks = _read_proc_cpu_ticks()
    if total_ticks is None:
        return None

    if not extra_pids:
        return total_ticks

    seen_pids = {os.getpid()}
    for pid in extra_pids:
        if pid <= 0 or pid in seen_pids:
            continue
        seen_pids.add(pid)
        cpu_ticks = _read_proc_cpu_ticks(pid=pid)
        if cpu_ticks is not None:
            total_ticks += cpu_ticks
    return total_ticks


def print_live_resource_usage(
    stage: str,
    completed: int,
    total: int,
    ram_bytes: int,
    cpu_cores: float | None,
    logical_cores_available: int,
) -> None:
    """
    Print one in-place live status line with progress, CPU, and RAM usage.
    """
    if total <= 0:
        progress_ratio = 1.0
    else:
        progress_ratio = min(max(completed / total, 0.0), 1.0)
    bar_width = 28
    filled = int(round(progress_ratio * bar_width))
    filled = min(max(filled, 0), bar_width)
    progress_bar = "#" * filled + "-" * (bar_width - filled)
    progress_percent = progress_ratio * 100.0

    if cpu_cores is None:
        cpu_label = "n/a"
    else:
        cpu_percent = (cpu_cores / max(1, logical_cores_available)) * 100.0
        cpu_label = f"{cpu_cores:5.2f} cores ({cpu_percent:5.1f}%)"

    line = (
        f"[{stage}] |{progress_bar}| {progress_percent:6.2f}% ({completed:,}/{total:,}) "
        f"| live_cpu_usage={cpu_label} "
        f"| live_ram_usage={format_bytes(ram_bytes)}"
    )
    previous_length = getattr(print_live_resource_usage, "_last_line_length", 0)
    padding = " " * max(0, previous_length - len(line))
    print(f"\r{line}{padding}", end="", flush=True)
    print_live_resource_usage._last_line_length = len(line)

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

def get_final_ray_angle(r_obs: float, alpha: float) -> float:
    """
    This function takes in the initial angle alpha that the photon from the observer and returns the final angle after the photon escapes.
    !!! THIS FUNCTION ASSUMES THAT THE PHOTON ESCAPES.

    Args:
        r_obs (float): Distance of the observer from the black hole center
        alpha (float): Initial viewing angle

    Returns:
        float: Final angle after the photon escapes
    """
    
    solution, _ = trace_ray(r_obs, alpha)
    r = solution.y[1] # type: ignore
    phi = solution.y[2] # type: ignore
    
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    
    dx = x[-1] - x[-2]
    dy = y[-1] - y[-2]

    return np.arctan2(dy, dx)


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
    
    final_heading = get_final_ray_angle(r_obs, alpha)
    final_alpha = world_heading_to_viewing_angle(final_heading)
    final_pixel = angles_to_pixel((final_alpha, theta), background_image.shape[:2], fov)
    
    # for debugging purposes, returns a specific color if the ray escapes but goes out of bounds of the background image
    if final_pixel[0] < 0 or final_pixel[0] >= background_image.shape[0] or final_pixel[1] < 0 or final_pixel[1] >= background_image.shape[1]:
        return np.array([1.0, 0.0, 1.0])
    
    return background_image[final_pixel]


def build_alpha_lookup(
    image_dimension: tuple[int, int],
    fov: tuple[float, float],
    decimals: int = 4,
) -> np.ndarray:
    """
    Build a per-pixel alpha lookup table with optional rounding.

    Args:
        image_dimension (tuple[int, int]): Image dimension (height, width).
        fov (tuple[float, float]): Field of view in radians (horizontal, vertical).
        decimals (int): Decimal precision for alpha binning.

    Returns:
        np.ndarray: Alpha lookup array with shape (height, width).
    """
    height, width = image_dimension
    alpha_lookup = np.empty((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            alpha = round(pixel_to_angles((y, x), (height, width), fov)[0], decimals)
            alpha_lookup[y, x] = alpha
    return alpha_lookup

def _trace_single_alpha(args: tuple[np.intp, float, float]) -> tuple[np.intp, float]:
    """
    Helper function to trace a single alpha value.

    Args:
        args (tuple[np.intp, float, float]): (index, alpha, r_obs)

    Returns:
        tuple[np.intp, float]: (index, final_alpha)
    """
    idx, alpha, r_obs = args
    final_heading = get_final_ray_angle(r_obs, alpha)
    final_alpha = world_heading_to_viewing_angle(final_heading)
    return (idx, final_alpha)

def precompute_final_alpha_lookup(
    alpha_lookup: np.ndarray,
    alpha_crit: float,
    r_obs: float,
    num_worker: int = None, # type: ignore
) -> tuple[np.ndarray, int, int, int, int, int, float]:
    """
    Precompute per-pixel final alpha by tracing only unique alpha bins that can escape.

    Args:
        alpha_lookup (np.ndarray): Per-pixel alpha array.
        alpha_crit (float): Critical angle in radians.
        r_obs (float): Observer radius.

    Returns:
        tuple[np.ndarray, int, int, int, int, int, float]: (
            final alpha lookup,
            unique alpha bins,
            traced alpha bins,
            worker cores used,
            current RAM bytes during precompute,
            peak RAM bytes during precompute,
            peak CPU cores used during precompute,
        ).
    """
    unique_alpha, inverse_alpha_idx = np.unique(alpha_lookup, return_inverse=True)
    valid_unique_mask = unique_alpha >= alpha_crit
    valid_indices = np.flatnonzero(valid_unique_mask)

    final_alpha_for_unique = np.full(unique_alpha.shape, np.nan, dtype=np.float32)
    
    tasks = [(idx, unique_alpha[idx], r_obs) for idx in valid_indices]
    worker_cores_used = min(resolve_worker_count(num_worker), len(tasks))
    current_ram_bytes = get_current_ram_bytes()
    peak_ram_bytes = current_ram_bytes
    current_cpu_cores: float | None = None
    peak_cpu_cores = 0.0
    logical_cores_available = int(os.cpu_count() or 1)

    if not tasks:
        final_alpha_lookup = final_alpha_for_unique[inverse_alpha_idx].reshape(alpha_lookup.shape)
        return (
            final_alpha_lookup,
            int(unique_alpha.size),
            int(valid_indices.size),
            worker_cores_used,
            current_ram_bytes,
            peak_ram_bytes,
            peak_cpu_cores,
        )

    with ProcessPoolExecutor(max_workers=worker_cores_used) as executor:
        futures = [executor.submit(_trace_single_alpha, task) for task in tasks]
        worker_pids: set[int] = set()
        total_futures = len(futures)
        completed_futures = 0
        last_print_time = perf_counter()
        live_print_interval_seconds = 0.25
        last_cpu_sample_time = perf_counter()
        last_cpu_ticks = get_total_cpu_ticks()
        print_live_resource_usage(
            "precompute_final_alpha",
            completed_futures,
            total_futures,
            current_ram_bytes,
            current_cpu_cores,
            logical_cores_available,
        )

        for future in as_completed(futures):
            processes = getattr(executor, "_processes", {})
            if isinstance(processes, dict):
                for process in processes.values():
                    pid = getattr(process, "pid", None)
                    if isinstance(pid, int):
                        worker_pids.add(pid)
            current_ram_bytes = get_total_ram_bytes(list(worker_pids))
            peak_ram_bytes = max(peak_ram_bytes, current_ram_bytes)

            current_total_cpu_ticks = get_total_cpu_ticks(list(worker_pids))
            now = perf_counter()
            if (
                current_total_cpu_ticks is not None
                and last_cpu_ticks is not None
                and current_total_cpu_ticks >= last_cpu_ticks
            ):
                elapsed_seconds = max(now - last_cpu_sample_time, 1e-12)
                delta_cpu_seconds = (current_total_cpu_ticks - last_cpu_ticks) / CLOCK_TICKS_PER_SECOND
                current_cpu_cores = max(0.0, delta_cpu_seconds / elapsed_seconds)
                peak_cpu_cores = max(peak_cpu_cores, current_cpu_cores)
                last_cpu_ticks = current_total_cpu_ticks
                last_cpu_sample_time = now

            idx, final_alpha = future.result()
            final_alpha_for_unique[idx] = final_alpha
            completed_futures += 1

            if (now - last_print_time) >= live_print_interval_seconds or completed_futures == total_futures:
                print_live_resource_usage(
                    "precompute_final_alpha",
                    completed_futures,
                    total_futures,
                    current_ram_bytes,
                    current_cpu_cores,
                    logical_cores_available,
                )
                last_print_time = now

        print()
            
        
    # for idx in valid_indices:
    #     alpha = float(unique_alpha[idx])
    #     final_heading = get_final_ray_angle(r_obs, alpha)
    #     final_alpha_for_unique[idx] = world_heading_to_viewing_angle(final_heading)

    current_ram_bytes = get_total_ram_bytes(list(worker_pids))
    peak_ram_bytes = max(peak_ram_bytes, current_ram_bytes)
    final_alpha_lookup = final_alpha_for_unique[inverse_alpha_idx].reshape(alpha_lookup.shape)
    return (
        final_alpha_lookup,
        int(unique_alpha.size),
        int(valid_indices.size),
        worker_cores_used,
        current_ram_bytes,
        peak_ram_bytes,
        peak_cpu_cores,
    )


def print_benchmark_summary(
    image_dimension: tuple[int, int],
    alpha_crit: float,
    unique_alpha_bins: int,
    traced_alpha_bins: int,
    core_metrics: dict[str, int],
    memory_metrics: dict[str, int],
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
    print(f"  logical CPU cores available: {core_metrics.get('logical_cores_available', 0):,}")
    print(f"  worker cores used: {core_metrics.get('worker_cores_used', 0):,}")
    print(f"  peak concurrent cores used: {core_metrics.get('peak_cores_used', 0):,}")
    peak_cpu_percent = (
        core_metrics.get("peak_cores_used", 0) / max(1, core_metrics.get("logical_cores_available", 1))
    ) * 100.0
    print(f"  {'live_cpu_usage_peak':<26}{core_metrics.get('peak_cores_used', 0):>5} cores ({peak_cpu_percent:5.1f}%)")
    print(f"  {'live_ram_usage_current':<26}{format_bytes(memory_metrics.get('current_ram_bytes', 0)):>12}")
    print(f"  {'live_ram_usage_peak':<26}{format_bytes(memory_metrics.get('peak_ram_bytes', 0)):>12}")
    print(f"  {'load_image':<26}{timings.get('load_image', 0.0):>10.3f} s")
    print(f"  {'build_alpha_lookup':<26}{timings.get('build_alpha_lookup', 0.0):>10.3f} s")
    print(f"  {'precompute_final_alpha':<26}{timings.get('precompute_final_alpha', 0.0):>10.3f} s")
    print(f"  {'render':<26}{timings.get('render', 0.0):>10.3f} s")
    print(f"  {'save_image':<26}{timings.get('save_image', 0.0):>10.3f} s")
    print(f"  {'total':<26}{timings.get('total', 0.0):>10.3f} s")
    print(f"  {'render_throughput':<26}{(pixel_count / render_time) / 1e6:>10.2f} MPix/s")
    print(f"  {'overall_throughput':<26}{(pixel_count / total_time) / 1e6:>10.2f} MPix/s")





def main(): 
    debug_benchmark: bool = True
    timings: dict[str, float] = {}
    total_start = perf_counter()
    logical_cores_available = int(os.cpu_count() or 1)

    # load image
    stage_start = perf_counter()
    img = mpimg.imread('image.jpg')
    
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    timings["load_image"] = perf_counter() - stage_start
    
    height, width = img.shape[:2]
    
    # output image 
    lensed_image = np.zeros_like(img)
    
    # obesrver properties
    r_obs = 100.0 * M
    alpha_crit_arg = B_CRIT * np.sqrt(1 - R_S / r_obs) / r_obs
    alpha_crit = np.arcsin(np.clip(alpha_crit_arg, -1.0, 1.0))
    
    vertical_fov_deg = 40.0
    vertical_fov: float = np.radians(vertical_fov_deg)
    horizontal_fov = 2 * np.arctan(np.tan(vertical_fov / 2) * width / height)
    fov = (horizontal_fov, vertical_fov)
    
    render_loop_around: bool = False # poor naming, I don't really know how to call this variable. If True, when the ray lands outside the bounds of the background image, it will be tiled instead of being marked as out of bounds.
    
    stage_start = perf_counter()
    alpha_lookup = build_alpha_lookup((height, width), fov)
    timings["build_alpha_lookup"] = perf_counter() - stage_start

    stage_start = perf_counter()
    (
        final_alpha_lookup,
        unique_alpha_bins,
        traced_alpha_bins,
        worker_cores_used,
        precompute_current_ram_bytes,
        precompute_peak_ram_bytes,
        precompute_peak_cpu_cores,
    ) = precompute_final_alpha_lookup(alpha_lookup, alpha_crit, r_obs)
    timings["precompute_final_alpha"] = perf_counter() - stage_start
    
    # render output image
    black = np.array([0.0, 0.0, 0.0], dtype=lensed_image.dtype)
    magenta = np.array([1.0, 0.0, 1.0], dtype=lensed_image.dtype)
    stage_start = perf_counter()
    render_total_rows = height
    render_completed_rows = 0
    render_current_ram_bytes = get_current_ram_bytes()
    render_current_cpu_cores: float | None = None
    render_peak_cpu_cores = 0.0
    render_last_print_time = perf_counter()
    render_print_interval_seconds = 0.25
    render_last_cpu_sample_time = perf_counter()
    render_last_cpu_ticks = get_total_cpu_ticks()
    print_live_resource_usage(
        "render",
        render_completed_rows,
        render_total_rows,
        render_current_ram_bytes,
        render_current_cpu_cores,
        logical_cores_available,
    )

    for y in range(height):
        for x in range(width):
            alpha = alpha_lookup[y, x]
            if alpha < alpha_crit:
                lensed_image[y, x] = black
                continue
            
            final_alpha = final_alpha_lookup[y, x]
            theta = pixel_to_angles((y, x), (height, width), fov)[1]
            final_pixel = angles_to_pixel((final_alpha, theta), img.shape[:2], fov)
            
            # for debugging purposes, returns a specific color if the ray escapes but goes out of bounds of the background image
            if not render_loop_around:
                if final_pixel[0] < 0 or final_pixel[0] >= img.shape[0] or final_pixel[1] < 0 or final_pixel[1] >= img.shape[1]:
                    lensed_image[y, x] = magenta
                    continue
            else:
                # Tile the background image when the ray lands outside bounds.
                final_pixel = (
                    final_pixel[0] % img.shape[0],
                    final_pixel[1] % img.shape[1],
                )
            
            color = img[final_pixel]
            lensed_image[y, x] = color
        render_completed_rows += 1
        now = perf_counter()
        if (now - render_last_print_time) >= render_print_interval_seconds or render_completed_rows == render_total_rows:
            render_current_ram_bytes = get_current_ram_bytes()
            current_cpu_ticks = get_total_cpu_ticks()
            if (
                current_cpu_ticks is not None
                and render_last_cpu_ticks is not None
                and current_cpu_ticks >= render_last_cpu_ticks
            ):
                elapsed_seconds = max(now - render_last_cpu_sample_time, 1e-12)
                delta_cpu_seconds = (current_cpu_ticks - render_last_cpu_ticks) / CLOCK_TICKS_PER_SECOND
                render_current_cpu_cores = max(0.0, delta_cpu_seconds / elapsed_seconds)
                render_peak_cpu_cores = max(render_peak_cpu_cores, render_current_cpu_cores)
                render_last_cpu_ticks = current_cpu_ticks
                render_last_cpu_sample_time = now

            print_live_resource_usage(
                "render",
                render_completed_rows,
                render_total_rows,
                render_current_ram_bytes,
                render_current_cpu_cores,
                logical_cores_available,
            )
            render_last_print_time = now
    print()
    timings["render"] = perf_counter() - stage_start
    
    
    """
    ! As I implement the alpha first look up table, this loop will become obsolete and will be replaced by a simple lookup from the precomputed table.
    for y in range(height):
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

    peak_cores_used = max(worker_cores_used, 1, int(np.ceil(max(precompute_peak_cpu_cores, render_peak_cpu_cores))))
    core_metrics = {
        "logical_cores_available": logical_cores_available,
        "worker_cores_used": worker_cores_used,
        "peak_cores_used": peak_cores_used,
    }
    peak_ram_bytes = max(get_peak_ram_bytes(), precompute_peak_ram_bytes)
    memory_metrics = {
        "current_ram_bytes": precompute_current_ram_bytes,
        "peak_ram_bytes": peak_ram_bytes,
    }

    if debug_benchmark:
        print_benchmark_summary(
            (height, width),
            alpha_crit,
            unique_alpha_bins,
            traced_alpha_bins,
            core_metrics,
            memory_metrics,
            timings,
        )

if __name__ == "__main__":
    main()
