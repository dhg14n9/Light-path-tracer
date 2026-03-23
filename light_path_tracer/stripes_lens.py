# Coordinates are ordered as (y, x).
# FOV pairs are (horizontal, vertical).

import numpy as np
import matplotlib.image as mpimg
from tqdm import tqdm
from time import perf_counter

from .metrics import Schwarzschild, Kerr


WINDING_DTYPE = np.uint16
WINDING_MAX = np.iinfo(WINDING_DTYPE).max
Y_AXIS_REFINE_FRAC = 0.07
DEFAULT_IMAGE_DIMENSION = (1080, 1920)
DEFAULT_N_X = 12
DEFAULT_N_Y = 12
STRIPE_WIDTH_FRAC = 0.12

LAST_COLOR_VISUALIZATION_DATA = None


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

    # Camera frame: +x right, +y down, +z forward; psi_y > 0 moves the BH up.
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

    # Tangent basis around the BH direction; matches image axes at psi=0.
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


def build_theta_lookup(image_dimension, fov, psi=(0.0, 0.0)):
    """Build a per-pixel theta lookup table in the local BH screen frame."""
    height, width = image_dimension
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
    return np.arctan2(
        vx * e_x[0] + vy * e_x[1] + vz * e_x[2],
        vx * e_y[0] + vy * e_y[1] + vz * e_y[2],
    ).astype(np.float32)


def _stripe_mask_from_directions(direction_x, direction_y, direction_z, n_x, n_y):
    """Return a boolean mask for rays that land on a stripe."""
    two_pi = 2.0 * np.pi

    yaw = np.mod(np.arctan2(direction_x, direction_z), two_pi)
    pitch = np.mod(np.arctan2(-direction_y, direction_z), two_pi)

    period_x = two_pi / n_x
    period_y = two_pi / n_y
    half_width_x = 0.5 * STRIPE_WIDTH_FRAC * period_x
    half_width_y = 0.5 * STRIPE_WIDTH_FRAC * period_y

    phase_x = np.mod(yaw, period_x)
    phase_y = np.mod(pitch, period_y)

    vertical_stripes = np.minimum(phase_x, period_x - phase_x) <= half_width_x
    horizontal_stripes = np.minimum(phase_y, period_y - phase_y) <= half_width_y
    return vertical_stripes | horizontal_stripes


def _reconstruct_source_direction_lookup(final_alpha_lookup, theta_lookup,
                                         psi=(0.0, 0.0)):
    """Reconstruct source directions from final alpha and screen theta."""
    shape = final_alpha_lookup.shape
    src_vx = np.full(shape, np.nan, dtype=np.float32)
    src_vy = np.full(shape, np.nan, dtype=np.float32)
    src_vz = np.full(shape, np.nan, dtype=np.float32)

    valid = np.isfinite(final_alpha_lookup)
    if not np.any(valid):
        return src_vx, src_vy, src_vz

    d, e_x, e_y, _ = _psi_frame(psi)
    fa = final_alpha_lookup[valid].astype(np.float64)
    th = theta_lookup[valid].astype(np.float64)

    sin_fa = np.sin(fa)
    cos_fa = np.cos(fa)
    sin_th = np.sin(th)
    cos_th = np.cos(th)

    src_vx[valid] = (
        cos_fa * d[0]
        + sin_fa * (sin_th * e_x[0] + cos_th * e_y[0])
    ).astype(np.float32)
    src_vy[valid] = (
        cos_fa * d[1]
        + sin_fa * (sin_th * e_x[1] + cos_th * e_y[1])
    ).astype(np.float32)
    src_vz[valid] = (
        cos_fa * d[2]
        + sin_fa * (sin_th * e_x[2] + cos_th * e_y[2])
    ).astype(np.float32)
    return src_vx, src_vy, src_vz


def _kerr_trace_direction_to_camera(direction_x, direction_y, direction_z,
                                    d, e_x, e_y):
    """Convert Kerr trace directions into the camera-coordinate frame."""
    canonical_x = direction_y
    # Kerr's a=0 limit uses +z for screen-up; the renderer uses +y down.
    canonical_y = direction_z
    canonical_z = -direction_x
    return _canonical_trace_direction_to_camera(
        canonical_x, canonical_y, canonical_z, d, e_x, e_y,
    )


def _canonical_trace_direction_to_camera(direction_x, direction_y, direction_z,
                                         d, e_x, e_y):
    """Lift canonical camera-frame directions into the current BH frame."""
    src_x = direction_x * e_x[0] + direction_y * e_y[0] + direction_z * d[0]
    src_y = direction_x * e_x[1] + direction_y * e_y[1] + direction_z * d[1]
    src_z = direction_x * e_x[2] + direction_y * e_y[2] + direction_z * d[2]
    return src_x, src_y, src_z


def bind_color_visualization_lookup(final_alpha_lookup, theta_lookup, n_x, n_y,
                                    psi=(0.0, 0.0),
                                    source_direction_lookup=None):
    """Bind final theta and stripe-hit state back to each starting pixel."""
    shape = final_alpha_lookup.shape
    final_theta_lookup = np.full(shape, np.nan, dtype=np.float32)
    ends_on_stripe_lookup = np.zeros(shape, dtype=bool)

    if source_direction_lookup is None:
        src_vx, src_vy, src_vz = _reconstruct_source_direction_lookup(
            final_alpha_lookup, theta_lookup, psi=psi,
        )
    else:
        src_vx, src_vy, src_vz = source_direction_lookup

    valid = np.isfinite(final_alpha_lookup)
    if np.any(valid):
        _, e_x, e_y, _ = _psi_frame(psi)
        dot_x = (src_vx[valid] * e_x[0]
                 + src_vy[valid] * e_x[1]
                 + src_vz[valid] * e_x[2])
        dot_y = (src_vx[valid] * e_y[0]
                 + src_vy[valid] * e_y[1]
                 + src_vz[valid] * e_y[2])
        final_theta_lookup[valid] = np.arctan2(dot_x, dot_y).astype(np.float32)
        ends_on_stripe_lookup[valid] = _stripe_mask_from_directions(
            src_vx[valid], src_vy[valid], src_vz[valid], n_x, n_y,
        )

    return {
        "final_theta_lookup": final_theta_lookup,
        "ends_on_stripe_lookup": ends_on_stripe_lookup,
        "source_direction_lookup": (src_vx, src_vy, src_vz),
    }


def get_last_color_visualization_data():
    """Return the most recent per-pixel color-visualization lookup data."""
    return LAST_COLOR_VISUALIZATION_DATA


def precompute_final_alpha_lookup(alpha_lookup, alpha_crit, r_obs, metric,
                                  fov=None, psi=(0.0, 0.0),
                                  return_direction_lookup=False,
                                  show_progress=False,
                                  progress_callback=None):
    """Trace one ray per pixel (1D alpha-only, spherically symmetric)."""
    alpha_flat = alpha_lookup.ravel().astype(np.float64)
    n = alpha_flat.size

    final_alpha_flat = np.full(n, np.nan, dtype=np.float64)
    winding_flat = np.zeros(n, dtype=np.int64)
    if return_direction_lookup:
        if fov is None:
            raise ValueError("fov is required when return_direction_lookup=True")
        theta_lookup = build_theta_lookup(alpha_lookup.shape, fov, psi=psi)
        theta_flat = theta_lookup.ravel().astype(np.float64)
        dir_x_flat = np.full(n, np.nan, dtype=np.float64)
        dir_y_flat = np.full(n, np.nan, dtype=np.float64)
        dir_z_flat = np.full(n, np.nan, dtype=np.float64)
    else:
        theta_flat = None
        dir_x_flat = None
        dir_y_flat = None
        dir_z_flat = None

    if n == 0:
        if progress_callback is not None:
            progress_callback(0, 0)
        return (np.full(alpha_lookup.shape, np.nan, dtype=np.float32),
                np.zeros(alpha_lookup.shape, dtype=WINDING_DTYPE),
                n, 0, None)

    chunk = 50_000
    for start in tqdm(range(0, n, chunk), desc="Tracing per-pixel rays",
                      unit="chunk", disable=not show_progress):
        end = min(start + chunk, n)
        if return_direction_lookup:
            metric.trace_rays_batch_with_dir(
                r_obs, alpha_flat[start:end], theta_flat[start:end],
                final_alpha_flat[start:end], winding_flat[start:end],
                dir_x_flat[start:end], dir_y_flat[start:end], dir_z_flat[start:end],
            )
        else:
            metric.trace_rays_batch(
                r_obs, alpha_flat[start:end],
                final_alpha_flat[start:end], winding_flat[start:end])
        if progress_callback is not None:
            progress_callback(end, n)

    fa_out = final_alpha_flat.astype(np.float32).reshape(alpha_lookup.shape)
    w_out = np.clip(winding_flat, 0, WINDING_MAX).astype(WINDING_DTYPE).reshape(alpha_lookup.shape)
    if return_direction_lookup:
        d, e_x, e_y, _ = _psi_frame(psi)
        src_x_flat, src_y_flat, src_z_flat = _canonical_trace_direction_to_camera(
            dir_x_flat, dir_y_flat, dir_z_flat, d, e_x, e_y,
        )
        source_direction_lookup = (
            src_x_flat.astype(np.float32).reshape(alpha_lookup.shape),
            src_y_flat.astype(np.float32).reshape(alpha_lookup.shape),
            src_z_flat.astype(np.float32).reshape(alpha_lookup.shape),
        )
    else:
        source_direction_lookup = None
    return fa_out, w_out, n, n, source_direction_lookup


# ============================================================================
# Alpha+theta lookup (2D, for non-spherically-symmetric metrics like Kerr)
# ============================================================================

def precompute_final_alpha_lookup_2d(
    alpha_lookup, fov, alpha_crit, r_obs, metric,
    theta_obs=np.pi / 2, psi=(0.0, 0.0),
    return_direction_lookup=False,
    show_progress=False, debug=False,
    progress_callback=None,
):
    """Trace one ray per pixel for non-spherically-symmetric metrics."""
    shape = alpha_lookup.shape
    height, width = shape

    hfov, _ = fov
    fx = (width / 2) / np.tan(hfov / 2)
    x_cam = (np.arange(width) - width / 2) / fx
    d, e_x, e_y, _ = _psi_frame(psi)
    theta_pixel = build_theta_lookup((height, width), fov, psi=psi)

    _, bh_x_cam, bh_proj_front = _psi_to_cam_projection(psi)
    if bh_proj_front:
        x_rel = x_cam - bh_x_cam
        x_cam_abs_max = max(float(np.max(np.abs(x_rel))), 1e-12)
        axis_refine_cols = np.abs(x_rel) <= (Y_AXIS_REFINE_FRAC * x_cam_abs_max)
    else:
        axis_refine_cols = np.zeros_like(x_cam, dtype=bool)

    use_tb_symmetry = (np.isclose(theta_obs, np.pi / 2)
                       and np.isclose(psi[0], 0.0)
                       and not return_direction_lookup)
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
        if return_direction_lookup:
            if not hasattr(metric, "trace_rays_batch_with_dir"):
                raise ValueError(
                    "Direction lookup requires metric.trace_rays_batch_with_dir"
                )
            dir_x_buf = np.full(alpha_f64.size, np.nan, dtype=np.float64)
            dir_y_buf = np.full(alpha_f64.size, np.nan, dtype=np.float64)
            dir_z_buf = np.full(alpha_f64.size, np.nan, dtype=np.float64)
        else:
            dir_x_buf = None
            dir_y_buf = None
            dir_z_buf = None

        chunk = 50_000
        for start in tqdm(range(0, alpha_f64.size, chunk),
                          desc="Tracing per-pixel rays", unit="chunk",
                          disable=not show_progress):
            end = min(start + chunk, alpha_f64.size)
            if return_direction_lookup:
                metric.trace_rays_batch_with_dir(
                    r_obs, alpha_f64[start:end], theta_f64[start:end],
                    theta_obs, axis_f64[start:end],
                    fa_buf[start:end], w_buf[start:end],
                    dir_x_buf[start:end], dir_y_buf[start:end],
                    dir_z_buf[start:end],
                )
            else:
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
    if return_direction_lookup:
        dir_x_out = np.full(shape, np.nan, dtype=np.float32)
        dir_y_out = np.full(shape, np.nan, dtype=np.float32)
        dir_z_out = np.full(shape, np.nan, dtype=np.float32)
    else:
        dir_x_out = None
        dir_y_out = None
        dir_z_out = None

    final_alpha_trace = final_alpha_trace_flat.reshape((trace_rows, width))
    winding_trace = winding_trace_flat.reshape((trace_rows, width))

    final_alpha_out[:trace_rows, :] = final_alpha_trace
    winding_out[:trace_rows, :] = winding_trace
    if return_direction_lookup and valid_indices.size:
        src_x_trace_flat, src_y_trace_flat, src_z_trace_flat = _kerr_trace_direction_to_camera(
            dir_x_buf, dir_y_buf, dir_z_buf, d, e_x, e_y,
        )
        dir_x_trace = src_x_trace_flat.astype(np.float32).reshape((trace_rows, width))
        dir_y_trace = src_y_trace_flat.astype(np.float32).reshape((trace_rows, width))
        dir_z_trace = src_z_trace_flat.astype(np.float32).reshape((trace_rows, width))
        dir_x_out[:trace_rows, :] = dir_x_trace
        dir_y_out[:trace_rows, :] = dir_y_trace
        dir_z_out[:trace_rows, :] = dir_z_trace

    if use_tb_symmetry:
        top_half = height // 2
        if top_half > 0:
            final_alpha_out[height - top_half:, :] = final_alpha_out[:top_half, :][::-1, :]
            winding_out[height - top_half:, :] = winding_out[:top_half, :][::-1, :]

    return (final_alpha_out,
            winding_out,
            int(alpha_lookup.size), int(valid_indices.size),
            None if dir_x_out is None else (dir_x_out, dir_y_out, dir_z_out))


# ============================================================================
# Rendering
# ============================================================================

def sample_background_sphere_stripes(direction_x, direction_y, direction_z,
                                     n_x, n_y):
    """Sample horizontal/vertical stripe lines on the celestial sphere."""
    stripes = _stripe_mask_from_directions(direction_x, direction_y, direction_z,
                                           n_x, n_y)
    colors = np.empty(direction_x.shape + (3,), dtype=np.float32)
    colors[stripes] = 0.0
    colors[~stripes] = 1.0
    return colors


def render_color_visualization(final_theta_lookup, ends_on_stripe_lookup):
    """Render the stored color-visualization lookup."""
    colors = np.zeros(final_theta_lookup.shape + (3,), dtype=np.float32)
    valid = np.isfinite(final_theta_lookup)
    non_stripe = valid & ~ends_on_stripe_lookup
    theta = final_theta_lookup

    red = non_stripe & (theta > 0.0) & (theta < (np.pi / 2.0))
    blue = non_stripe & (theta > (np.pi / 2.0)) & (theta < np.pi)
    green = non_stripe & (theta < 0.0) & (theta > (-np.pi / 2.0))
    yellow = non_stripe & (theta < (-np.pi / 2.0)) & (theta > -np.pi)

    colors[red] = (1.0, 0.0, 0.0)
    colors[blue] = (0.0, 0.0, 1.0)
    colors[green] = (0.0, 1.0, 0.0)
    colors[yellow] = (1.0, 1.0, 0.0)
    # Stripe hits stay black (already zeros)
    return colors


def render_lensed_image(final_alpha_lookup, fov, n_x, n_y,
                        color_visualization=False,
                        source_direction_lookup=None,
                        psi=(0.0, 0.0)):
    """Render the output image by sampling stripes on the background sphere."""
    global LAST_COLOR_VISUALIZATION_DATA

    height, width = final_alpha_lookup.shape
    lensed = np.zeros((height, width, 3), dtype=np.float32)

    theta_lookup = build_theta_lookup((height, width), fov, psi=psi)
    d, e_x, e_y, _ = _psi_frame(psi)

    if color_visualization:
        LAST_COLOR_VISUALIZATION_DATA = bind_color_visualization_lookup(
            final_alpha_lookup,
            theta_lookup,
            n_x,
            n_y,
            psi=psi,
            source_direction_lookup=source_direction_lookup,
        )
        return render_color_visualization(
            LAST_COLOR_VISUALIZATION_DATA["final_theta_lookup"],
            LAST_COLOR_VISUALIZATION_DATA["ends_on_stripe_lookup"],
        )

    valid = np.isfinite(final_alpha_lookup)
    n_escaped = np.count_nonzero(valid)

    if n_escaped > 0:
        fa = final_alpha_lookup[valid].astype(np.float64)
        th = theta_lookup[valid]

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
        lensed[valid] = sample_background_sphere_stripes(
            src_vx, src_vy, src_vz, n_x, n_y)

    LAST_COLOR_VISUALIZATION_DATA = None

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
    print(f"  {'build_lookup':<26}{timings.get('build_lookup', 0.0):>10.3f} s")
    print(f"  {'precompute':<26}{timings.get('precompute', 0.0):>10.3f} s")
    print(f"  {'render':<26}{timings.get('render', 0.0):>10.3f} s")
    print(f"  {'save_image':<26}{timings.get('save_image', 0.0):>10.3f} s")
    print(f"  {'total':<26}{timings.get('total', 0.0):>10.3f} s")
    print(f"  {'render_throughput':<26}"
          f"{(pixel_count / render_time) / 1e6:>10.2f} MPix/s")
    print(f"  {'overall_throughput':<26}"
          f"{(pixel_count / total_time) / 1e6:>10.2f} MPix/s")


def _spin_mass_units_to_kerr_a(M, spin_over_mass):
    return float(M) * float(spin_over_mass)


def _metric_spin_over_mass(metric):
    metric_mass = float(getattr(metric, "M", 0.0))
    if np.isclose(metric_mass, 0.0):
        return 0.0
    return float(getattr(metric, "a", 0.0)) / metric_mass


# ============================================================================
# Main
# ============================================================================

def main(metric=None, M=1.0, a=0.0, r_obs_mult=100.0,
         psi=(0.0, 0.0), vertical_fov_deg=40.0,
         width=DEFAULT_IMAGE_DIMENSION[1],
         height=DEFAULT_IMAGE_DIMENSION[0],
         n_x=DEFAULT_N_X, n_y=DEFAULT_N_Y,
         color_visualization=False,
         debug=False, benchmark=False):
    if width <= 0 or height <= 0:
        raise ValueError("width and height must both be positive integers")
    if n_x <= 0 or n_y <= 0:
        raise ValueError("n_x and n_y must both be positive integers")

    if metric is None:
        if abs(a) > 1.0:
            raise ValueError("Dimensionless BH spin a/M must be between -1 and 1")
        if np.isclose(a, 0.0):
            metric = Schwarzschild(M=M)
        else:
            metric = Kerr(M=M, a=_spin_mass_units_to_kerr_a(M, a))

    _debug_log(
        debug,
        f"Metric: {type(metric).__name__} "
        f"(M={metric.M}, a/M={_metric_spin_over_mass(metric):g}, "
        f"a={getattr(metric, 'a', 0):g})",
    )

    timings = {}
    total_start = _bench_start(benchmark)
    _debug_log(debug, f"Output: {width}x{height}")

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
    _debug_log(debug, f"Stripe counts: n_x={n_x}, n_y={n_y}")
    if color_visualization:
        _debug_log(
            debug,
            "Color visualization enabled; rendering stripe hits in black and "
            "final-theta quadrants in red/blue/green/yellow",
        )
    if metric.is_spherically_symmetric:
        _debug_log(debug, "Building per-pixel alpha lookup...")
        stage_start = _bench_start(benchmark)
        alpha_lookup = build_alpha_lookup((height, width), fov, psi=psi)
        _bench_stop(benchmark, timings, "build_lookup", stage_start)

        stage_start = _bench_start(benchmark)
        (final_alpha_lookup, _winding_lookup,
         total_rays, traced_rays,
         source_direction_lookup) = precompute_final_alpha_lookup(
            alpha_lookup, alpha_crit, r_obs, metric,
            fov=fov,
            psi=psi,
            return_direction_lookup=color_visualization,
            show_progress=debug)
        _bench_stop(benchmark, timings, "precompute", stage_start)
    else:
        _debug_log(debug, "Building per-pixel (alpha, theta) lookup...")
        stage_start = _bench_start(benchmark)
        alpha_lookup = build_alpha_lookup((height, width), fov, psi=psi)
        _bench_stop(benchmark, timings, "build_lookup", stage_start)

        stage_start = _bench_start(benchmark)
        (final_alpha_lookup, _winding_lookup,
         total_rays, traced_rays, source_direction_lookup) = precompute_final_alpha_lookup_2d(
            alpha_lookup, fov, alpha_crit, r_obs, metric,
            psi=psi,
            return_direction_lookup=color_visualization,
            show_progress=debug, debug=debug)
        _bench_stop(benchmark, timings, "precompute", stage_start)

    stage_start = _bench_start(benchmark)
    lensed_image = render_lensed_image(
        final_alpha_lookup, fov, n_x, n_y,
        color_visualization=color_visualization,
        source_direction_lookup=source_direction_lookup,
        psi=psi,
    )
    _bench_stop(benchmark, timings, "render", stage_start)

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
                        help="Dimensionless BH spin a/M (-1 <= a/M <= 1, 0 = Schwarzschild)")
    parser.add_argument("--r-obs", type=float, default=100.0,
                        help="Observer distance in units of M (default: 100)")
    parser.add_argument("--psi-y", type=float, default=0.0,
                        help="BH vertical offset in deg (+ = top, - = bottom)")
    parser.add_argument("--psi-x", type=float, default=0.0,
                        help="BH horizontal offset in deg (+ = right, - = left)")
    parser.add_argument("--fov-v", type=float, default=40.0,
                        help="Vertical field of view in deg")
    parser.add_argument("--width", type=int, default=DEFAULT_IMAGE_DIMENSION[1],
                        help="Output image width in pixels")
    parser.add_argument("--height", type=int, default=DEFAULT_IMAGE_DIMENSION[0],
                        help="Output image height in pixels")
    parser.add_argument("--n-x", type=int, default=DEFAULT_N_X,
                        help="Number of vertical stripe sectors (spacing 2pi/n_x)")
    parser.add_argument("--n-y", type=int, default=DEFAULT_N_Y,
                        help="Number of horizontal stripe sectors (spacing 2pi/n_y)")
    parser.add_argument("--color-visualization", action="store_true",
                        help=("Render stripe hits in black and color the "
                              "remaining escaped rays by final theta "
                              "quadrant"))
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logs and progress bars")
    parser.add_argument("--benchmark", action="store_true",
                        help="Enable benchmark timing summary")
    args = parser.parse_args()
    main(M=args.M, a=args.a, r_obs_mult=args.r_obs,
         psi=(np.radians(args.psi_y), np.radians(args.psi_x)),
         vertical_fov_deg=args.fov_v,
         width=args.width, height=args.height,
         n_x=args.n_x, n_y=args.n_y,
         color_visualization=args.color_visualization,
         debug=args.debug, benchmark=args.benchmark)
