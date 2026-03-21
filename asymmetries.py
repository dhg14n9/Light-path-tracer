"""Utilities for measuring directional shadow asymmetries."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np


_ALPHA_UPPER_LIMIT = 0.5 * np.pi - 1e-6
_DEFAULT_ASYMMETRY_PROFILE = "normal"
_CIRCLE_FIT_METHOD_ALIASES = {
    "cardinal": "cardinal_points_best_fit_circle",
    "cardinal_points": "cardinal_points_best_fit_circle",
    "cardinal_points_best_fit_circle": "cardinal_points_best_fit_circle",
    "global": "global_least_squares_circle",
    "global_least_squares": "global_least_squares_circle",
    "global_least_squares_circle": "global_least_squares_circle",
}
_CARDINAL_POINT_SPECS = (
    ("top", "y", "min"),
    ("bottom", "y", "max"),
    ("left", "x", "min"),
    ("right", "x", "max"),
)
_MEASUREMENT_RUNTIME_STAT_DEFAULTS = {
    "trace_outcome_calls": 0,
    "trace_ray_calls": 0,
    "invalid_trace_results": 0,
    "alpha_crit_calls": 0,
    "boundary_point_requests": 0,
    "boundary_samples_requested": 0,
    "point_cache_hits": 0,
    "point_cache_misses": 0,
    "trace_time": 0.0,
}
ASYMMETRY_PERFORMANCE_PROFILE_PRESETS = {
    "quick": {
        "n_bracket_samples": 33,
        "tol": 5e-6,
        "max_iter": 40,
        "n_theta_samples": 91,
        "n_refine_samples": 9,
        "refine_levels": 2,
        "n_boundary_samples": 181,
    },
    "normal": {
        "n_bracket_samples": 65,
        "tol": 1e-8,
        "max_iter": 64,
        "n_theta_samples": 181,
        "n_refine_samples": 17,
        "refine_levels": 4,
        "n_boundary_samples": 361,
    },
    "accurate": {
        "n_bracket_samples": 97,
        "tol": 1e-10,
        "max_iter": 96,
        "n_theta_samples": 361,
        "n_refine_samples": 17,
        "refine_levels": 5,
        "n_boundary_samples": 721,
    },
    "ultra_accurate": {
        "n_bracket_samples": 129,
        "tol": 1e-12,
        "max_iter": 128,
        "n_theta_samples": 721,
        "n_refine_samples": 21,
        "refine_levels": 6,
        "n_boundary_samples": 1081,
    },
}
ASYMMETRY_PERFORMANCE_PROFILE_NAMES = tuple(ASYMMETRY_PERFORMANCE_PROFILE_PRESETS)
ASYMMETRY_CIRCLE_FIT_CHOICES = ("global", "cardinal")
ASYMMETRY_ALPHA_CRIT_OPTION_NAMES = (
    "alpha_upper",
    "n_bracket_samples",
    "tol",
    "max_iter",
)
ASYMMETRY_SHADOW_EXTREMUM_OPTION_NAMES = (
    "n_theta_samples",
    "n_refine_samples",
    "refine_levels",
)
ASYMMETRY_BOUNDARY_OPTION_NAMES = ("n_boundary_samples",)
ASYMMETRY_CIRCULARITY_OPTION_NAMES = ("circle_fit",)
ASYMMETRY_MEASUREMENT_OPTION_NAMES = {
    "right_left_tangent_ratio": ASYMMETRY_ALPHA_CRIT_OPTION_NAMES,
    "top_bottom_outer_circle_gap_ratio": (
        ASYMMETRY_SHADOW_EXTREMUM_OPTION_NAMES + ASYMMETRY_ALPHA_CRIT_OPTION_NAMES
    ),
    "x_span_over_y_span": (
        ASYMMETRY_SHADOW_EXTREMUM_OPTION_NAMES + ASYMMETRY_ALPHA_CRIT_OPTION_NAMES
    ),
    "circularity_metrics": (
        ASYMMETRY_CIRCULARITY_OPTION_NAMES
        + ASYMMETRY_SHADOW_EXTREMUM_OPTION_NAMES
        + ASYMMETRY_BOUNDARY_OPTION_NAMES
        + ASYMMETRY_ALPHA_CRIT_OPTION_NAMES
    ),
}


def asymmetry_performance_profile_preset(profile):
    """Return a copy of one named asymmetry-performance preset."""
    preset = ASYMMETRY_PERFORMANCE_PROFILE_PRESETS.get(profile)
    if preset is None:
        preset = ASYMMETRY_PERFORMANCE_PROFILE_PRESETS[_DEFAULT_ASYMMETRY_PROFILE]
    return dict(preset)


def asymmetry_measurement_option_names(name):
    """Return the accepted tuning-option names for one public measurement."""
    try:
        return tuple(ASYMMETRY_MEASUREMENT_OPTION_NAMES[str(name)])
    except KeyError as exc:
        available = ", ".join(sorted(ASYMMETRY_MEASUREMENT_OPTION_NAMES))
        raise ValueError(
            f"Unknown asymmetry measurement {name!r}. Available measurements: {available}."
        ) from exc


def filter_asymmetry_measurement_kwargs(name, kwargs):
    """Return only the tuning kwargs that apply to one measurement."""
    allowed = set(asymmetry_measurement_option_names(name))
    return {
        key: value
        for key, value in dict(kwargs).items()
        if key in allowed
    }


def normalize_circle_fit_method(circle_fit):
    """Return the canonical circle-fit method name for one accepted alias."""
    try:
        return _CIRCLE_FIT_METHOD_ALIASES[str(circle_fit)]
    except KeyError as exc:
        available = ", ".join(sorted(_CIRCLE_FIT_METHOD_ALIASES))
        raise ValueError(
            f"Unknown circle fit method {circle_fit!r}. Available methods: {available}."
        ) from exc


def _wrap_screen_theta(theta):
    """Wrap a screen azimuth into [-pi, pi], keeping the top direction at +pi."""
    wrapped = (float(theta) + np.pi) % (2.0 * np.pi) - np.pi
    if np.isclose(wrapped, -np.pi, atol=1e-12):
        return float(np.pi)
    if np.isclose(wrapped, 0.0, atol=1e-12):
        return 0.0
    return float(wrapped)


def _screen_tangent_coordinates(alpha, theta):
    """Return tangent-plane screen coordinates with +x right and +y down."""
    rho = np.tan(alpha)
    return float(rho * np.sin(theta)), float(rho * np.cos(theta))


def _shadow_boundary_point(alpha, theta):
    """Return one boundary point in both angular and tangent-plane coordinates."""
    theta = _wrap_screen_theta(theta)
    x, y = _screen_tangent_coordinates(alpha, theta)
    return {
        "alpha": float(alpha),
        "theta": theta,
        "x": x,
        "y": y,
    }


def _circle_result(center_x, center_y, radius):
    """Return a standardized circle-fit result in screen tangent-plane coordinates."""
    return {
        "center_x": float(center_x),
        "center_y": float(center_y),
        "radius": float(radius),
    }


def _circumcircle_from_points(p1, p2, p3):
    """Return ``(center_x, center_y, radius)`` for the circumcircle through 3 points."""
    x1, y1 = float(p1["x"]), float(p1["y"])
    x2, y2 = float(p2["x"]), float(p2["y"])
    x3, y3 = float(p3["x"]), float(p3["y"])

    matrix = np.array(
        [
            [2.0 * (x2 - x1), 2.0 * (y2 - y1)],
            [2.0 * (x3 - x1), 2.0 * (y3 - y1)],
        ],
        dtype=np.float64,
    )
    rhs = np.array(
        [
            x2 * x2 + y2 * y2 - x1 * x1 - y1 * y1,
            x3 * x3 + y3 * y3 - x1 * x1 - y1 * y1,
        ],
        dtype=np.float64,
    )

    det = float(np.linalg.det(matrix))
    scale = max(1.0, float(np.max(np.abs(matrix))))
    if abs(det) <= 1e-14 * scale * scale:
        raise RuntimeError(
            "Could not fit a circumcircle: the chosen shadow points are nearly collinear."
        )

    center_x, center_y = np.linalg.solve(matrix, rhs)
    radius = float(np.hypot(x1 - center_x, y1 - center_y))
    if radius <= 0.0:
        raise RuntimeError(
            "Computed a non-positive circumcircle radius for the chosen shadow points."
        )

    return float(center_x), float(center_y), radius


def _least_squares_circle_from_points(points):
    """Return ``(center_x, center_y, radius)`` from an algebraic least-squares fit."""
    if len(points) < 3:
        raise ValueError("Need at least 3 points to fit a circle.")

    coords = np.array(
        [[float(point["x"]), float(point["y"])] for point in points],
        dtype=np.float64,
    )
    design = np.column_stack(
        (coords[:, 0], coords[:, 1], np.ones(coords.shape[0], dtype=np.float64))
    )
    rhs = -(coords[:, 0] ** 2 + coords[:, 1] ** 2)
    coeffs, _, rank, _ = np.linalg.lstsq(design, rhs, rcond=None)
    if rank < 3:
        raise RuntimeError(
            "Could not fit a least-squares circle: the sampled shadow points are degenerate."
        )

    a_coeff, b_coeff, c_coeff = coeffs
    center_x = -0.5 * float(a_coeff)
    center_y = -0.5 * float(b_coeff)
    radius_sq = center_x * center_x + center_y * center_y - float(c_coeff)
    scale = max(1.0, float(np.max(np.abs(coords))))
    if radius_sq <= 0.0 and abs(radius_sq) <= 1e-12 * scale * scale:
        radius_sq = 0.0
    if radius_sq <= 0.0:
        raise RuntimeError(
            "Computed a non-positive least-squares circle radius for the shadow points."
        )

    return center_x, center_y, float(np.sqrt(radius_sq))


def asymmetry_measurement(method):
    """Mark a method as a public asymmetry measurement."""
    method._is_asymmetry_measurement = True
    return method


class AsymmetryMeasurements:
    """Asymmetry helpers for a fixed metric and observer configuration."""

    def __init__(self, metric, r_obs, theta_obs=np.pi / 2):
        self.metric = metric
        self.r_obs = float(r_obs)
        self.theta_obs = float(theta_obs)
        self._measurement_progress_callback = None
        self._measurement_runtime_stats = None

    @classmethod
    def measurement_names(cls):
        names = []
        for name in dir(cls):
            method = getattr(cls, name)
            if getattr(method, "_is_asymmetry_measurement", False):
                names.append(name)
        return tuple(sorted(names))

    def available_measurements(self):
        return self.measurement_names()

    def _is_spherically_symmetric(self):
        return bool(getattr(self.metric, "is_spherically_symmetric", False))

    @staticmethod
    def _new_measurement_runtime_stats(name, estimated_work_units):
        stats = dict(_MEASUREMENT_RUNTIME_STAT_DEFAULTS)
        stats.update(
            {
                "measurement": str(name),
                "estimated_work_units": float(max(0.0, estimated_work_units)),
                "completed_work_units": 0.0,
            }
        )
        return stats

    def _record_runtime_stat(self, key, increment=1.0):
        if self._measurement_runtime_stats is None:
            return
        self._measurement_runtime_stats[key] += increment

    @staticmethod
    def _estimate_alpha_crit_work_units(n_bracket_samples=65, max_iter=64):
        if n_bracket_samples < 2:
            raise ValueError("n_bracket_samples must be at least 2.")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        return int(1 + 2 * (n_bracket_samples - 1) + max_iter)

    @staticmethod
    def _estimate_cardinal_boundary_requests(
        n_theta_samples=181,
        n_refine_samples=17,
        refine_levels=4,
    ):
        AsymmetryMeasurements._validate_shadow_extremum_search(
            n_theta_samples,
            n_refine_samples,
            refine_levels,
        )
        return int(n_theta_samples + 4 * refine_levels * n_refine_samples)

    @classmethod
    def estimate_measurement_work_units(cls, name: str, **kwargs) -> int:
        name = str(name)
        alpha_kwargs = filter_asymmetry_measurement_kwargs(name, kwargs)
        alpha_units = cls._estimate_alpha_crit_work_units(
            n_bracket_samples=int(alpha_kwargs.get("n_bracket_samples", 65)),
            max_iter=int(alpha_kwargs.get("max_iter", 64)),
        )

        if name == "right_left_tangent_ratio":
            return int(2 * alpha_units)

        if name in {"top_bottom_outer_circle_gap_ratio", "x_span_over_y_span"}:
            boundary_requests = cls._estimate_cardinal_boundary_requests(
                n_theta_samples=int(alpha_kwargs.get("n_theta_samples", 181)),
                n_refine_samples=int(alpha_kwargs.get("n_refine_samples", 17)),
                refine_levels=int(alpha_kwargs.get("refine_levels", 4)),
            )
            return int(boundary_requests * alpha_units)

        if name == "circularity_metrics":
            boundary_requests = int(alpha_kwargs.get("n_boundary_samples", 361))
            if boundary_requests < 8:
                raise ValueError("n_boundary_samples must be at least 8.")
            circle_fit = normalize_circle_fit_method(
                alpha_kwargs.get("circle_fit", "global_least_squares_circle")
            )
            if circle_fit == "cardinal_points_best_fit_circle":
                boundary_requests += cls._estimate_cardinal_boundary_requests(
                    n_theta_samples=int(alpha_kwargs.get("n_theta_samples", 181)),
                    n_refine_samples=int(alpha_kwargs.get("n_refine_samples", 17)),
                    refine_levels=int(alpha_kwargs.get("refine_levels", 4)),
                )
            return int(boundary_requests * alpha_units)

        asymmetry_measurement_option_names(name)
        return 0

    def begin_measurement_run(
        self,
        name: str,
        *,
        estimated_work_units=0,
        progress_callback=None,
    ) -> None:
        self._measurement_progress_callback = progress_callback
        self._measurement_runtime_stats = self._new_measurement_runtime_stats(
            name,
            estimated_work_units,
        )
        self._emit_measurement_progress()

    def finish_measurement_run(self, *, completed=True):
        if self._measurement_runtime_stats is None:
            return {}

        if completed:
            estimated = float(self._measurement_runtime_stats["estimated_work_units"])
            self._measurement_runtime_stats["completed_work_units"] = max(
                float(self._measurement_runtime_stats["completed_work_units"]),
                estimated,
            )
            self._emit_measurement_progress()

        stats = dict(self._measurement_runtime_stats)
        self._measurement_progress_callback = None
        self._measurement_runtime_stats = None
        return stats

    def _emit_measurement_progress(self):
        if self._measurement_progress_callback is None or self._measurement_runtime_stats is None:
            return
        self._measurement_progress_callback(
            float(self._measurement_runtime_stats["completed_work_units"]),
            float(self._measurement_runtime_stats["estimated_work_units"]),
        )

    def _advance_measurement_progress(self, work_units=1.0):
        if self._measurement_runtime_stats is None:
            return
        self._measurement_runtime_stats["completed_work_units"] += float(work_units)
        self._emit_measurement_progress()

    def measure(self, name: str, *args, **kwargs) -> Any:
        method = getattr(self, name, None)
        class_method = getattr(type(self), name, None)
        if method is None or not getattr(class_method, "_is_asymmetry_measurement", False):
            available = ", ".join(self.available_measurements()) or "<none>"
            raise ValueError(
                f"Unknown asymmetry measurement {name!r}. Available measurements: {available}."
            )
        return method(*args, **kwargs)

    def _trace_outcome(self, alpha, theta):
        """Trace one ray and retry with axis refinement if the first pass is invalid."""
        self._record_runtime_stat("trace_outcome_calls")

        for axis_refine in (False, True):
            trace_start = perf_counter() if self._measurement_runtime_stats is not None else None
            _, _, outcome = self.metric.trace_ray(
                self.r_obs,
                alpha,
                theta=theta,
                theta_obs=self.theta_obs,
                axis_refine=axis_refine,
            )
            self._record_runtime_stat("trace_ray_calls")
            if trace_start is not None:
                self._record_runtime_stat("trace_time", perf_counter() - trace_start)
            if outcome == "invalid":
                self._record_runtime_stat("invalid_trace_results")
            if outcome != "invalid":
                self._advance_measurement_progress(1.0)
                return outcome

        self._advance_measurement_progress(1.0)
        raise RuntimeError(
            f"Ray tracing stayed invalid at alpha={alpha:.12f}, theta={theta:.12f}."
        )

    @staticmethod
    def _validate_shadow_extremum_search(n_theta_samples, n_refine_samples, refine_levels):
        if n_theta_samples < 8:
            raise ValueError("n_theta_samples must be at least 8.")
        if n_refine_samples < 3:
            raise ValueError("n_refine_samples must be at least 3.")
        if refine_levels < 0:
            raise ValueError("refine_levels must be non-negative.")

    @staticmethod
    def _choose_extremum_index(kind):
        if kind == "min":
            return np.argmin
        if kind == "max":
            return np.argmax
        raise ValueError("kind must be 'min' or 'max'.")

    @staticmethod
    def _spherical_shadow_extremum_point(alpha, axis, kind):
        if kind not in {"min", "max"}:
            raise ValueError("kind must be 'min' or 'max'.")
        if axis == "x":
            theta = -0.5 * np.pi if kind == "min" else 0.5 * np.pi
        elif axis == "y":
            theta = np.pi if kind == "min" else 0.0
        else:
            raise ValueError("axis must be 'x' or 'y'.")
        return _shadow_boundary_point(alpha, theta)

    def _shadow_boundary_point_for_theta(self, theta, point_cache=None, **alpha_kwargs):
        """Return a boundary point dict at one screen azimuth, optionally cached."""
        theta = _wrap_screen_theta(theta)
        self._record_runtime_stat("boundary_point_requests")
        if point_cache is not None:
            cached = point_cache.get(theta)
            if cached is not None:
                self._record_runtime_stat("point_cache_hits")
                return cached

        self._record_runtime_stat("point_cache_misses")
        alpha = self.alpha_crit_for_theta(theta, **alpha_kwargs)
        point = _shadow_boundary_point(alpha, theta)
        if point_cache is not None:
            point_cache[theta] = point
        return point

    def _exact_shadow_circle(self):
        """Return the exact circular shadow for spherically symmetric metrics."""
        alpha = float(self.metric.alpha_crit(self.r_obs, self.theta_obs))
        return _circle_result(0.0, 0.0, np.tan(alpha))

    def _sample_shadow_boundary_points(
        self,
        *,
        n_boundary_samples=361,
        point_cache=None,
        **alpha_kwargs,
    ):
        """Return boundary samples around the full shadow silhouette."""
        if n_boundary_samples < 8:
            raise ValueError("n_boundary_samples must be at least 8.")
        self._record_runtime_stat("boundary_samples_requested", int(n_boundary_samples))

        theta_samples = np.linspace(
            -np.pi,
            np.pi,
            n_boundary_samples,
            endpoint=False,
            dtype=np.float64,
        )
        return [
            self._shadow_boundary_point_for_theta(
                float(theta),
                point_cache=point_cache,
                **alpha_kwargs,
            )
            for theta in theta_samples
        ]

    def _refine_shadow_extremum(
        self,
        axis,
        kind,
        theta_center,
        half_width,
        *,
        n_refine_samples=17,
        refine_levels=4,
        point_cache=None,
        **alpha_kwargs,
    ):
        """Refine one screen-coordinate extremum near ``theta_center``."""
        if axis not in {"x", "y"}:
            raise ValueError("axis must be 'x' or 'y'.")
        if n_refine_samples < 3:
            raise ValueError("n_refine_samples must be at least 3.")
        if refine_levels < 0:
            raise ValueError("refine_levels must be non-negative.")

        theta_center = _wrap_screen_theta(theta_center)
        half_width = float(abs(half_width))
        choose_index = self._choose_extremum_index(kind)

        best_point = self._shadow_boundary_point_for_theta(
            theta_center,
            point_cache=point_cache,
            **alpha_kwargs,
        )

        for _ in range(refine_levels):
            offsets = np.linspace(-half_width, half_width, n_refine_samples, dtype=np.float64)
            sample_thetas = np.empty(n_refine_samples, dtype=np.float64)
            coordinate_values = np.empty(n_refine_samples, dtype=np.float64)
            sample_points = []

            for i, offset in enumerate(offsets):
                theta = _wrap_screen_theta(theta_center + float(offset))
                point = self._shadow_boundary_point_for_theta(
                    theta,
                    point_cache=point_cache,
                    **alpha_kwargs,
                )
                sample_thetas[i] = theta
                coordinate_values[i] = float(point[axis])
                sample_points.append(point)

            best_idx = int(choose_index(coordinate_values))
            theta_center = _wrap_screen_theta(sample_thetas[best_idx])
            best_point = sample_points[best_idx]
            half_width = 2.0 * half_width / (n_refine_samples - 1)

        return best_point

    def _shadow_extremum_point(
        self,
        axis,
        kind,
        *,
        n_theta_samples=181,
        n_refine_samples=17,
        refine_levels=4,
        point_cache=None,
        **alpha_kwargs,
    ):
        """Return one min/max shadow boundary point in tangent-plane coordinates."""
        if axis not in {"x", "y"}:
            raise ValueError("axis must be 'x' or 'y'.")
        choose_index = self._choose_extremum_index(kind)
        self._validate_shadow_extremum_search(n_theta_samples, n_refine_samples, refine_levels)

        if self._is_spherically_symmetric():
            alpha = float(self.metric.alpha_crit(self.r_obs, self.theta_obs))
            return self._spherical_shadow_extremum_point(alpha, axis, kind)

        theta_samples = np.linspace(
            -np.pi,
            np.pi,
            n_theta_samples,
            endpoint=False,
            dtype=np.float64,
        )
        coordinate_values = np.empty(n_theta_samples, dtype=np.float64)

        for i, theta in enumerate(theta_samples):
            point = self._shadow_boundary_point_for_theta(
                float(theta),
                point_cache=point_cache,
                **alpha_kwargs,
            )
            coordinate_values[i] = float(point[axis])

        coarse_step = 2.0 * np.pi / n_theta_samples
        best_idx = int(choose_index(coordinate_values))
        return self._refine_shadow_extremum(
            axis,
            kind,
            float(theta_samples[best_idx]),
            coarse_step,
            n_refine_samples=n_refine_samples,
            refine_levels=refine_levels,
            point_cache=point_cache,
            **alpha_kwargs,
        )

    def alpha_crit_for_theta(
        self,
        theta,
        *,
        alpha_upper=None,
        n_bracket_samples=65,
        tol=1e-8,
        max_iter=64,
    ):
        """Return the capture boundary at one screen azimuth ``theta``."""
        self._record_runtime_stat("alpha_crit_calls")

        if self._is_spherically_symmetric():
            return float(self.metric.alpha_crit(self.r_obs, self.theta_obs))

        if n_bracket_samples < 2:
            raise ValueError("n_bracket_samples must be at least 2.")
        if tol <= 0.0:
            raise ValueError("tol must be positive.")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")

        theta = float(theta)

        if alpha_upper is None:
            alpha_upper = 1.05 * float(self.metric.alpha_crit(self.r_obs, self.theta_obs))
        alpha_upper = float(np.clip(alpha_upper, 1e-6, _ALPHA_UPPER_LIMIT))

        low = 0.0
        if self._trace_outcome(low, theta) != "captured":
            raise RuntimeError(
                "Expected the on-axis ray (alpha=0) to be captured while bracketing "
                "alpha_crit(theta)."
            )

        high = None
        for upper in (alpha_upper, _ALPHA_UPPER_LIMIT):
            samples = np.linspace(low, upper, n_bracket_samples, dtype=np.float64)
            prev_alpha = float(samples[0])

            for alpha in samples[1:]:
                alpha = float(alpha)
                outcome = self._trace_outcome(alpha, theta)
                if outcome == "escaped":
                    low = prev_alpha
                    high = alpha
                    break
                prev_alpha = alpha

            if high is not None:
                break

        if high is None:
            raise RuntimeError(
                "Could not bracket alpha_crit(theta) before alpha reached pi/2."
            )

        for _ in range(max_iter):
            if (high - low) <= tol:
                break

            mid = 0.5 * (low + high)
            outcome = self._trace_outcome(mid, theta)
            if outcome == "captured":
                low = mid
            elif outcome == "escaped":
                high = mid
            else:
                raise RuntimeError(
                    f"Unexpected ray outcome {outcome!r} while refining alpha_crit(theta)."
                )

        return 0.5 * (low + high)

    def top_bottom_shadow_points(
        self,
        *,
        n_theta_samples=181,
        n_refine_samples=17,
        refine_levels=4,
        **alpha_kwargs,
    ):
        """Return the top-most and bottom-most shadow points on the screen."""
        cardinal = self._shadow_cardinal_points(
            n_theta_samples=n_theta_samples,
            n_refine_samples=n_refine_samples,
            refine_levels=refine_levels,
            **alpha_kwargs,
        )
        return {"top": cardinal["top"], "bottom": cardinal["bottom"]}

    def _shadow_cardinal_points(
        self,
        *,
        n_theta_samples=181,
        n_refine_samples=17,
        refine_levels=4,
        **alpha_kwargs,
    ):
        """Return the top, bottom, left, and right shadow extrema on the screen."""
        point_cache = {}
        return {
            name: self._shadow_extremum_point(
                axis,
                kind,
                n_theta_samples=n_theta_samples,
                n_refine_samples=n_refine_samples,
                refine_levels=refine_levels,
                point_cache=point_cache,
                **alpha_kwargs,
            )
            for name, axis, kind in _CARDINAL_POINT_SPECS
        }

    def left_right_shadow_points(
        self,
        *,
        n_theta_samples=181,
        n_refine_samples=17,
        refine_levels=4,
        **alpha_kwargs,
    ):
        """Return the left-most and right-most shadow points on the screen."""
        cardinal = self._shadow_cardinal_points(
            n_theta_samples=n_theta_samples,
            n_refine_samples=n_refine_samples,
            refine_levels=refine_levels,
            **alpha_kwargs,
        )
        return {"left": cardinal["left"], "right": cardinal["right"]}

    def cardinal_points_best_fit_circle(
        self,
        *,
        n_theta_samples=181,
        n_refine_samples=17,
        refine_levels=4,
        **alpha_kwargs,
    ):
        """Return a least-squares circle fitted to top, bottom, left, and right."""
        if self._is_spherically_symmetric():
            return self._exact_shadow_circle()

        cardinal = self._shadow_cardinal_points(
            n_theta_samples=n_theta_samples,
            n_refine_samples=n_refine_samples,
            refine_levels=refine_levels,
            **alpha_kwargs,
        )
        center_x, center_y, radius = _least_squares_circle_from_points(
            [cardinal["top"], cardinal["bottom"], cardinal["left"], cardinal["right"]]
        )
        return _circle_result(center_x, center_y, radius)

    def global_least_squares_circle(
        self,
        *,
        n_boundary_samples=361,
        **alpha_kwargs,
    ):
        """Return a least-squares circle fitted to many shadow boundary samples."""
        if self._is_spherically_symmetric():
            return self._exact_shadow_circle()

        boundary_points = self._sample_shadow_boundary_points(
            n_boundary_samples=n_boundary_samples,
            point_cache={},
            **alpha_kwargs,
        )
        center_x, center_y, radius = _least_squares_circle_from_points(boundary_points)
        return _circle_result(center_x, center_y, radius)

    def _shadow_radius_profile(
        self,
        *,
        center_x,
        center_y,
        n_boundary_samples=721,
        boundary_points=None,
        point_cache=None,
        **alpha_kwargs,
    ):
        """Return center-based polar angles and radii along the shadow boundary."""
        if boundary_points is None:
            boundary_points = self._sample_shadow_boundary_points(
                n_boundary_samples=n_boundary_samples,
                point_cache=point_cache,
                **alpha_kwargs,
            )

        center_x = float(center_x)
        center_y = float(center_y)
        center_thetas = np.empty(len(boundary_points), dtype=np.float64)
        radii = np.empty(len(boundary_points), dtype=np.float64)

        for i, point in enumerate(boundary_points):
            dx = float(point["x"]) - center_x
            dy = float(point["y"]) - center_y
            center_thetas[i] = _wrap_screen_theta(np.arctan2(dx, dy))
            radii[i] = float(np.hypot(dx, dy))

        order = np.argsort(center_thetas)
        center_thetas = center_thetas[order]
        radii = radii[order]

        return {
            "theta": center_thetas,
            "radius": radii,
        }

    def _circularity_metrics(
        self,
        *,
        circle_fit="global_least_squares_circle",
        n_theta_samples=181,
        n_refine_samples=17,
        refine_levels=4,
        n_boundary_samples=361,
        **alpha_kwargs,
    ):
        """Return the shared RMS radius-profile circularity metrics."""
        circle_fit = normalize_circle_fit_method(circle_fit)
        point_cache = {}

        if circle_fit == "global_least_squares_circle":
            if self._is_spherically_symmetric():
                fitted_circle = self._exact_shadow_circle()
                boundary_points = None
            else:
                boundary_points = self._sample_shadow_boundary_points(
                    n_boundary_samples=n_boundary_samples,
                    point_cache=point_cache,
                    **alpha_kwargs,
                )
                center_x, center_y, radius = _least_squares_circle_from_points(boundary_points)
                fitted_circle = _circle_result(center_x, center_y, radius)
        else:
            fitted_circle = self.cardinal_points_best_fit_circle(
                n_theta_samples=n_theta_samples,
                n_refine_samples=n_refine_samples,
                refine_levels=refine_levels,
                **alpha_kwargs,
            )
            boundary_points = None

        profile = self._shadow_radius_profile(
            center_x=fitted_circle["center_x"],
            center_y=fitted_circle["center_y"],
            n_boundary_samples=n_boundary_samples,
            boundary_points=boundary_points,
            point_cache=point_cache,
            **alpha_kwargs,
        )

        theta = profile["theta"]
        radius = profile["radius"]
        if len(theta) < 3:
            raise RuntimeError(
                "Need at least 3 sampled boundary points to compute circularity metrics."
            )

        theta_extended = np.concatenate((theta, [theta[0] + 2.0 * np.pi]))
        radius_extended = np.concatenate((radius, [radius[0]]))

        mean_radius = float(np.trapezoid(radius_extended, theta_extended) / (2.0 * np.pi))
        if mean_radius <= 0.0:
            raise RuntimeError("Computed a non-positive mean shadow radius.")

        variance_term = float(
            np.trapezoid((radius_extended - mean_radius) ** 2, theta_extended) / (2.0 * np.pi)
        )
        if variance_term < 0.0 and abs(variance_term) <= 1e-14 * mean_radius * mean_radius:
            variance_term = 0.0
        if variance_term < 0.0:
            raise RuntimeError("Computed a negative circularity variance term.")

        asymmetry_parameter = float(2.0 * np.sqrt(variance_term))
        asymmetry_tol = 1e-14 * mean_radius
        if asymmetry_parameter <= asymmetry_tol:
            asymmetry_parameter = 0.0
        circularity_deviation = float(asymmetry_parameter / (2.0 * mean_radius))
        return {
            "A": asymmetry_parameter,
            "deltaC": circularity_deviation,
        }

    @asymmetry_measurement
    def right_left_tangent_ratio(self, **alpha_kwargs):
        """Return tan(alpha_right) / tan(alpha_left) on the observer screen."""
        alpha_right = self.alpha_crit_for_theta(np.pi / 2, **alpha_kwargs)
        alpha_left = self.alpha_crit_for_theta(-np.pi / 2, **alpha_kwargs)
        return float(np.tan(alpha_right) / np.tan(alpha_left))

    @asymmetry_measurement
    def top_bottom_outer_circle_gap_ratio(
        self,
        *,
        n_theta_samples=181,
        n_refine_samples=17,
        refine_levels=4,
        **alpha_kwargs,
    ):
        """Return D/R for the top-bottom-outer circumcircle and inner horizontal point."""
        cardinal = self._shadow_cardinal_points(
            n_theta_samples=n_theta_samples,
            n_refine_samples=n_refine_samples,
            refine_levels=refine_levels,
            **alpha_kwargs,
        )
        top = cardinal["top"]
        bottom = cardinal["bottom"]
        left = cardinal["left"]
        right = cardinal["right"]

        abs_left = abs(float(left["x"]))
        abs_right = abs(float(right["x"]))
        tie_tol = 1e-12 * max(1.0, abs_left, abs_right)
        if abs_right >= abs_left - tie_tol:
            outer = right
            inner = left
        else:
            outer = left
            inner = right

        center_x, center_y, radius = _circumcircle_from_points(top, bottom, outer)
        inner_radius = float(np.hypot(float(inner["x"]) - center_x, float(inner["y"]) - center_y))
        gap = radius - inner_radius
        gap_tol = 1e-10 * max(1.0, radius)
        if gap < 0.0 and abs(gap) <= gap_tol:
            gap = 0.0
        if gap < 0.0:
            raise RuntimeError(
                "The inner horizontal shadow point lies outside the fitted circumcircle."
            )

        return float(gap / radius)

    @asymmetry_measurement
    def x_span_over_y_span(
        self,
        *,
        n_theta_samples=181,
        n_refine_samples=17,
        refine_levels=4,
        **alpha_kwargs,
    ):
        """Return (X_max - X_min) / (Y_max - Y_min) on the observer screen."""
        cardinal = self._shadow_cardinal_points(
            n_theta_samples=n_theta_samples,
            n_refine_samples=n_refine_samples,
            refine_levels=refine_levels,
            **alpha_kwargs,
        )
        x_span = float(cardinal["right"]["x"]) - float(cardinal["left"]["x"])
        y_span = float(cardinal["bottom"]["y"]) - float(cardinal["top"]["y"])
        if x_span < 0.0:
            raise RuntimeError("Expected X_max to be greater than or equal to X_min.")
        if y_span <= 0.0:
            raise RuntimeError("Expected Y_max to be greater than Y_min for the shadow boundary.")
        return float(x_span / y_span)

    @asymmetry_measurement
    def circularity_metrics(
        self,
        *,
        circle_fit="global_least_squares_circle",
        n_theta_samples=181,
        n_refine_samples=17,
        refine_levels=4,
        n_boundary_samples=361,
        **alpha_kwargs,
    ):
        """Return the RMS asymmetry parameter A and circularity deviation deltaC."""
        return self._circularity_metrics(
            circle_fit=circle_fit,
            n_theta_samples=n_theta_samples,
            n_refine_samples=n_refine_samples,
            refine_levels=refine_levels,
            n_boundary_samples=n_boundary_samples,
            **alpha_kwargs,
        )


def alpha_crit_for_theta(
    metric,
    r_obs,
    theta,
    theta_obs=np.pi / 2,
    *,
    alpha_upper=None,
    n_bracket_samples=65,
    tol=1e-8,
    max_iter=64,
):
    """Convenience wrapper around ``AsymmetryMeasurements.alpha_crit_for_theta``."""
    return AsymmetryMeasurements(metric, r_obs, theta_obs).alpha_crit_for_theta(
        theta,
        alpha_upper=alpha_upper,
        n_bracket_samples=n_bracket_samples,
        tol=tol,
        max_iter=max_iter,
    )


def top_bottom_shadow_points(
    metric,
    r_obs,
    theta_obs=np.pi / 2,
    *,
    n_theta_samples=181,
    n_refine_samples=17,
    refine_levels=4,
    **alpha_kwargs,
):
    """Convenience wrapper around ``AsymmetryMeasurements.top_bottom_shadow_points``."""
    return AsymmetryMeasurements(metric, r_obs, theta_obs).top_bottom_shadow_points(
        n_theta_samples=n_theta_samples,
        n_refine_samples=n_refine_samples,
        refine_levels=refine_levels,
        **alpha_kwargs,
    )


def left_right_shadow_points(
    metric,
    r_obs,
    theta_obs=np.pi / 2,
    *,
    n_theta_samples=181,
    n_refine_samples=17,
    refine_levels=4,
    **alpha_kwargs,
):
    """Convenience wrapper around ``AsymmetryMeasurements.left_right_shadow_points``."""
    return AsymmetryMeasurements(metric, r_obs, theta_obs).left_right_shadow_points(
        n_theta_samples=n_theta_samples,
        n_refine_samples=n_refine_samples,
        refine_levels=refine_levels,
        **alpha_kwargs,
    )


def cardinal_points_best_fit_circle(
    metric,
    r_obs,
    theta_obs=np.pi / 2,
    *,
    n_theta_samples=181,
    n_refine_samples=17,
    refine_levels=4,
    **alpha_kwargs,
):
    """Convenience wrapper around ``AsymmetryMeasurements.cardinal_points_best_fit_circle``."""
    return AsymmetryMeasurements(metric, r_obs, theta_obs).cardinal_points_best_fit_circle(
        n_theta_samples=n_theta_samples,
        n_refine_samples=n_refine_samples,
        refine_levels=refine_levels,
        **alpha_kwargs,
    )


def global_least_squares_circle(
    metric,
    r_obs,
    theta_obs=np.pi / 2,
    *,
    n_boundary_samples=361,
    **alpha_kwargs,
):
    """Convenience wrapper around ``AsymmetryMeasurements.global_least_squares_circle``."""
    return AsymmetryMeasurements(metric, r_obs, theta_obs).global_least_squares_circle(
        n_boundary_samples=n_boundary_samples,
        **alpha_kwargs,
    )


def circularity_metrics(
    metric,
    r_obs,
    theta_obs=np.pi / 2,
    *,
    circle_fit="global_least_squares_circle",
    n_theta_samples=181,
    n_refine_samples=17,
    refine_levels=4,
    n_boundary_samples=361,
    **alpha_kwargs,
):
    """Convenience wrapper around ``AsymmetryMeasurements.circularity_metrics``."""
    return AsymmetryMeasurements(metric, r_obs, theta_obs).circularity_metrics(
        circle_fit=circle_fit,
        n_theta_samples=n_theta_samples,
        n_refine_samples=n_refine_samples,
        refine_levels=refine_levels,
        n_boundary_samples=n_boundary_samples,
        **alpha_kwargs,
    )


__all__ = [
    "ASYMMETRY_ALPHA_CRIT_OPTION_NAMES",
    "ASYMMETRY_BOUNDARY_OPTION_NAMES",
    "ASYMMETRY_CIRCLE_FIT_CHOICES",
    "ASYMMETRY_CIRCULARITY_OPTION_NAMES",
    "ASYMMETRY_MEASUREMENT_OPTION_NAMES",
    "ASYMMETRY_PERFORMANCE_PROFILE_NAMES",
    "ASYMMETRY_PERFORMANCE_PROFILE_PRESETS",
    "ASYMMETRY_SHADOW_EXTREMUM_OPTION_NAMES",
    "AsymmetryMeasurements",
    "alpha_crit_for_theta",
    "asymmetry_measurement",
    "asymmetry_measurement_option_names",
    "asymmetry_performance_profile_preset",
    "cardinal_points_best_fit_circle",
    "circularity_metrics",
    "filter_asymmetry_measurement_kwargs",
    "global_least_squares_circle",
    "left_right_shadow_points",
    "normalize_circle_fit_method",
    "top_bottom_shadow_points",
]
