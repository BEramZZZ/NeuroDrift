import math
import numpy as np
from env.track import line_intersection_point


def cast_rays(car, track, ray_angles=(-90, -45, 0, 45, 90), max_distance=200):
    """
    Casts multiple rays from the car at given angles (relative to car's heading).
    Returns a list of normalized distances (0.0 = wall right at car, 1.0 = no wall within max_distance).

    Vectorized with NumPy: checks every ray against every wall segment in one
    batched operation instead of a Python for-loop per segment. This was the
    main training bottleneck (each call used to take ~1.5ms with ~350 wall
    segments; with 60 cars per frame that alone was ~90ms/frame).
    """
    segs = track.wall_segments_np
    if len(segs) == 0:
        return [1.0] * len(ray_angles)

    x1, y1, x2, y2 = segs[:, 0], segs[:, 1], segs[:, 2], segs[:, 3]
    sx = (x2 - x1).reshape(1, -1)   # segment direction, shape (1, M)
    sy = (y2 - y1).reshape(1, -1)
    ex = (x1 - car.x).reshape(1, -1)  # vector from car to segment start, shape (1, M)
    ey = (y1 - car.y).reshape(1, -1)

    angles_rad = np.radians(car.angle + np.array(ray_angles))
    dx = np.cos(angles_rad).reshape(-1, 1)  # ray direction, shape (R, 1)
    dy = np.sin(angles_rad).reshape(-1, 1)

    # solving Cramer's rule for: car + t*ray_dir == wall_start + u*wall_dir
    det = sx * dy - sy * dx  # shape (R, M), broadcasts across rays and segments

    with np.errstate(divide="ignore", invalid="ignore"):
        t = (sx * ey - sy * ex) / det   # distance along the ray (world units)
        u = (dx * ey - dy * ex) / det   # position along the wall segment (0-1)

    valid = (det != 0) & (u >= 0) & (u <= 1) & (t >= 0) & (t <= max_distance)

    t_masked = np.where(valid, t, np.inf)
    closest_per_ray = t_masked.min(axis=1)
    closest_per_ray = np.where(np.isinf(closest_per_ray), max_distance, closest_per_ray)

    normalized = closest_per_ray / max_distance
    return normalized.tolist()


def _point_on_segment(seg_start, seg_end, point, tolerance=1e-6):
    """Checks if 'point' (known to be on the infinite line through seg_start-seg_end)
    actually lies within the segment's start-to-end range. Kept for reference/tests -
    the vectorized cast_rays above no longer uses this."""
    min_x, max_x = min(seg_start[0], seg_end[0]), max(seg_start[0], seg_end[0])
    min_y, max_y = min(seg_start[1], seg_end[1]), max(seg_start[1], seg_end[1])

    return (min_x - tolerance <= point[0] <= max_x + tolerance and
            min_y - tolerance <= point[1] <= max_y + tolerance)