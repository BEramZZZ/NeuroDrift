import math

import numpy as np


def segments_intersect(p1, p2, p3, p4):
    """Returns True if line segment p1-p2 intersects line segment p3-p4.
    Standard orientation-based segment intersection test."""

    def orientation(a, b, c):
        val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
        if val == 0:
            return 0
        return 1 if val > 0 else 2

    def on_segment(a, b, c):
        return (min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and
                min(a[1], c[1]) <= b[1] <= max(a[1], c[1]))

    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and on_segment(p1, p3, p2):
        return True
    if o2 == 0 and on_segment(p1, p4, p2):
        return True
    if o3 == 0 and on_segment(p3, p1, p4):
        return True
    if o4 == 0 and on_segment(p3, p2, p4):
        return True

    return False


def line_intersection_point(p1, p2, p3, p4):
    """Returns the (x, y) point where segment p1-p2 crosses p3-p4, or None if parallel/no cross."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # parallel lines

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    px = x1 + t * (x2 - x1)
    py = y1 + t * (y2 - y1)
    return (px, py)


class Track:
    def __init__(self, inner_wall, outer_wall, checkpoints, spawn_point, spawn_angle=0.0):
        """
        inner_wall / outer_wall: list of (x, y) points forming the two track boundary lines
        checkpoints: list of ((x1,y1),(x2,y2)) line segments across the track, in order
        spawn_point: (x, y) where cars start
        spawn_angle: starting heading in degrees
        """
        self.inner_wall = inner_wall
        self.outer_wall = outer_wall
        self.checkpoints = checkpoints
        self.spawn_point = spawn_point
        self.spawn_angle = spawn_angle

        # convert wall point lists into segment pairs, e.g. [p0,p1,p2] -> [(p0,p1),(p1,p2)]
        self.inner_segments = list(zip(inner_wall, inner_wall[1:]))
        self.outer_segments = list(zip(outer_wall, outer_wall[1:]))
        self.all_wall_segments = self.inner_segments + self.outer_segments

        # precomputed as a flat NumPy array [x1,y1,x2,y2] per row - lets
        # raycast.py check a ray against every wall segment in one vectorized
        # operation instead of a Python loop per segment (this was the main
        # training speed bottleneck: ~90ms/frame across a 60-car population)
        import numpy as np
        self.wall_segments_np = np.array(
            [[p1[0], p1[1], p2[0], p2[1]] for p1, p2 in self.all_wall_segments],
            dtype=np.float64
        )

    def check_collision(self, car_corners):
        """car_corners: list of 4 (x,y) points from car.get_corners().
        Returns True if any car edge crosses any wall segment.

        Vectorized with NumPy against self.wall_segments_np, same idea as
        raycast.py's cast_rays(): checks one car edge against every wall
        segment in a single batched operation instead of a Python loop
        calling segments_intersect() once per (edge, wall) pair. This was
        by far the biggest bottleneck in the whole simulation - profiling
        showed it eating ~94% of every env.step() call, since the old
        version did ~2000 individual Python function calls per car per
        frame (4 edges x ~500 wall segments).

        Uses the same orientation cross-product test as segments_intersect(),
        just batched, and skips its exact-collinear-touch special case
        (on_segment) since that's a zero-probability edge case with floats
        and not worth the extra vectorized complexity for a real-time game.
        """
        walls = self.wall_segments_np
        if len(walls) == 0:
            return False

        x3, y3, x4, y4 = walls[:, 0], walls[:, 1], walls[:, 2], walls[:, 3]
        car_edges = list(zip(car_corners, car_corners[1:] + car_corners[:1]))

        for (ax, ay), (bx, by) in car_edges:
            d1 = (x4 - x3) * (ay - y3) - (y4 - y3) * (ax - x3)
            d2 = (x4 - x3) * (by - y3) - (y4 - y3) * (bx - x3)
            d3 = (bx - ax) * (y3 - ay) - (by - ay) * (x3 - ax)
            d4 = (bx - ax) * (y4 - ay) - (by - ay) * (x4 - ax)

            crosses = ((d1 > 0) != (d2 > 0)) & ((d3 > 0) != (d4 > 0))
            if np.any(crosses):
                return True
        return False

    def check_checkpoint(self, car_corners, checkpoint_index):
        """Checks if the car has crossed the next expected checkpoint.
        Returns True if crossed (caller should then increment their checkpoint counter)."""
        if checkpoint_index >= len(self.checkpoints):
            return False

        cp_start, cp_end = self.checkpoints[checkpoint_index]
        car_edges = list(zip(car_corners, car_corners[1:] + car_corners[:1]))

        for c_start, c_end in car_edges:
            if segments_intersect(c_start, c_end, cp_start, cp_end):
                return True
        return False