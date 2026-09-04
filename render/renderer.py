import pygame
import pygame.gfxdraw
import math
import random

# color palette
BG_COLOR = (32, 44, 34)            # grass base
GRASS_DETAIL = (40, 54, 42)        # subtle grass texture dots
ROAD_COLOR = (60, 62, 66)          # asphalt
ROAD_CENTERLINE = (110, 112, 116)  # faint lane line down the middle
WALL_LINE = (235, 235, 235)
CURB_RED = (195, 60, 55)
CURB_WHITE = (235, 235, 235)
CHECKPOINT_COLOR = (230, 195, 70)
CAR_BODY = (225, 225, 230)
CAR_CABIN = (150, 155, 165)
CAR_OUTLINE = (25, 25, 30)
CAR_WINDOW = (80, 90, 115)
CAR_SHADOW = (15, 20, 16)
CAR_TAILLIGHT = (200, 50, 45)
RAY_GLOW = (255, 255, 255)
RAY_LABEL_BG = (20, 24, 28)
HUD_BG = (20, 24, 28)
HUD_TEXT = (235, 235, 235)

_camera_offset = (0, 0)
_camera_scale = 1.0
_grass_dots_cache = None


def set_camera(scale, offset_x, offset_y):
    global _camera_scale, _camera_offset
    _camera_scale = scale
    _camera_offset = (offset_x, offset_y)


def compute_camera_transform(track, screen_width, screen_height, padding=60):
    """Computes a scale + offset so the track's full bounding box fits inside
    the screen (with padding) and is centered. Used for a full-track overview
    (e.g. watching an entire NEAT population at once)."""
    all_points = track.outer_wall + track.inner_wall
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]

    track_width = max(xs) - min(xs)
    track_height = max(ys) - min(ys)
    track_center_x = (min(xs) + max(xs)) / 2
    track_center_y = (min(ys) + max(ys)) / 2

    available_w = screen_width - padding * 2
    available_h = screen_height - padding * 2
    scale = min(available_w / track_width, available_h / track_height)
    scale = min(scale, 1.0)

    offset_x = screen_width / 2 - track_center_x * scale
    offset_y = screen_height / 2 - track_center_y * scale
    return scale, offset_x, offset_y


def compute_follow_camera(target_x, target_y, screen_width, screen_height, zoom=2.5):
    """Computes a scale + offset that keeps (target_x, target_y) centered on
    screen at a fixed zoom level - used for a car-following camera instead of
    a static full-track view. Call this every frame with the car's current
    position, since the target moves."""
    offset_x = screen_width / 2 - target_x * zoom
    offset_y = screen_height / 2 - target_y * zoom
    return zoom, offset_x, offset_y


def compute_population_camera(cars, track, screen_width, screen_height,
                               padding=120, min_zoom=0.4, max_zoom=2.5):
    """Fits the camera to the BOUNDING BOX of all given cars' positions,
    instead of their average point. Averaging breaks down once a population
    spreads around a loop-shaped track - e.g. half the cars on the far side
    and half near the start average out to a point in the empty grass hole
    in the middle, showing nothing useful. Fitting a bounding box always
    keeps every alive car actually on screen.

    Falls back to the full-track view if there are no cars (e.g. everyone
    just crashed) or all cars are in virtually the same spot (avoids
    zooming in absurdly far on a single point).
    """
    if not cars:
        return compute_camera_transform(track, screen_width, screen_height)

    xs = [c.x for c in cars]
    ys = [c.y for c in cars]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    bbox_width = max(max_x - min_x, 1.0)
    bbox_height = max(max_y - min_y, 1.0)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    available_w = screen_width - padding * 2
    available_h = screen_height - padding * 2
    scale = min(available_w / bbox_width, available_h / bbox_height)
    scale = max(min_zoom, min(scale, max_zoom))  # clamp: never too far in or out

    offset_x = screen_width / 2 - center_x * scale
    offset_y = screen_height / 2 - center_y * scale
    return scale, offset_x, offset_y


def _t(point):
    s = _camera_scale
    ox, oy = _camera_offset
    return (point[0] * s + ox, point[1] * s + oy)


def _t_list(points):
    s = _camera_scale
    ox, oy = _camera_offset
    return [(x * s + ox, y * s + oy) for x, y in points]


def clear_screen(screen):
    screen.fill(BG_COLOR)


def _draw_grass_texture(screen, track):
    """Speckle pattern for the grass, generated once in WORLD-space coordinates
    (not screen pixels) covering the track's area, so it scrolls correctly
    with a moving/zooming camera instead of staying stuck to the screen."""
    global _grass_dots_cache

    if _grass_dots_cache is None:
        all_points = track.outer_wall + track.inner_wall
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        margin = 150
        min_x, max_x = min(xs) - margin, max(xs) + margin
        min_y, max_y = min(ys) - margin, max(ys) + margin

        rng = random.Random(42)
        area = (max_x - min_x) * (max_y - min_y)
        num_dots = max(200, int(area / 4000))  # density scales with track area
        _grass_dots_cache = [
            (rng.uniform(min_x, max_x), rng.uniform(min_y, max_y), rng.randint(1, 2))
            for _ in range(min(num_dots, 3000))
        ]

    screen_w, screen_h = screen.get_size()
    for x, y, size in _grass_dots_cache:
        sx, sy = _t((x, y))
        if -10 <= sx <= screen_w + 10 and -10 <= sy <= screen_h + 10:
            pygame.draw.circle(screen, GRASS_DETAIL, (int(sx), int(sy)), size)


def draw_track(screen, track):
    _draw_grass_texture(screen, track)

    inner = _t_list(track.inner_wall)
    outer = _t_list(track.outer_wall)

    if len(inner) > 2 and len(outer) > 2:
        road_polygon = inner + outer[::-1]
        pygame.gfxdraw.filled_polygon(screen, road_polygon, ROAD_COLOR)

    _draw_centerline(screen, inner, outer)
    _draw_curbed_wall(screen, inner)
    _draw_curbed_wall(screen, outer)

    for cp_start, cp_end in track.checkpoints:
        _draw_dashed_line(screen, _t(cp_start), _t(cp_end), CHECKPOINT_COLOR, dash_len=8)


def _draw_centerline(screen, inner, outer):
    """Faint dashed line down the middle of the road, purely cosmetic."""
    n = min(len(inner), len(outer))
    mid_points = [
        ((inner[i][0] + outer[i][0]) / 2, (inner[i][1] + outer[i][1]) / 2)
        for i in range(n)
    ]
    for i in range(0, n - 1, 2):
        pygame.draw.line(screen, ROAD_CENTERLINE, mid_points[i], mid_points[i + 1], max(1, round(2 * _camera_scale)))


def _draw_curbed_wall(screen, points):
    n = len(points)
    if n < 2:
        return
    for i in range(n - 1):
        color = CURB_RED if (i // 3) % 2 == 0 else CURB_WHITE
        pygame.draw.line(screen, color, points[i], points[i + 1], max(1, round(5 * _camera_scale)))


def _draw_dashed_line(screen, start, end, color, dash_len=6):
    x1, y1 = start
    x2, y2 = end
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist == 0:
        return
    dx, dy = (x2 - x1) / dist, (y2 - y1) / dist
    num_dashes = int(dist / dash_len)
    for i in range(0, num_dashes, 2):
        sx = x1 + dx * dash_len * i
        sy = y1 + dy * dash_len * i
        ex = x1 + dx * dash_len * (i + 1)
        ey = y1 + dy * dash_len * (i + 1)
        pygame.draw.line(screen, color, (sx, sy), (ex, ey), 3)


def _car_local_shape(half_length, half_width):
    """Defines a car silhouette in the car's local space (forward = +x,
    sideways = +y): a pointed nose, straight sides, and a rounded-ish rear —
    reads as an actual car instead of a plain rectangle."""
    return [
        (half_length, 0),                              # nose tip
        (half_length * 0.75, -half_width),              # front-right shoulder
        (half_length * 0.2, -half_width),                # right side, front
        (-half_length * 0.7, -half_width),               # right side, rear
        (-half_length, -half_width * 0.5),               # rear-right corner
        (-half_length, half_width * 0.5),                # rear-left corner
        (-half_length * 0.7, half_width),                # left side, rear
        (half_length * 0.2, half_width),                 # left side, front
        (half_length * 0.75, half_width),                # front-left shoulder
    ]


def draw_car(screen, car, color=None):
    body_color = color if color else CAR_BODY
    rad = math.radians(car.angle)

    half_length = car.height / 2
    half_width = car.width / 2
    local_shape = _car_local_shape(half_length, half_width)

    # shadow: same shape, offset slightly for a subtle 3D feel
    shadow_offset = 4
    shadow_points = [(x + shadow_offset, y + shadow_offset) for x, y in car.to_world(local_shape)]
    pygame.gfxdraw.filled_polygon(
        screen, [(int(x), int(y)) for x, y in _t_list(shadow_points)], CAR_SHADOW
    )

    body_points = _t_list(car.to_world(local_shape))
    int_points = [(int(x), int(y)) for x, y in body_points]
    pygame.gfxdraw.filled_polygon(screen, int_points, body_color)
    pygame.gfxdraw.aapolygon(screen, int_points, CAR_OUTLINE)

    # cabin/roof: a narrower inset shape toward the rear-center, for a top-down "car" read
    cabin_shape = [
        (half_length * 0.15, -half_width * 0.6),
        (-half_length * 0.55, -half_width * 0.6),
        (-half_length * 0.55, half_width * 0.6),
        (half_length * 0.15, half_width * 0.6),
    ]
    cabin_points = [(int(x), int(y)) for x, y in _t_list(car.to_world(cabin_shape))]
    pygame.gfxdraw.filled_polygon(screen, cabin_points, CAR_CABIN)
    pygame.gfxdraw.aapolygon(screen, cabin_points, CAR_OUTLINE)

    # windshield toward the front
    front_x = car.x + math.cos(rad) * (half_length * 0.35)
    front_y = car.y + math.sin(rad) * (half_length * 0.35)
    fx, fy = _t((front_x, front_y))
    pygame.draw.circle(screen, CAR_WINDOW, (int(fx), int(fy)), max(3, car.width // 5))

    # taillights toward the back, one on each side
    back_x = car.x - math.cos(rad) * (half_length * 0.9)
    back_y = car.y - math.sin(rad) * (half_length * 0.9)
    perp_x = -math.sin(rad) * (half_width * 0.65)
    perp_y = math.cos(rad) * (half_width * 0.65)
    for sign in (-1, 1):
        lx, ly = _t((back_x + sign * perp_x, back_y + sign * perp_y))
        pygame.draw.circle(screen, CAR_TAILLIGHT, (int(lx), int(ly)), 3)


def draw_rays(screen, car, distances, ray_angles, max_distance, font):
    glow_surface = pygame.Surface(screen.get_size(), pygame.SRCALPHA)

    for offset, dist_normalized in zip(ray_angles, distances):
        ray_angle_rad = math.radians(car.angle + offset)
        actual_dist = dist_normalized * max_distance

        end_x = car.x + math.cos(ray_angle_rad) * actual_dist
        end_y = car.y + math.sin(ray_angle_rad) * actual_dist

        start_screen = _t((car.x, car.y))
        end_screen = _t((end_x, end_y))

        pygame.draw.line(glow_surface, (*RAY_GLOW, 40), start_screen, end_screen, 5)
        pygame.draw.aaline(screen, (255, 255, 255), start_screen, end_screen)

        label_text = f"{dist_normalized:.2f}"
        label = font.render(label_text, True, HUD_TEXT)
        label_rect = label.get_rect(center=(end_screen[0], end_screen[1] - 8))
        bg_rect = label_rect.inflate(6, 4)
        pygame.draw.rect(screen, RAY_LABEL_BG, bg_rect, border_radius=3)
        screen.blit(label, label_rect)

    screen.blit(glow_surface, (0, 0))


def draw_hud(screen, font, generation=None, alive_count=None, extra_text=None):
    lines = []
    if generation is not None:
        lines.append(f"Generation {generation}")
    if alive_count is not None:
        lines.append(f"Alive: {alive_count}")
    if extra_text is not None:
        lines.append(extra_text)

    if not lines:
        return

    padding = 8
    line_height = 22
    box_width = max(font.size(line)[0] for line in lines) + padding * 2
    box_height = line_height * len(lines) + padding

    panel = pygame.Surface((box_width, box_height), pygame.SRCALPHA)
    pygame.draw.rect(panel, (*HUD_BG, 180), panel.get_rect(), border_radius=6)
    screen.blit(panel, (10, 10))

    for i, line in enumerate(lines):
        text = font.render(line, True, HUD_TEXT)
        screen.blit(text, (10 + padding, 10 + padding // 2 + i * line_height))