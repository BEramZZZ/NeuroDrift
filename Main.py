import json
import pygame

from env.car import Car
from env.track import Track
from env.raycast import cast_rays
from render import renderer

SCREEN_WIDTH, SCREEN_HEIGHT = 900, 700
RAY_ANGLES = (-90, -45, 0, 45, 90)
MAX_RAY_DISTANCE = 200


def load_track(path):
    with open(path) as f:
        data = json.load(f)

    inner_wall = [tuple(p) for p in data["inner_wall"]]
    outer_wall = [tuple(p) for p in data["outer_wall"]]
    checkpoints = [(tuple(a), tuple(b)) for a, b in data["checkpoints"]]
    spawn_point = tuple(data["spawn_point"])
    spawn_angle = data.get("spawn_angle", 0.0)

    return Track(inner_wall, outer_wall, checkpoints, spawn_point, spawn_angle)


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("NeuroDrift - Manual Test Drive")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16)

    track = load_track("assets/track1.json")
    car = Car(track.spawn_point[0], track.spawn_point[1], track.spawn_angle)

    CAMERA_ZOOM = 1.5

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        steer = 0.0
        throttle = 0.0
        if keys[pygame.K_LEFT]:
            steer = -1.0
        if keys[pygame.K_RIGHT]:
            steer = 1.0
        if keys[pygame.K_UP]:
            throttle = 1.0
        if keys[pygame.K_DOWN]:
            throttle = -1.0

        car.move(steer, throttle)

        if track.check_collision(car.get_corners()):
            car.kill()

        camera_scale, camera_ox, camera_oy = renderer.compute_follow_camera(
            car.x, car.y, SCREEN_WIDTH, SCREEN_HEIGHT, zoom=CAMERA_ZOOM
        )
        renderer.set_camera(camera_scale, camera_ox, camera_oy)

        distances = cast_rays(car, track, RAY_ANGLES, MAX_RAY_DISTANCE)

        renderer.clear_screen(screen)
        renderer.draw_track(screen, track)
        renderer.draw_rays(screen, car, distances, RAY_ANGLES, MAX_RAY_DISTANCE, font)
        renderer.draw_car(screen, car)
        status = "CRASHED - press R to restart" if not car.alive else None
        renderer.draw_hud(screen, font, extra_text=status)

        if not car.alive and keys[pygame.K_r]:
            car = Car(track.spawn_point[0], track.spawn_point[1], track.spawn_angle)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()