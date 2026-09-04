import argparse

import pygame
from stable_baselines3 import PPO

from env.raycast import cast_rays
from env.car import Car
from render import renderer
from rl_mode.race_env import RAY_ANGLES, MAX_RAY_DISTANCE, load_track

SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 460  # no brain panel for RL, so a shorter window than play_neat.py
FOLLOW_ZOOM = 2.2
LEADER_COLOR = (255, 205, 80)
TRACK_PATH = "assets/track1.json"
DEFAULT_MODEL_PATH = "rl_mode/models/ppo_car.zip"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to a trained PPO model (.zip)")
    parser.add_argument("--track", default=TRACK_PATH, help="Path to a track .json")
    args = parser.parse_args()

    track = load_track(args.track)
    model = PPO.load(args.model)

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(f"NeuroDrift - RL Playback ({args.model})")
    font = pygame.font.SysFont("Arial", 16)
    clock = pygame.time.Clock()

    car = Car(track.spawn_point[0], track.spawn_point[1], track.spawn_angle)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                # respawn on the same model if it crashes or you just want a fresh run
                car = Car(track.spawn_point[0], track.spawn_point[1], track.spawn_angle)

        if car.alive:
            distances = cast_rays(car, track, RAY_ANGLES, MAX_RAY_DISTANCE)
            normalized_speed = car.speed / car.max_speed
            obs = distances + [normalized_speed]
            action, _ = model.predict(obs, deterministic=True)
            steer, throttle = float(action[0]), float(action[1])
            car.move(steer, throttle)
            if track.check_collision(car.get_corners()):
                car.kill()

        renderer.clear_screen(screen)
        camera = renderer.compute_follow_camera(car.x, car.y, SCREEN_WIDTH, SCREEN_HEIGHT, zoom=FOLLOW_ZOOM)
        renderer.set_camera(*camera)
        renderer.draw_track(screen, track)
        renderer.draw_car(screen, car, color=LEADER_COLOR)
        renderer.draw_hud(screen, font, generation=None, alive_count=1 if car.alive else 0)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
