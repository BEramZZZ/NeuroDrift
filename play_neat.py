import argparse
import json
import pickle

import neat
import pygame

from env.car import Car
from env.raycast import cast_rays
from env.track import Track
from neat_mode import render_brain
from neat_mode.population_runner import (
    RAY_ANGLES, MAX_RAY_DISTANCE, PANEL_HEIGHT, FOLLOW_ZOOM,
    LEADER_COLOR, INPUT_LABELS, OUTPUT_LABELS,
)
from render import renderer

SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 720
TRACK_PATH = "assets/track1.json"
CONFIG_PATH = "neat_mode/neat-config.txt"
DEFAULT_GENOME_PATH = "neat_mode/models/best_genome.pkl"


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--genome", default=DEFAULT_GENOME_PATH, help="Path to a pickled genome (.pkl)")
    parser.add_argument("--track", default=TRACK_PATH, help="Path to a track .json")
    args = parser.parse_args()

    track = load_track(args.track)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH,
    )

    with open(args.genome, "rb") as f:
        genome = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(f"NeuroDrift - Playback ({args.genome})")
    font = pygame.font.SysFont("Arial", 16)
    clock = pygame.time.Clock()

    car = Car(track.spawn_point[0], track.spawn_point[1], track.spawn_angle)
    track_h = SCREEN_HEIGHT - PANEL_HEIGHT

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                # respawn on the same genome if it crashes or you just want a fresh run
                car = Car(track.spawn_point[0], track.spawn_point[1], track.spawn_angle)

        if car.alive:
            distances = cast_rays(car, track, RAY_ANGLES, MAX_RAY_DISTANCE)
            normalized_speed = car.speed / car.max_speed
            steer, throttle = net.activate(distances + [normalized_speed])
            car.move(steer, throttle)
            if track.check_collision(car.get_corners()):
                car.kill()

        renderer.clear_screen(screen)

        track_surface = screen.subsurface(pygame.Rect(0, 0, SCREEN_WIDTH, track_h))
        camera = renderer.compute_follow_camera(car.x, car.y, SCREEN_WIDTH, track_h, zoom=FOLLOW_ZOOM)
        renderer.set_camera(*camera)
        renderer.draw_track(track_surface, track)
        renderer.draw_car(track_surface, car, color=LEADER_COLOR)
        renderer.draw_hud(track_surface, font, generation=None, alive_count=1 if car.alive else 0)

        panel_surface = screen.subsurface(pygame.Rect(0, track_h, SCREEN_WIDTH, PANEL_HEIGHT))
        render_brain.draw_network(
            panel_surface, (0, 0, SCREEN_WIDTH, PANEL_HEIGHT),
            genome, config, net,
            input_labels=INPUT_LABELS, output_labels=OUTPUT_LABELS,
            title=f"Fitness: {genome.fitness:.1f}" if genome.fitness is not None else "Playback",
        )

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()