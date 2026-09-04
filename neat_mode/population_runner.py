import neat
import pygame

from env.car import Car
from env.raycast import cast_rays
from neat_mode.fitness import CarStats, compute_fitness, is_stuck
from neat_mode import render_brain
from render import renderer

RAY_ANGLES = (-90, -45, 0, 45, 90)
MAX_RAY_DISTANCE = 200
MAX_FRAMES_PER_GENERATION = 1800  # 30 sec at 60fps - a safety cap so a generation can't run forever

PANEL_HEIGHT = 260  # bottom strip of the window reserved for the live network diagram
FOLLOW_ZOOM = 2.2  # how tightly the camera follows the leader car
LEADER_COLOR = (255, 205, 80)  # highlight color so the followed car stands out
INPUT_LABELS = [f"{a}°" for a in RAY_ANGLES] + ["Speed"]
OUTPUT_LABELS = ["Steer", "Throttle"]


def run_generation(genomes, config, track, generation_number,
                    screen=None, font=None, clock=None, fps=60):
    """Runs one full NEAT generation: spawns a car per genome, steps all of
    them together each frame, kills cars that crash or stall, and assigns
    each genome's fitness once the generation ends. If screen is provided,
    renders every alive car live, with the camera following the current
    leader and a live network diagram of the leader's genome in a panel at
    the bottom of the window. If screen is None, runs headless (no
    rendering) for much faster training.
    """
    cars = []
    nets = []
    stats_list = []
    genome_list = []

    for genome_id, genome in genomes:
        genome.fitness = 0.0
        car = Car(track.spawn_point[0], track.spawn_point[1], track.spawn_angle)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        stats = CarStats(car.x, car.y)

        cars.append(car)
        nets.append(net)
        stats_list.append(stats)
        genome_list.append(genome)

    frame = 0
    any_alive = True

    while any_alive and frame < MAX_FRAMES_PER_GENERATION:
        frame += 1
        any_alive = False

        if screen is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise SystemExit

        for car, net, stats in zip(cars, nets, stats_list):
            if not car.alive:
                continue

            distances = cast_rays(car, track, RAY_ANGLES, MAX_RAY_DISTANCE)
            normalized_speed = car.speed / car.max_speed
            net_inputs = distances + [normalized_speed]

            steer, throttle = net.activate(net_inputs)
            car.move(steer, throttle)

            stats.update(car, track)

            if track.check_collision(car.get_corners()) or is_stuck(stats):
                car.kill()
            else:
                any_alive = True

        if screen is not None:
            leader_idx = max(
                range(len(stats_list)),
                key=lambda i: (stats_list[i].checkpoints_passed, stats_list[i].total_distance),
            )
            _render_population(
                screen, font, track, cars, generation_number,
                leader_idx=leader_idx,
                leader_genome=genome_list[leader_idx],
                leader_net=nets[leader_idx],
                config=config,
            )
            if clock is not None:
                clock.tick(fps)

    for genome, stats in zip(genome_list, stats_list):
        genome.fitness = compute_fitness(stats)

    best_fitness = max((g.fitness for g in genome_list), default=0)
    print(f"Generation {generation_number} done - frames: {frame}, best fitness: {best_fitness:.1f}")


def _render_population(screen, font, track, cars, generation_number,
                        leader_idx=None, leader_genome=None, leader_net=None, config=None):
    alive_cars = [c for c in cars if c.alive]
    screen_w, screen_h = screen.get_size()
    track_h = screen_h - PANEL_HEIGHT

    renderer.clear_screen(screen)

    track_surface = screen.subsurface(pygame.Rect(0, 0, screen_w, track_h))

    leader_car = cars[leader_idx] if leader_idx is not None else None
    if leader_car is not None:
        camera = renderer.compute_follow_camera(leader_car.x, leader_car.y, screen_w, track_h, zoom=FOLLOW_ZOOM)
    else:
        camera = renderer.compute_camera_transform(track, screen_w, track_h)
    renderer.set_camera(*camera)

    renderer.draw_track(track_surface, track)
    for car in alive_cars:
        color = LEADER_COLOR if car is leader_car else None
        renderer.draw_car(track_surface, car, color=color)
    renderer.draw_hud(track_surface, font, generation=generation_number, alive_count=len(alive_cars))

    if leader_genome is not None and leader_net is not None and config is not None:
        panel_surface = screen.subsurface(pygame.Rect(0, track_h, screen_w, PANEL_HEIGHT))
        render_brain.draw_network(
            panel_surface, (0, 0, screen_w, PANEL_HEIGHT),
            leader_genome, config, leader_net,
            input_labels=INPUT_LABELS, output_labels=OUTPUT_LABELS,
            title=f"Best genome — Gen {generation_number}",
        )

    pygame.display.flip()