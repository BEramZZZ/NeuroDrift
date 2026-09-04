import json
import os
import pickle

import neat
import pygame

from env.track import Track
from neat_mode.population_runner import run_generation

SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 720
NUM_GENERATIONS = 300
TRACK_PATH = "assets/track1.json"
CONFIG_PATH = "neat_mode/neat-config.txt"
BEST_GENOME_PATH = "neat_mode/models/best_genome.pkl"


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
    track = load_track(TRACK_PATH)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH,
    )

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats_reporter = neat.StatisticsReporter()
    population.add_reporter(stats_reporter)

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("NeuroDrift - NEAT Training")
    font = pygame.font.SysFont("Arial", 16)
    clock = pygame.time.Clock()

    generation_counter = {"n": 0}
    best_seen = {"genome": None, "fitness": float("-inf")}

    def eval_genomes(genomes, cfg):
        generation_counter["n"] += 1
        run_generation(
            genomes, cfg, track, generation_counter["n"],
            screen=screen, font=font, clock=clock, fps=60
        )
        # snapshot right after fitness is computed for this generation -
        # elitism reuses genome objects across generations, so if the
        # window closes mid-generation the reset at the top of the next
        # run_generation() call can zero out this same object before we
        # read winner.fitness at the end. Storing the float separately
        # protects against that.
        for _, genome in genomes:
            if genome.fitness is not None and genome.fitness > best_seen["fitness"]:
                best_seen["fitness"] = genome.fitness
                best_seen["genome"] = genome

    try:
        winner = population.run(eval_genomes, NUM_GENERATIONS)
    except SystemExit:
        winner = best_seen["genome"] or population.best_genome

    if winner is not None and best_seen["genome"] is winner:
        winner.fitness = best_seen["fitness"]  # restore in case it got zeroed by the reset described above

    os.makedirs(os.path.dirname(BEST_GENOME_PATH), exist_ok=True)
    with open(BEST_GENOME_PATH, "wb") as f:
        pickle.dump(winner, f)

    print(f"Training finished. Best genome saved to {BEST_GENOME_PATH}")
    print(f"Best fitness achieved: {best_seen['fitness']:.1f}")

    pygame.quit()
if __name__ == "__main__":
    main()