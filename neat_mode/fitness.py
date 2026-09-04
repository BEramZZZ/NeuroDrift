import math


class CarStats:
    """Tracks everything needed to compute fitness for one car over its lifetime.
    Kept separate from the Car class itself, since Car only knows about physics -
    this is specifically about scoring driving performance for NEAT."""

    def __init__(self, start_x, start_y):
        self.last_x = start_x
        self.last_y = start_y
        self.total_distance = 0.0
        self.checkpoints_passed = 0
        self.next_checkpoint_index = 0
        self.frames_alive = 0
        self.frames_since_last_checkpoint = 0

    def update(self, car, track):
        """Call this once per frame for a living car. Updates distance traveled
        and checks if the car crossed its next expected checkpoint."""
        dx = car.x - self.last_x
        dy = car.y - self.last_y
        self.total_distance += math.hypot(dx, dy)
        self.last_x, self.last_y = car.x, car.y

        self.frames_alive += 1
        self.frames_since_last_checkpoint += 1

        if track.check_checkpoint(car.get_corners(), self.next_checkpoint_index):
            self.checkpoints_passed += 1
            self.next_checkpoint_index = (self.next_checkpoint_index + 1) % len(track.checkpoints)
            self.frames_since_last_checkpoint = 0


def compute_fitness(stats: CarStats):
    """Combines checkpoint progress (primary signal) with a small distance
    bonus (encourages movement even before reaching the first checkpoint,
    and rewards taking efficient racing lines) into one fitness number.

    Checkpoints dominate the score deliberately: a car that loops the track
    slowly but correctly must always beat a car that drives fast in circles
    without making real progress.
    """
    checkpoint_score = stats.checkpoints_passed * 1000
    distance_score = stats.total_distance * 0.1
    return checkpoint_score + distance_score


def is_stuck(stats: CarStats, patience_frames=300):
    """Returns True if the car hasn't reached a new checkpoint in too long -
    used by the population runner to kill cars that are stalled or driving
    in useless circles, so a generation doesn't run forever waiting on them."""
    return stats.frames_since_last_checkpoint > patience_frames