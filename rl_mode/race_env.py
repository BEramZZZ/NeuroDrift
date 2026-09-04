import json

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.car import Car
from env.raycast import cast_rays
from env.track import Track

RAY_ANGLES = (-90, -45, 0, 45, 90)
MAX_RAY_DISTANCE = 200
MAX_EPISODE_FRAMES = 1800   # 30 sec at 60fps, same cap as NEAT training
STUCK_PATIENCE_FRAMES = 300  # 5 sec without a new checkpoint -> episode ends

# reward shaping - mirrors compute_fitness() in neat_mode/fitness.py so both
# modes are being pushed toward the same definition of "good driving", but
# expressed as a per-step reward instead of an end-of-run score:
CHECKPOINT_REWARD = 10.0
PROGRESS_REWARD_SCALE = 0.01
CRASH_PENALTY = -10.0
STUCK_PENALTY = -10.0        # keep this part — matching crash was the right call
TIME_PENALTY = -0.0015       # ~1.5x original, not 5x — caps at -2.7 over a full episode, not -9

def load_track(path):
    with open(path) as f:
        data = json.load(f)

    inner_wall = [tuple(p) for p in data["inner_wall"]]
    outer_wall = [tuple(p) for p in data["outer_wall"]]
    checkpoints = [(tuple(a), tuple(b)) for a, b in data["checkpoints"]]
    spawn_point = tuple(data["spawn_point"])
    spawn_angle = data.get("spawn_angle", 0.0)

    return Track(inner_wall, outer_wall, checkpoints, spawn_point, spawn_angle)


class RaceEnv(gym.Env):
    """Gymnasium environment for the RL mode. Same Car/Track/raycast simulation
    as the NEAT mode, so both agents are learning to drive on identical physics -
    only the *how* (evolution vs gradient descent on a reward signal) differs.

    Observation (6 floats, matches the NEAT inputs exactly):
        5 normalized ray distances (0 = wall right at the car, 1 = clear to MAX_RAY_DISTANCE)
        + normalized speed (current speed / max_speed, range roughly -0.36 to 1.0)

    Action (2 floats, both in [-1, 1]):
        steer:    -1 (full left) to 1 (full right)
        throttle: -1 (full brake/reverse) to 1 (full gas)
    """

    metadata = {"render_modes": []}

    def __init__(self, track_path="assets/track1.json"):
        super().__init__()
        self.track = load_track(track_path)

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.car = None
        self.frame = 0
        self.checkpoints_passed = 0
        self.next_checkpoint_index = 0
        self.frames_since_last_checkpoint = 0
        self._prev_checkpoint_distance = None

    def _get_obs(self):
        distances = cast_rays(self.car, self.track, RAY_ANGLES, MAX_RAY_DISTANCE)
        normalized_speed = self.car.speed / self.car.max_speed
        return np.array(distances + [normalized_speed], dtype=np.float32)

    def _distance_to_next_checkpoint(self):
        cp_start, cp_end = self.track.checkpoints[self.next_checkpoint_index]
        cp_mid_x = (cp_start[0] + cp_end[0]) / 2
        cp_mid_y = (cp_start[1] + cp_end[1]) / 2
        return float(np.hypot(self.car.x - cp_mid_x, self.car.y - cp_mid_y))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.car = Car(self.track.spawn_point[0], self.track.spawn_point[1], self.track.spawn_angle)
        self.frame = 0
        self.checkpoints_passed = 0
        self.next_checkpoint_index = 0
        self.frames_since_last_checkpoint = 0
        self._prev_checkpoint_distance = self._distance_to_next_checkpoint()

        return self._get_obs(), {}

    def step(self, action):
        steer, throttle = float(action[0]), float(action[1])
        self.car.move(steer, throttle)
        self.frame += 1
        self.frames_since_last_checkpoint += 1

        reward = TIME_PENALTY
        terminated = False
        truncated = False

        # progress shaping: reward getting closer to the next checkpoint,
        # penalize moving away from it - this is what lets the agent learn
        # something useful long before it ever crosses its first checkpoint,
        # instead of only getting signal on the rare checkpoint-crossing frame
        new_distance = self._distance_to_next_checkpoint()
        reward += (self._prev_checkpoint_distance - new_distance) * PROGRESS_REWARD_SCALE
        self._prev_checkpoint_distance = new_distance

        if self.track.check_checkpoint(self.car.get_corners(), self.next_checkpoint_index):
            self.checkpoints_passed += 1
            self.next_checkpoint_index = (self.next_checkpoint_index + 1) % len(self.track.checkpoints)
            self.frames_since_last_checkpoint = 0
            self._prev_checkpoint_distance = self._distance_to_next_checkpoint()
            reward += CHECKPOINT_REWARD

        if self.track.check_collision(self.car.get_corners()):
            reward += CRASH_PENALTY
            terminated = True
        elif self.frames_since_last_checkpoint > STUCK_PATIENCE_FRAMES:
            reward += STUCK_PENALTY
            terminated = True
        elif self.frame >= MAX_EPISODE_FRAMES:
            truncated = True

        info = {
            "checkpoints_passed": self.checkpoints_passed,
            "frame": self.frame,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        pass  # visual playback is handled separately by play_rl.py

    def close(self):
        pass
    
    
