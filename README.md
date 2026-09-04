# NeuroDrift

A top-down racing simulation comparing two fundamentally different
approaches to teaching an agent to drive: **neuroevolution (NEAT)** and
**reinforcement learning (PPO)**. Both agents learn on identical physics,
the same track, and the same sensor input — only the learning method
differs, making this a direct, controlled comparison between evolutionary
and gradient-based approaches to a continuous control task.

## How it works

The car perceives its environment through a 6-dimensional observation:
five raycast distance sensors at −90°, −45°, 0°, 45°, and 90° relative to
its heading (normalized 0 = wall right at the car, 1 = clear out to the
200px max range), plus its current normalized speed. From this input
alone, an agent must learn to steer and accelerate around a closed circuit
without crashing, using two continuous outputs in [-1, 1]: steering and
throttle.

The car itself is simulated with a **bicycle model**: turning rate depends
on both the current steering angle and current speed (a stationary car
can't turn at all, and sharp turns at high speed sweep a wider arc), the
steering wheel eases toward the requested angle instead of snapping to it,
and there's passive drag plus stronger active braking. Collision and
checkpoint-crossing checks, and the five raycasts, are all vectorized with
NumPy against the track's wall segments — this was the main performance
bottleneck early on (~94% of each simulation step) before vectorizing.

Both learning methods are evaluated against the same underlying goal —
reach as many checkpoints as possible around the track before an episode
ends (crash, prolonged lack of progress, or a time limit of 1,800 frames /
30 seconds) — but they pursue it in very different ways:

- **NEAT** (`neat_mode/`) evolves a population of neural networks across
  generations. Each genome is scored by a fitness function
  (`checkpoints_passed * 1000 + total_distance * 0.1`), and the fittest
  genomes are selected, mutated, and recombined to produce the next
  generation. No gradients, no reward shaping — just selection pressure.

- **PPO** (`rl_mode/`) trains a single policy network via reinforcement
  learning, receiving a shaped reward signal on every simulation step
  (Stable-Baselines3's PPO implementation, on a custom Gymnasium
  environment).

## Project structure

```
env/                    Car physics (bicycle model), track geometry,
                         vectorized collision/checkpoint detection, vectorized raycasting
render/                 Rendering (camera modes, track, car, HUD)
neat_mode/
  neat-config.txt         NEAT hyperparameters (population, mutation, speciation)
  fitness.py               CarStats tracking + compute_fitness() + is_stuck()
  population_runner.py     Steps a whole generation, live-renders the population + leader's brain
  render_brain.py          Live network diagram (nodes colored by activation, edges by weight)
  train_neat.py            Entry point — runs NEAT for 300 generations, saves the best genome
rl_mode/
  race_env.py              Gymnasium environment wrapping the same car/track/raycast sim
  live_render.py           Renders all N parallel training envs in a grid during training
  train_rl.py              Entry point — trains PPO for 2M timesteps
assets/                  Track definitions (JSON: walls, checkpoints, spawn point)
Main.py                  Manual driving mode (arrow keys)
play_neat.py             Watch a trained NEAT genome drive (--genome, --track)
play_rl.py               Watch a trained PPO model drive (--model, --track)
```

## Installation

```bash
python -m venv venv
source venv/bin/activate        # or venv\Scripts\activate on Windows
pip install neat-python gymnasium stable-baselines3 pygame numpy tqdm rich tensorboard
```

## Usage

```bash
python Main.py                          # drive manually (arrow keys, R to restart after a crash)
python -m neat_mode.train_neat          # train NEAT (renders the whole population + leader's brain live)
python play_neat.py                     # watch the best saved NEAT genome (R to respawn)
python play_neat.py --genome <path.pkl> --track <path.json>

python -m rl_mode.train_rl              # train PPO (renders a live grid of all parallel envs)
python play_rl.py                       # watch the trained PPO model (R to respawn)
python play_rl.py --model <path.zip> --track <path.json>
```

Run training/playback scripts as modules (`python -m package.script`) from
the project root rather than as standalone files, so their internal package
imports resolve correctly.

## The two methods, briefly

**NEAT (NeuroEvolution of Augmenting Topologies)** is a genetic algorithm
for neural networks. It starts a population of small, mostly-random
networks, lets each one drive a lap, scores them by fitness, then breeds
the next generation from the fittest: recombining and mutating both the
network's weights _and its structure_ (it can grow new nodes and
connections over generations). There's no notion of a reward signal
guiding a single network step by step — only survival-of-the-fittest
across generations. This makes it gradient-free and remarkably good at
escaping local optima, at the cost of needing many parallel rollouts per
generation.

**PPO (Proximal Policy Optimization)** is a gradient-based reinforcement
learning algorithm. A single neural network (the policy) drives the car,
receives a reward at every frame, and is updated via gradient descent to
make the actions that led to higher reward more likely in the future. Its
updates are "proximal" because it clips how far the policy can shift in
one update, which keeps training stable. Unlike NEAT, PPO learns
continuously within one network's lifetime — but everything it learns is
only as good as the reward function it's handed, which is exactly what
the case study below runs into.

## NEAT configuration

Population of 60 genomes per generation, run for up to 300 generations
(`neat_mode/neat-config.txt`, `train_neat.py`). Networks start fully
connected with no hidden nodes (`initial_connection = full_direct`) and
grow structure over time via `node_add_prob = 0.3` / `conn_add_prob =
0.5`, balanced against `node_delete_prob = 0.2` / `conn_delete_prob =
0.3`. Weight and bias mutation rates are high (0.8 / 0.7) with a
mutate-power of 0.5, so most generations perturb existing connections
rather than growing the network. Genomes are grouped into species by
topological/weight similarity (`compatibility_threshold = 3.0`), each
species is protected from extinction for up to 20 stagnant generations,
and the top `survival_threshold = 0.2` of each species reproduces, with
2 elite genomes carried over unchanged each generation.

## PPO configuration

8 parallel environments (`N_ENVS = 8`, via `make_vec_env`'s
`DummyVecEnv`) feeding a single `MlpPolicy`, trained for 2,000,000
timesteps total (`rl_mode/train_rl.py`). Mostly Stable-Baselines3's
default continuous-control hyperparameters: `n_steps=2048`,
`batch_size=256`, `n_epochs=10`, `gamma=0.99`, `gae_lambda=0.95`,
`learning_rate=3e-4`, plus a small `ent_coef=0.01` entropy bonus to keep
the policy exploring instead of collapsing onto "always drive straight"
early on. A `CheckpointCallback` saves a model snapshot roughly every
50,000 timesteps.

## Live visualization

Both training loops render live rather than running as a black box.
NEAT's `population_runner.py` draws every car currently alive on the
track, follows whichever car is in the lead, and renders that leader's
actual network topology in a panel underneath — nodes colored by layer
(input/hidden/output) and sized-by-nothing but colored by their current
activation value, edges colored green/red by weight sign and faded by
magnitude, so you can watch which sensors are actually driving a
decision in real time. PPO's `live_render.py` instead tiles all 8
parallel training environments into a grid with each car's checkpoint
count and running episode reward overlaid, the closest RL equivalent of
watching a NEAT population race together.

## Results

### NEAT

NEAT reliably masters the track, reaching near-maximum fitness
(~121,000) within roughly **30 generations**. This figure is close to the
actual ceiling for the task: with a 1,800-frame (30-second) episode cap,
even a car driving at maximum speed for the entire episode can only
complete ~2.5–3 laps, which bounds achievable fitness to roughly
117,000–130,000. NEAT essentially exhausts the available headroom on this
track well before training would otherwise be considered "done."

### PPO — a reward shaping case study

Getting PPO to comparable performance required three iterations on the
reward function, and the process is worth documenting because it's a
concrete, reproducible example of **reward hacking**: an agent optimizing
the reward function exactly as specified, while failing the actual
underlying objective.

**Iteration 1 — the exploit.** With the initial reward structure:

| Event                              | Reward                            |
| ---------------------------------- | --------------------------------- |
| Crash                              | −10                               |
| Stuck (no progress for 300 frames) | −5                                |
| Idle time                          | ≈0 (negligible per-frame penalty) |

the agent converged on stopping before difficult corners rather than
attempting them. This wasn't a training failure — the policy correctly
identified that repeatedly risking a crash (−10) cost more than giving up
once (−5, or effectively nothing at inference time, since playback has no
timeout to end a frozen episode). Halting was the rational move under this
exact reward function.

This behavior is also a direct consequence of PPO's discount factor
(γ = 0.99). PPO optimizes _discounted future return_, not immediate reward,
and with γ = 0.99 the effective planning horizon extends roughly
1/(1−γ) ≈ 100 steps ahead. A myopic agent would simply chase the immediate
positive shaping reward toward the next checkpoint, crash, and only
register the penalty afterward. Instead, this agent learned to anticipate
that continuing toward a difficult corner leads to a crash a few frames
later, discounted that anticipated −10 back to the present, and found it
outweighed the small reward gained from approaching the checkpoint — so it
stopped before ever reaching the corner.

**Iteration 2 — overcorrection.** The first fix attempt raised the stuck
penalty to match the crash penalty and increased the idle-time penalty
5x:

| Event     | Reward         |
| --------- | -------------- |
| Crash     | −10            |
| Stuck     | −10            |
| Idle time | −0.005 / frame |

This removed the idling exploit but introduced a different problem: over a
full 1,800-frame episode, the time penalty alone totals −9.0 — nearly as
costly as an entire checkpoint (+10). Survival itself became expensive,
with nothing strong enough counterbalancing it. The result: `ep_len_mean`
converged to almost exactly 302 frames — just past the 300-frame
stuck-timeout — meaning nearly every episode across the entire 2M-step run
ended in a stall, not a successful run. `ep_rew_mean` settled at 4.75.

**Iteration 3 — the fix.** The fix wasn't a bigger penalty, and it wasn't a
new reward term either — it was scaling the existing pieces sensibly. The
environment already shapes reward continuously via progress toward the
next checkpoint (`(prev_distance - new_distance) * PROGRESS_REWARD_SCALE`
every frame, see `race_env.py`), but that signal had been getting drowned
out by a time penalty large enough to rival an entire checkpoint. Pulling
the time penalty back to a more modest level let the progress-shaping term
actually do its job as the counterbalance to idling/stalling, instead of
competing against an oversized penalty:

| Constant                | Iteration 2                                           | Iteration 3                                |
| ----------------------- | ----------------------------------------------------- | ------------------------------------------ |
| `CRASH_PENALTY`         | −10                                                   | −10                                        |
| `STUCK_PENALTY`         | −10 (unchanged — matching crash was correct)          | −10                                        |
| `TIME_PENALTY`          | −0.005 / frame (−9.0 over a full episode)             | −0.0015 / frame (−2.7 over a full episode) |
| `PROGRESS_REWARD_SCALE` | 0.01 (unchanged, but overwhelmed by the time penalty) | 0.01                                       |

This produced a clear, unambiguous improvement:

| Metric               | Iteration 2 | Iteration 3 |
| -------------------- | ----------- | ----------- |
| `ep_len_mean`        | 302         | 1,280       |
| `ep_rew_mean`        | 4.75        | 883         |
| `std` (action noise) | 1.76        | 0.662       |
| `explained_variance` | 0.401       | 0.699       |

Episodes now run most of their full length instead of dying near the
stuck-timeout, the policy is far more decisive (lower action noise), and
the value function predicts outcomes substantially better — all consistent
with a policy that's actually solving the task, rather than exploiting a
gap in the reward function.

**Takeaway:** in reinforcement learning, a policy will always find the
cheapest path to satisfying the reward function as literally specified.
Penalizing failure alone doesn't reliably produce good behavior — it can
just as easily produce risk-avoidance strategies, and even a well-intentioned
fix can overcorrect if a penalty is scaled without checking it against the
magnitude of the rewards it's meant to compete with.

## Author

BELAIDI Ramzy Zakaria — Master's student, USTHB (AI / Bioinformatics / CS)
