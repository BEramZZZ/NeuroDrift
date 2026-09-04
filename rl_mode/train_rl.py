import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env

from rl_mode.live_render import LiveRenderCallback
from rl_mode.race_env import RaceEnv

TRACK_PATH = "assets/track1.json"
MODEL_PATH = "rl_mode/models/ppo_car.zip"
CHECKPOINT_DIR = "rl_mode/models/checkpoints"
TOTAL_TIMESTEPS = 2_000_000
N_ENVS = 8  # parallel simulations - PPO is on-policy and benefits a lot from
            # collecting experience from many environments at once instead of
            # one car at a time (this is RL's version of NEAT's whole-population-
            # per-generation parallelism)


def make_env():
    return RaceEnv(track_path=TRACK_PATH)


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # make_vec_env wraps each sub-env with a Monitor automatically, which is
    # what feeds the ep_rew_mean / ep_len_mean numbers you'll see in the
    # verbose=1 training log below
    vec_env = make_vec_env(make_env, n_envs=N_ENVS)

    # PPO hyperparameters: mostly SB3 defaults, which are already well-tuned
    # for continuous-control tasks like this one. n_steps*n_envs = the batch
    # of experience collected before each policy update.
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        device="cpu",  # "cuda" is faster if you have a GPU, but this is small enough that CPU is fine
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        ent_coef=0.01,  # small entropy bonus - keeps exploring instead of
                        # collapsing onto "always drive straight" too early
        tensorboard_log="rl_mode/tb_logs",
    )

    # saves a snapshot every save_freq*N_ENVS timesteps, so - like the NEAT
    # side's checkpointing fix - a crash or an early stop never loses all
    # progress, just whatever happened since the last checkpoint
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1),
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_car",
    )

    # live_render_callback draws all N_ENVS cars in a grid every render_every
    # steps (see rl_mode/live_render.py) - this slows training down since it
    # gates step speed to 60fps, raise render_every once you just want speed
    live_render_callback = LiveRenderCallback(render_every=50)
    callback = CallbackList([checkpoint_callback, live_render_callback])

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, progress_bar=True)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Training finished. Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()