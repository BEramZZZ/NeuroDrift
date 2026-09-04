import pygame
from stable_baselines3.common.callbacks import BaseCallback

from render import renderer

GRID_ROWS = 2
GRID_COLS = 4
CELL_WIDTH = 260
CELL_HEIGHT = 240
HUD_HEIGHT = 50
SCREEN_WIDTH = CELL_WIDTH * GRID_COLS
SCREEN_HEIGHT = CELL_HEIGHT * GRID_ROWS + HUD_HEIGHT
FOLLOW_ZOOM = 1.3
CELL_BORDER_COLOR = (60, 62, 66)
HUD_BG_COLOR = (20, 24, 28)
HUD_TEXT_COLOR = (235, 235, 235)


class LiveRenderCallback(BaseCallback):
    """Renders all N parallel training environments at once, in a grid -
    the RL equivalent of watching the whole NEAT population race together.

    Only works with DummyVecEnv (all sub-envs living in this same process,
    not SubprocVecEnv), since it reaches directly into each sub-env's car
    and track to draw it - make_vec_env() defaults to DummyVecEnv, so this
    just works as long as train_rl.py doesn't override that.
    """

    def __init__(self, render_every=1, verbose=0):
        super().__init__(verbose)
        self.render_every = render_every
        self.screen = None
        self.font = None
        self.clock = None
        self.episode_rewards = None

    def _on_training_start(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("NeuroDrift - RL Training (live)")
        self.font = pygame.font.SysFont("Arial", 14)
        self.clock = pygame.time.Clock()
        self.episode_rewards = [0.0] * self.training_env.num_envs

    def _on_step(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False  # SB3 stops model.learn() cleanly, no exception needed

        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        for i in range(self.training_env.num_envs):
            self.episode_rewards[i] += rewards[i]

        if self.n_calls % self.render_every == 0:
            self._draw(infos)
            self.clock.tick(60)

        # reset AFTER drawing, so a just-finished episode's final total is
        # visible for one frame instead of snapping straight to 0
        for i in range(self.training_env.num_envs):
            if dones[i]:
                self.episode_rewards[i] = 0.0

        return True

    def _draw(self, infos):
        renderer.clear_screen(self.screen)

        for i in range(self.training_env.num_envs):
            row, col = divmod(i, GRID_COLS)
            cell_rect = pygame.Rect(col * CELL_WIDTH, row * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT)
            cell_surface = self.screen.subsurface(cell_rect)

            env = self.training_env.envs[i].unwrapped
            camera = renderer.compute_follow_camera(env.car.x, env.car.y, CELL_WIDTH, CELL_HEIGHT, zoom=FOLLOW_ZOOM)
            renderer.set_camera(*camera)
            renderer.draw_track(cell_surface, env.track)
            renderer.draw_car(cell_surface, env.car)

            checkpoints = infos[i].get("checkpoints_passed", env.checkpoints_passed)
            renderer.draw_hud(
                cell_surface, self.font,
                extra_text=f"Env {i}  cp:{checkpoints}  R:{self.episode_rewards[i]:.1f}",
            )
            pygame.draw.rect(self.screen, CELL_BORDER_COLOR, cell_rect, width=2)

        hud_rect = pygame.Rect(0, GRID_ROWS * CELL_HEIGHT, SCREEN_WIDTH, HUD_HEIGHT)
        hud_surface = self.screen.subsurface(hud_rect)
        hud_surface.fill(HUD_BG_COLOR)

        status = f"Timesteps: {self.num_timesteps}/{self.locals.get('total_timesteps', '?')}"
        if len(self.model.ep_info_buffer) > 0:
            ep_rew_mean = sum(ep["r"] for ep in self.model.ep_info_buffer) / len(self.model.ep_info_buffer)
            status += f"   ep_rew_mean: {ep_rew_mean:.1f}"
        text = self.font.render(status, True, HUD_TEXT_COLOR)
        hud_surface.blit(text, (10, HUD_HEIGHT // 2 - 8))

        pygame.display.flip()

    def _on_training_end(self):
        pygame.quit()