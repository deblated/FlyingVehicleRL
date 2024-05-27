import pygame.display
import gym
import random

from typing import Optional
from FlyingVehicle import *
from gym import spaces

from Utils import pygame_events


class FlyingVehicleEnv(gym.Env):
    def __init__(self, max_time_steps=500):
        self.screen = None
        self.draw_options = None
        self.screen_width = 1280
        self.screen_height = 720
        self.clock = pygame.time.Clock()
        self.force_mag = 1000
        self.max_time_steps = max_time_steps
        self.steps_count = 0
        self.background = pygame.image.load("background.png")

        self.left_force = -1
        self.right_force = -1
        self.x_target = random.uniform(50, self.screen_width - 50)
        self.y_target = random.uniform(50, self.screen_height - 50)

        self._init_pymunk()

        action_high = np.array([1, 1], dtype=np.float32)
        self.action_space = spaces.Box(low=-action_high, high=action_high, dtype=np.float32)

        observation_high = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=-observation_high, high=observation_high, dtype=np.float32)

    def _init_pymunk(self):
        pymunk.pygame_util.positive_y_is_up = True
        self.space = pymunk.Space()
        self.space.gravity = Vec2d(0, -1000)

        self.vehicle = FlyingVehicle(random.uniform(200, self.screen_width - 200),
                                     random.uniform(200, self.screen_height - 200),
                                     random.uniform(-np.pi / 4, np.pi / 4),
                                     20,
                                     100,
                                     0.2,
                                     0.4,
                                     0.4,
                                     self.space)

        self.vehicle_radius = self.vehicle.vehicle_radius

    def step(self, action):
        self.steps_count += 1
        terminated = False
        truncated = False

        self.left_force = (action[0] / 2 + 0.5) * self.force_mag
        self.right_force = (action[1] / 2 + 0.5) * self.force_mag
        self.vehicle.frame_shape.body.apply_force_at_local_point(Vec2d(0, self.left_force), (-self.vehicle_radius, 0))
        self.vehicle.frame_shape.body.apply_force_at_local_point(Vec2d(0, self.right_force), (self.vehicle_radius, 0))

        self.space.step(1 / 60)

        obs = self._get_obs()
        reward = (1 / (np.abs(obs[4]) + 0.1)) + (1 / (np.abs(obs[5]) + 0.1))

        if np.abs(obs[3]) == 1 or np.abs(obs[6]) == 1 or np.abs(obs[7]) == 1:
            terminated = True
            reward = -10

        if self.steps_count == self.max_time_steps:
            truncated = True

        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            self.draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES
        pygame_events(self.space, self, self.screen_height)
        self.screen.blit(self.background, (0, 0))
        pygame.draw.rect(self.screen, (51, 25, 0), pygame.Rect(0, 0, self.screen_width, self.screen_height), 10)

        self.space.debug_draw(self.draw_options)

        pygame.draw.circle(self.screen, (255, 0, 0), (self.x_target, self.screen_height - self.y_target), 5)

        pygame.display.flip()
        self.clock.tick(60)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.__init__(self.max_time_steps)
        return self._get_obs(), {}

    def _get_obs(self):
        velocity_x, velocity_y = self.vehicle.frame_shape.body.velocity_at_local_point((0, 0))
        velocity_x = np.clip(velocity_x / 1500, -1, 1)
        velocity_y = np.clip(velocity_y / 1500, -1, 1)

        omega = self.vehicle.frame_shape.body.angular_velocity
        omega = np.clip(omega / 10, -1, 1)

        alpha = self.vehicle.frame_shape.body.angle
        alpha = np.clip(alpha / (np.pi / 2), -1, 1)

        x, y = self.vehicle.frame_shape.body.position

        if x < self.x_target:
            distance_x = np.clip((x / self.x_target) - 1, -1, 0)
        else:
            distance_x = np.clip(
                (-x / (self.x_target - self.screen_width) + self.x_target / (self.x_target - self.screen_width)), 0, 1)

        if y < self.y_target:
            distance_y = np.clip((y / self.y_target) - 1, -1, 0)
        else:
            distance_y = np.clip(
                (-y / (self.y_target - self.screen_height) + self.y_target / (self.y_target - self.screen_height)), 0,
                1)

        pos_x = np.clip(x / (self.screen_width / 2) - 1, -1, 1)
        pos_y = np.clip(y / (self.screen_height / 2) - 1, -1, 1)

        return np.array([velocity_x, velocity_y, omega, alpha, distance_x, distance_y, pos_x, pos_y])

    def change_target_pos(self, x, y):
        self.x_target = x
        self.y_target = y
