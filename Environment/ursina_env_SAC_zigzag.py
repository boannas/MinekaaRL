import gym
from gym import spaces
import numpy as np
from ursina import *
import cv2
from panda3d.core import PNMImage, loadPrcFileData
import os
from stable_baselines3.common.callbacks import BaseCallback

# Headless config
loadPrcFileData('', 'window-type offscreen')
loadPrcFileData('', 'clock-mode limited')
loadPrcFileData('', 'clock-frame-rate 0')
os.environ['URSINA_HEADLESS'] = '1'

class UrsinaParkourEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.app = Ursina()
        window.borderless = True
        window.visible = False

        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=0, high=255, shape=(3, 84, 84), dtype=np.uint8)  # Format: HWC

        self.max_steps = 50  # Max steps per episode
        self.goal_pos = Vec3(4, 0.05, 12)
        self.done = False
        self.step_time_pernalty = 0.01
        self._build_world()


    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

    def _build_world(self):
        Sky()

        for x in range(-2, 8):
            for z in range(-2, 19):
                Entity(model='cube', color=color.red.tint(-0.3), position=(x, 0, z), scale=(1, 0.1, 1), collider='box', name='lava')

        # Row z=2: S at x=3,4
        Entity(model='cube', color=color.rgb(194, 178, 128), position=(3, 0.05, 2), scale=(1, 0.2, 1), collider='box', name='sandstone')
        Entity(model='cube', color=color.rgb(194, 178, 128), position=(4, 0.05, 2), scale=(1, 0.2, 1), collider='box', name='sandstone')

        # Row z=3: S at x=2,3
        Entity(model='cube', color=color.rgb(194, 178, 128), position=(2, 0.05, 3), scale=(1, 0.2, 1), collider='box', name='sandstone')
        Entity(model='cube', color=color.rgb(194, 178, 128), position=(3, 0.05, 3), scale=(1, 0.2, 1), collider='box', name='sandstone')

        # Row z=4: S at x=3,4
        Entity(model='cube', color=color.rgb(194, 178, 128), position=(3, 0.05, 4), scale=(1, 0.2, 1), collider='box', name='sandstone')
        Entity(model='cube', color=color.rgb(194, 178, 128), position=(4, 0.05, 4), scale=(1, 0.2, 1), collider='box', name='sandstone')

        # Row z=5: S at x=4,5
        Entity(model='cube', color=color.rgb(194, 178, 128), position=(4, 0.05, 5), scale=(1, 0.2, 1), collider='box', name='sandstone')
        Entity(model='cube', color=color.rgb(194, 178, 128), position=(5, 0.05, 5), scale=(1, 0.2, 1), collider='box', name='sandstone')

        # Row z=6: S at x=3,4
        Entity(model='cube', color=color.rgb(194, 178, 128), position=(3, 0.05, 6), scale=(1, 0.2, 1), collider='box', name='sandstone')
        Entity(model='cube', color=color.rgb(194, 178, 128), position=(4, 0.05, 6), scale=(1, 0.2, 1), collider='box', name='sandstone')

        # Row z=7: S at x=2,3
        Entity(model='cube', color=color.rgb(194, 178, 128), position=(2, 0.05, 7), scale=(1, 0.2, 1), collider='box', name='sandstone')
        Entity(model='cube', color=color.rgb(194, 178, 128), position=(3, 0.05, 7), scale=(1, 0.2, 1), collider='box', name='sandstone')

        # Row z=8: S at x=1,2
        Entity(model='cube', color=color.rgb(194, 178, 128), position=(1, 0.05, 8), scale=(1, 0.2, 1), collider='box', name='sandstone')
        Entity(model='cube', color=color.rgb(194, 178, 128), position=(2, 0.05, 8), scale=(1, 0.2, 1), collider='box', name='sandstone')

        # Row z=9: S at x=0,1
        Entity(model='cube', color=color.rgb(194, 178, 128), position=(0, 0.05, 9), scale=(1, 0.2, 1), collider='box', name='sandstone')
        Entity(model='cube', color=color.rgb(194, 178, 128), position=(1, 0.05, 9), scale=(1, 0.2, 1), collider='box', name='sandstone')

        # Row z=10: S at x=1,2
        Entity(model='cube', color=color.rgb(194, 178, 128), position=(1, 0.05, 10), scale=(1, 0.2, 1), collider='box', name='sandstone')
        Entity(model='cube', color=color.rgb(194, 178, 128), position=(2, 0.05, 10), scale=(1, 0.2, 1), collider='box', name='sandstone')

        # Row z=11: S at x=2,3
        Entity(model='cube', color=color.rgb(194, 178, 128), position=(2, 0.05, 11), scale=(1, 0.2, 1), collider='box', name='sandstone')
        Entity(model='cube', color=color.rgb(194, 178, 128), position=(3, 0.05, 11), scale=(1, 0.2, 1), collider='box', name='sandstone')

        # Row z=12: S at x=3
        Entity(model='cube', color=color.rgb(194, 178, 128), position=(3, 0.05, 12), scale=(1, 0.2, 1), collider='box', name='sandstone')

        # Start and goal blocks
        Entity(model='cube', color=color.green, position=(4, 0.05, 1), scale=(1, 0.2, 1), collider='box', name='start')
        Entity(model='cube', color=color.cyan, position=(4, 0.05, 12), scale=(1, 0.2, 1), collider='box', name='goal')

        self.player = Entity(model='cube', color=color.orange, scale=(0.5, 1, 0.5), position=(4, 1.1, 1), collider='box')
        self.player.velocity = Vec3(0, 0, 0)
        self.gravity = -0.05
        self.jump_force = 0.5
        self.on_ground = False
        self.prev_distance = self._distance_to_goal()
        self.rotation_angles = [0, 90, 180, 270]
        self.current_angle_index = 0

        camera.parent = self.player
        camera.position = (0, 1, 0)
        camera.rotation = (0, 0, 0)
        camera.fov = 90

    def reset(self, *, seed=None, options=None):
        """Reset the environment with seed and options parameters (Gym v0.21+ API)"""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset the environment state
        self.player.position = Vec3(4, 1.1, 1)
        self.player.velocity = Vec3(0, 0, 0)
        self.done = False
        self.prev_distance = self._distance_to_goal()
        self.step_count = 0
        
        return self._get_obs(), {}

    def step(self, action):
        """Execute one step in the environment (updated for Gym v0.26+ API)"""
        if self.done:
            return self._get_obs(), 0, True, False, {}  # Added truncated=False

        self._apply_gravity()
        self._move(action)
        self._check_ground()

        reward = self._compute_reward()
        self.app.taskMgr.step()
        obs = self._get_obs()
        terminated = self._check_done()
        truncated = self.step_count >= self.max_steps

        if truncated:
            self.done = True

        
        return obs, reward, terminated, truncated, {}

    def _apply_gravity(self):
        self.player.velocity.y += self.gravity
        self.player.y += self.player.velocity.y

    def _move(self, action):
        forward = Vec3(self.player.forward.x, 0, self.player.forward.z).normalized()
        right = Vec3(self.player.right.x, 0, self.player.right.z).normalized()
        speed = 0.5

        if action == 0:
            self.player.position += forward * speed
        elif action == 1:
            self.player.position -= forward * speed
        elif action == 2:
            self.player.position -= right * speed
        elif action == 3:
            self.player.position += right * speed
        elif action == 4:
            self.player.rotation_x = 45
        elif action == 5:
            self.player.rotation_x = 90
        elif action == 6:
            self.player.rotation_x = 0
        elif action == 7:
            self.current_angle_index = (self.current_angle_index - 1) % 4
            self.player.rotation_y = self.rotation_angles[self.current_angle_index]
        elif action == 8:
            self.current_angle_index = (self.current_angle_index + 1) % 4
            self.player.rotation_y = self.rotation_angles[self.current_angle_index]
        elif action == 9 and self.on_ground:
            self.player.velocity.y = self.jump_force
            self.player.position += forward * 0.2
            self.on_ground = False

    def _check_ground(self):
        hit_info = raycast(self.player.world_position, direction=Vec3(0, -1, 0), distance=0.6, ignore=(self.player,))
        if hit_info.hit:
            self.player.y = hit_info.world_point.y + 0.5
            self.player.velocity.y = 0
            self.on_ground = True
        else:
            self.on_ground = False

    def _distance_to_goal(self):
        return distance(self.player.position, self.goal_pos)

    def _compute_reward(self):
        current_distance = self._distance_to_goal()
        reward = (self.prev_distance - current_distance) * 10.0 if self.prev_distance else 0
        self.prev_distance = current_distance

        self.step_count += 1
        reward -= self.step_time_pernalty * self.step_count

        hit_info = raycast(self.player.world_position, direction=Vec3(0, -1, 0), distance=0.6, ignore=(self.player,))
        if hit_info.hit:
            if hit_info.entity.name == 'lava':
                self.done = True
                return -50
            elif hit_info.entity.name == 'goal':
                self.done = True
                return 100
        return reward

    def _check_done(self):
        return self.done or self.player.y < -5

    def _get_obs(self):
        pnm_image = PNMImage()
        application.base.win.getScreenshot(pnm_image)
        w, h = pnm_image.get_x_size(), pnm_image.get_y_size()
        rgb_array = np.zeros((h, w, 3), dtype=np.uint8)

        for y in range(h):
            for x in range(w):
                r = int(pnm_image.getRedVal(x, y))
                g = int(pnm_image.getGreenVal(x, y))
                b = int(pnm_image.getBlueVal(x, y))
                rgb_array[h - y - 1, x] = (r, g, b)

        img_resized = cv2.resize(rgb_array, (84, 84))
        img_chw = np.transpose(img_resized, (2, 0, 1))
        return img_chw


    def render(self, mode='human'):
        window.visible = True

    def close(self):
        self.done = True
        if hasattr(application, "base") and hasattr(application.base, "userExit"):
            try:
                application.base.userExit()
            except SystemExit:
                pass


class EpisodeLogger(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_count += 1
                ep_reward = info['episode']['r']
                print(f"Episode {self.episode_count} — Reward: {ep_reward:.2f}")
        return True

