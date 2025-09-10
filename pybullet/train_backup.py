import gym
import torch
import glob
import cv2
import os
import re
import pybullet as p

import trainer
import time
from datetime import datetime
import shutil
import yaml

#MPC part
from IPython.display import clear_output
import os
import time

import numpy as np
import casadi
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import sys
import ur5e_rope
import shutil

# Add the parent directory to the path to allow imports from other directories
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def get_day_time():
    """Get the current day time.

    Returns:
        str: Current day time in format 'YYYY-MM-DD HH:MM:SS'
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# Define your environment (ensure UR5eRopeEnv is correctly implemented and accessible)
class UR5eRopeEnvWrapper(gym.Env):
    def __init__(self, fps, step_episode,youngs_modulus,radius_rope, client_id):
        self.env = ur5e_rope.UR5eRopeEnv(
            fps=fps, step_episode=step_episode, youngs_modulus=youngs_modulus, radius_rope=radius_rope, client_id=client_id
        )

    @property
    def youngs_modulus(self):
        return self.env.youngs_modulus

    @property
    def radius_rope(self):
        return self.env.radius_rope

    def set_youngs_modulus(self, youngs_modulus):
        self.env.set_youngs_modulus(youngs_modulus)

    def step(self, p0, pN):
        return self.env.step(p0, pN)

    def reset(self, rope_length, p0, pN,youngs_modulus,radius_rope):
        return self.env.reset(rope_length, p0, pN,youngs_modulus,radius_rope)

    def seed(self, seed=None):
        self.env.seed(seed)

    def render(self, mode="human"):
        if mode == "rgb_array":
            frame = self.env.render(mode)
            return frame
        else:
            self.env.render(mode)

    def close(self):
        self.env.close()

class MakeVideo:
    def __init__(self, fps, imgsize, src, videoname):
        self.fps = 5#fps
        self.imgsize = imgsize
        self.src = src
        self.videoname = videoname
        self.main()

    def atoi(self, text):
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        return [self.atoi(c) for c in re.split(r"(\d+)", text)]

    def main(self):
        # Check if source directory exists
        if not os.path.exists(self.src):
            raise FileNotFoundError(f"Source directory {self.src} does not exist")

        # Get sorted list of PNG files
        png_files = sorted(glob.glob(f"{self.src}/*.png"), key=self.natural_keys)
        if not png_files:
            raise FileNotFoundError(f"No PNG files found in {self.src}")

        # Read first image to get dimensions
        first_img = cv2.imread(png_files[0])
        if first_img is None:
            raise ValueError(f"Could not read image: {png_files[0]}")

        height, width, layers = first_img.shape
        size = (width, height)
        #print(f"size={size}")
        #print(f"self.fps={self.fps},png_files={png_files}")

        # Validate all images have same dimensions
        for filename in png_files[1:]:
            img = cv2.imread(filename)
            if img is None:
                print(f"Warning: Could not read {filename}, skipping...")
                continue
            if img.shape[:2] != (height, width):
                raise ValueError(f"Image {filename} has different dimensions: {img.shape[:2]} vs {(height, width)}")

        # Create video writer with proper error handling
        try:
            out = cv2.VideoWriter(
                self.videoname, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, size
            )

            if not out.isOpened():
                # Try alternative codec
                out = cv2.VideoWriter(
                    self.videoname, cv2.VideoWriter_fourcc(*"avc1"), self.fps, size
                )
                if not out.isOpened():
                    raise RuntimeError("Could not create video writer with any codec")

            # Write images to video
            for filename in png_files:
                img = cv2.imread(filename)
                if img is not None:
                    out.write(img)

        finally:
            out.release()


# Initialize PPO algorithm
"""CHECK HERE"""
params_sim = {
    "fps":50,
    "step_episode": 3,  # 20 seconds.
    "num_eval_episodes": 3,  # number of episodes per evaluation
    "root_dir": os.getcwd(),
    "root_config": r"C:\Users\kawaw\python\mpc\casadi_mpc_nyuumon\src\pybullet\config",
    "youngs_modulus": 0.64e9,#0.64GPa, nylon
    "radius_rope": 0.008, #8 mm.
    "base_init": [0.0, 0.0, 0.0],
    "tip_init": [0.0, 0.0, 2.0],
    "rope_length": 2.0
}

rootDir = params_sim["root_dir"]
"""End : tocheck"""

postfix = f"result_{get_day_time()}"
saveDir = os.path.join(rootDir, postfix)
os.makedirs(saveDir, exist_ok=True)
video_folder = os.path.join(saveDir, "video")
#simulation control frequency.
_fps = params_sim["fps"]
_replay_speed = 0.5
_replay_speed = int(
    _replay_speed * _fps / 5
)  # replay_speed [frames/s] = _fps/(10frame).rendering is every 10 frames.
_step_episode = _fps * params_sim["step_episode"]

# Initialize environments
id_render = p.connect(p.DIRECT)
env_render = UR5eRopeEnvWrapper(
    fps=_fps, step_episode=_step_episode, youngs_modulus=params_sim["youngs_modulus"],
    radius_rope=params_sim["radius_rope"], client_id=id_render
)

# Initialize Trainer
trainer_instance = trainer.Trainer(
    env_render=env_render,
    fps=_fps,
    seed=0,
    step_episode=_step_episode,
    rootDir=rootDir,
    saveDir=saveDir,
    video_folder=video_folder,
)

# Optionally, visualize the trained policy
trainer_instance.visualize(params_sim["base_init"], params_sim["tip_init"],params_sim["rope_length"])

# make a video
video_path = os.path.join(saveDir, "result.mp4")
imgsize = (640, 480)
mkVideo_left = MakeVideo(
    fps=_replay_speed, imgsize=imgsize, src=video_folder, videoname=video_path
)

# Clean up video folder after video creation
if os.path.exists(video_folder):
    shutil.rmtree(video_folder)
