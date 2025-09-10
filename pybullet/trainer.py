import time
import numpy as np
import torch
from datetime import timedelta
import os
import gym
import glob
from base64 import b64encode
from IPython.display import HTML
from gym.wrappers.monitoring import video_recorder
import cv2
import matplotlib.pyplot as plt
import pandas as pd
# Add the parent directory to the path to allow imports from other directories
import sys

"""CHECK HERE"""
rootDir = r"C:\Users\kawaw\python\pybullet\ur5-bullet\UR5\code\robot_RL_curriculum\residual_policy\residual_target"
type_reward = "ppo"
lr_type = "constant"
bool_residual_lr = False  # Redidual learning or not.
file_base = os.path.join(rootDir, "ppo_actor_critic.pt")  # Path to the baseline model.
"""CHECK HERE"""


class Trainer:

    def __init__(
        self,
        env_render,
        fps=200,
        seed=0,
        turning_radius=0.5,
        turning_speed=4.0,
        step_episode=4 * 10**3,
        rootDir=rootDir,
        saveDir=rootDir,
        video_folder=rootDir,
    ):
        self.env_render = env_render
        self.fps = fps
        self.saveDir = saveDir
        self.video_folder = video_folder
        self.turning_radius = turning_radius
        self.turning_speed = turning_speed
        os.makedirs(self.saveDir, exist_ok=True)
        os.makedirs(self.video_folder, exist_ok=True)
        # Dictionary to store average returns
        self.returns = {"step": [], "return": [], "return_base": [], "return_diff": []}

        self.step_episode = step_episode


    def visualize(self,base_init,tip_init,rope_length):
        """Visualize a single episode using the trained policy."""
        # env = gym.make(self.env.unwrapped.spec.id)

        rope_state = self.env_render.reset(rope_length=rope_length, p0=base_init, pN=tip_init,
        youngs_modulus=self.env_render.youngs_modulus,radius_rope=self.env_render.radius_rope)
        rope_state_list = [rope_state]
        vel_base = np.zeros(3)
        vel_tip = np.zeros(3)
        done = False

        #Simulation setting.
        t = 0
        time_list = [0.0]
        fps = self.fps
        period_save = 5
        counter = 0
        while not done:
            #update velocity
            vel_base[1] = self.turning_radius * np.sin(self.turning_speed * t)
            vel_base[2] = self.turning_radius * np.cos(self.turning_speed * t)

            vel_tip[1] = self.turning_radius * np.sin(self.turning_speed * t)
            vel_tip[2] = self.turning_radius * np.cos(self.turning_speed * t)
            #update position
            base_current = base_init + vel_base
            tip_current = tip_init + vel_tip

            t += 1/fps


            rope_state, done = self.env_render.step(p0=base_current, pN=tip_current)
            rope_state_list.append(rope_state)
            counter += 1
            time_list.append(t)
            if counter % period_save == 1:
                print(f"counter={counter}/{self.step_episode}")
                frame = self.env_render.render(mode="rgb_array")
                if isinstance(frame, np.ndarray):
                    file_img = os.path.join(self.video_folder, f"{counter:05d}.png")
                    # print("frame.shape=",frame.shape)
                    cv2.imwrite(file_img, frame)

        time_list = np.array(time_list)
        rope_state_list = np.array(rope_state_list)
        self.plot_eval(time_list, rope_state_list)

    def plot_eval(self, times, rope_state_list):
        """
        Parameters:
        -----------
        times : list
            time list
        rope_state_list : list
            rope state list (N_step,N_segment,3)
        """
        #save all the rope state in a csv file.
        rope_state_list = np.array(rope_state_list)
        rope_state_csv = rope_state_list.reshape(-1, rope_state_list.shape[1]*rope_state_list.shape[2]) #(N_step,N_segment*3)
        df = pd.DataFrame(rope_state_csv)
        df.to_csv(os.path.join(self.saveDir, "rope_state.csv"), index=False)

        fig,ax = plt.subplots(1,3,figsize=(20,5))
        ax[0].plot(times, rope_state_list[:,0,0])
        ax[1].plot(times, rope_state_list[:,0,1])
        ax[2].plot(times, rope_state_list[:,0,2])
        ax[0].set_title("x")
        ax[1].set_title("y")
        ax[2].set_title("z")
        ax[0].set_xlabel("time [s]")
        ax[1].set_xlabel("time [s]")
        ax[2].set_xlabel("time [s]")
        ax[0].set_ylabel("x")
        ax[1].set_ylabel("y")
        ax[2].set_ylabel("z")
        ax[0].grid()
        ax[1].grid()
        ax[2].grid()
        fig.savefig(os.path.join(self.saveDir, "rope_state_initial.png"))

        #plot the middle point transition.
        fig,ax = plt.subplots(1,3,figsize=(20,5))
        ax[0].plot(times, rope_state_list[:,int(rope_state_list.shape[1]/2),0])
        ax[1].plot(times, rope_state_list[:,int(rope_state_list.shape[1]/2),1])
        ax[2].plot(times, rope_state_list[:,int(rope_state_list.shape[1]/2),2])
        ax[0].set_title("x")
        ax[1].set_title("y")
        ax[2].set_title("z")
        ax[0].set_xlabel("time [s]")
        ax[1].set_xlabel("time [s]")
        ax[2].set_xlabel("time [s]")
        ax[0].set_ylabel("x")
        ax[1].set_ylabel("y")
        ax[2].set_ylabel("z")
        ax[0].grid()
        ax[1].grid()
        ax[2].grid()
        fig.savefig(os.path.join(self.saveDir, "rope_state_middle.png"))

        #plot the end point transition.
        fig,ax = plt.subplots(1,3,figsize=(20,5))
        ax[0].plot(times, rope_state_list[:,-1,0])
        ax[1].plot(times, rope_state_list[:,-1,1])
        ax[2].plot(times, rope_state_list[:,-1,2])
        ax[0].set_title("x")
        ax[1].set_title("y")
        ax[2].set_title("z")
        ax[0].set_xlabel("time [s]")
        ax[1].set_xlabel("time [s]")
        ax[2].set_xlabel("time [s]")
        ax[0].set_ylabel("x")
        ax[1].set_ylabel("y")
        ax[2].set_ylabel("z")
        ax[0].grid()
        ax[1].grid()
        ax[2].grid()
        fig.savefig(os.path.join(self.saveDir, "rope_state_end.png"))


    @property
    def time(self):
        """Calculate the elapsed training time."""
        return str(timedelta(seconds=int(time.time() - self.start_time)))
