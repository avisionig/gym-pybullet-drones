"""Script demonstrating the joint use of velocity input.

The simulation is run by a `VelocityAviary` environment.

Example
-------
In a terminal, run as:

    $ python pid_velocity.py

Notes
-----
The drones use interal PID control to track a target velocity.

"""
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from stable_baselines3 import PPO
from gym_pybullet_drones.envs.SpaceshipAviary import SpaceshipAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType, SpaceshipModel

DEFAULT_DRONE = DroneModel("cf2x")
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 15
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

def run(
        trained_model_path="C:/Users/Ayan/Desktop/endka/gym-pybullet-drones/spaceship_results/save-12.01.2024_15.42.29/final_model.zip",
        gui=True,
        record_video=True,
        plot=True,
        simulation_freq_hz=240,
        control_freq_hz=48,
        duration_sec=15,
        output_folder="spaceship_results",
        colab=False
        ):
    """
    Test a trained model for the spaceship.
    """

    #### Create the environment ################################
    env = SpaceshipAviary(
        obs=ObservationType('kin'),
        act=ActionType('rpm'),
        spaceship_model=SpaceshipModel.BASIC,
        gui=gui,
        record=record_video,
        pyb_freq=simulation_freq_hz,
        ctrl_freq=control_freq_hz,
    )

    #### Load the trained model ################################
    print(f"[INFO] Loading trained model from: {trained_model_path}")
    model = PPO.load(trained_model_path)

    #### Initialize the logger #################################
    logger = Logger(
        logging_freq_hz=control_freq_hz,
        num_drones=1,  # Single entity (spaceship)
        output_folder=output_folder,
        colab=colab
    )

    #### Run the simulation ####################################
    obs, info = env.reset()
    START = time.time()
    for i in range(0, int(duration_sec * env.CTRL_FREQ)):
        #### Predict action using the trained model ############
        action, _ = model.predict(obs, deterministic=True)

        #### Step the simulation ################################
        obs, reward, terminated, truncated, info = env.step(action)

        #### Log the simulation #################################
        logger.log(
            drone=0,  # Replace "drone" with "spaceship"
            timestamp=i/env.CTRL_FREQ,
            state=obs,  # Log spaceship state
            control=action
        )

        #### Printout and render ###############################
        env.render()

        #### Sync the simulation ################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

        #### Reset environment if terminated ####################
        if terminated:
            obs, info = env.reset()

    #### Close the environment #################################
    env.close()

    #### Plot the simulation results ###########################
    logger.save_as_csv("test_spaceship")  # Save logs as CSV
    if plot:
        logger.plot()


if __name__ == "__main__":
    import argparse

    #### Define and parse arguments for the script ##
    parser = argparse.ArgumentParser(description='Spaceship velocity control testing')
    parser.add_argument('--gui', default=True, type=bool, help='Whether to use GUI (default: True)')
    parser.add_argument('--record_video', default=True, type=bool, help='Whether to record a video (default: True)')
    parser.add_argument('--plot', default=True, type=bool, help='Whether to plot results (default: True)')
    parser.add_argument('--simulation_freq_hz', default=240, type=int, help='Simulation frequency in Hz (default: 240)')
    parser.add_argument('--control_freq_hz', default=48, type=int, help='Control frequency in Hz (default: 48)')
    parser.add_argument('--duration_sec', default=15, type=int, help='Duration of the simulation in seconds (default: 15)')
    parser.add_argument('--output_folder', default="spaceship_results", type=str, help='Output folder for results')
    parser.add_argument('--trained_model_path', default="C:/Users/Ayan/Desktop/endka/gym-pybullet-drones/spaceship_results/save-12.01.2024_15.42.29/final_model.zip", type=str, help='Path to the trained model')
    args = parser.parse_args()
    run(
        trained_model_path=args.trained_model_path,
        gui=args.gui,
        record_video=args.record_video,
        plot=args.plot,
        simulation_freq_hz=args.simulation_freq_hz,
        control_freq_hz=args.control_freq_hz,
        duration_sec=args.duration_sec,
        output_folder=args.output_folder
    )

