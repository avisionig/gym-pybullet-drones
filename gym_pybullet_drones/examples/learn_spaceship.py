"""Script demonstrating the use of `gym_pybullet_drones`'s Gymnasium interface for spaceship training.

The SpaceshipAviary environment is used with the PPO algorithm.

Example
-------
In a terminal, run as:

    $ python learn_spaceship.py

Notes
-----
This is a minimal working example integrating `gym-pybullet-drones` with 
reinforcement learning library `stable-baselines3` for a spaceship task.

"""
import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.SpaceshipAviary import SpaceshipAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType, SpaceshipModel

# Default settings
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = True
DEFAULT_OUTPUT_FOLDER = 'spaceship_results'
DEFAULT_OBS = ObservationType('kin')  # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('rpm')  # Action space specific to SpaceshipAviary

def run(output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, record_video=DEFAULT_RECORD_VIDEO, local=True):
    """Main training and evaluation loop."""

    # Set up directories
    filename = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    # Initialize training and evaluation environments
    train_env = make_vec_env(SpaceshipAviary,
                            env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT, spaceship_model=SpaceshipModel.BASIC),
                            n_envs=1,
                            seed=0
                            )

    eval_env = SpaceshipAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)

    # Check environment spaces
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    # Initialize the PPO model
    model = PPO('MlpPolicy',
                train_env,
                verbose=1)

    # Set reward thresholds for stopping training
    target_reward = 100.0  # Adjust based on the reward scaling in SpaceshipAviary
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward, verbose=1)
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=1000,
                                 deterministic=True,
                                 render=False)

    # Train the model
    model.learn(total_timesteps=int(10000),  # Set training duration
                callback=eval_callback,
                log_interval=1000)

    # Save the final model
    model.save(filename+'/final_model.zip')
    print(f"[INFO] Model saved at {filename}")

    # Evaluate and log the training progress
    with np.load(filename+'/evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            print(f"Step: {data['timesteps'][j]}, Reward: {data['results'][j][0]}")

    # Test the trained model
    test_env = SpaceshipAviary(gui=gui, obs=DEFAULT_OBS, act=DEFAULT_ACT, record=record_video)
    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ), num_drones=1, output_folder=output_folder)

    mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=10)
    print("\n\nMean reward:", mean_reward, "±", std_reward, "\n\n")

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        
        # Ensure action is flattened
        action_flat = action.flatten()
        
        # Log the state
        logger.log(
            drone=0,
            timestamp=i / test_env.CTRL_FREQ,
            state=np.hstack([obs[0:3], np.zeros(4), obs[3:15], action_flat]),
            control=np.zeros(12)
        )
        
        test_env.render()
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            obs = test_env.reset(seed=42, options={})
    test_env.close()
    
    if plot:
        logger.plot()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Spaceship reinforcement learning script')
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool, help='Use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VIDEO, type=str2bool, help='Record a video (default: False)', metavar='')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str, help='Folder to save logs (default: "spaceship_results")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
