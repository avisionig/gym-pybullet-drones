import pybullet as p
import numpy as np
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import SpaceshipModel, Physics, ActionType, ObservationType
from gymnasium import spaces
import pkg_resources

class SpaceshipAviary(BaseRLAviary):
    def __init__(self, 
                 spaceship_model: SpaceshipModel = SpaceshipModel.BASIC,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM
                 ):
        # Initialize your attributes
        # ... [Your initialization code here]

        # Set NUM_DRONES to 1
        self.NUM_DRONES = 1
        self.spaceship_id = None
        # Initialize other necessary attributes
        self.INIT_XYZS = initial_xyzs
        self.INIT_RPYS = initial_rpys
        self.PYB_FREQ = pyb_freq
        self.CTRL_FREQ = ctrl_freq
        self.GUI = gui
        self.RECORD = record
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        self.PHYSICS = physics
        self.episode_counter = 0
        self.FUEL_CAPACITY = 100
        self.TARGET_POS = np.array([0, 0, 10])
        self.EPISODE_LEN_SEC = 20

        # Define action space (add this line)
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        # Action format: [thrust, roll, pitch, yaw]
        
        # Define observation space (already present, but ensure it's here)
        self.observation_space = self._observationSpace()

        # Call the base class initializer
        super().__init__(
            drone_model=spaceship_model,
            num_drones=self.NUM_DRONES,
            initial_xyzs=self.INIT_XYZS,
            initial_rpys=self.INIT_RPYS,
            physics=self.PHYSICS,
            pyb_freq=self.PYB_FREQ,
            ctrl_freq=self.CTRL_FREQ,
            gui=self.GUI,
            record=self.RECORD,
            obs=self.OBS_TYPE,
            act=self.ACT_TYPE
        )

        self._addSpaceship()
        self._addAssets()
        self.DRONE_IDS = [self.spaceship_id]

    def _addAssets(self):

        self.DRONE_IDS = [self.spaceship_id]

        wall_urdf_path = r"C:\Users\Hp\Desktop\RL\gym-pybullet-drones\gym_pybullet_drones\assets\tall_wall.urdf"
        
        self.wall_id = p.loadURDF(
            wall_urdf_path,
            physicsClientId=self.CLIENT
        )
        if self.wall_id < 0:
            raise ValueError("Failed to load wall URDF!")
        
        target_urdf_path = pkg_resources.resource_filename('gym_pybullet_drones', 'assets/target_marker.urdf')  # Assuming you have a URDF for the target marker
        target_position = [0, 7, 0.01]
        self.target_id = p.loadURDF(
            target_urdf_path,
            physicsClientId=self.CLIENT
        )
        if self.target_id < 0:
            raise ValueError("Failed to load target marker URDF!")
    
        self.TARGET_POS = np.array(target_position)

    def _housekeeping(self):
        """Override to include base class housekeeping."""
        # Call the base class's _housekeeping() method
        super()._housekeeping()
        # Additional customizations if necessary
        # For example, update DRONE_IDS if needed
        if self.spaceship_id is not None:
            self.DRONE_IDS = [self.spaceship_id]

    def _addSpaceship(self):
        """Adds the spaceship to the environment."""
        # Construct the full path to the URDF file
        # urdf_file = pkg_resources.resource_filename('gym_pybullet_drones', 'assets/basic_spaceship.urdf')
        self.spaceship_id = p.loadURDF(
            r'..\assets\basic_spaceship.urdf',
            basePosition=self.INIT_XYZS[0],
            baseOrientation=p.getQuaternionFromEuler(self.INIT_RPYS[0]),
            physicsClientId=self.CLIENT
        )
        if self.spaceship_id is None:
            raise ValueError("Failed to load spaceship URDF!")

    def _computeInfo(self):
        """
        Computes additional information about the environment.

        Returns
        -------
        dict
            A dictionary with additional environment-specific information.
        """
        return {"message": "Spaceship environment info"}  # Replace with meaningful data

    def reset(self, seed=None, options=None):
        """Resets the environment."""
        # Reset internal variables
        self.step_counter = 0
        self.episode_counter += 1
        self.has_flown = False
        self.fuel = self.FUEL_CAPACITY

        # Reset the simulation
        p.resetSimulation(physicsClientId=self.CLIENT)
        p.setTimeStep(1 / self.PYB_FREQ, physicsClientId=self.CLIENT)

        # Re-add the spaceship
        self._addSpaceship()
        self._addAssets()
        self.DRONE_IDS = [self.spaceship_id, self.wall_id]

        # Update kinematic information
        self._updateAndStoreKinematicInformation()

        # Compute the initial observation
        initial_obs = self._computeObs()
        return initial_obs, {}



        # # Reset internal variables
        # self.step_counter = 0
        # self.episode_counter += 1
        # self.last_action = np.zeros((self.num_drones, self.action_dim))
        # self.gui_input = np.zeros(4)
        # self._housekeeping()
        # # Reset the simulation
        # self._resetSimulation()
        # # Add the spaceship before computing the initial observation
        # self._addSpaceship()
        # # Start video recording if enabled
        # self._startVideoRecording()
        # # Compute the initial observation
        # initial_obs = self._computeObs()
        # return initial_obs, {}

    def _observationSpace(self):
        """
        Define the observation space with the correct size.
        """
        low = np.array([-np.inf] * 73, dtype=np.float32)  # 73 elements
        high = np.array([np.inf] * 73, dtype=np.float32)  # Adjust to 72
        return spaces.Box(low=low, high=high, dtype=np.float32)


    def _computeReward(self):
        """
        Computes the reward value, focusing on progress, movement, and proximity to the target.
        """
        state = self._getSpaceshipState()
        position = state["position"]
        velocity = state["velocity"]

        # Current distance to target
        current_distance_to_target = np.linalg.norm(self.TARGET_POS - position)

        # Initialize tracking variables for progress
        if not hasattr(self, 'prev_distance_to_target'):  # Initialize if not already present
            self.prev_distance_to_target = current_distance_to_target
            self.prev_altitude = position[2]

        # Horizontal and vertical distances
        horizontal_distance_to_target = np.linalg.norm(self.TARGET_POS[:2] - position[:2])
        altitude_to_target = abs(self.TARGET_POS[2] - position[2])

        # Track progress (reward for reducing horizontal distance)
        distance_delta = self.prev_distance_to_target - current_distance_to_target
        self.prev_distance_to_target = current_distance_to_target
        progress_reward = max(0, distance_delta * 10)  # Strong reward for reducing distance

        # Vertical progress: Reward for moving toward the target altitude
        vertical_delta = position[2] - self.prev_altitude
        self.prev_altitude = position[2]
        vertical_progress_reward = max(0, vertical_delta * 10)

        # Movement reward: Encourage any movement (speed above a threshold)
        speed = np.linalg.norm(velocity)
        movement_reward = 5.0 if speed > 0.2 else -5.0  # Strong penalty for being stationary

        # Proximity reward: Reward for being close to the target
        proximity_reward = 1 / (1 + current_distance_to_target)  # Avoids exponential scaling

        # Penalize increases in distance to target
        distance_penalty = -10.0 if distance_delta < 0 else 0.0

        # Altitude penalty: Penalize flying too high
        altitude_penalty = 0.0
        if position[2] > 15:  # Example threshold for high altitude
            altitude_penalty = (position[2] - 15) * 0.2

        # Combine all rewards and penalties
        reward = (
            progress_reward        # Reward for reducing distance
            + vertical_progress_reward  # Reward for upward movement
            + movement_reward      # Reward for movement
            + proximity_reward     # Reward for being close to the target
            + distance_penalty     # Penalize increasing distance
            - altitude_penalty     # Penalize flying too high
        )

        # Debugging logs for reward components
        print(f"[DEBUG] Progress reward: {progress_reward:.2f}, Vertical progress reward: {vertical_progress_reward:.2f}, "
            f"Movement reward: {movement_reward:.2f}, Proximity reward: {proximity_reward:.2f}, "
            f"Distance penalty: {distance_penalty:.2f}, Altitude penalty: {altitude_penalty:.2f}, "
            f"Total reward: {reward:.2f}")

        return reward

        
    def _getSpaceshipState(self):
        """Returns the spaceship's state or a default state if not initialized."""
        if not hasattr(self, 'spaceship_id') or self.spaceship_id is None:
            return {
                "position": np.zeros(3),
                "velocity": np.zeros(3),
                "orientation": np.zeros(4),
                "angular_velocity": np.zeros(3)
            }
        try:
            position, orientation = p.getBasePositionAndOrientation(self.spaceship_id, physicsClientId=self.CLIENT)
            velocity, angular_velocity = p.getBaseVelocity(self.spaceship_id, physicsClientId=self.CLIENT)
            return {
                "position": np.array(position),
                "velocity": np.array(velocity),
                "orientation": np.array(orientation),
                "angular_velocity": np.array(angular_velocity)
            }
        except p.error as e:
            if self.step_counter > 0:
                print("Warning: Failed to get spaceship state:", e)
            return {
                "position": np.zeros(3),
                "velocity": np.zeros(3),
                "orientation": np.zeros(4),
                "angular_velocity": np.zeros(3)
            }

    # def _computeTerminated(self):
    #     """Determines if the episode should be terminated."""
    #     state = self._getSpaceshipState()
    #     position = state["position"]
    #     altitude = position[2]  # Z-coordinate represents altitude

    #     if self.fuel <= 0:
    #         return True
    #     if np.linalg.norm(position[:2]) > 10.0:  # X-Y plane boundary radius
    #         return True

    #     if np.linalg.norm(self.TARGET_POS - position) < 0.1:
    #         return True

    #     if self.has_flown and altitude <= 0.1:  # Ground level threshold
    #         return True

    #     return False

    def _computeTerminated(self):
        """Determines if the episode should be terminated."""
        state = self._getSpaceshipState()
        position = state["position"]
        velocity = state["velocity"]

        # Compute termination conditions
        distance_to_target = np.linalg.norm(position[:2] - self.TARGET_POS[:2])  # X-Y plane distance
        altitude_difference = abs(position[2] - self.TARGET_POS[2])  # Z-coordinate difference
        speed = np.linalg.norm(velocity)

        # Debugging outputs
        # print(f"Current position: {position}")
        # print(f"Target position: {self.TARGET_POS}")
        # print(f"Distance to target: {distance_to_target}")
        # print(f"Altitude difference: {altitude_difference}")
        # print(f"Speed: {speed}")

        # Landing criteria
        landed_successfully = distance_to_target < 0.5 and altitude_difference < 0.1 and speed < 0.2

        if landed_successfully:  # Successful landing
            print("Rocket successfully landed on the target marker!")
            return True
        if self.fuel <= 0:  # Out of fuel
            return True
        if np.linalg.norm(position[:2]) > 10.0:  # X-Y plane boundary radius
            return True
        if self.has_flown and position[2] <= 0.1:  # Crashed or landed off-target
            return True

        return False


    def _computeTruncated(self):
        """Check if the episode is truncated."""
        state = self._getSpaceshipState()
        position = state["position"]

        out_of_bounds = np.any(np.abs(position) > 50)
        no_fuel = self.fuel <= 0
        timeout = self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC
        return out_of_bounds or no_fuel or timeout

    
    def _computeObs(self):
        """
        Return the observation trimmed to the correct size.
        """
        state = self._getSpaceshipState()
        position = state["position"]
        altitude = position[2]  

        if altitude > 0.5:  
            self.has_flown = True
        action_buffer_array = np.array(self.action_buffer).flatten()

        obs = np.concatenate([
            state["position"],          
            state["velocity"],         
            state["orientation"],      
            state["angular_velocity"],  
            action_buffer_array         
        ])

        return obs[:73]

    # def _preprocessAction(self, action):
    #     """Handle spaceship-specific action preprocessing."""
    #     self.fuel -= np.linalg.norm(action) * 0.1  # Decrease fuel with action intensity
    #     if self.fuel < 0:
    #         self.fuel = 0  # Prevent negative fuel
    #     return super()._preprocessAction(action)

    def _preprocessAction(self, action):
        """
        Preprocesses the action to include thrust and lateral movement.
        """
        # Ensure action is a 1D array
        if len(action.shape) == 2:
            action = action.squeeze(0)  # Remove the extra dimension

        # Debug the shape after squeezing
        # print(f"[DEBUG] Processed Action shape: {action.shape}, Action: {action}")

        # Check the action shape
        if action.shape[0] != 4:
            raise ValueError(f"Action shape mismatch. Expected 4, got {action.shape[0]}")

        # Parse the action components
        thrust = action[0]  # Upward thrust
        roll = action[1]    # Roll for lateral X movement
        pitch = action[2]   # Pitch for lateral Y movement
        yaw = action[3]     # Yaw for rotation

        # Decrease fuel based on action intensity
        self.fuel -= abs(thrust) * 0.1 + abs(roll) * 0.05 + abs(pitch) * 0.05 + abs(yaw) * 0.05
        if self.fuel < 0:
            self.fuel = 0

        # Apply forces and torques to the spaceship
        p.applyExternalForce(self.spaceship_id, -1, [roll, pitch, thrust], [0, 0, 0], p.LINK_FRAME)
        p.applyExternalTorque(self.spaceship_id, -1, [0, 0, yaw], p.LINK_FRAME)

        return action