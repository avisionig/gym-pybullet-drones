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

        # After base class initialization, override or add any additional setup
        # For example, add the spaceship
        self._addSpaceship()
        # Update DRONE_IDS to include the spaceship ID
        self.DRONE_IDS = [self.spaceship_id]

    def _addAssets(self):
        """Override to add the spaceship instead of drones."""
        # Add your spaceship
        self._addSpaceship()
        # Update DRONE_IDS
        self.DRONE_IDS = [self.spaceship_id]


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
            'C:\\Users\\Ayan\\Desktop\\endka\\gym-pybullet-drones\\gym_pybullet_drones\\assets\\basic_spaceship.urdf',
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
        self.DRONE_IDS = [self.spaceship_id]

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
        Computes the current reward value, including penalties and rewards.
        """
        state = self._getSpaceshipState()
        position = state["position"]
        velocity = state["velocity"]

        # Compute speed
        speed = np.linalg.norm(velocity)

        stationary_threshold = 0.1  # Define a threshold for considering the spaceship as stationary
        if speed < stationary_threshold:
            stationary_penalty = 1.0  # Penalize for being stationary
        else:
            stationary_penalty = 0.0

        flying_reward = speed * 0.1  # Reward proportional to speed
        distance_to_target = np.linalg.norm(self.TARGET_POS - position)
        proximity_reward = max(0, 1 - distance_to_target * 0.1)
        reward = flying_reward + proximity_reward - stationary_penalty
        fuel_penalty = (self.FUEL_CAPACITY - self.fuel) * 0.01
        reward -= fuel_penalty

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

    def _computeTerminated(self):
        """Determines if the episode should be terminated."""
        state = self._getSpaceshipState()
        position = state["position"]
        altitude = position[2]  # Z-coordinate represents altitude

        if self.fuel <= 0:
            return True
        if np.linalg.norm(position[:2]) > 10.0:  # X-Y plane boundary radius
            return True

        if np.linalg.norm(self.TARGET_POS - position) < 0.1:
            return True

        if self.has_flown and altitude <= 0.1:  # Ground level threshold
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

    def _preprocessAction(self, action):
        """Handle spaceship-specific action preprocessing."""
        self.fuel -= np.linalg.norm(action) * 0.1  # Decrease fuel with action intensity
        if self.fuel < 0:
            self.fuel = 0  # Prevent negative fuel
        return super()._preprocessAction(action)