import numpy as np
import math
from physics_sim import PhysicsSim

# adapted from https://github.com/udacity/RL-Quadcopter/blob/master/quad_controller_rl/src/quad_controller_rl/tasks/takeoff.py
class TakeoffTask():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_height, target_height, runtime=5):
        """Initialize a Task object.
        Params
        ======
            init_height: initial height of the quadcopter in z dimension to start from
            target_height: target height of the quadcopter in z dimension to reach for successful takeoff
            runtime: time limit for each episode
        """
        # Simulation
        self.init_height = init_height
        self.target_height = target_height
        self.runtime = runtime
        self.sim = PhysicsSim(init_pose = np.array([0., 0., init_height, 0., 0., 0.]),
                              init_velocities = np.array([0., 0., 0.]),
                              init_angle_velocities = np.array([0., 0., 0.]),
                              runtime = runtime) 

        self.state_size = len(self.create_state())
        self.action_low = 0
        self.action_high = 900
        self.action_size = 1
        
    def get_reward(self, actual_height):
        return -abs(actual_height - self.target_height)

    def step(self, rotor_speed):
        """Uses action to obtain next state, reward, done."""
        done = self.sim.next_timestep(rotor_speed * 4)
        return self.create_state(), self.get_reward(self.sim.pose[2]), self.sim.pose[2] >= self.target_height or done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = [self.create_state()]
        return state

    def create_state(self):
        return np.array([self.sim.pose[2], self.sim.v[2], self.sim.linear_accel[2]])
