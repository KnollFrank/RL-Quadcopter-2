import numpy as np
import math
from physics_sim import PhysicsSim

# adapted from https://github.com/udacity/RL-Quadcopter/blob/master/quad_controller_rl/src/quad_controller_rl/tasks/takeoff.py
# rename to Takeoff and file to takeoff.py
class TakeoffTask():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
        """
        # Simulation
        self.runtime = runtime
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 1

        self.state_size = self.action_repeat * len(self.create_non_repeated_state())
        self.action_low = 0
        self.action_high = 900
        self.action_size = 1
        
        self.target_z = 10.0 # target height (z position) to reach for successful takeoff

    def get_reward_done(self):
        """Uses current pose of sim to return reward."""
        done = False
        reward = -min(abs(self.target_z - self.sim.pose[2]), 20.0)
        if self.sim.pose[2] >= self.target_z:
            reward += 10.0
            done = True
        elif self.sim.time > self.runtime:
            reward -= 10.0
            done = True

        return reward, done

    def step(self, rotor_speed):
        """Uses action to obtain next state, reward, done."""
        rotor_speeds = rotor_speed * 4
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds)
            non_repeated_reward, non_repeated_done = self.get_reward_done() 
            reward += non_repeated_reward 
            pose_all.append(self.create_non_repeated_state())
        next_state = np.concatenate(pose_all)
        return next_state, reward, done or non_repeated_done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.create_non_repeated_state()] * self.action_repeat)
        return state

    def create_non_repeated_state(self):
        return np.array([self.sim.pose[2], self.sim.v[2], self.sim.linear_accel[2]])
