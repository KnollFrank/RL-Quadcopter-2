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
        self.runtime = runtime
        self.sim = PhysicsSim(init_pose = np.array([0., 0., init_height, 0., 0., 0.]),
                              init_velocities = np.array([0., 0., 0.]),
                              init_angle_velocities = np.array([0., 0., 0.]),
                              runtime = runtime) 
        self.action_repeat = 1

        self.state_size = self.action_repeat * len(self.create_non_repeated_state())
        self.action_low = 0
        self.action_high = 900
        self.action_size = 1
        
        self.target_height = target_height

    def get_reward_done(self, target_height, actual_height, actual_time, runtime):
        """Uses current pose of sim to return reward."""
        done = False
        reward = -min(abs(target_height - actual_height), 20.0)
        if actual_height >= target_height: # agent has crossed the target height
            reward += 10.0 # bonus reward
            done = True
        elif actual_time > runtime: # agent has run out of time
            reward -= 10.0 # extra penalty
            done = True

        return reward, done

    def step(self, rotor_speed):
        """Uses action to obtain next state, reward, done."""
        rotor_speeds = rotor_speed * 4
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds)
            non_repeated_reward, non_repeated_done = self.get_reward_done(self.target_height, self.sim.pose[2], self.sim.time, self.runtime) 
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
