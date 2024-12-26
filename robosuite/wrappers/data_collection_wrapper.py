"""
This file implements a wrapper for saving simulation states to disk.
This data collection wrapper is useful for collecting demonstrations.
"""

import json
import os
import time

import numpy as np

from robosuite.utils.mjcf_utils import save_sim_model
from robosuite.utils import transform_utils as T
from robosuite.wrappers import Wrapper


class DataCollectionWrapper(Wrapper):
    def __init__(self, env, directory, collect_freq=1, flush_freq=100):
        """
        Initializes the data collection wrapper.

        Args:
            env (MujocoEnv): The environment to monitor.
            directory (str): Where to store collected data.
            collect_freq (int): How often to save simulation state, in terms of environment steps.
            flush_freq (int): How frequently to dump data to disk, in terms of environment steps.
        """
        super().__init__(env)

        # the base directory for all logging
        self.directory = directory

        # in-memory cache for simulation states and action info
        self.states = []
        self.action_infos = []  # stores information about actions taken
        self.successful = False  # stores success state of demonstration

        # how often to save simulation state, in terms of environment steps
        self.collect_freq = collect_freq

        # how frequently to dump data to disk, in terms of environment steps
        self.flush_freq = flush_freq

        if not os.path.exists(directory):
            print("DataCollectionWrapper: making new directory at {}".format(directory))
            os.makedirs(directory)

        # store logging directory for current episode
        self.ep_directory = None

        # remember whether any environment interaction has occurred
        self.has_interaction = False

        # some variables for remembering the current episode's initial state and model xml
        self._current_task_instance_state = None
        self._current_task_instance_xml = None

    def _start_new_episode(self):
        """
        Bookkeeping to do at the start of each new episode.
        """

        # flush any data left over from the previous episode if any interactions have happened
        if self.has_interaction:
            self._flush()

        # timesteps in current episode
        self.t = 0
        self.has_interaction = False

        # save the task instance (will be saved on the first env interaction)

        # NOTE: was previously self.env.model.get_xml(). Was causing the following issue in rare cases:
        # ValueError: Error: eigenvalues of mesh inertia violate A + B >= C
        # switching to self.env.sim.model.get_xml() does not create this issue
        self._current_task_instance_xml = self.env.sim.model.get_xml()
        self._current_task_instance_state = np.array(self.env.sim.get_state().flatten())

        # trick for ensuring that we can play MuJoCo demonstrations back
        # deterministically by using the recorded actions open loop
        self.env.reset_from_xml_string(self._current_task_instance_xml)
        self.env.sim.reset()
        self.env.sim.set_state_from_flattened(self._current_task_instance_state)
        self.env.sim.forward()

    def _on_first_interaction(self):
        """
        Bookkeeping for first timestep of episode.
        This function is necessary to make sure that logging only happens after the first
        step call to the simulation, instead of on the reset (people tend to call
        reset more than is necessary in code).

        Raises:
            AssertionError: [Episode path already exists]
        """

        self.has_interaction = True

        # create a directory with a timestamp
        t1, t2 = str(time.time()).split(".")
        self.ep_directory = os.path.join(self.directory, "ep_{}_{}".format(t1, t2))
        assert not os.path.exists(self.ep_directory)
        print("DataCollectionWrapper: making folder at {}".format(self.ep_directory))
        os.makedirs(self.ep_directory)

        # save the model xml
        xml_path = os.path.join(self.ep_directory, "model.xml")
        with open(xml_path, "w") as f:
            f.write(self._current_task_instance_xml)

        # save the episode info to json file
        ep_meta_path = os.path.join(self.ep_directory, "ep_meta.json")
        with open(ep_meta_path, "w") as f:
            json.dump(self.env.get_ep_meta(), f)

        # save initial state and action
        assert len(self.states) == 0
        self.states.append(self._current_task_instance_state)

    def _flush(self):
        """
        Method to flush internal state to disk.
        """
        t1, t2 = str(time.time()).split(".")
        state_path = os.path.join(self.ep_directory, "state_{}_{}.npz".format(t1, t2))
        if hasattr(self.env, "unwrapped"):
            env_name = self.env.unwrapped.__class__.__name__
        else:
            env_name = self.env.__class__.__name__
        np.savez(
            state_path,
            states=np.array(self.states),
            action_infos=self.action_infos,
            successful=self.successful,
            env=env_name,
        )
        self.states = []
        self.action_infos = []
        self.successful = False

    def reset(self):
        """
        Extends vanilla reset() function call to accommodate data collection

        Returns:
            OrderedDict: Environment observation space after reset occurs
        """
        ret = super().reset()
        self._start_new_episode()
        return ret

    def world_to_gripper_frame(self, action, robot, arm):
        """Convert world frame action to gripper frame."""
        # Get gripper pose using proper robot attributes
        gripper_pos = self.sim.data.site_xpos[robot.eef_site_id[arm]]
        gripper_rot = self.sim.data.site_xmat[robot.eef_site_id[arm]].reshape(3, 3)
        
        # Split translation and rotation from action
        pos_delta = action[:3]
        rot_delta = action[3:6]
        
        # Transform to gripper frame
        pos_gripper = gripper_rot.T @ pos_delta
        rot_gripper = gripper_rot.T @ rot_delta
        
        return np.concatenate([pos_gripper, rot_gripper, action[6:]])  # Keep gripper action

    def get_joint_positions(self, robot, arm):
        """Get current joint positions for the specified robot arm."""
        if hasattr(robot, '_ref_joint_pos_indexes'):
            # For standard robots
            start_idx = 0 if arm == "right" else robot._joint_split_idx
            end_idx = robot._joint_split_idx if arm == "right" else None
            return robot.sim.data.qpos[robot._ref_joint_pos_indexes[start_idx:end_idx]].copy()
        return None

    def step(self, action):
        """
        Extends vanilla step() function call to accommodate data collection

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (OrderedDict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        ret = super().step(action)
        self.t += 1

        # on the first time step, make directories for logging
        if not self.has_interaction:
            self._on_first_interaction()

        # collect the current simulation state if necessary
        if self.t % self.collect_freq == 0:
            state = self.env.sim.get_state().flatten()
            self.states.append(state)

            # Get robot and arm (assuming single robot, single arm)
            robot = self.env.robots[0]
            arm = robot.arms[0]  # Usually 'right' for single arm

            # Store actions
            info = {}
            
            # 1. World frame (original)
            info["actions"] = np.array(action)
            
            try:
                # 2. Gripper frame
                # Use proper robot model attributes
                eef_site_id = robot.eef_site_id[arm]
                gripper_pos = self.env.sim.data.site_xpos[eef_site_id]
                gripper_rot = self.env.sim.data.site_xmat[eef_site_id].reshape(3, 3)
                
                # Convert world frame deltas to gripper frame
                pos_delta = action[:3]
                rot_delta = action[3:6]
                gripper_pos_delta = gripper_rot.T @ pos_delta
                gripper_rot_delta = gripper_rot.T @ rot_delta
                
                info["actions_gripper_frame"] = np.concatenate([
                    gripper_pos_delta,
                    gripper_rot_delta,
                    action[6:]  # Keep gripper action unchanged
                ])

                # 3. Joint positions - use robot's joint indexes
                if hasattr(robot, '_ref_joint_pos_indexes'):
                    info["joint_positions"] = robot.sim.data.qpos[robot._ref_joint_pos_indexes].copy()
                    info["joint_velocities"] = robot.sim.data.qvel[robot._ref_joint_vel_indexes].copy()

            except Exception as e:
                print(f"Warning: Error collecting additional action formats: {e}")
                
            self.action_infos.append(info)

        # check if the demonstration is successful
        if self.env._check_success():
            self.successful = True

        # flush collected data to disk if necessary
        if self.t % self.flush_freq == 0:
            self._flush()

        return ret

    def close(self):
        """
        Override close method in order to flush left over data
        """
        if self.has_interaction:
            self._flush()
        self.env.close()
