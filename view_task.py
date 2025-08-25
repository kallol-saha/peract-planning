#!/usr/bin/env python3
"""
Script to launch CoppeliaSim with visual window and load a specific task.
"""

import os
import sys
from os.path import dirname, abspath, join

# Add RLBench to path
rlbench_path = join(dirname(abspath(__file__)), 'RLBench')
sys.path.append(rlbench_path)

from pyrep import PyRep
from rlbench.backend.task import Task
from rlbench.backend.robot import Robot
from rlbench.backend.scene import Scene
from rlbench.robots.arms.panda import Panda
from rlbench.robots.grippers.panda_gripper import PandaGripper
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig


class PutGroceriesInCupboardTask(Task):
    """Simple task class to load the put groceries in cupboard task."""
    
    def init_task(self) -> None:
        # This will be called when the task is loaded
        pass
    
    def init_episode(self, index: int) -> list:
        # Return task descriptions
        return ['put groceries in cupboard']
    
    def variation_count(self) -> int:
        return 1


def main():
    print("Launching CoppeliaSim with visual window...")
    
    # Initialize PyRep
    pyrep = PyRep()
    
    # Get the path to the base scene file
    rlbench_dir = join(dirname(abspath(__file__)), 'RLBench', 'rlbench')
    base_scene = join(rlbench_dir, 'task_design.ttt')
    
    # Launch CoppeliaSim with visual window (headless=False)
    print(f"Loading base scene: {base_scene}")
    pyrep.launch(base_scene, headless=False)
    
    # Create robot
    arm = Panda()
    gripper = PandaGripper()
    robot = Robot(arm, gripper)
    
    # Create scene
    obs_config = ObservationConfig()
    scene = Scene(pyrep, robot, obs_config, 'panda')
    
    # Create task
    task = PutGroceriesInCupboardTask(pyrep, robot, 'put_groceries_in_cupboard')
    
    # Load the task (this loads the .ttm file)
    print("Loading put_groceries_in_cupboard task...")
    task.load()
    
    print("CoppeliaSim launched successfully!")
    print("Task loaded: put_groceries_in_cupboard")
    print("You should see the CoppeliaSim window with the task scene.")
    print("Press Ctrl+C to exit.")
    
    try:
        # Keep the simulation running
        while True:
            pyrep.step()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        pyrep.shutdown()
        print("CoppeliaSim closed.")


if __name__ == "__main__":
    main()
