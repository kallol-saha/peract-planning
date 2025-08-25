#!/usr/bin/env python3
"""
Simple script to launch CoppeliaSim and load a task TTM file directly.
"""

import os
import sys
from os.path import dirname, abspath, join

from pyrep import PyRep


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
    
    # Get the path to the task TTM file
    task_ttm = join(rlbench_dir, 'task_ttms', 'put_groceries_in_cupboard.ttm')
    
    if not os.path.exists(task_ttm):
        print(f"Error: Task file not found: {task_ttm}")
        print("Available task files:")
        task_dir = join(rlbench_dir, 'task_ttms')
        if os.path.exists(task_dir):
            for f in os.listdir(task_dir):
                if f.endswith('.ttm'):
                    print(f"  - {f}")
        pyrep.shutdown()
        return
    
    # Load the task TTM file directly
    print(f"Loading task: {task_ttm}")
    imported_model = pyrep.import_model(task_ttm)
    
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
