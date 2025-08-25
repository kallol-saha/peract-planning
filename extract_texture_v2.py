#!/usr/bin/env python3
"""
Alternative script to extract texture files from CoppeliaSim objects.
Saves the scene and attempts to extract textures from the saved scene.
"""

import os
import sys
import numpy as np
from os.path import dirname, abspath, join
import subprocess
import shutil

from pyrep import PyRep
from pyrep.objects.shape import Shape
from pyrep.objects.object import Object
from pyrep.const import ObjectType


def extract_texture_from_saved_scene(shape, output_dir, texture_name):
    """
    Extract texture by saving the scene and then extracting from the saved file.
    
    Args:
        shape: PyRep Shape object
        output_dir: Directory to save the texture
        texture_name: Name for the texture file
    
    Returns:
        str: Path to the extracted texture file, or None if failed
    """
    try:
        from pyrep.backend import sim
        
        # Get texture ID
        texture = shape.get_texture()
        if texture is None:
            print(f"No texture found for shape {shape.get_name()}")
            return None
            
        texture_id = texture.get_texture_id()
        print(f"Found texture ID: {texture_id}")
        
        # Save the current scene to a temporary file
        temp_scene_path = join(output_dir, "temp_scene_with_textures.ttt")
        print(f"Saving scene to: {temp_scene_path}")
        sim.simSaveScene(temp_scene_path)
        
        # Try to extract texture using CoppeliaSim command line
        texture_path = join(output_dir, f"{texture_name}.png")
        
        # Method 1: Try to use CoppeliaSim's command line interface
        coppelia_root = os.environ.get('COPPELIASIM_ROOT')
        if coppelia_root:
            coppelia_exe = join(coppelia_root, 'coppeliaSim')
            if os.path.exists(coppelia_exe):
                print("Attempting to extract texture using CoppeliaSim command line...")
                
                # Create a Lua script to extract the texture
                lua_script_content = f"""
                function extractTexture()
                    local textureId = {texture_id}
                    local texturePath = "{texture_path}"
                    
                    -- Try to save the texture
                    local result = sim.saveTexture(textureId, texturePath, 0, 0, 0, 0, 0, 0, 0, 0)
                    
                    if result == 0 then
                        print("Texture extracted successfully to: " .. texturePath)
                        return true
                    else
                        print("Failed to extract texture. Error code: " .. result)
                        return false
                    end
                end
                
                extractTexture()
                """
                
                lua_script_path = join(output_dir, "extract_texture.lua")
                with open(lua_script_path, 'w') as f:
                    f.write(lua_script_content)
                
                # Run CoppeliaSim with the script
                cmd = [coppelia_exe, '-h', '-s', lua_script_path, temp_scene_path]
                print(f"Running command: {' '.join(cmd)}")
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    print(f"Command output: {result.stdout}")
                    if result.stderr:
                        print(f"Command errors: {result.stderr}")
                    
                    if os.path.exists(texture_path):
                        print(f"✅ Texture extracted successfully: {texture_path}")
                        return texture_path
                except subprocess.TimeoutExpired:
                    print("Command timed out")
                except Exception as e:
                    print(f"Command failed: {e}")
        
        # Method 2: Try to find texture in the saved scene file
        print("Searching for texture files in saved scene...")
        if os.path.exists(temp_scene_path):
            # The saved scene might contain embedded textures
            # Try to extract them using a different approach
            print("Scene saved, but direct texture extraction not implemented")
            print("You may need to manually extract textures from the saved scene file")
            print(f"Saved scene: {temp_scene_path}")
        
        return None
        
    except Exception as e:
        print(f"Error in texture extraction: {e}")
        return None


def create_texture_placeholder(output_dir, texture_name, color_info):
    """
    Create a placeholder texture based on the object's color.
    
    Args:
        output_dir: Directory to save the texture
        texture_name: Name for the texture file
        color_info: Color information from the object
    
    Returns:
        str: Path to the created texture file
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        # Create a simple colored texture
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        
        # Create a rectangle with the object's color
        rect = patches.Rectangle((0, 0), 1, 1, 
                               facecolor=(color_info['r'], color_info['g'], color_info['b'], color_info['a']),
                               edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # Add text label
        ax.text(0.5, 0.5, f'{texture_name}\nPlaceholder', 
               ha='center', va='center', fontsize=12, fontweight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Save as PNG
        texture_path = join(output_dir, f"{texture_name}.png")
        plt.savefig(texture_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        print(f"✅ Created placeholder texture: {texture_path}")
        return texture_path
        
    except ImportError:
        print("matplotlib not available, cannot create placeholder texture")
        return None
    except Exception as e:
        print(f"Error creating placeholder texture: {e}")
        return None


def main():
    # Configuration
    OBJECT_NAME = "crackers_visual"
    TASK_NAME = "put_groceries_in_cupboard"
    OUTPUT_DIR = "./exports"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Extracting texture from: {OBJECT_NAME}")
    print(f"From task: {TASK_NAME}")
    print(f"To directory: {OUTPUT_DIR}")
    print("-" * 50)
    
    # Initialize PyRep
    pyrep = PyRep()
    
    # Get the path to the base scene file
    rlbench_dir = join(dirname(abspath(__file__)), 'RLBench', 'rlbench')
    base_scene = join(rlbench_dir, 'task_design.ttt')
    
    # Launch CoppeliaSim
    print(f"Loading base scene: {base_scene}")
    pyrep.launch(base_scene, headless=True)
    
    # Get the path to the task TTM file
    task_ttm = join(rlbench_dir, 'task_ttms', f'{TASK_NAME}.ttm')
    
    if not os.path.exists(task_ttm):
        print(f"Error: Task file not found: {task_ttm}")
        pyrep.shutdown()
        return False
    
    # Load the task TTM file
    print(f"Loading task: {task_ttm}")
    imported_model = pyrep.import_model(task_ttm)
    
    # Find the object
    print(f"Looking for object: {OBJECT_NAME}")
    obj = Object.get_object(OBJECT_NAME)
    
    if obj is None:
        print(f"Object '{OBJECT_NAME}' not found!")
        pyrep.shutdown()
        return False
    
    # Check if it's a shape
    obj_type = Object.get_object_type(OBJECT_NAME)
    if obj_type == ObjectType.SHAPE:
        shape = Shape(obj.get_handle())
        
        # Get color information for placeholder
        color = shape.get_color()
        transparency = shape.get_transparency()
        color_info = {
            'r': color[0],
            'g': color[1],
            'b': color[2],
            'a': 1.0 - transparency
        }
        
        # Try to extract texture
        texture_path = extract_texture_from_saved_scene(shape, OUTPUT_DIR, OBJECT_NAME)
        
        if texture_path is None:
            print("\nCreating placeholder texture based on object color...")
            texture_path = create_texture_placeholder(OUTPUT_DIR, OBJECT_NAME, color_info)
        
        if texture_path:
            print(f"\n✅ Texture extraction/completion completed successfully!")
            print(f"📁 Texture file: {texture_path}")
            
            # Update the URDF info file with texture path
            urdf_info_path = join(OUTPUT_DIR, f"{OBJECT_NAME}_urdf_info.json")
            if os.path.exists(urdf_info_path):
                import json
                with open(urdf_info_path, 'r') as f:
                    urdf_info = json.load(f)
                
                urdf_info['texture']['texture_file'] = f"{OBJECT_NAME}.png"
                urdf_info['texture']['texture_path'] = texture_path
                
                with open(urdf_info_path, 'w') as f:
                    json.dump(urdf_info, f, indent=2)
                
                print(f"📝 Updated URDF info file with texture path")
        else:
            print(f"\n❌ Texture extraction failed!")
            print("Note: You may need to manually extract textures from the CoppeliaSim scene file.")
        
        pyrep.shutdown()
        return texture_path is not None
    else:
        print(f"Object '{OBJECT_NAME}' is not a shape (type: {obj_type})")
        pyrep.shutdown()
        return False


if __name__ == "__main__":
    main()

