#!/usr/bin/env python3
"""
Script to extract texture files from CoppeliaSim objects.
Uses CoppeliaSim's Lua API to extract texture data.
"""

import os
import sys
import numpy as np
from os.path import dirname, abspath, join

from pyrep import PyRep
from pyrep.objects.shape import Shape
from pyrep.objects.object import Object
from pyrep.const import ObjectType


def extract_texture_from_shape(shape, output_dir, texture_name):
    """
    Extract texture from a shape using CoppeliaSim's Lua API.
    
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
        
        # Create Lua script to extract texture
        lua_script = f"""
        function extractTexture()
            local textureId = {texture_id}
            local texturePath = "{join(output_dir, texture_name)}.png"
            
            -- Try to get texture data and save it
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
        
        # Execute Lua script
        print("Executing texture extraction script...")
        result = sim.simExecuteScriptFunction(sim.sim_scripttype_childscript, 
                                            "extractTexture", 
                                            [], 
                                            lua_script)
        
        texture_path = join(output_dir, f"{texture_name}.png")
        if os.path.exists(texture_path):
            print(f"✅ Texture extracted successfully: {texture_path}")
            return texture_path
        else:
            print("❌ Texture extraction failed")
            return None
            
    except Exception as e:
        print(f"Error extracting texture: {e}")
        return None


def extract_texture_alternative_method(shape, output_dir, texture_name):
    """
    Alternative method to extract texture using different CoppeliaSim API.
    
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
        
        # Try to get texture data using different methods
        lua_script = f"""
        function extractTextureAlternative()
            local textureId = {texture_id}
            local texturePath = "{join(output_dir, texture_name)}.png"
            
            -- Method 1: Try sim.saveTexture
            local result = sim.saveTexture(textureId, texturePath, 0, 0, 0, 0, 0, 0, 0, 0)
            if result == 0 then
                print("Method 1 successful: " .. texturePath)
                return true
            end
            
            -- Method 2: Try sim.getTextureData
            local textureData = sim.getTextureData(textureId)
            if textureData then
                -- Save texture data as PNG
                local file = io.open(texturePath, "wb")
                if file then
                    file:write(textureData)
                    file:close()
                    print("Method 2 successful: " .. texturePath)
                    return true
                end
            end
            
            -- Method 3: Try sim.getTextureId
            local textureInfo = sim.getTextureId(textureId)
            if textureInfo then
                print("Texture info available but extraction method not implemented")
                return false
            end
            
            print("All extraction methods failed")
            return false
        end
        
        extractTextureAlternative()
        """
        
        # Execute Lua script
        print("Trying alternative texture extraction methods...")
        result = sim.simExecuteScriptFunction(sim.sim_scripttype_childscript, 
                                            "extractTextureAlternative", 
                                            [], 
                                            lua_script)
        
        texture_path = join(output_dir, f"{texture_name}.png")
        if os.path.exists(texture_path):
            print(f"✅ Texture extracted successfully: {texture_path}")
            return texture_path
        else:
            print("❌ All texture extraction methods failed")
            return None
            
    except Exception as e:
        print(f"Error in alternative texture extraction: {e}")
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
        
        # Try to extract texture
        texture_path = extract_texture_from_shape(shape, OUTPUT_DIR, OBJECT_NAME)
        
        if texture_path is None:
            print("Trying alternative extraction method...")
            texture_path = extract_texture_alternative_method(shape, OUTPUT_DIR, OBJECT_NAME)
        
        if texture_path:
            print(f"\n✅ Texture extraction completed successfully!")
            print(f"📁 Texture file: {texture_path}")
        else:
            print(f"\n❌ Texture extraction failed!")
            print("Note: Texture extraction from CoppeliaSim may require additional setup.")
            print("You may need to manually extract textures from the CoppeliaSim scene file.")
        
        pyrep.shutdown()
        return texture_path is not None
    else:
        print(f"Object '{OBJECT_NAME}' is not a shape (type: {obj_type})")
        pyrep.shutdown()
        return False


if __name__ == "__main__":
    main()

