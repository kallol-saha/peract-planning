#!/usr/bin/env python3
"""
Script to export a specific object from a RLBench task to OBJ format.
"""

import os
import sys
import numpy as np
from os.path import dirname, abspath, join

from pyrep import PyRep
from pyrep.objects.shape import Shape
from pyrep.objects.object import Object
from pyrep.const import ObjectType


def export_shape_to_obj(shape, output_path):
    """
    Export a PyRep Shape object to OBJ format.
    
    Args:
        shape: PyRep Shape object
        output_path: Path to save the OBJ file
    """
    try:
        # Get mesh data from the shape
        vertices, indices, normals = shape.get_mesh_data()
        
        # Convert to numpy arrays if needed
        if not isinstance(vertices, np.ndarray):
            vertices = np.array(vertices)
        if not isinstance(indices, np.ndarray):
            indices = np.array(indices)
        
        # Get the object's pose (position and orientation)
        pose = shape.get_pose()
        position = pose[:3]
        orientation = pose[3:]  # quaternion
        
        print(f"Exporting object: {shape.get_name()}")
        print(f"  Position: {position}")
        print(f"  Orientation: {orientation}")
        print(f"  Vertices: {len(vertices)}")
        print(f"  Faces: {len(indices)}")
        
        # Write OBJ file
        with open(output_path, 'w') as f:
            f.write(f"# Exported from RLBench task\n")
            f.write(f"# Object: {shape.get_name()}\n")
            f.write(f"# Position: {position}\n")
            f.write(f"# Orientation: {orientation}\n\n")
            
            # Write vertices
            for vertex in vertices:
                f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            
            # Write normals if available
            if normals is not None and len(normals) > 0:
                for normal in normals:
                    f.write(f"vn {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
            
            # Note: Texture coordinates (UVs) are not available from get_mesh_data()
            # If needed, they would need to be obtained through other methods
            
            # Write faces (OBJ uses 1-based indexing)
            for face in indices:
                if len(face) == 3:  # Triangle
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
                elif len(face) == 4:  # Quad
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1} {face[3]+1}\n")
        
        print(f"Successfully exported to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error exporting object {shape.get_name()}: {e}")
        return False


def export_shape_for_urdf(shape, output_dir, object_name):
    """
    Export a PyRep Shape object for URDF use with texture and material information.
    
    Args:
        shape: PyRep Shape object
        output_dir: Directory to save the exported files
        object_name: Name of the object for file naming
    
    Returns:
        dict: Dictionary containing URDF visual information
    """
    try:
        from pyrep.backend import sim
        
        # Export mesh to OBJ
        obj_path = join(output_dir, f"{object_name}.obj")
        success = export_shape_to_obj(shape, obj_path)
        if not success:
            return None
        
        # Get material properties
        color = shape.get_color()
        transparency = shape.get_transparency()
        
        # Get texture information
        texture_info = {}
        try:
            texture = shape.get_texture()
            if texture is not None:
                texture_id = texture.get_texture_id()
                texture_info['texture_id'] = texture_id
                
                # Get texture mapping parameters (using correct PyRep API)
                try:
                    texture_x = sim.simGetObjectFloatParameter(shape.get_handle(), sim.sim_shapefloatparam_texture_x)
                    texture_y = sim.simGetObjectFloatParameter(shape.get_handle(), sim.sim_shapefloatparam_texture_y)
                    texture_scaling_x = sim.simGetObjectFloatParameter(shape.get_handle(), sim.sim_shapefloatparam_texture_scaling_x)
                    texture_scaling_y = sim.simGetObjectFloatParameter(shape.get_handle(), sim.sim_shapefloatparam_texture_scaling_y)
                except:
                    # Fallback if texture parameters are not available
                    texture_x = texture_y = texture_scaling_x = texture_scaling_y = 0.0
                
                texture_info.update({
                    'texture_x': texture_x,
                    'texture_y': texture_y,
                    'texture_scaling_x': texture_scaling_x,
                    'texture_scaling_y': texture_scaling_y
                })
                
                print(f"  Texture ID: {texture_id}")
                print(f"  Texture mapping: x={texture_x:.3f}, y={texture_y:.3f}")
                print(f"  Texture scaling: x={texture_scaling_x:.3f}, y={texture_scaling_y:.3f}")
        except Exception as e:
            print(f"  No texture found: {e}")
        
        # Create URDF visual information
        urdf_info = {
            'mesh_file': f"{object_name}.obj",
            'mesh_path': obj_path,
            'color': {
                'r': color[0],
                'g': color[1], 
                'b': color[2],
                'a': 1.0 - transparency
            },
            'texture': texture_info,
            'object_name': object_name,
            'shape_name': shape.get_name()
        }
        
        # Save URDF info to JSON for reference
        json_path = join(output_dir, f"{object_name}_urdf_info.json")
        import json
        with open(json_path, 'w') as f:
            json.dump(urdf_info, f, indent=2)
        
        print(f"  Color: RGB({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})")
        print(f"  Transparency: {transparency:.3f}")
        print(f"  URDF info saved to: {json_path}")
        
        return urdf_info
        
    except Exception as e:
        print(f"Error exporting object {shape.get_name()} for URDF: {e}")
        return None


def find_object_by_name(pyrep, object_name):
    """
    Find an object by name in the scene.
    
    Args:
        pyrep: PyRep instance
        object_name: Name of the object to find
    
    Returns:
        Object if found, None otherwise
    """
    try:
        # Try to get the object directly
        obj = Object.get_object(object_name)
        return obj
    except:
        # If not found, search through all objects
        all_objects = pyrep.get_objects_in_tree()
        for obj in all_objects:
            if obj.get_name() == object_name:
                return obj
        return None


def list_available_objects(pyrep, show_all=False):
    """
    List available objects in the scene.
    
    Args:
        pyrep: PyRep instance
        show_all: If True, show all objects. If False, show only exportable shape objects.
    """
    # Note: Using get_objects_in_tree() instead of get_objects() 
    # as the latter method doesn't exist in PyRep
    
    if show_all:
        print("Available objects in the scene:")
        print("-" * 60)
        
        # Get all objects in the scene tree
        all_objects = pyrep.get_objects_in_tree()
        shape_objects = []
        other_objects = []
        
        for obj in all_objects:
            obj_type = Object.get_object_type(obj.get_name())
            obj_name = obj.get_name()
            
            if obj_type == ObjectType.SHAPE:  # Shape type
                shape_objects.append((obj_name, obj_type))
            else:
                other_objects.append((obj_name, obj_type))
        
        # List shape objects first (these can be exported)
        if shape_objects:
            print("📦 SHAPE OBJECTS (can be exported to OBJ):")
            print("-" * 40)
            for i, (obj_name, obj_type) in enumerate(shape_objects, 1):
                print(f"{i:3d}. {obj_name}")
            print()
        
        # List other objects
        if other_objects:
            print("🔧 OTHER OBJECTS (cannot be exported):")
            print("-" * 40)
            for i, (obj_name, obj_type) in enumerate(other_objects, 1):
                print(f"{i:3d}. {obj_name} (type: {obj_type})")
            print()
        
        print(f"Total objects: {len(all_objects)}")
        print(f"Shape objects: {len(shape_objects)} (exportable)")
        print(f"Other objects: {len(other_objects)} (not exportable)")
    else:
        print("📦 EXPORTABLE OBJECTS (Shape objects only):")
        print("-" * 50)
        
        # Get only shape objects
        shape_objects = pyrep.get_objects_in_tree(object_type=ObjectType.SHAPE)
        
        if shape_objects:
            for i, obj in enumerate(shape_objects, 1):
                print(f"{i:3d}. {obj.get_name()}")
            print()
            print(f"Total exportable objects: {len(shape_objects)}")
        else:
            print("No exportable objects found in the scene.")


def export_object(object_name, task_name="put_groceries_in_cupboard", output_dir="./exports", list_objects=False, show_all_objects=False):
    """
    Export a specific object from a RLBench task to OBJ format.
    
    Args:
        object_name: Name of the object to export (or "list" to list all objects)
        task_name: Name of the task (default: "put_groceries_in_cupboard")
        output_dir: Directory to save the exported OBJ file (default: "./exports")
        list_objects: If True, just list all available objects and return
    
    Returns:
        bool: True if export was successful, False otherwise
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Launching CoppeliaSim...")
    
    # Initialize PyRep
    pyrep = PyRep()
    
    # Get the path to the base scene file
    rlbench_dir = join(dirname(abspath(__file__)), 'RLBench', 'rlbench')
    base_scene = join(rlbench_dir, 'task_design.ttt')
    
    # Launch CoppeliaSim in headless mode for export
    print(f"Loading base scene: {base_scene}")
    pyrep.launch(base_scene, headless=True)
    
    # Get the path to the task TTM file
    task_ttm = join(rlbench_dir, 'task_ttms', f'{task_name}.ttm')
    
    if not os.path.exists(task_ttm):
        print(f"Error: Task file not found: {task_ttm}")
        print("Available task files:")
        task_dir = join(rlbench_dir, 'task_ttms')
        if os.path.exists(task_dir):
            for f in os.listdir(task_dir):
                if f.endswith('.ttm'):
                    print(f"  - {f}")
        pyrep.shutdown()
        return False
    
    # Load the task TTM file
    print(f"Loading task: {task_ttm}")
    imported_model = pyrep.import_model(task_ttm)
    
    # List all objects if requested
    if list_objects or object_name.lower() == "list":
        list_available_objects(pyrep, show_all=show_all_objects)
        pyrep.shutdown()
        return True
    
    # Find the object
    print(f"Looking for object: {object_name}")
    obj = find_object_by_name(pyrep, object_name)
    
    if obj is None:
        print(f"Object '{object_name}' not found!")
        print("\nAvailable objects:")
        list_available_objects(pyrep, show_all=show_all_objects)
        pyrep.shutdown()
        return False
    
    # Check if it's a shape (has mesh data)
    obj_type = Object.get_object_type(object_name)
    if obj_type == ObjectType.SHAPE:  # Shape type
        shape = Shape(obj.get_handle())
        
        # Use URDF export method for visual objects
        if object_name.endswith('_visual'):
            print(f"Exporting visual object for URDF use...")
            urdf_info = export_shape_for_urdf(shape, output_dir, object_name)
            
            if urdf_info:
                print(f"\n✅ Visual object '{object_name}' exported successfully for URDF!")
                print(f"📁 Files created:")
                print(f"   - Mesh: {urdf_info['mesh_path']}")
                print(f"   - URDF info: {join(output_dir, f'{object_name}_urdf_info.json')}")
                print(f"\n🎨 Material properties:")
                print(f"   - Color: RGB({urdf_info['color']['r']:.3f}, {urdf_info['color']['g']:.3f}, {urdf_info['color']['b']:.3f})")
                print(f"   - Alpha: {urdf_info['color']['a']:.3f}")
                if urdf_info['texture']:
                    print(f"   - Texture: ID {urdf_info['texture']['texture_id']}")
                pyrep.shutdown()
                return True
            else:
                print(f"\n❌ Failed to export visual object '{object_name}'")
                pyrep.shutdown()
                return False
        else:
            # Regular OBJ export for non-visual objects
            output_path = join(output_dir, f"{object_name}.obj")
            success = export_shape_to_obj(shape, output_path)
            
            if success:
                print(f"\nObject '{object_name}' exported successfully!")
                print(f"Output file: {output_path}")
                pyrep.shutdown()
                return True
            else:
                print(f"\nFailed to export object '{object_name}'")
                pyrep.shutdown()
                return False
    else:
        print(f"Object '{object_name}' is not a shape (type: {obj_type})")
        print("Only Shape objects can be exported to OBJ format.")
        pyrep.shutdown()
        return False


def main():
    # ============================================================================
    # CONFIGURATION - Modify these variables as needed
    # ============================================================================
    
    # Object to export (set to "list" to see all available objects)
    OBJECT_NAME = "crackers_visual"  # Change this to the object you want to export
    
    # Task name (default: "put_groceries_in_cupboard")
    TASK_NAME = "put_groceries_in_cupboard"  # Change this if needed
    
    # Output directory for exported OBJ files
    OUTPUT_DIR = "./exports"  # Change this if needed
    
    # Set to True to just list all objects without exporting
    LIST_OBJECTS_ONLY = False  # Set to True to just list objects
    
    # Set to True to show all objects (including non-exportable ones)
    SHOW_ALL_OBJECTS = False  # Set to True to show all objects, False for exportable only
    
    # ============================================================================
    # END CONFIGURATION
    # ============================================================================
    
    if LIST_OBJECTS_ONLY:
        if SHOW_ALL_OBJECTS:
            print("Listing all available objects...")
        else:
            print("Listing exportable objects only...")
        export_object("list", TASK_NAME, OUTPUT_DIR, list_objects=True, show_all_objects=SHOW_ALL_OBJECTS)
    else:
        print(f"Exporting object: {OBJECT_NAME}")
        print(f"From task: {TASK_NAME}")
        print(f"To directory: {OUTPUT_DIR}")
        print("-" * 50)
        
        success = export_object(OBJECT_NAME, TASK_NAME, OUTPUT_DIR)
        
        if success:
            print(f"\n✅ Export completed successfully!")
        else:
            print(f"\n❌ Export failed!")


if __name__ == "__main__":
    main()
