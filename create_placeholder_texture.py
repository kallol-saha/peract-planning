#!/usr/bin/env python3
"""
Create a placeholder texture based on object color for URDF use.
"""

import os
import json
from os.path import join

def create_placeholder_texture(output_dir, object_name, color_info):
    """
    Create a placeholder texture based on the object's color.
    
    Args:
        output_dir: Directory to save the texture
        object_name: Name for the texture file
        color_info: Color information from the object
    
    Returns:
        str: Path to the created texture file
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        # Create a simple colored texture
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        
        # Create a rectangle with the object's color
        rect = patches.Rectangle((0, 0), 1, 1, 
                               facecolor=(color_info['r'], color_info['g'], color_info['b'], color_info['a']),
                               edgecolor='black', linewidth=3)
        ax.add_patch(rect)
        
        # Add text label
        ax.text(0.5, 0.5, f'{object_name}\nPlaceholder Texture\nRGB({color_info["r"]:.2f}, {color_info["g"]:.2f}, {color_info["b"]:.2f})\nAlpha: {color_info["a"]:.2f}', 
               ha='center', va='center', fontsize=14, fontweight='bold',
               color='white' if sum(color_info['r':'b']) < 1.5 else 'black')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Save as PNG
        texture_path = join(output_dir, f"{object_name}.png")
        plt.savefig(texture_path, dpi=150, bbox_inches='tight', pad_inches=0.1, 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ Created placeholder texture: {texture_path}")
        return texture_path
        
    except ImportError:
        print("matplotlib not available, creating simple text file instead")
        # Create a simple text file with color information
        texture_path = join(output_dir, f"{object_name}_color_info.txt")
        with open(texture_path, 'w') as f:
            f.write(f"Placeholder texture for {object_name}\n")
            f.write(f"Color: RGB({color_info['r']:.3f}, {color_info['g']:.3f}, {color_info['b']:.3f})\n")
            f.write(f"Alpha: {color_info['a']:.3f}\n")
            f.write(f"Use this color information in your URDF material definition.\n")
        return texture_path
    except Exception as e:
        print(f"Error creating placeholder texture: {e}")
        return None


def main():
    # Read the existing URDF info file
    output_dir = "./exports"
    object_name = "crackers_visual"
    urdf_info_path = join(output_dir, f"{object_name}_urdf_info.json")
    
    if not os.path.exists(urdf_info_path):
        print(f"URDF info file not found: {urdf_info_path}")
        print("Please run the export_object.py script first to generate the URDF info.")
        return
    
    # Load color information
    with open(urdf_info_path, 'r') as f:
        urdf_info = json.load(f)
    
    color_info = urdf_info['color']
    
    print(f"Creating placeholder texture for: {object_name}")
    print(f"Color: RGB({color_info['r']:.3f}, {color_info['g']:.3f}, {color_info['b']:.3f})")
    print(f"Alpha: {color_info['a']:.3f}")
    
    # Create placeholder texture
    texture_path = create_placeholder_texture(output_dir, object_name, color_info)
    
    if texture_path:
        # Update the URDF info file with texture path
        urdf_info['texture']['texture_file'] = f"{object_name}.png"
        urdf_info['texture']['texture_path'] = texture_path
        
        with open(urdf_info_path, 'w') as f:
            json.dump(urdf_info, f, indent=2)
        
        print(f"📝 Updated URDF info file with texture path")
        print(f"\n✅ Placeholder texture created successfully!")
        print(f"📁 Texture file: {texture_path}")
        print(f"\nYou can now use this texture in your URDF:")
        print(f'<texture filename="package://your_package/textures/{object_name}.png"/>')
    else:
        print("❌ Failed to create placeholder texture")


if __name__ == "__main__":
    main()

