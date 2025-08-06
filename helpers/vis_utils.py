import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import torch
import cv2
from tqdm import tqdm

def gaussian_3d_pcd(mean, std, num_points):
    """
    Generate a point cloud sampled uniformly from inside a 3D ellipsoid
    (centered at mean, axes lengths given by std),
    and color the points using the plasma colormap based on Mahalanobis distance.
    """
    # Uniformly sample inside a unit sphere
    points = []
    while len(points) < num_points:
        p = np.random.uniform(-1, 1, 3)
        if np.linalg.norm(p) <= 1:
            points.append(p)
    points = np.array(points)
    # Scale by std and shift by mean
    points = points * std + mean

    # Color by Mahalanobis distance (for visualization)
    mahal = np.sqrt(np.sum(((points - mean) / std) ** 2, axis=1))
    mahal = mahal / mahal.max()  # Normalize to [0, 1]
    colors = plt.cm.plasma(1 - mahal)[:, :3]

    return points, (colors * 255).astype(np.uint8)

def transform_pcd(pcd: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Transforms the given point cloud by the given transformation matrix.

    Args:
    -----
        pcd (np.ndarray): Nx3 point cloud
        transform (np.ndarray): 4x4 transformation matrix

    Returns:
    --------
            pcd_new (np.ndarray): Nx3 transformed point cloud
    """

    if pcd.shape[1] != 4:
        pcd = np.concatenate((pcd, np.ones((pcd.shape[0], 1))), axis=1)
    pcd_new = np.matmul(transform, pcd.T)[:-1, :].T
    return pcd_new


def reshape_to_points(data):
    """
    Reshape data to have shape (-1, 3) by finding the dimension with length 3
    and moving it to the end, then flattening all other dimensions.
    Also creates a writable copy of the data.
    
    Args:
        data: numpy array with one dimension of length 3
        
    Returns:
        writable reshaped data with shape (-1, 3)
    """
    # Ensure data has 2-3 dimensions
    if len(data.shape) > 3 or len(data.shape) < 2:
        raise ValueError("Data must have 2 or 3 dimensions")

    # Find dimension with length 3 and move it to end
    three_dim = None
    for i, dim in enumerate(data.shape):
        if dim == 3:
            three_dim = i
            break
    
    if three_dim is None:
        raise ValueError("Data must have one dimension of length 3")
        
    # Move dimension with length 3 to end and reshape
    if three_dim != len(data.shape)-1:
        dims = list(range(len(data.shape)))
        dims.remove(three_dim)
        dims.append(three_dim)
        data = np.transpose(data, dims)
    
    data = data.reshape(-1, 3)
    
    # Create writable copy
    data_new = np.zeros(data.shape)
    data_new[:] = data[:]
    
    return data_new

def plot_pcd(pcd, colors=None, frame=False):

    if type(pcd) == torch.Tensor:
        pcd = pcd.cpu().detach().numpy()
    if colors is not None and type(colors) == torch.Tensor:
        colors = colors.cpu().detach().numpy()

    # Reshape point cloud to (-1, 3) and create writable copy
    pcd_new = reshape_to_points(pcd)

    pts_vis = o3d.geometry.PointCloud()
    pts_vis.points = o3d.utility.Vector3dVector(pcd_new)

    if colors is not None:
        # Apply the same reshaping to colors as we did to pcd
        colors_new = reshape_to_points(colors)
        
        # Ensure colors are in the right range [0, 1]
        if colors_new.max() > 1.0:
            colors_new = colors_new / 255.0
        
        pts_vis.colors = o3d.utility.Vector3dVector(colors_new)

    geometries = [pts_vis]

    if frame:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.2, origin=[0, 0, 0]
        )
        geometries.append(frame)

    o3d.visualization.draw_geometries(geometries)



def plot_voxel_grid_with_action(voxel_grid: torch.Tensor, 
                    action_voxels: torch.Tensor,
                    action_colors: torch.Tensor):
    """
    Plot the voxel grid with the action translation in the voxel grid
    Args:
        voxel_grid: (10, D, H, W)
        action_voxels: (N, 3)
        action_colors: (N, 3)
    """
    
    vis_grid = voxel_grid.permute(1, 2, 3, 0)

    # Remove the action voxels from the voxel grid:
    vis_grid[action_voxels[:, 0], 
             action_voxels[:, 1], 
             action_voxels[:, 2], 
             :3] = 0.

    # Mask out the points that are not in the voxel grid:
    mask = torch.norm(vis_grid[..., :3], dim = -1) > 0      # Could just use occupancy instead...
    vis_pts = vis_grid[torch.where(mask)][..., 6:9]
    vis_rgb = vis_grid[torch.where(mask)][..., 3:6]

    # Add the action voxels to the voxel grid point cloud
    action_voxel_center = vis_grid[action_voxels[:, 0], 
                                   action_voxels[:, 1], 
                                   action_voxels[:, 2], 
                                   6:9]
    
    vis_pts = torch.cat([vis_pts, action_voxel_center], dim=0)
    vis_rgb = torch.cat([vis_rgb, action_colors], dim=0)

    plot_pcd(vis_pts, vis_rgb)


def create_cube_without_points(cube_size=64):
    """
    Create a single transparent cube with only its edges visible (no points).
    
    Args:
        cube_size: Size of the cube
    
    Returns:
        Open3D LineSet geometry for the cube edges
    """
    # Create the cube edges (12 edges of a cube)
    cube_points = [
        [0, 0, 0],           # 0: bottom front left
        [cube_size, 0, 0],   # 1: bottom front right
        [cube_size, cube_size, 0], # 2: bottom back right
        [0, cube_size, 0],   # 3: bottom back left
        [0, 0, cube_size],   # 4: top front left
        [cube_size, 0, cube_size], # 5: top front right
        [cube_size, cube_size, cube_size], # 6: top back right
        [0, cube_size, cube_size]  # 7: top back left
    ]
    
    # Define the 12 edges of the cube
    cube_edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
    ]
    
    # Create line set for cube edges
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(cube_points)
    line_set.lines = o3d.utility.Vector2iVector(cube_edges)
    line_set.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5]] * len(cube_edges))  # Grey edges
    
    return line_set


def plot_voxel_grid_with_action_cubes(voxel_grid, action_voxels, action_colors, cube_size=1., action_cube_size=0.03):
    """
    Plot voxel grid with action voxels as larger cubes and enclose everything in a transparent cube.
    
    Args:
        voxel_grid: Voxel grid tensor/array of shape (features, X, Y, Z)
        action_voxels: Action voxel indices tensor/array of shape (num_actions, 3)
        action_colors: Colors for action voxels tensor/array of shape (num_actions, 3)
        cube_size: Size of the main voxel grid cube
        action_cube_size: Size multiplier for action cubes (relative to voxel size)
    """
    geometries = []
    
    # Create the main transparent cube that encloses everything
    main_cube = create_cube_without_points(cube_size)
    geometries.append(main_cube)
    
    # Convert inputs to numpy arrays if they are torch tensors
    if torch.is_tensor(voxel_grid):
        voxel_grid = voxel_grid.detach().cpu().numpy()
    if torch.is_tensor(action_voxels):
        action_voxels = action_voxels.detach().cpu().numpy()
    if torch.is_tensor(action_colors):
        action_colors = action_colors.detach().cpu().numpy()
        
    # Convert voxel grid to (X, Y, Z, features) format
    if voxel_grid.shape[0] < voxel_grid.shape[-1]:  # If in (features, X, Y, Z) format
        vis_grid = np.transpose(voxel_grid, (1, 2, 3, 0))
    else:
        vis_grid = voxel_grid
    
    # Remove the action voxels from the voxel grid for point cloud visualization
    for i in range(action_voxels.shape[0]):
        vis_grid[action_voxels[i, 0], 
                 action_voxels[i, 1], 
                 action_voxels[i, 2], 
                 :3] = 0.
    
    # Mask out the points that are not in the voxel grid
    mask = np.linalg.norm(vis_grid[..., :3], axis=-1) > 0
    vis_pts = vis_grid[np.where(mask)][..., 6:9]  # Voxel center coordinates
    vis_rgb = vis_grid[np.where(mask)][..., 3:6]  # RGB colors
    
    # Create point cloud from voxel grid
    if len(vis_pts) > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vis_pts)
        if vis_rgb.max() > 1.0:
            vis_rgb = vis_rgb / 255.
        pcd.colors = o3d.utility.Vector3dVector(vis_rgb)        # 0 TO 1 range
        geometries.append(pcd)
    
    # Create larger cubes for action voxels
    for i in range(action_voxels.shape[0]):
        # Get action voxel center coordinates
        action_voxel_center = vis_grid[action_voxels[i, 0], 
                                       action_voxels[i, 1], 
                                       action_voxels[i, 2], 
                                       6:9]  # Voxel center coordinates
        
        # Create cube for this action voxel
        cube = o3d.geometry.TriangleMesh.create_box(
            width=action_cube_size, 
            height=action_cube_size, 
            depth=action_cube_size
        )
        
        # Position the cube at the action voxel center
        cube.translate(action_voxel_center - action_cube_size/2)
        
        # Color the cube with action color
        action_color = action_colors[i]
        if action_color.max() > 1.0:  # Normalize color if needed
            action_color = action_color / 255.
        cube.paint_uniform_color(action_color)
        
        geometries.append(cube)
    
    # Create coordinate frame for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=cube_size * 0.1, origin=[0, 0, 0]
    )
    geometries.append(coordinate_frame)
    
    # Visualize
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Voxel Grid with Action Cubes",
        width=1200,
        height=800,
        point_show_normal=False,
        mesh_show_back_face=True
    )


def visualize_image(pcd_vis, point_size=0.4, zoom_out=0.1):
    """Render point cloud image with standard camera setup"""
    # Set up Open3D visualization and render
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=512, height=512)
    vis.add_geometry(pcd_vis)

    # Set point size
    render_option = vis.get_render_option()
    render_option.point_size = point_size

    ctr = vis.get_view_control()
    camera_params = ctr.convert_to_pinhole_camera_parameters()
    extrinsic = np.eye(4)
    extrinsic[2, 3] = zoom_out  # Zoom out to capture everything
    camera_params.extrinsic = extrinsic
    ctr.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)

    # Render and capture the view
    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer(do_render=True)

    # Convert image to numpy array
    image_np = (np.asarray(image) * 255).astype(np.uint8)

    # Clean up
    vis.destroy_window()
    return image_np