from helpers.vis_utils import plot_voxel_grid_with_action_cubes
import numpy as np
import os
import torch
import wandb
from tqdm import tqdm

from pytorch3d.structures import Pointclouds

from helpers.vis_utils import render_360_gif, voxel_points_and_features_from_voxel_grid

wandb.init(project="PerAct_Planning_All_Objects", name=f"run_0")

weights_dir = '/data/ModelBasedPlanning/PerAct/peract_train_log/multi/PERACT_BC/seed0/weights'
all_epochs = os.listdir(weights_dir)
# COnvert list of strs to list of ints:
all_epochs = [int(epoch) for epoch in all_epochs]

# Sort the list of epochs:
all_epochs.sort()


for epoch_num in tqdm(all_epochs):
    data = np.load(os.path.join(weights_dir, f'{epoch_num}',f'voxel_grid_{epoch_num}.npz'), 
                allow_pickle=True)

    voxel_grid = data['voxel_grid']
    action_voxels = data['action_voxels']
    action_colors = data['action_colors']
    loss = data['total_losses']

    points, features = voxel_points_and_features_from_voxel_grid(voxel_grid[0], action_voxels, action_colors)

    device = torch.device('cuda')

    # convert to tensor if it is not already
    points = torch.tensor(points, dtype=torch.float32, device=device)
    features = torch.tensor(features, dtype=torch.float32, device=device)

    point_cloud = Pointclouds(
        points=[points.to(device)],
        features=[features]
    ).to(device)

    # Create media folder if it doesn't exist
    os.makedirs('media', exist_ok=True)

    # Example 1: Default 360-degree rotation (original behavior)
    gif_path = os.path.join('media', f'{epoch_num}_default.gif')
    render_360_gif(point_cloud, 
                gif_path,
                image_size=1024, 
                light_location=[0, 0, -3], 
                fps=60, 
                dist=1., 
                elev=40,
                azimuth_range=(0, 360, 2),
                point_radius=0.005,
                fov=90)

    wandb.log({
        "visualization": wandb.Image(gif_path),
        "loss": loss
    })

wandb.finish()