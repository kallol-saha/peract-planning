from helpers.vis_utils import plot_voxel_grid_with_action_cubes
import numpy as np
import os

epoch_num = 34000
weights_dir = '/data/kallol/PerAct/peract_train_log/multi/PERACT_BC/seed0/weights'
all_epochs = os.listdir(weights_dir)

for epoch in all_epochs:
    data = np.load(os.path.join(weights_dir, f'{epoch}',f'voxel_grid_{epoch}.npz'), 
                allow_pickle=True)

    voxel_grid = data['voxel_grid']
    action_voxels = data['action_voxels']
    action_colors = data['action_colors']

    print("Visualizing epoch: ", epoch)
    plot_voxel_grid_with_action_cubes(voxel_grid[0], action_voxels, action_colors)






