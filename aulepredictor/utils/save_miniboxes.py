"""
Functions for saving protein regions 3D representations.

Copyright by Raúl Fernández Díaz
"""

import os
import torch
import numpy as np

def save_miniboxes(output_dir : str, minibox_size : int) -> None:
    """
    Store 3D representations of the protein regions separeting them according
    to whether they are metal-binding or not.

    Parameters
    ----------

    output_dir : str
        Directory where the results are to be stored.

    minibox_size : int
        Dimensions of the regions to be generated in number of voxels.
    """

    half_size = minibox_size // 2
    with open(os.path.join(output_dir, 'metal_binding.csv')) as fi:
        fi.readline()
        counter = {}
        for line in fi:
            info = line.split(',')

            try:
                counter[info[1]] += 1
            except KeyError:
                counter[info[1]] = 0

            protein_vox = np.load(os.path.join(output_dir, 'whole', info[1] + '_vox.npy'), 
                                                allow_pickle=True)
            minibox = torch.from_numpy(protein_vox[:, :, 
                                                int(info[2])-half_size:int(info[2])+half_size, 
                                                int(info[3])-half_size:int(info[3])+half_size, 
                                                int(info[4])-half_size:int(info[4])+half_size])
            if int(info[5]) == 0:
                torch.save(minibox, os.path.join(output_dir, 'not_binding', info[1] + 
                                                '_' + str(counter[info[1]]) + '.pt'))
                print("Saving not binding: " + info[1] + ' ' + str(counter[info[1]]))
            else:
                torch.save(minibox, os.path.join(output_dir, 'metal_binding', info[1] 
                                                + '_' + str(counter[info[1]]) + '.pt'))
                print("Saving binding: " + info[1] + ' ' + str(counter[info[1]]))
            
