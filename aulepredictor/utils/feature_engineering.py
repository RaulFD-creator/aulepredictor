import os
import random
import numpy as np
import pandas as pd

def feature_engineering(output_dir : str, minibox_size : int, distance : float) -> None: 
    """ 
    Traverse all protein voxelised images, load their metal coordinates and execute the 
    create_database function.

    Parameters
    ----------
    output_dir : string
        Directory where the data set is located.

    minibox_size : int
        Dimensions of the regions to be generated in number of voxels.

    distance : float
        Distance from the center of a minibox to a metal center to
        consider the minibox as metal-binding in A.
    """
    if not os.path.exists(os.path.join(output_dir, 'metal_binding.csv')):
        with open(os.path.join(output_dir, 'metal_binding.csv'), 'w') as fo:
            fo.write("protein_ID,x,y,z,binding\n")
    
    already_processed = []
    if not os.path.exists(os.path.join(output_dir, 'minibox_log.txt')):
        with open(os.path.join(output_dir, 'minibox_log.txt'), 'w') as fo:
            fo.write("")

    with open(os.path.join(output_dir, "minibox_log.txt")) as fi:
        for line in fi:
            already_processed.append(line.strip('\n'))
    print("Beginning analysis")            

    for file in os.listdir(os.path.join(output_dir, 'whole')):           
        if file not in already_processed:
            print(f"Analizing: {file}")
            protein_name = file.split("_")[0]
            protein_vox = np.load(os.path.join(output_dir, 'whole', file), allow_pickle=True)
            k = 0
            for file2 in os.listdir(f"{output_dir}/metals"):
                if file2.split("_")[0] == protein_name:

                    metal_coordinates = np.loadtxt(os.path.join(output_dir, 'metals', file2))
                    if metal_coordinates.size == 0:
                        print("Avoiding incomplete reccord")
                        continue

                    else:
                        outfilename = os.path.join(output_dir, 'metal_binding.csv')
                        
                        with open(outfilename, "a") as fo:
                            _register_regions(protein_vox, metal_coordinates, protein_name, fo,
                                       k < 1, minibox_size, distance) # If first time a protein
                                                               # is processed flag is True
                                                               # subsequent iterations
                                                               # will be False.
                        k += 1

        with open(os.path.join(output_dir, 'minibox_log.txt'), "a") as fo:
            fo.write(f"{file}\n")
        already_processed.append(file)

    _filtering_data(output_dir)

def _register_regions(protein_vox, metal_coordinates, protein_name, fo,
                        not_binding=False, minibox_size=16, distance=3):
    
    """
    Function to create subsets from the greater voxelized image. The function
    divides the greater voxelised image into smaller subsets of dimensions (minibox_size,
    minibox_size, minibox_size) and then it stores them in their corresponding directory
    (metal_binding or not_binding). To avoid introducing bias, only 15 % of not_binding miniboxes
    will be stored, to create a more balanced dataset and to not clutter the memory.

    This miniboxes will be stored as torch tensors, to facilitate access during
    MC-DCNN training.

    not_binding flag allows for not saving not_binding boxes when there are more than 1 
    metal in a protein. Otherwise, there would be redundant entries in our dataset.

    Parameters
    ----------
    protein_vox : np.array
        Contains the voxelized representation of the protein.

    metal_coordinates : np.array
        Contains the coordinates of the voxels that are close to a metallic center.

    protein_name : str
        PDB ID of the protein of interest.

    output_dir : str
        Directory where the results are to be stored.

    not_binding : bool
        Controls whether the not_binding miniboxes should be stored. By default, False.

    minibox_size : int
        Dimensions of the regions to be generated in number of voxels.
    """

    size_x = protein_vox.shape[2]
    size_y = protein_vox.shape[3]
    size_z = protein_vox.shape[4]
    half_size = minibox_size // 2
    for i in range(half_size, size_x-half_size, minibox_size//4):
        for j in range(half_size, size_y-half_size, minibox_size//4):
            for k in range(half_size, size_z-half_size, minibox_size//4):
                metal_binding = False
                for metal_coord in metal_coordinates:
                    if (metal_coord[0] in np.arange(i-distance+1, i+distance, 0.1) and 
                        metal_coord[1] in np.arange(j-distance+1, j+distance, 0.1) and
                        metal_coord[2] in np.arange(k-distance+1, k+distance, 0.1)):
    
                        fo.write(f"{protein_name},{i},{j},{k},1\n")
                        metal_binding = True
                        break
                if not metal_binding and not_binding and random.random() > 0.925:
                    fo.write(f"{protein_name},{i},{j},{k},0\n")


def _filtering_data(output_dir : str):
    """
    Function that goes through all possible metal binding sites and filters them to get a more balanced
    dataset. It generates a '.csv' file where it stores the coordinates of the centers of 
    the resulting putative binding sites.

    Parameters
    ----------
    output_dir : string
                Directory where the information is located
    
    Returns
    -------
    None
    """
    df = pd.read_csv(os.path.join(output_dir, 'metal_binding.csv'))

    df = df.sample(frac=1)
    k = 0
    for protein in df['protein_ID'].unique():
        k += 1
        print(f"\n\nProteins analysed: {k}")
        protein_counts = len(df[df['protein_ID']==protein])
        a = df[df['protein_ID']==protein]
        if protein_counts > 20:
            difference = protein_counts - 20
            counter = 0
            to_remove = []
            print(f"To be removed: {difference}")
            for idx in a.index:
                if counter < difference:
                    to_remove.append(idx)
                    counter += 1
                else:
                    break
            df.drop(to_remove, axis=0, inplace=True)

        else:
            continue

    print("Stopped first stage")
    print(df)
    binding_counts = df['binding'].value_counts()
    print(binding_counts)
    difference = binding_counts[0] - binding_counts[1] * 0.4
    total = binding_counts[0] + binding_counts[1]
    correction = difference / total
    print(correction)
    print(df['binding'].value_counts())
    print(df['protein_ID'].value_counts())

    k = 0
    for idx, entry in df[df["binding"] == 0].iterrows():
        if random.random() < correction and not df[df['protein_ID']==entry['protein_ID']]['binding'].value_counts().sum() < 5:
            df.drop(idx, axis=0, inplace=True)
        k += 1
        if k % 10000 == 0:
            print(k)

    print(df['binding'].value_counts())
    print(df['protein_ID'].value_counts())

    df.to_csv(os.path.join(output_dir, 'metal_binding.csv'))

