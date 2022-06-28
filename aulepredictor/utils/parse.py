"""
Functions for parsing MetalPDB TSV files.

Copyright by Raúl Fernández Díaz
"""

import os
import sys
import numpy as np

from moleculekit.molecule import Molecule
from moleculekit.tools.atomtyper import prepareProteinForAtomtyping
from moleculekit.tools.voxeldescriptors import getCenters, getVoxelDescriptors


def parse(metal_database : str, output_dir : str, metal : str, channels : list, 
          voxel_resolution : float=1.0, buffer : int=8, distance_to_metal : float=1.5) -> None:
    """
    Function to parse a database file from metalPDB. It reads all entries and proceeds
    to voxelize the proteins and to store the coordinates of the metals.

    Parameters
    ----------
    metal_database : str
        Path to where the database is located.

    output_dir : str
        Path where the outputs will be stored.

    metal : str
        Name of the metal of interest.

    channels : list of str
        Channels to be used for voxelization.
        Supported channels:
            ['hydrophobic', 'aromatic', 'hbond_acceptor', 'hbond_donor',
            'positive_ionizable', 'negative_ionizable', 'metal', 'occupancies']
    
    voxel_resolution : float
        Resolution of the 3D representation in A.
        By default: 1.0

    buffer : int
        Number of voxels to be introduced in the outside of the protein with
        value 0.
        By default: 8

    distance_to_metal : float
        Distance from the center of one voxel to a metal center
        so that it is considered to contain a metal in A.
        By default: 1.5

    Examples
    --------
    >>> channels = ['hydrophobic', 'hbond_acceptor', 'hbond_donor', 
                'positive_ionizable', 'negative_ionizable', 'occupancies']
    >>> parse('./databases/Li.tsv', './dataset/Li', channels)
    """

    # Make sure the output directory has all necessary subdirectories; if not, create them
    _prepare_output_dir(output_dir)
    # Check what entries have already been processed. 
    # To do so, read the file 'log.txt' where they will be stored
    with open(os.path.join(output_dir, "log.txt"), "rt") as fi:
        already_processed = []
        for line in fi: 
            already_processed.append(line.split("_")[0].strip("\n"))

    # For all proteins in database, if a protein was not already processed, extract its pdb
    # and voxelise it. If there is an error save the information in its corresponding Exceptions
    # file; if properly executed save it in the appropriate 'log.txt' to avoid reanalysing it.
    with open(f"{metal_database}") as fi1:
        fi1.readline()
        for line1 in fi1:
            with open(os.path.join(output_dir, "log.txt"), "rt") as fi2:
                for line2 in fi2: 
                    if line2.split("_")[0].strip("\n") not in already_processed:
                        already_processed.append(line2.split("_")[0].strip("\n"))

            site_id = line1.split("\t")[0].split("_")[0].strip("\n")

            if site_id not in already_processed:
                pdb_code = site_id.split("_")[0]
                try:
                    voxelize(pdb_code, output_dir, metal, channels, voxel_resolution, 
                            buffer, distance_to_metal)
                except KeyboardInterrupt:
                    sys.exit()
                except:
                    with open(os.path.join(output_dir, "Exception.txt"), "a") as fo:
                        fo.write(f"{pdb_code}\n")
                with open(os.path.join(output_dir, "log.txt"), "a") as fo:
                    fo.write(f"{pdb_code}\n")


def voxelize(protein_name : str, output_dir : str, metal : str, channels : list,
             voxel_resolution : float, buffer : int, distance_to_metal : float) -> None:
    """
    Function to perform voxelization. It takes the PDB ID of a protein and creates a voxelixed
    image with 3 dimensions and up to 6 channels. It also saves a list with the voxel center coordinates
    that are within 3 A (euclidean distance) from a metallic center.

    Parameters
    ----------
    protein_name : str
        PDB ID of the protein we wish to voxelize.

    output_dir : str
        Name of the directory where the information is to be stored. 

    metal : str
                Name of the metal of interest.

    channels : list of str
                Supported channels:
                    ['hydrophobic', 'aromatic', 'hbond_acceptor', 'hbond_donor',
                    'positive_ionizable', 'negative_ionizable', 'metal', 'excluded_volume']
    
    voxel_resolution : float
        Resolution of the 3D representation in A.

    buffer : int
        Number of voxels to be introduced in the outside of the protein with
        value 0.

    distance_to_metal : float
        Distance from the center of one voxel to a metal center
        so that it is considered to contain a metal in A.

    Examples
    --------
    >>> protein_name = '3jys'
    >>> output_dir = MG
    >>> channels = ['hydrophobic', 'hbond_acceptor', 'hbond_donor', 
                    'positive_ionizable', 'negative_ionizable', 'excluded_volume']

    >>> voxelize(protein_name, output_dir, 'Mg', channels)
    """


    protein = Molecule(protein_name) # Create a Molecule (moleculekit package) with the information of the protein

    metal_coordinates = []

    for index in protein.get("index"): # Go through every entry in the PDB
        if protein.get("name", sel=index) == metal.upper(): # If an entry of the PDB contains the metal of interest
            metal_coordinates.append(protein.get("coords", sel=index)) # Save its coordinates


    protein.remove("not protein") # Remove all non-protein atoms within our Molecule
      
    protein = prepareProteinForAtomtyping(protein, verbose=0) # Prepare the Molecule for voxelization
                                                    # Introduce H, check bonds, etc.

    centers = getCenters(protein) # Get a list of the coordinates of all atoms
    uchannels = np.ones((len(centers[0]),8)) # Create an array with dimensions (mol.num_atoms, num_channels)
    
    # Set to 0 the channels we are not interested in calculating
    undesired_channels = _get_undesired_channels(channels)
    for channel in undesired_channels:
        uchannels[:,channel] = 0 

    # Perform the voxelization
    protein_vox, protein_centers, protein_N = getVoxelDescriptors(protein,voxelsize=voxel_resolution, 
                                                    buffer=buffer, 
                                                    validitychecks=False,
                                                    userchannels = uchannels
                                                    )

    # Eliminate the channels that we are not interested in
    new_protein_vox = np.zeros((len(protein_centers), 8-len(undesired_channels)))
    j = -1
    for i in range(8):
        if i not in undesired_channels: # No aromatic, no metal
            j += 1
            new_protein_vox[:,j] = protein_vox[:,j]
    
    # From the 2D output create the proper 3D output
    nchannels = new_protein_vox.shape[1]
    protein_vox_t = new_protein_vox.transpose().reshape([1, nchannels, protein_N[0], protein_N[1], protein_N[2]])

    # Store the dimensions of the voxelized image
    nvoxels = [protein_N[0], protein_N[1], protein_N[2]]
    k = 0 # Initialise a counter of the number of metal-containing voxels
    for metal_coordinate in metal_coordinates:

        # Use find_metal function to find the metal-containing voxels
        metal_locations = _find_metal(protein_centers, nvoxels, metal_coordinate[0], distance_to_metal)

        # Save a list with the metal-containing voxels
        np.savetxt(os.path.join(output_dir, 'metals', f'{protein_name}_{k}_mp.txt'), metal_locations)

        k += 1 # This counter will serve as a discriminating identifier for different
                # lists for proteins bound to more than 1 metal center

    # Save the voxelised image of the protein
    np.save(os.path.join(output_dir, 'whole', f'{protein_name}_vox.npy'), protein_vox_t)

def _get_undesired_channels(channels):
    """
    Helper function to voxelize() that reads the desired channels for voxelization and determines what
    channels should be turned off. It outputs a list of ints with the indexes of the undesired channels.
    """

    possible_channels = {'hydrophobic': 0, 'aromatic': 1, 'hbond_acceptor': 2, 
                        'hbond_donor': 3, 'positive_ionizable': 4, 'negative_ionizable': 5, 
                        'metal': 6, 'excluded_volume': 7}
    working_channels = []
    undesired_channels = []

    for channel in channels:
        if channel not in possible_channels.keys():
            raise Exception(f"Unsupported channel: {channel}.\n Supported channels: {possible_channels.keys()}.")
        working_channels.append(possible_channels[channel])

    for i in range(8):
        if i not in working_channels:
            undesired_channels.append(i)
    return undesired_channels

def _find_metal(centers : np.array, nvoxels : int, coords : np.array, 
                distance_to_metal: float) -> np.array:
    """
    Function to obtain the voxel coordinates of the metals within the voxelized image.

    distance_to_metal : float
        Distance from the center of one voxel to a metal center
        so that it is considered to contain a metal in A.
    """

    i, j, k = 0, 0, 0 # Start from the initial coordinates upper-left-front corner of the 3D-image
    results = [] # Initialize a list with voxel coordinates close to the real coordinates of the metal
    
    for center in centers: # Go through every voxel analysing whether their coordinates are sufficiently
                            # close to our metal
        if k == nvoxels[2]:
            k = 0
            j += 1
            if j == nvoxels[1]:
                j = 0
                i += 1
        if _euclidean_distance(center, coords) < distance_to_metal: # Maximum distance to consider a voxel close enough to metal
                                                    # semi-arbitrary value, based on average van der waals radii  
                                                    # of metalic atoms and the average resolution of protein structures
            results.append([i,j,k])
        k += 1
    results = np.array(results)   # Turn the python list into numpy array to improve performance in subsequent steps
    return results

def _euclidean_distance(A, B):
    """
    Function to calculate the euclidean distance between 2 different points in 
    cartesian space.
    """
    return np.sqrt((A[0]-B[0])**2+(A[1]-B[1])**2+(A[2]-B[2])**2)

def _prepare_output_dir(output_dir):
    """
    Helper function to parse() function that checks that all necessary subdirectories
    have been created and when not it creates them. It also creates log and Exception files.
    """

    tree_directory = {"metals": False, "whole": False, 
                    "metal_binding": False, "not_binding": False}
    
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for entry in os.listdir(output_dir):
        candidate_subdir = os.path.join(output_dir, entry)
        if os.path.isdir(candidate_subdir):
            for key in tree_directory.keys():
                if entry == key:
                    tree_directory[key] = True
    
    for key, value in tree_directory.items():
        if not value:
            os.mkdir(os.path.join(output_dir, key))

    if not os.path.exists(os.path.join(output_dir, 'log.txt')):
        with open(os.path.join(output_dir, 'log.txt'), 'w') as fo:
            fo.write("")

    if not os.path.exists(os.path.join(output_dir, 'Exception.txt')):
        with open(os.path.join(output_dir, 'Exception.txt'), 'w') as fo:
            fo.write("")
