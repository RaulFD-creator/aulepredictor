"""
Data Mining Pipeline used to extract the information for training DeepBioMetAll.

Copyright by Raúl Fernández Díaz
"""

import os
import argparse
import random

from warnings import WarningMessage

def parse_cli():
    """
    Parse arguments.
    """
    p = argparse.ArgumentParser()
    p.add_argument('database_path', type=str,
                    help='Path to the database with metal-binding sites.')
    p.add_argument('output_path', type=str,
                    help='Path where the data sets are to be stored.')
    p.add_argument('metal', type=str, 
                    help='Name of the metal from which information is to be extracted.')
    p.add_argument('name_folds', type=str,
                    help='Name to be given to the training/validation/testing folds to be generated.')
    p.add_argument('--channels', type=str, default="hydrophobic,hbond_acceptor,hbond_donor,positive_ionizable,negative_ionizable,excluded_volume",
                    help='Channels that will be computed during voxelization.')
    p.add_argument('--num_threads', type=int, default=8,
                    help='Number of threads used for parallelization.')
    p.add_argument('--num_folds', type=int, default=5,
                    help='Number of folds for cross-validation to be generated.')
    p.add_argument('--random_seed', type=int, default=None,
                    help='Random seed to be used, by default no random seed.')
    p.add_argument('--voxel_resolution', type=float, default=1.0,
                    help='Resolution of the 3D representation in A.')
    p.add_argument('--buffer', type=int, default=8,
                    help='Number of voxels to be introduced in the outside of the protein with value 0.')
    p.add_argument('--distance_to_metal', type=float, default=3,
                    help='Distance from the center of one voxel to a metal center so that it is considered to contain a metal in A.')
    p.add_argument('--minibox_size', type=int, default=16,
                    help='Number of voxels each regions will contain.')

    return p.parse_args()

def prepare_environment(args : argparse.Namespace):
    """
    Prepare environment for beginnning the data mining process.
    """
    
    message = ' Using DeepBioMetAll Data Mining Pipeline '
    print("-" * (len(message)+2))
    print("|" + message + "|")
    print("-" * (len(message)+2))

    args = vars(args)
    os.environ['NUMEXPR_MAX_THREADS'] = str(args['num_threads'])
    channels = args['channels'].split(",")

    if args['random_seed'] is not None:
        random.seed(args['random_seed']) 
    
    if args['minibox_size'] // 2 < args['buffer']:
        raise WarningMessage('Buffer is less than half the minibox size.\nInformation in the outer regions of the protein might be lost')

    with open(os.path.join(args['output_path'], 'data_mining.config'), 'w') as fo:
        fo.write('DeepBioMetall Data Mining: \n')
        for key, value in args.items():
            fo.write(f'{key}: {value}\n')
    
    return args, channels

args, channels = prepare_environment(parse_cli())

from utils.parse import parse
from utils.feature_engineering import feature_engineering
from utils.save_miniboxes import save_miniboxes
from utils.register_data import register_data

parse(args['database_path'], args['output_path'], args['metal'], 
        channels, voxel_resolution=args['voxel_resolution'], 
        buffer=args['buffer'], distance_to_metal=args['distance_to_metal'])

feature_engineering(args['output_path'], args['minibox_size'], args['distance_to_metal'])
save_miniboxes(args['output_path'], args['minibox_size'])
register_data(args['output_path'], args['name_folds'], args['metal'], args['num_folds'])