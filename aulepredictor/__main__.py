from ast import parse
from aulepredictor import aule
import argparse
import sys

def parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument("target", type=str,
                    help="Molecule PDB file to be analysed.")
    p.add_argument("--model", type=str, default=None,
                    help="Name of the model to be used.")
    p.add_argument("--device", type=str, default='cpu',
                    help="Device in which calculations will be run.")
    p.add_argument("--device_id", type=int, default=0,
                    help="GPU ID to run the calculations.")
    p.add_argument("--output_dir", type=str, default='.',
                    help='Directory where the output file should be generated.')
    p.add_argument("--candidates", type=str, default=None,
                    help='Path to where a file with candidate coordinates is stored.')
    p.add_argument("--stride", type=int, default=1,
                    help="Step of the sliding window when evaluating the protein.")
    p.add_argument("--threshold", type=float, default=0.9,
                    help="Threshold for considering predictions positive.")
    p.add_argument("--voxelsize", type=float, default=1.0,
                    help="Resolution of the 3D representation. In Arnstrongs.")
    p.add_argument("--verbose", type=int, default=1, 
                    help="Information that will be displayed. 0: Only Moleculekit, 1: All.")
    
    return p.parse_args()

def main():
    message = "Using AulePredictor by Raúl Fernández-Díaz"
    print("-" * (len(message) + 4))
    print("| " + message + " |")
    print("-" * (len(message) + 4))
    print("\n")
    args = parse_cli()
    Aule = aule(**vars(args))
    Aule.predict(**vars(args))

if __name__ == '__main__':
    main()

    

