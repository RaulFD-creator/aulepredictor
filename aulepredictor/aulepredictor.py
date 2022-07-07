"""
AulePredictor, computational tool for the prediction of
metal-binding regions in proteins by leveraging Deep Convolutional
Neural Networks (DCNNs).

Copyright 2022 by Raúl Fernández Díaz
"""

from moleculekit.molecule import Molecule
from moleculekit.tools.voxeldescriptors import getVoxelDescriptors, getCenters
from moleculekit.tools.atomtyper import prepareProteinForAtomtyping
import torch
import numpy as np
import os
import time

class aule():
    """
    Aule predictor class. It is defined by a DCNN or MC-DCNN model 
    that will predict the metal-bindingness of the desired protein regions.

    Attributes
    ----------
    model : torch.nn.Module
        DCNN or MC-DCNN model for the prediction of 
        metal-binding regions.
    device : str
        Device where the computations will be performed.
        Options:
            'cpu': Computations performed in CPU.
            'cuda': Computations performed in GPU.

    Methods
    -------
    predict(target, output_dir, output_option, candidates=None, 
        stride=1, threshold=0.75, verbose=1, 
        voxelsize=1, buffer=1, validitychecks=False, 
        minibox_size=16) : Perform voxelization of the target protein
                            and evaluate the regions to determine which are most
                            likely to be metal-binding sites.
    voxelize(protein_name, voxelsize=1, buffer=1, validitychecks=False,
             channels=['hydrophobic', 'hbond_acceptor', 'hbond_donor', 
                        'positive_ionizable', 'negative_ionizable', 
                        'occupancies']) : Perform the voxelization of the target
                                          protein
    evaluate(protein_vox, candidates, minibox_size=16, 
            stride=1, threshold=0.75) : Evaluate the regions of the voxelized
                                        protein focusing on certain candidate regions
                                        that have been predefined.
    """

    def __init__(self, model : torch.nn.Module, device : str='cpu', device_id : int=0):
        """
        Creates an instance of the aule predictor class. 
        It loads a trained DCNN or MC-DCNN model and prepares
        the Python session to take optimal advantage of GPUs
        if required.

        Parameters
        ----------
        model : str
            Path to where the trained model is stored
        device : str, optional
            Device where the computations will be performed.
            Options:
                'cpu': Computations performed in CPU.
                'cuda': Computations performed in GPU.
            By default: 'cpu'
        device_id : int, optional
            If there are more than 1 GPU devices, which
            one should be used.
            By default: 0

        Returns
        -------
        aule : aule class object
            Aule predictor class instance
        """
        if device == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.fastest = True
            torch.cuda.set_device(device_id)

        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def __str__(self):
        """
        String representation of the aule predictor class.

        Returns
        -------
        aule_str : str
            String representation of the aule predictor class
        """
        return f"Aule predictor class powered by model: {self.model}"

    def predict(self, target : str,  output_dir : str='.', output_option : str='pdb',
            candidates : str or np.array or list=None, stride: int=1, threshold_1 : float=0.75, threshold_2 : float=0.6, 
            verbose : int=1, voxelsize : float=1, validitychecks : bool=False, 
            minibox_size : int=16, occupancy_restrictions : float=0.4) -> np.array:
        """
        Take a target protein either from its PDB ID or through a path to a 
        locally stored PDB file and create from it a voxelized representation. 
        If candidate points have been provided, evaluate only those points; 
        else, traverse all points in the  voxelized representation and evaluate 
        them. Finally, generate the appropriate output files which can be a 
        Biometall-style PDB file,  a text file with the cartesian coordinates of 
        the regions evaluated and their metal-bindingness score, both or neither.
        In any case, it ouputs a 2D np.array where the first dimension contains 
        the cartesian coordinates  of the centers of all voxels in the voxelized 
        representation and the second dimentsion contains their predicted score.
        The function can work with next to none parametrization, just with its
        defaults but can also be customized with a great variety of different options.
        
        Parameters
        ---------
        target : str
            Target protein PDB ID or path to a PDB file locally stored.
        output_dir : str, optional
            Path to the directory where output files are to be stored.
            By default: '.'
        output_option : str, optional
            Output files that will be created.
            Options: 
                'PDB': A Biometall-style PDB file will be generated
                       with Ar atoms indicating regions with a 
                       metal-bidingness greater than 0.5 and lower
                       than threshold and Kr atoms indicating regions
                       with scores greater than threshold.
                'txt': A text file with the cartesian coordinates of the regions
                       and the metal-bindingness score they obtained.
                'all': Generates both files.
                'none': Does not create neither file. Output will remain
                        only in the Python session.
            By default: 'all'
        candidates : str or np.array or list, optional
            List of cartesian coordinates of possible metal-binding centers
            or path to the Biometall output with the predicted regions
            to be further evaluated.
            By default: None
        stride : int, optional
            Step at which the algorithm moves through the voxelized
            representation. Should not be less than voxelsize. If it 
            were greater than voxelsize, resolution would be reduced 
            though computation time would also decrease.
            By default: 1 
        threshold : float, optional
            Metal-bindingness score value that a region has to achieve
            to be considered metal-binding. 
            Please note that this threshold does not represent a statistical
            metric. Thus, a threshold of 0.75 does not mean that there is a 75 % 
            chance that it is truly metal-binding. Deep learning scores do not allow 
            for such statistical interpretations. To know the certainty of the 
            predictions it is necessary to study the benchmarking results of 
            the individual model to be used.
            By default: 0.75
        verbose : int, optional
            Flag that indicates how much output the program should provide
            to inform the user about the current tasks being computed.
            Options:
                0: There will be no information about current tasks,
                   only relevant warnings that may arise from use 
                   of third party software (Moleculekit package)
                1: The program will inform about which task is currently 
                   being computed.
            By default: 1              
        voxelsize : float, optional
            Resolution of the voxelized images to be generated from the 
            target protein. Units in Armstrongs. Lower values will increase
            precision but will multiply computational costs, conversely greater
            values decrease precision but reduce computational cost.
            By default: 1
        buffer : int, optional
            Number of voxels introduced as padding surrounding the voxelized molecule.
            By default: 1
        validitychecks : bool, optional
            Flag that indicates whether a series of validity checks should be performed
            during voxelization to ensure that there are no problems. Activating it
            increases computational cost and may generate errors that have to be manually
            handled.
            By default: False
        minibox_size : int, optional         print(target)

            Size of the region to be evaluated. Units in Armstrongs.
            By default: 16

        Raises
        ------
        FileNotFoundError : candidates is a string but no file cannot be found where
                            its path is indicating. Error raised by function evaluate().
        TypeError : If target is not a string, raise the error and inform of the type
                    parameter target should contain.
        TypeError : candidates parameter does not contain neither a list, 
                    an np.array nor a string. Error raised by function evaluate().
        ValueError : If an output_option has been indicated that is not supported, raise
                     the error and inform of the supported options.
        ValueError : If any of the channels provided is not supported, 
                     raise the error and inform of the supported channels. 
                     Error raised by function voxelize().

        Returns
        -------
        protein_scores : np.array
            2D np.array where the first dimension contains the cartesian 
            coordinates of all the voxels in the voxelized representation
            of the target protein and the second dimension the predicted
            metal-bindingness score.

        Examples
        --------
        >>> protein_name = '3jys'
        >>> aule.predict(protein_name)
        """
        start = time.time()
        # Check that the input has the appropriate format
        if not isinstance(target, str):
            raise TypeError('Target has to be:\n  a) string with PDB ID or\n  b) path to local PDB file.')

        # Print current status if appropriate
        if verbose == 1:
            print(f"Voxelizing target: {target}")
        
        # Perform the voxelization
        centers, protein_vox, protein_centers, nvoxels = self.voxelize(target, voxelsize=voxelsize, 
                                                                        buffer=minibox_size//2,
                                                                        validitychecks=validitychecks)
        # Print current status if appropriate
        if verbose == 1:
            print(f"Evaluating target: {target}")

        # Evaluate the indicated regions to determine whether they are or not metal-binding
        protein_scores = self.evaluate(protein_vox, centers, protein_centers=protein_centers, candidates=candidates, 
                                        voxelsize=voxelsize, minibox_size=minibox_size, stride=stride, 
                                        occupancy_restrictions=occupancy_restrictions)
        # Save results with the appropriate files
        if output_option.lower() == 'pdb': self.create_PDB(output_dir, target, protein_scores, threshold_1, threshold_2)
        elif output_option.lower() == 'txt': self.create_txt(output_dir, protein_scores)
        elif str(output_option).lower() == 'none': return protein_scores
        elif output_option.lower() == 'all':
            self.create_PDB(output_dir, target, protein_scores, threshold_1, threshold_2)
            self.create_txt(output_dir, protein_scores)
        else:
            raise ValueError(f"Output option {output_option} not supported.\nPlease use one of the supported options: 'PDB', 'txt', 'all', or 'none'")
        
        end = time.time()
        print(f"Computation took {end-start} s.")
        # Return protein_scores 
        return protein_scores
      
    def voxelize(self, protein_name : str, voxelsize : int or float=1, buffer : int=1, validitychecks : bool=False, 
                channels : list=['hydrophobic', 'hbond_acceptor', 'hbond_donor', 
                          'positive_ionizable', 'negative_ionizable', 'occupancies']):
        """
        Function to perform voxelization. It takes a molecule object (moleculekit) and 
        creates a voxelixed image with 3 dimensions and up to 8 channels. 
        
        Parameters
        ----------
        protein_name : str
            PDB ID of the target protein.
        voxelsize : float, optional
            Resolution of the voxelized images to be generated from the 
            target protein. Units in Armstrongs. Lower values will increase
            precision but will multiply computational costs, conversely greater
            values decrease precision but reduce computational cost.
            By default: 1        
        buffer : int, optional
            Number of voxels introduced as padding surrounding the voxelized molecule.
            By default: 1
        validitychecks : bool, optional
            Flag that indicates whether a series of validity checks should be performed
            during voxelization to ensure that there are no problems. Activating it
            increases computational cost and may generate errors that have to be manually
            handled.
            By default: False
        channels : list, optional
            List of channels to be computed for voxelization.
            Supported channels:
                ['hydrophobic', 'aromatic', 'hbond_acceptor', 'hbond_donor',
                'positive_ionizable', 'negative_ionizable', 'metal', 'occupancies'].
            By default:
                ['hydrophobic', 'hbond_acceptor', 'hbond_donor', 'positive_ionizable', 
                'negative_ionizable', 'occupancies'].
        Raises
        ------
        ValueError : If any of the channels provided is not supported, 
                    raise the error and inform of the supported channels. 
                    Error raised by helper function _get_undesired_channels().

        Outputs
        -------
        protein_vox : np.array
            Numpy array with the voxelized representation of the protein.
        protein_centers : np.array
            Numpy array with the coordinates of the centers of the voxels.
        nvoxels : np.array 
            Number of voxels in each dimension of the voxelized image [x, y, z].

        Examples
        --------
        >>> protein_name = '3jys'
        >>> metal_dir = MG
        >>> channels = ['hydrophobic', 'hbond_acceptor', 'hbond_donor', 
                        'positive_ionizable', 'negative_ionizable', 'occupancies']
        >>> protein_vox, protein_centers, nvoxels = aule.voxelize(protein_name, channels)
        """
        # Create a Molecule object (moleculekit package) with the information of the protein
        protein = Molecule(protein_name) 

        # Remove all non-protein atoms within the Molecule
        protein.remove("not protein") 

        # Prepare the Molecule for voxelization: introduce H, check bonds, etc.
        protein = prepareProteinForAtomtyping(protein, verbose=0) 

        # Get a list of the coordinates of all atoms
        centers = getCenters(protein) 

        # Create an array with dimensions (mol.num_atoms, num_channels)
        uchannels = np.ones((len(centers[0]),8))
        
        # Set to 0 the channels we are not interested in calculating
        undesired_channels = self._get_undesired_channels(channels)
        for channel in undesired_channels:
            uchannels[:,channel] = 0 

        # Perform the voxelization
        protein_vox, protein_centers, protein_N = getVoxelDescriptors(protein,
                                                        voxelsize=voxelsize, 
                                                        buffer=buffer, 
                                                        validitychecks=validitychecks,
                                                        userchannels=uchannels
                                                        )

        # Eliminate the channels that we are not interested in
        new_protein_vox = np.zeros((len(protein_centers), 8-len(undesired_channels)))
        j = -1
        for i in range(8):
            if i not in undesired_channels:
                j += 1
                new_protein_vox[:,j] = protein_vox[:,j]
        
        # From the 2D output create the proper 3D output
        nchannels = new_protein_vox.shape[1]
        protein_vox = new_protein_vox.transpose().reshape([1, nchannels, protein_N[0], protein_N[1], protein_N[2]])
        nvoxels = np.array([protein_N[0], protein_N[1], protein_N[2]])

        return centers, protein_vox, protein_centers, nvoxels

    def evaluate(self, protein_vox : np.array, centers : np.array, protein_centers : np.array, candidates : str or np.array or list,
                voxelsize : float=1, minibox_size : int=16, stride : int=1, occupancy_restrictions : float=0.4) -> np.array:
        """
        Traverse a voxelized representation of the target protein and evaluate 
        the metal-bindingness of all its regions. If candidates have been provided
        either in the form of a PDB biometall output file or in the form of 
        a np.array or list with predefined cartesian coordinates, only evaluate
        these candidate points.watch -n0.1 nvidia-smi

        Parameters
        ----------
        protein_vox : np.array
            Voxelized representation of the target protein.
        protein_centers : np.array
            List with the cartesian coordinates of the voxel centers.
        candidates : list, np.array or str, optional
            List of cartesian coordinates of possible metal-binding centers
            or path to the biometall output with the predicted regions
            to be further evaluated.
        voxelsize : float, optional
            Resolution of the voxelized image. Units in Armstrongs. 
            By default: 1
        minibox_size : int, optional 
            Size of the region to be evaluated. Units in Armstrongs.
            By default: 16
        stride : int, optional
            Step at which the algorithm moves through the voxelized
            representation. Should not be less than voxelsize, if it 
            is greater than voxelsize resolution will be lower but
            computation time will decrease greatly.
            By default: 1 

        Raises
        -------
        TypeError: candidates parameter is not None and does not contain neither a list, 
                    an np.array nor a string.
        FileNotFoundError : candidates is a string but no file cannot be found where
                    its path is indicating. This error is raised by helper function
                    _get_candidate_voxels().

        Returns
        -------
        evaluated_vox : np.array
            2D np.array where the first dimension contains the cartesian 
            coordinates of all the voxels in the voxelized representation
            of the target protein and the second dimension the predicted
            metal-bindingness score.
        """    
        # Obtain information from the voxelized representation
        size_x = protein_vox.shape[2]
        size_y = protein_vox.shape[3]
        size_z = protein_vox.shape[4]
        # Determine the step from one region to the next
        half_size = minibox_size // 2
        # Create an empty representation of the protein to be filled
        # with the metal-bindingness prediction
        evaluated_vox = np.zeros((size_x, size_y, size_z))
        if self.device == 'cuda':
            evaluated_vox = torch.Tensor(evaluated_vox).to(self.device)

        # Preprocess candidate points to locate their corresponding coordinates in the voxelized
        # representation.
        if candidates is not None:
            if isinstance(candidates, list) or isinstance(candidates, str):
                candidate_voxels = self._get_candidate_voxels(candidates, protein_centers, np.array([size_x, size_z, size_z]), voxelsize)
            else:
                raise TypeError('Candidates should be either:\n  a) string with the path to a Biometall PDB output or\n  b) list or np.array with cartesian coordinates')

        # Move input and output data to GPU device, if required.
        protein_vox = torch.tensor(protein_vox, device=self.device).float()

        # If there are no candidate voxels, traverse the whole voxelized representation
        # and evaluate each voxel as center of a possible metal-binding region;
        counter = 0
        if candidates is None:
            x_dim = len(range(half_size, size_x-half_size, stride))
            y_dim = len(range(half_size, size_y-half_size, stride))
            z_dim = len(range(half_size, size_z-half_size, stride))
            for i in range(half_size, size_x-half_size, stride):
                for j in range(half_size, size_y-half_size, stride):
                    for k in range(half_size, size_z-half_size, stride):
                        if protein_vox[:,5,i,j,k] < occupancy_restrictions:
                            if self.device == 'cuda':
                                evaluated_vox[i,j,k] = self.model(protein_vox[:, :,
                                                                        i-half_size:i+half_size,
                                                                        j-half_size:j+half_size,
                                                                        k-half_size:k+half_size
                                                                        ]).detach()

                            else:
                                evaluated_vox[i,j,k] = self.model(protein_vox[:, :,
                                        i-half_size:i+half_size,
                                        j-half_size:j+half_size,
                                        k-half_size:k+half_size
                                        ]).detach().numpy()

                        print(f"Analysed: {(counter/(x_dim*y_dim*z_dim))*100} %")
                        counter += 1
        # else, traverse the candidate voxels and only evaluate them.
        else:
            for candidate in candidate_voxels:
                evaluated_vox[candidate[0], 
                                candidate[1], 
                                candidate[2]] = self.model(protein_vox[:, :, 
                                                                        candidate[0]-half_size:candidate[0]+half_size,
                                                                        candidate[1]-half_size:candidate[1]+half_size,
                                                                        candidate[2]-half_size:candidate[2]+half_size])
        # evaluated_vox = evaluated_vox.detach.numpy()
        # Reshape 3D np.array to 1D np.array
        evaluated_vox = evaluated_vox.reshape(-1)

        # Add voxel centers column to the 1D np.array to create a 2D array
        # where cartesian coordinates are correlated to metal-bindingness
        if self.device == 'cuda': evaluated_vox = evaluated_vox.detach().cpu().numpy()
        evaluated_vox = np.c_[protein_centers, evaluated_vox]
        return evaluated_vox

    def create_PDB(self, output_dir : str, target_name : str, protein_scores : np.array, 
                    threshold_1 : float, threshold_2 : float):
        """
        """
        with open(os.path.join(output_dir, target_name+'_results_aule.pdb'), "w") as fo:
            num_at = 0
            num_res = 0
            for entry in protein_scores:
                if entry[3] > threshold_2:
                    num_at += 1
                    num_res = 1
                    ch = "A"
                    prb_str = ""

                    for idx in range(3):
                        prb_center = "{:.8s}".format(str(round(float(entry[idx]),3)))
                        if len(prb_center) < 8:
                            prb_center = " "*(8-len(prb_center)) + prb_center
                            prb_str += prb_center
                        else:
                            prb_str += prb_center
                    score = "{:.8s}".format(str(round(float(entry[3]),3)))
                    if entry[3] < threshold_1:
                        atom = "N"
                        atom2 = "NE"
                    elif entry[3] >= threshold_1:
                        atom = "N"
                        atom2 = "NE"
                    fo.write("ATOM" +" "*(7-len(str(num_at))) + "%s  %s  SLN %s" %(num_at, atom2, ch))
                    fo.write(" "*(3-len(str(num_res))) + "%s     %s  1.00  0.00          %s\n" %(num_res, prb_str, atom2))
    
    def create_txt(self, output_dir : str, protein_scores : np.array):
        raise NotImplementedError("Method still not implemented")

    def find_candidates(self, centers : np.array, nvoxels : np.array, coords : np.array,
                        voxelsize : float) -> list:
        """
        Helper function to evaluate() that obtains the voxel coordinates closer than the 
        voxelsize to a given cartesian coordinate.

        Returns
        -------
        results : list
            List with coordinates of the voxels with centers that are closer
            than the voxelsize to the cartesian coordinates
        """
        # Start from the initial coordinates (upper-left-front corner of the 3D image)
        i, j, k = 0, 0, 0 
        # Initialize a list with voxel coordinates close to the cartesian coordinates
        results = [] 
        
        # Go through every voxel center analysing whether they are close enough
        # to the cartesian coordinates we are evaluating. If they are,
        # save the voxel coordinates they will correspond to in the images.
        for center in centers: 
            # If reached the end of the depth, go back to the front plane 
            # and move one voxel to the right
            if k == nvoxels[2]:
                k = 0
                j += 1
                # If reached the end of the width, go back to the left
                # and move one voxel down
                if j == nvoxels[1]:
                    j = 0
                    i += 1
            # If a voxel center is located closer than the voxelsize to the cartesian coordinates
            if self._euclidean_distance(center, coords) < voxelsize: 
                # append it to the results
                results.append((i,j,k))
            # Move one voxel deeper
            k += 1
        # Return the list as np.array to improve performance down the line
        return np.array(results)
    
    def _not_in_backbone(self, A : np.array, centers : np.array) -> bool:
        ""
        for center in centers[0]:
            if self._euclidean_distance(A, center) <= 0.75:
                return False
        return True

    def _euclidean_distance(self, A : np.array, B : np.array) -> float:
        """
        Helper function to find_candidates() that calculate the euclidean distance between 2 
        different points in cartesian space.

        Returns
        -------
        euclidean_distance : float
            Euclidean distance between two points A and B.
        """
        return np.sqrt((A[0]-B[0])**2+(A[1]-B[1])**2+(A[2]-B[2])**2)

    def _get_undesired_channels(self, channels : list) -> list:
        """
        Helper function to voxelize() that reads the desired channels for
        voxelization and determines what channels should be turned off. 
        It outputs a list of ints with the indexes of the undesired channels.

        Raises
        ------
        ValueError : If any of the channels provided is not supported, 
                     raise the error and inform of the supported channels.

        Returns
        -------
        undesired_channels : list
            List of undesired channels.
        """
        # Set a dictionary that relates the channel label to its dimension.
        supported_channels = {'hydrophobic': 0, 'aromatic': 1, 'hbond_acceptor': 2, 
                            'hbond_donor': 3, 'positive_ionizable': 4, 'negative_ionizable': 5, 
                            'metal': 6, 'occupancies': 7}
        # Initialise necessary lists
        working_channels = []
        undesired_channels = []

        # Go through every channel in channels and
        for channel in channels:
            # first verify that they are supported. If not raise ValueError.
            if channel not in supported_channels.keys():
                raise ValueError(f"Unsupported channel: {channel}.\nSupported channels: {supported_channels.keys()}.")
            # If they are supported append their corresponding int to the working_channels list
            working_channels.append(supported_channels[channel])

        # For all possible channels,
        for i in range(8):
            # if they will not be used,
            if i not in working_channels:
                # append them to the undesired_channels list.
                undesired_channels.append(i)
        return undesired_channels

    def _get_candidate_voxels(self, candidates : str or np.array or list, protein_centers : np.array, 
                              nvoxels : np.array, voxelsize : float) -> np.array:
        """
        Check candidates and obtain the candidate voxels associated with them. 
        There are two situations the algorithm is equiped to handle: 1) candidates 
        contains the path to a biometall output or 2) it contains a list or 
        np.array with the cartesian coordinates of candidate regions to be evaluated.
        The second situation is trivial to solve and does not deserve further comment,
        first situation requires analysing the PDB and locating the cartesian coordinates
        for HE and XE atoms which are the markers of putative metal-binding regions.

        Raises
        ------
        FileNotFoundError : candidates is a string but no file cannot be found where
                            its path is indicating.

        Returns
        -------
        candidate_voxels : np.array
            List of voxel coordinates which have centers that are closer than voxelsize
            to any of the cartesian coordinates of the candidates.
        """
        candidates_list = []
        candidate_voxels = []

        # If candidates is a string, then it will be a biometall output path
        # so it requires a preprocessing
        if isinstance(candidates, str):
            try:
                molecule = Molecule(candidates, validateElements=False)
            except FileNotFoundError:
                raise FileNotFoundError(f"File {candidates} does not exist. Please check the path to verify its spelling.")

            # Go through every entry in the PDB
            for index in molecule.get("index"): 
                # If an entry of the PDB contains the XE or HE, biometall marks of putative metal-binding,
                if molecule.get("resname", sel=index)[0] == "XE" or molecule.get("resname", sel=index)[0] == "HE":
                    # save its coordinates
                    candidates_list.append(molecule.get("coords", sel=index)) 

        # For each set of candidate cartesian coordinates
        for candidate in candidates_list:
            # Find voxels with a center at voxelsize of the cartesian coordinate
            candidate_voxels.append(*self.find_candidates(protein_centers, nvoxels, candidate[0], voxelsize))
        return np.array(candidate_voxels)
                

class aule_trainer():
    """
    Class to be created
    """
    def __init__(self, trainee):
        self.trainee = trainee

if __name__ == "__main__":
    # Do something if this file is invoked on its own
    help(aule)
