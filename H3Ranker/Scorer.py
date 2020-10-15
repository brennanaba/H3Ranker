import numpy as np
from Bio.PDB import PDBParser, Polypeptide
from numba import jit

from H3Ranker.geometries import geom_from_residues
from H3Ranker.network import deep2d_model, one_hot, bins, dist_bins
import os


current_directory = os.path.dirname(os.path.realpath(__file__))

def get_models(fread_output):
    res = PDBParser(QUIET=True).get_structure("outs",fread_output)
    with open(fread_output) as file:
        ids = [x.split()[1] for x in file.readlines() if x.split()[0] == "MODEL"]
        
    assert len([x.get_id() for x in res.get_models()]) == len(ids), "Error Reading File: " + fread_output
    return {ids[i] : res.child_list[i] for i in range(len(ids))}

def get_anchors(pdb_file, chain):
    og = PDBParser(QUIET=True).get_structure("outs",pdb_file)
    heavy_chain = [chn for chn in og.get_chains() if chn.get_id() == chain][0]
    
    # Chose 91 to 104 as the aminoacids on these residues ar always the same
    residues = [r for r in heavy_chain.get_residues() if r.get_id()[0] == " " and 91 < r.get_id()[1] < 105]
    return residues


@jit
def sort_distance_into_bins(x, dist_bins):
    x = np.where(np.isnan(x), -1, x)
    x = np.where((0 < x) & (x < dist_bins[0]), dist_bins[0], x)
    return np.digitize(x, dist_bins)

@jit
def sort_angles_into_bins(x, bins):
    x = np.where(np.isnan(x), -1e5, x)
    return np.digitize(x, bins)


@jit
def log_likelihood(probabilty_map, binned_map):
    """Calculates the log likelihood of a structure according to the DeepH3 network

    :param probabilty_map:
        Probability for each inter residue geometry to be in one of the
        predefined bins according to the model. This is basically the output
        of the NN.
    :param binned_map:
        Binned values of each inter residue geometry calculated for a given
        (pdb) structure.
    """
    score = 0.0
    xsize, ysize = len(binned_map), len(binned_map[0])
    for i in range(xsize):
        for j in range(ysize):
            filt = probabilty_map[i, j, binned_map[i, j]] > 1/len(bins)
            if filt:
                score = score + np.log(probabilty_map[i, j, binned_map[i, j]])
            else:
                score = score + np.log(1/len(bins))
    return -score



class DecoyScorer:
    def __init__(self, pdb_file, chain, network_weights  = os.path.join(current_directory, "models/kullback_centered_gaussian_15layers_50drop.h5"), model = deep2d_model()):
        self.pdb_file = pdb_file
        self.chain = chain
        self.model = model
        self.model.load_weights(network_weights)
        
        self.original_residues = get_anchors(pdb_file, chain)
        self.sequence = Polypeptide.Polypeptide(self.original_residues).get_sequence()
        to_numbers = Polypeptide.d1_to_index
        self.numerical_sequence = [to_numbers[x] for x in self.sequence]
        model_input = np.expand_dims(one_hot(np.array(self.numerical_sequence)),0)

        self.distance_predictions, self.omega_predictions, self.theta_predictions, self.phi_predictions = self.model.predict(model_input)
        self.distance_predictions = np.squeeze(self.distance_predictions) 
        self.omega_predictions = np.squeeze(self.omega_predictions)
        self.theta_predictions = np.squeeze(self.theta_predictions)
        self.phi_predictions = np.squeeze(self.phi_predictions)
            
    def score(self,loop_residues, one_score = True):
        start = [x.get_id() for x in self.original_residues].index(loop_residues[0].get_id())
        
        # Mount decoy onto anchor residues:
        loop_with_anchor = self.original_residues.copy()
        loop_with_anchor[start:start+len(loop_residues)] = loop_residues
        
        # Get Geometry maps from list of residues
        decoy_dist, decoy_omega, decoy_theta, decoy_phi = geom_from_residues(loop_with_anchor)
        
        # Sort values into corresponding bins
        decoy_dist_binned = sort_distance_into_bins(decoy_dist,dist_bins)
        decoy_omega_binned = sort_angles_into_bins(decoy_omega,bins)
        decoy_theta_binned = sort_angles_into_bins(decoy_theta,bins)
        decoy_phi_binned = sort_angles_into_bins(decoy_phi,bins)
        
        # Get scores for distance and omega
        dist_score = log_likelihood(self.distance_predictions, decoy_dist_binned)
        omega_score = log_likelihood(self.omega_predictions, decoy_omega_binned)
        theta_score = log_likelihood(self.theta_predictions, decoy_theta_binned)
        phi_score = log_likelihood(self.phi_predictions, decoy_phi_binned)
        
        if one_score:
            return  dist_score + omega_score + theta_score + phi_score
        else:
            return [dist_score, omega_score, theta_score, phi_score]
        
        
    def score_fread_output(self, fread_output, decoys = []):
        loops = get_models(fread_output)
        
        scores = []
        scored_decoys = []
        
        for loop in loops:
            if loop in decoys or decoys == []:
                scores.append(self.score(list(loops[loop].get_residues())))
                scored_decoys.append(loop)
        
        return scored_decoys, scores


