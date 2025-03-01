
# with OpenMM 7.7.0, the import calls have changed. So, try both, if needed
try:
    try:
        # >=7.7.0
        from openmm.app import *
    except:
        # earlier
        print('Unable to load OpenMM as \'openmm\'. Will try the older way \'simtk.openmm\'')
        from simtk.openmm.app import *
except:
    print('Failed to load OpenMM. Check your configuration.')

import numpy as np
import random
from scipy.spatial import distance
import scipy as sc
import itertools
from scipy.stats.stats import pearsonr
from sklearn.preprocessing import normalize
import os
import pandas as pd
import warnings

class Energy_Landscape_Optimizer:
    
    def __init__(self, mu=2.0, rc = 2.0, method="Adam",eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, it=1, error_pca_weight=0.0):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta
        self.mu = mu
        self.rc = rc
        self.NFrames = 0
        self.t = it
        self.method = method.lower()
        self.opt_params = {}
        self.reduce_error_dimension = error_pca_weight
        
    def load_HiC(self, HiC, cutoff_low=0.0, cutoff_high=1.0, neighbors=0):
        R"""
        Receives the experimental Hi-C map (Full dense matrix) in a text format and performs the data normalization from Hi-C frequency/counts/reads to probability.
        
        Args:
            HiC (file, required):
                Experimental Hi-C map (Full dense matrix) in a text format.
            centerRemove (bool, optional):
                Whether to set the contact probability of the centromeric region to zero. (Default value: :code:`False`).
            centrange (list, required if **centerRemove** = :code:`True`)):
                Range of the centromeric region, *i.e.*, :code:`centrange=[i,j]`, where *i* and *j*  are the initial and final beads in the centromere. (Default value = :code:`[0,0]`).
            cutoff (float, optional):
                Cutoff value for reducing the noise in the original data. Values lower than the **cutoff** are considered :math:`0.0`.
        """

        # get the file extension
        _, file_extension = os.path.splitext(HiC)
        assert file_extension == '.txt', "Input Hi-C file should be a TXT file that can be handled by np.loadtxt"
        
        hic_mat = np.loadtxt(HiC)
        
        assert self.is_symmetric(hic_mat), "Experimental HiC input is NOT symmetric"
                
        #remove noise by cutoff 
        if cutoff_low > 0.0:
            hic_mat[hic_mat < cutoff_low] = 0.0
        
        if cutoff_high < 1.0:
            hic_mat[hic_mat > cutoff_high] = 0.0

        # Remove the number of Neighbors to optimize.
        neighbor_mask = np.abs(np.subtract.outer(np.arange(len(hic_mat)), np.arange(len(hic_mat)))) <= neighbors
        hic_mat[neighbor_mask] = 0.0

        self.phi_exp = hic_mat
        
        self.mask = hic_mat == 0.0

        self.reset_Pi()
        self.init_optimization_params()
    
    
    def is_symmetric(self, mat, rtol=1e-05, atol=1e-08):
        return np.allclose(mat, mat.T, rtol=rtol, atol=atol)

    
    def reset_Pi(self):
        R"""
        Resets Pi matrix to zeros
        """
        if not hasattr(self, "phi_exp"):
            print("Cannot reset Pi; HiC map shape unknown. Load HiC map first!")
        else:              
            self.Pi = np.zeros(self.phi_exp.shape)
            self.NFrames = 0

    
    def init_optimization_params(self,):
        
        if self.method == "adam":
            
            self.opt_params["m_dw"] = np.zeros_like(self.phi_exp)
            self.opt_params["v_dw"] = np.zeros_like(self.phi_exp)
        
        elif self.method == "nadam":
            
            self.opt_params["m_dw"] = np.zeros_like(self.phi_exp)
            self.opt_params["v_dw"] = np.zeros_like(self.phi_exp)
        
        elif self.method == 'rmsprop':
            self.opt_params["v_dw"] = np.zeros_like(self.phi_exp)
        
        elif self.method == 'adagrad':
            self.opt_params["G_dw"] = np.zeros_like(self.phi_exp)
        
    
    def update(self,):
        R"""Adam optimization step. This function updates weights and biases for each step.
        """
        
        grad = self.get_error_gradient()
        
        if self.method == 'adam':
            
            m_dw = self.opt_params["m_dw"] * self.beta1 + (1 - self.beta1) * grad
            v_dw = self.opt_params["v_dw"] * self.beta2 + (1 - self.beta2) * (grad ** 2)
            
            m_dw_corr = m_dw / (1 - self.beta1 ** self.t)
            v_dw_corr = v_dw / (1 - self.beta2 ** self.t)
            
            w = self.force_field - self.eta * m_dw_corr / (np.sqrt(v_dw_corr) + self.epsilon)
            
            self.opt_params["m_dw"] = m_dw
            self.opt_params["v_dw"] = v_dw
        
        elif self.method == 'rmsprop':
            
            v_dw = self.opt_params["v_dw"] * self.beta1 + (1 - self.beta1) * (grad ** 2)
            w = self.force_field - self.eta * grad / (np.sqrt(v_dw) + self.epsilon)
            
            self.opt_params["v_dw"] = v_dw
        
        elif self.method == 'adagrad':
            
            G_dw = self.opt_params["G_dw"] + (grad ** 2)
            w = self.force_field - self.eta * grad / (np.sqrt(G_dw) + self.epsilon)
            
            self.opt_params["G_dw"] = G_dw
            
        elif self.method == 'nadam':
            
            m_dw = self.opt_params["m_dw"] * self.beta1 + (1 - self.beta1) * grad
            v_dw = self.opt_params["v_dw"] * self.beta2 + (1 - self.beta2) * (grad ** 2)
            
            m_dw_corr = m_dw / (1 - self.beta1 ** self.t)
            v_dw_corr = v_dw / (1 - self.beta2 ** self.t)
            
            lookahead_gradient = (1 - self.beta1) * grad / (1 - self.beta1 ** self.t)
            
            w = self.force_field - self.eta * (m_dw_corr + lookahead_gradient) / (np.sqrt(v_dw_corr) + self.epsilon)
            
            self.opt_params["m_dw"] = m_dw
            self.opt_params["v_dw"] = v_dw
            
        self.t += 1
        return w


    def compute_contact_prob(self, state):
        R"""
        Calculates the contact probability matrix for a given state.
        """

        Pi = 0.5*(1.0 + np.tanh(self.mu*(self.rc - distance.cdist(state,state, 'euclidean'))))
    
        self.Pi += Pi
        self.NFrames += 1


    def get_error_gradient(self):
        R"""
        Calcultes the gradient function.
        """
        g = -self.phi_sim + self.phi_exp
        np.fill_diagonal(g, 0.0)
        g -= np.diagflat(np.diag(g,k=1),k=1)
        g = np.triu(g) + np.triu(g).T
        
        if self.regularize > 0.0:
            print(f"Removing gradient dimensions where eignevalues are smaller than {self.reduce_error_dimension} times principal eignevalue")
            eig_vals, eig_vecs = np.linalg.eig(g)
            max_eig = eig_vals[0]
            removed_components = []
            for xx, eig in enumerate(eig_vals):
                if abs(eig/max_eig) < self.reduce_error_dimension:
                    g -= eig * np.outer(eig_vecs[:,xx], eig_vecs[:,xx])
                    removed_components.append(xx)
            print(f"Removed components: {removed_components}")
        
        return g


    def compute_force_field(self, ff_current):
        R"""
        Calculates the Lagrange multipliers of each pair of interaction and returns the matrix containing the energy values for the optimization step.
        
        Args:
            Lambdas (file, required):
                The matrix containing the energies values used to make the simulation in that step. 
            fixedPoints (list, optional):
                List of all pairs (i,j) of interactions that will remain unchanged throughout the optimization procedure.
        
        Returns:
            :math:`(N,N)` :class:`numpy.ndarray`:
                Returns an updated matrix of interactions between each pair of bead.
        """
        self.phi_sim = self.Pi/self.NFrames
        self.phi_sim[self.mask] = 0.0

        df = pd.read_csv(ff_current, sep=None, engine='python')
        
        current_force_field = df.values
        current_force_field = np.triu(current_force_field) + np.triu(current_force_field).T
        np.fill_diagonal(current_force_field, 0.0)
        self.force_field = current_force_field
        self.updated_force_field = self.update()
        
        df_updated_ff  = pd.DataFrame(self.updated_force_field,columns=list(df.columns.values))
        
        self.error = np.sum(np.absolute(np.triu(self.phi_sim, k=2) - np.triu(self.phi_exp, k=2)))/np.sum(np.triu(self.phi_exp, k=2))

        return (df_updated_ff)