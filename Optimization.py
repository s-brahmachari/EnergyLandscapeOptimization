import numpy as np
import logging
import os
import pandas as pd
from scipy.spatial import distance

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class EnergyLandscapeOptimizer:
    
    def __init__(self, mu: float = 2.0, rc: float = 2.0, method: str = "adam",
                 eta: float = 0.01, beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8, it: int = 1, error_pca_weight: float = 0.0):
        """
        Initializes the Energy Landscape Optimizer with given hyperparameters.
        """
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
        self.regularize = error_pca_weight
        self.phi_exp = None  # Experimental Hi-C data
        self.force_field = None
        self.updated_force_field = None
        self.phi_sim = None
        self.Pi = None

    def load_HiC(self, hic_file: str, cutoff_low: float = 0.0, cutoff_high: float = 1.0, neighbors: int = 0) -> None:
        """
        Loads the Hi-C matrix from a text file, applies cutoffs, and initializes optimization parameters.
        """
        if not hic_file.endswith('.txt'):
            raise ValueError("Input Hi-C file should be a TXT file that can be handled by np.loadtxt.")

        hic_mat = np.loadtxt(hic_file)
        
        if not self.is_symmetric(hic_mat):
            raise ValueError("Experimental HiC input is NOT symmetric.")
        
        # Apply cutoffs to remove noise
        hic_mat = np.clip(hic_mat, a_min=cutoff_low, a_max=cutoff_high)
        
        # Remove neighbor interactions within the given range
        neighbor_mask = np.abs(np.subtract.outer(np.arange(len(hic_mat)), np.arange(len(hic_mat)))) <= neighbors
        hic_mat[neighbor_mask] = 0.0

        self.phi_exp = hic_mat
        self.mask = hic_mat == 0.0

        self.reset_Pi()
        self.init_optimization_params()

    @staticmethod
    def is_symmetric(mat: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Checks if a matrix is symmetric."""
        return np.allclose(mat, mat.T, rtol=rtol, atol=atol)

    def reset_Pi(self) -> None:
        """Resets Pi matrix to zeros."""
        if self.phi_exp is None:
            raise ValueError("Cannot reset Pi; HiC map shape unknown. Load HiC map first.")
        self.Pi = np.zeros_like(self.phi_exp)
        self.NFrames = 0

    def init_optimization_params(self) -> None:
        """Initializes optimization parameters for different optimizers."""
        self.opt_params.clear()
        shape = self.phi_exp.shape

        if self.method in {"adam", "nadam"}:
            self.opt_params["m_dw"] = np.zeros(shape)
            self.opt_params["v_dw"] = np.zeros(shape)
        elif self.method == "rmsprop":
            self.opt_params["v_dw"] = np.zeros(shape)
        elif self.method == "adagrad":
            self.opt_params["G_dw"] = np.zeros(shape)

    def update_step(self, grad) -> np.ndarray:
        """Performs an optimization step based on the selected method."""
        

        if self.method in {"adam", "nadam"}:
            self.opt_params["m_dw"] *= self.beta1
            self.opt_params["m_dw"] += (1 - self.beta1) * grad
            self.opt_params["v_dw"] *= self.beta2
            self.opt_params["v_dw"] += (1 - self.beta2) * (grad ** 2)

            m_dw_corr = self.opt_params["m_dw"] / (1 - self.beta1 ** self.t)
            v_dw_corr = self.opt_params["v_dw"] / (1 - self.beta2 ** self.t)

            if self.method == "nadam":
                lookahead_gradient = (1 - self.beta1) * grad / (1 - self.beta1 ** self.t)
                m_dw_corr += lookahead_gradient

            w = self.force_field - self.eta * m_dw_corr / (np.sqrt(v_dw_corr) + self.epsilon)

        elif self.method == "rmsprop":
            self.opt_params["v_dw"] *= self.beta1
            self.opt_params["v_dw"] += (1 - self.beta1) * (grad ** 2)
            w = self.force_field - self.eta * grad / (np.sqrt(self.opt_params["v_dw"]) + self.epsilon)

        elif self.method == "adagrad":
            self.opt_params["G_dw"] += grad ** 2
            w = self.force_field - self.eta * grad / (np.sqrt(self.opt_params["G_dw"]) + self.epsilon)
        
        elif self.method == "sgd":
            w = self.force_field - self.eta * grad

        self.t += 1
        return w

    def compute_contact_prob(self, state: np.ndarray) -> None:
        """Calculates the contact probability matrix for a given state."""
        Pi = 0.5 * (1.0 + np.tanh(self.mu * (self.rc - distance.cdist(state, state, 'euclidean'))))
        self.Pi += Pi
        self.NFrames += 1

    def get_error_gradient(self) -> np.ndarray:
        """Calculates the gradient of the optimization objective."""
        if self.phi_sim is None:
            raise ValueError("phi_sim is not initialized. Ensure force field computation before calling this method.")

        gt = self.phi_exp - self.phi_sim
        np.fill_diagonal(gt, 0.0)
        gt -= np.diagflat(np.diag(gt, k=1), k=1)
        gt = np.triu(gt) + np.triu(gt).T  # Ensure symmetry

        if self.regularize > 0.0:
            logging.info(f"Removing gradient dimensions where eigenvalues are smaller than {self.regularize} times principal eigenvalue")
            eig_vals, eig_vecs = np.linalg.eigh(gt)
            max_eig = eig_vals[-1]
            removed_components = []

            for idx, eig in enumerate(eig_vals):
                if abs(eig / max_eig) < self.regularize:
                    gt -= eig * np.outer(eig_vecs[:, idx], eig_vecs[:, idx])
                    removed_components.append(idx)

            logging.info(f"Removed components: {removed_components}")

        return gt

    def compute_force_field(self, ff_current: str) -> pd.DataFrame:
        """Computes and updates the force field from the given file."""
        if self.Pi is None or self.NFrames == 0:
            raise ValueError("Contact probability matrix not initialized. Call compute_contact_prob before force field computation.")

        self.phi_sim = self.Pi / self.NFrames
        self.phi_sim[self.mask] = 0.0  # Apply the mask to filter out noise

        df = pd.read_csv(ff_current, sep=None, engine='python')
        current_force_field = df.values
        self.force_field = current_force_field
        grad = self.get_error_gradient()
        self.updated_force_field = self.update_step(grad)

        df_updated_ff = pd.DataFrame(self.updated_force_field, columns=list(df.columns.values))

        self.error = np.sum(np.abs(np.triu(self.phi_sim, k=2) - np.triu(self.phi_exp, k=2))) / np.sum(np.triu(self.phi_exp, k=2))
        return df_updated_ff