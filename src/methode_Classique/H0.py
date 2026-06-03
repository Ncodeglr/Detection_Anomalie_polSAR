import numpy as np
from pathlib import Path

class DepthCalibrator:
    """ Gère la génération de l'espace de projection et les stats Stahel-Donoho. """
    def __init__(self, M=5000, dim=16):
        self.M = M
        self.dim = dim
        self.V = None
        self.medians = None
        self.mads = None

    def generate_vectors(self):
        """ Génère les directions de projection uniformes. """
        N_rand = np.random.randn(self.M, self.dim) 
        norms = np.linalg.norm(N_rand, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.V = N_rand / norms 
        return self.V

    def calibrate(self, X_train):
        """ Calcule les projections, médianes, MADs et scores finaux. """
        if self.V is None: self.generate_vectors()
        
        # Projections [N, M]
        projections = X_train @ self.V.T
        
        # Stats par axe
        self.medians = np.median(projections, axis=0) 
        self.mads = np.median(np.abs(projections - self.medians), axis=0) 
        self.mads[self.mads == 0] = 1e-10  
        
        # Scores finaux [N]
        abs_diff = np.abs(projections - self.medians) 
        max_deviations = np.max(abs_diff / self.mads, axis=1) 
        D = 1.0 / (1.0 + max_deviations) 
        return D

    def save(self, path: Path, scores):
        np.save(path / "V_directions.npy", self.V)
        np.save(path / "Train_medians.npy", self.medians)
        np.save(path / "Train_mads.npy", self.mads)
        np.save(path / "Train_Depth_Scores.npy", scores)

    def load_and_score(self, X_test, calib_dir: Path):
        """ Charge les paramètres appris (H0) et calcule les scores sur de nouvelles données. """
        self.V = np.load(calib_dir / "V_directions.npy")
        self.medians = np.load(calib_dir / "Train_medians.npy")
        self.mads = np.load(calib_dir / "Train_mads.npy")
        
        projections = X_test @ self.V.T
        abs_diff = np.abs(projections - self.medians)
        max_deviations = np.max(abs_diff / self.mads, axis=1)
        return 1.0 / (1.0 + max_deviations)

