import numpy as np
from pathlib import Path

class DepthCalibrator:
    """ Gère la génération de l'espace de projection et les stats Stahel-Donoho. """
    def __init__(self, M=8000, dim=16):
        self.M = M
        self.dim = dim
        self.V = None
        self.medians = None
        self.mads = None

    def generate_vectors(self):
        """ Génère les directions de projection uniformes. """
        """Retourne cette matrice V de 8000 vecteurs unitaires de 16 dimensions, qui seront ensuite utilisés pour "projeter" les données et calculer les scores d'anomalie."""
        
        N_rand = np.random.randn(self.M, self.dim) #Matrice de M vecteurs aléatoires de dimension dim
        norms = np.linalg.norm(N_rand, axis=1, keepdims=True) #Vecteur contenant la norme de chacun des 8000 vecteurs (axis=1 signifie que la norme est calculée pour chaque ligne, keepdims=True garde la dimension pour permettre la division élément par élément)
        norms[norms == 0] = 1e-10
        self.V = N_rand / norms  #Normalisation pour obtenir des vecteurs unitaires uniformément répartis sur la sphère de rayon 1 de dimension dim
        return self.V

    def calibrate(self, X_train, sample_size=100000):
        """ Calcule les projections, médianes, MADs et scores finaux. """
        if self.V is None: self.generate_vectors()
        
        #Sous-échantillonnage pour éviter la saturation RAM lors de l'estimation des statistiques
        if len(X_train) > sample_size:
            np.random.seed(42)
            indices = np.random.choice(len(X_train), sample_size, replace=False)
            X_sample = X_train[indices]
        else:
            X_sample = X_train
            
        #Projections uniquement sur le sous-échantillon
        projections = X_sample @ self.V.T
        
        self.medians = np.median(projections, axis=0) 
        self.mads = np.median(np.abs(projections - self.medians), axis=0) 
        self.mads[self.mads == 0] = 1e-10  #Sécurité pour éviter la division par zéro
        
        #Calcul des scores sur TOUTES les données par lots pour économiser la RAM
        return self.score_batched(X_train)

    def score_batched(self, X, batch_size=10000):
        """ Calcule les distances par petits lots de données (Évite de saturer la RAM). """
        N = X.shape[0]
        scores = np.zeros(N)
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            proj = X[i:end] @ self.V.T
            abs_diff = np.abs(proj - self.medians)
            max_deviations = np.max(abs_diff / self.mads, axis=1)
            scores[i:end] = 1.0 / (1.0 + max_deviations)
        return scores

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
        
        return self.score_batched(X_test)
