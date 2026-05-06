import sys
import os
import datetime
import numpy as np
from pathlib import Path

# Ajout des chemins pour importer vos modules cvnn et l'extracteur local
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "cvnn", "src"))
sys.path.append(os.path.dirname(__file__))

from cvnn.config import load_config
from cvnn.utils import set_seed
from cvnn.data import azimut_split
from feature_extraction import extract_features_from_loader

class DepthCalibrator:
    """ Gère la génération de l'espace de projection et les stats Stahel-Donoho. """
    def __init__(self, M=10000, dim=16):
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


class CoherenceCalibrator:
    """ Gère le calcul de la cohérence globale (somme des modules gamma). """
    def compute_scores(self, X_train):
        # On ne prend que les 12 premières colonnes (index 0 à 11)
        X_complex_parts = X_train[:, :12].reshape(-1, 6, 2) 
        modules = np.sqrt(np.sum(X_complex_parts**2, axis=2))
        return np.sum(modules, axis=1)

    def save(self, path: Path, scores):
        np.save(path / "Train_Coherence_Scores.npy", scores)


class SpectralEntropyCalibrator:
    """ Gère la décomposition en valeurs propres et le calcul de l'entropie de référence. """
    def compute_scores(self, X_train):
        X_geom = X_train[:, :12] 
        N = X_train.shape[0]
        
        # Reconstruction des matrices 4x4
        Gamma = np.zeros((N, 4, 4), dtype=complex)
        for i in range(4):
            Gamma[:, i, i] = 1.0 + 0j 
            
        idx = 0
        for i in range(4):
            for j in range(i):
                comp = X_geom[:, idx*2] + 1j * X_geom[:, idx*2+1] 
                Gamma[:, i, j] = comp
                Gamma[:, j, i] = np.conj(comp) 
                idx += 1
                
        # Extraction des valeurs propres
        lambdas = np.linalg.eigvalsh(Gamma)
        
        # Calcul de l'Entropie H
        lambdas_safe = np.clip(lambdas, 1e-10, None) 
        P = lambdas_safe / np.sum(lambdas_safe, axis=1, keepdims=True)
        H = -np.sum(P * np.log(P) / np.log(4), axis=1)
        return H

    def save(self, path: Path, scores):
        np.save(path / "Train_Entropy_Scores.npy", scores)

if __name__ == "__main__":
    # 1. Setup et chargement
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))
    
    # Résolution absolue du chemin des données par rapport à la racine du projet
    trainpath = Path(cfg["data"]["dataset"]["trainpath"])
    if not trainpath.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        cfg["data"]["dataset"]["trainpath"] = str((repo_root / trainpath).resolve())

    print("\n[*] Chargement des données saines (H0)...")
    loaders_dict = azimut_split(cfg, use_cuda=False)
    train_loader, valid_loader, test_loader = loaders_dict["part1_loaders"]
    
    X_train = extract_features_from_loader(train_loader, desc="Extraction Train (Zone 1)")
    X_valid = extract_features_from_loader(valid_loader, desc="Extraction Valid (Zone 1)")
    X_test  = extract_features_from_loader(test_loader, desc="Extraction Test (Zone 1)")

    # Combinaison des 3 loaders de la Zone 1
    X_zone1 = np.vstack([X_train, X_valid, X_test])
    print(f"[*] Taille totale des données saines (Zone 1) : {X_zone1.shape}")

    # 2. Dossier de sortie
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("calibration_results") / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarde de la matrice des features pour les méthodes Machine Learning
    np.save(out_dir / "X_zone1_features.npy", X_zone1)

    # 3. Calibration
    print("\n[*] Exécution du DepthCalibrator (Stahel-Donoho)...")
    depth_calib = DepthCalibrator(dim=X_zone1.shape[1])
    d_scores = depth_calib.calibrate(X_zone1)
    depth_calib.save(out_dir, d_scores)

    print("[*] Exécution du CoherenceCalibrator...")
    coh_calib = CoherenceCalibrator()
    c_scores = coh_calib.compute_scores(X_zone1)
    coh_calib.save(out_dir, c_scores)

    print("[*] Exécution du SpectralEntropyCalibrator...")
    ent_calib = SpectralEntropyCalibrator()
    e_scores = ent_calib.compute_scores(X_zone1)
    ent_calib.save(out_dir, e_scores)

    print(f"\n[+] Calibration H0 terminée avec succès. Résultats sauvés dans '{out_dir}/'")