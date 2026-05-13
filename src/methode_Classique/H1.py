import sys
import os
import datetime
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm

# Ajout des chemins pour importer vos modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "cvnn", "src"))
sys.path.append(os.path.dirname(__file__))

from cvnn.config import load_config
from cvnn.utils import set_seed
from cvnn.data import azimut_split

# Import depuis vos propres scripts
from feature_extraction import compute_batched_global_covariance, extract_batched_correlation_features
from feature_extraction import compute_batched_global_covariance, extract_batched_correlation_features, extract_features_from_loader
from H0 import CoherenceCalibrator, SpectralEntropyCalibrator


# ==============================================================================
# 1. GÉNÉRATEURS D'ANOMALIES PHYSIQUES (H1)
# ==============================================================================

class PhysicalH1Generator(ABC):
    """ 
    Classe de base abstraite pour générer des anomalies physiques H1. 
    L'anomalie s'applique directement sur la matrice de covariance C.
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def apply_corruption(self, C_batched: np.ndarray) -> np.ndarray:
        """
        Prend un batch de matrices de covariance PURES (B, 4, 4).
        Retourne un batch de matrices CORROMPUES (B, 4, 4).
        """
        pass

class Crosstalk(PhysicalH1Generator):
    """
    Simule une mauvaise isolation de l'antenne (Diaphonie / Cross-talk).
    Une partie de l'énergie Horizontale fuit dans le canal Vertical et inversement.
    """
    def __init__(self, delta=0.05):
        super().__init__(f"Crosstalk_Error_d{delta}".replace(".", "p"))
        self.delta = delta

    def apply_corruption(self, C_batched: np.ndarray) -> np.ndarray:
        D_2x2 = np.array([
            [1.0, self.delta],
            [self.delta, 1.0]
        ], dtype=complex)

        D_4x4 = np.kron(D_2x2, D_2x2)
        D_4x4_H = np.conjugate(D_4x4.T)

        C_corrupted = np.einsum('ij,njk,kl->nil', D_4x4, C_batched, D_4x4_H)
        return C_corrupted  

class ChannelGainImbalance(PhysicalH1Generator):
    """
    Simule un déséquilibre du gain entre les canaux (Channel Gain Imbalance).
    """
    def __init__(self, g: complex = 1.029):
        name_val = str(np.round(np.abs(g), 3)).replace(".", "p")
        super().__init__(f"ChannelGain_Error_g{name_val}")
        self.g = g

    def apply_corruption(self, C_batched: np.ndarray) -> np.ndarray:
        D_4x4 = np.diag([1.0, self.g, self.g, self.g**2]).astype(complex)
        D_4x4_H = np.conjugate(D_4x4.T)
        C_corrupted = np.einsum('ij,njk,kl->nil', D_4x4, C_batched, D_4x4_H)
        return C_corrupted


# ==============================================================================
# 2. EXTRACTEUR AVEC CORRUPTION
# ==============================================================================

def extract_anomalous_features(dataloader, anomaly_generator: PhysicalH1Generator, desc="Extraction H1"):
    """ 
    Extrait les caractéristiques (16 Features) après avoir altéré physiquement 
    les matrices de covariance du flux de données.
    """
    X_list = []

    for batch in tqdm(dataloader, desc=f"[{anomaly_generator.name}]"):
        # Gestion du format de batch
        if isinstance(batch, (list, tuple)): inputs = batch[0]
        elif isinstance(batch, dict): inputs = batch.get("inputs", batch.get("data"))
        else: inputs = batch
            
        x_np = inputs.cpu().numpy()
        
        # 1. Calcul de la covariance pure
        mat_C_batched = compute_batched_global_covariance(x_np)
        
        # 2. INJECTION DE L'ANOMALIE
        mat_C_corrupted = anomaly_generator.apply_corruption(mat_C_batched)
        
        # 3. Extraction des corrélations altérées
        features_batched = extract_batched_correlation_features(mat_C_corrupted)
        X_list.append(features_batched)

    return np.vstack(X_list)


# ==============================================================================
# 3. ÉVALUATION SUR LE MODÈLE H0
# ==============================================================================

def score_depth_against_H0(X_h1: np.ndarray, calib_dir: Path) -> np.ndarray:
    """ Calcule la Profondeur (Depth) en projetant sur les axes sains (H0) de Stahel-Donoho. """
    V = np.load(calib_dir / "V_directions.npy")
    medians = np.load(calib_dir / "Train_medians.npy")
    mads = np.load(calib_dir / "Train_mads.npy")
    
    projections = X_h1 @ V.T
    abs_diff = np.abs(projections - medians)
    max_deviations = np.max(abs_diff / mads, axis=1)
    return 1.0 / (1.0 + max_deviations)

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))
    
    # Résolution absolue du chemin des données par rapport à la racine du projet
    trainpath = Path(cfg["data"]["dataset"]["trainpath"])
    if not trainpath.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        cfg["data"]["dataset"]["trainpath"] = str((repo_root / trainpath).resolve())

    print("\n[*] Chargement des zones 2.1 et 2.2 pour les tests...")
    loaders_dict = azimut_split(cfg, use_cuda=False)
    loader_test_2_1, _, _ = loaders_dict["loader2_1_full"]
    loader_test_2_2, _, _ = loaders_dict["loader2_2_full"]

    base_calib_dir = Path("calibration_results")
    if not base_calib_dir.exists() or not list(base_calib_dir.glob("run_*")):
        print("[!] ERREUR: Aucun dossier de calibration H0 trouvé. Lancez H0.py en premier.")
        sys.exit(1)

    # Récupérer le dossier de calibration le plus récent
    calib_dir = max([d for d in base_calib_dir.glob("run_*") if d.is_dir()], key=os.path.getmtime)
    print(f"\n[*] Utilisation de la calibration trouvée : {calib_dir}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("test_results") / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ==============================================================================
    # A. ÉVALUATION DES DONNÉES PURES (Zone 2.1 et Zone 2.2)
    # ==============================================================================
    print("\n[*] Extraction des caractéristiques sur les données PURES (Zone 2.1)...")
    X_test_pure_2_1 = extract_features_from_loader(loader_test_2_1, desc="Extraction Pure 2.1")
    
    print("[*] Calcul et sauvegarde des scores pour les données PURES 2.1...")
    np.save(out_dir / "Pure_2_1_Depth_Scores.npy", score_depth_against_H0(X_test_pure_2_1, calib_dir))
    np.save(out_dir / "Pure_2_1_Coherence_Scores.npy", CoherenceCalibrator().compute_scores(X_test_pure_2_1))
    np.save(out_dir / "Pure_2_1_Entropy_Scores.npy", SpectralEntropyCalibrator().compute_scores(X_test_pure_2_1))
    np.save(out_dir / "X_test_pure_2_1_features.npy", X_test_pure_2_1) #Sauvegarde pour methode_MachineLearning

    print("\n[*] Extraction des caractéristiques sur les données PURES (Zone 2.2)...")
    X_test_pure_2_2 = extract_features_from_loader(loader_test_2_2, desc="Extraction Pure 2.2")
    
    print("[*] Calcul et sauvegarde des scores pour les données PURES 2.2...")
    np.save(out_dir / "Pure_2_2_Depth_Scores.npy", score_depth_against_H0(X_test_pure_2_2, calib_dir))
    np.save(out_dir / "Pure_2_2_Coherence_Scores.npy", CoherenceCalibrator().compute_scores(X_test_pure_2_2))
    np.save(out_dir / "Pure_2_2_Entropy_Scores.npy", SpectralEntropyCalibrator().compute_scores(X_test_pure_2_2))
    # Sauvegarde ML
    np.save(out_dir / "X_test_pure_2_2_features.npy", X_test_pure_2_2)

    # ==============================================================================
    # B. INJECTION ET ÉVALUATION DES ANOMALIES (H1_test)
    # ==============================================================================
    anomalies_to_test = [
        Crosstalk(delta=0.05),
        ChannelGainImbalance(g=1.029)
    ]

    for anomaly in anomalies_to_test:
        print(f"\n[*] Injection de l'anomalie sur la Zone 2.2 : {anomaly.name}")
        X_test_h1 = extract_anomalous_features(loader_test_2_2, anomaly)
        
        # Calcul des scores d'anomalies
        depth_scores = score_depth_against_H0(X_test_h1, calib_dir)
        coh_scores = CoherenceCalibrator().compute_scores(X_test_h1)
        ent_scores = SpectralEntropyCalibrator().compute_scores(X_test_h1)
        
        # Sauvegarde
        np.save(out_dir / f"{anomaly.name}_Depth_Scores.npy", depth_scores)
        np.save(out_dir / f"{anomaly.name}_Coherence_Scores.npy", coh_scores)
        np.save(out_dir / f"{anomaly.name}_Entropy_Scores.npy", ent_scores)
        # Sauvegarde ML
        np.save(out_dir / f"X_{anomaly.name}_features.npy", X_test_h1)
        print(f"[+] Scores sauvegardés pour l'anomalie : {anomaly.name}")