import sys
import os
import copy
import torch
import math 
import datetime
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Ajout des chemins pour importer vos modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "cvnn", "src"))
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cvnn.config import load_config
from cvnn.utils import set_seed
from cvnn.data import azimut_split
from cvnn.data import azimut_split, get_full_image_dataloader

# Import depuis vos propres scripts
from feature_extraction import compute_batched_global_covariance, extract_batched_correlation_features, extract_features_from_loader
from H0 import CoherenceCalibrator, DepthCalibrator
from H1 import Crosstalk, ChannelGainImbalance
from synthetic_parameter_generator import SyntheticParameterGenerator


# ==============================================================================
# 2. EXTRACTEUR AVEC CORRUPTION
# ==============================================================================

def extract_anomalous_features(dataloader, anomaly_generator, desc="Extraction H1"):
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

    # --- Division de la Zone 2.2 en 3 parties sur l'axe du range (colonnes) ---
    print("\n[*] Division de la Zone 2.2 en 3 sous-zones (range)...")
    dataset_cfg = cfg["data"]["dataset"]
    crop_cfg = dataset_cfg.get("crop_coordinates", {})
    
    zone2_2_start_row = dataset_cfg.get("azimut_split_x2")
    zone2_2_end_row = crop_cfg.get("end_row_crop")
    zone2_2_start_col = crop_cfg.get("start_col", 0)
    zone2_2_end_col = crop_cfg.get("end_col_crop")

    col_split_points = np.linspace(zone2_2_start_col, zone2_2_end_col, 4, dtype=int)
    print(f"   - Points de split (colonnes) : {col_split_points}")

    loaders_2_2_parts = []
    for i in range(3):
        cfg_part = copy.deepcopy(cfg)
        cfg_part["data"]["dataset"]["crop_coordinates"] = {
            "start_row": zone2_2_start_row, "end_row": zone2_2_end_row,
            "start_col": col_split_points[i], "end_col": col_split_points[i+1],
            "max_rows": crop_cfg.get("max_rows"), "max_cols": crop_cfg.get("max_cols")
        }
        loader, _, _ = get_full_image_dataloader(cfg_part, use_cuda=False)
        loaders_2_2_parts.append(loader)
        print(f"   - Sous-zone {i+1} créée ({col_split_points[i]} -> {col_split_points[i+1]}) avec {len(loader.dataset)} patchs.")

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
    np.save(out_dir / "Pure_2_1_Depth_Scores.npy", DepthCalibrator().load_and_score(X_test_pure_2_1, calib_dir))
    np.save(out_dir / "Pure_2_1_Coherence_Scores.npy", CoherenceCalibrator().compute_scores(X_test_pure_2_1))
    np.save(out_dir / "X_test_pure_2_1_features.npy", X_test_pure_2_1) #Sauvegarde pour methode_MachineLearning

    print("\n[*] Extraction des caractéristiques sur les données PURES (Zone 2.2 divisée)...")
    X_test_pure_2_2_parts = []
    for i, loader_part in enumerate(loaders_2_2_parts):
        zone_index = i + 1
        anomaly_name = f"Zone_2_2_Part_{zone_index}"
        X_test_pure_part = extract_features_from_loader(loader_part, desc=f"Extraction Pure {anomaly_name}")
        X_test_pure_2_2_parts.append(X_test_pure_part)
        
    # Concaténation pour sauvegarder une Pure_2_2 globale (compatibilité avec plot_metrics et SVDD)
    X_test_pure_2_2 = np.vstack(X_test_pure_2_2_parts)
    print("\n[*] Calcul et sauvegarde des scores pour l'ensemble des données PURES 2.2...")
    np.save(out_dir / "Pure_2_2_Depth_Scores.npy", DepthCalibrator().load_and_score(X_test_pure_2_2, calib_dir))
    np.save(out_dir / "Pure_2_2_Coherence_Scores.npy", CoherenceCalibrator().compute_scores(X_test_pure_2_2))
    np.save(out_dir / "X_test_pure_2_2_features.npy", X_test_pure_2_2)

    # ==============================================================================
    # B. INJECTION ET ÉVALUATION DES ANOMALIES (H1_test)
    # ==============================================================================
    print("\n[*] Génération des anomalies (Crosstalk et Gain testés distinctement sur chaque sous-zone)...")
    
    # 1. Instanciation avec les nouveaux paramètres
    # Pour "delta" : par exemple une amplitude à -30 dB et une phase concentrée autour de 45°
    delta_generator = SyntheticParameterGenerator(
        mean_db=-15.0,                # Niveau typique de cross-talk cité dans l'article
        std_dev_amp=0.01,               # Légère variation
        phase_mean_rad=0.0,           # Peu importe si le kappa est à 0
        phase_concentration=1e-5       # Kappa = 0 donne une phase aléatoire uniforme (typiques des bruits de couplage)
    )

    # Pour "g" : par exemple une amplitude légèrement différente (-25 dB) 
    # ou une dispersion de phase plus large (kappa plus faible) pour simuler un comportement différent
    g_generator = SyntheticParameterGenerator(
        mean_db=0.0,                  # Le gain est proche de 1 (0 dB)
        std_dev_amp=0.01,             # Faible variation pour que |g|^4 reste < 0.5 dB
        phase_mean_rad=0.0,           # Pas de déphasage massif par défaut
        phase_concentration=10.0
    )

    # Utilisation d'une seed fixe pour la reproductibilité entre les méthodes
    anomaly_seed = cfg.get("anomaly_seed", 1234)

    # 2. Génération des valeurs
    nombre_echantillons = 3

    delta_values = delta_generator(num_samples=nombre_echantillons, seed=anomaly_seed)
    g_values = g_generator(num_samples=nombre_echantillons, seed=anomaly_seed + 1)
    
    
    final_anomaly_definitions = []
    for i in range(3):
        crosstalk_anomaly = Crosstalk(delta=delta_values[i].item())
        gain_anomaly = ChannelGainImbalance(g=g_values[i].item())
        final_anomaly_definitions.extend([crosstalk_anomaly, gain_anomaly])
        print(f"   - Anomalies pour sous-zone {i+1}: Crosstalk (delta={crosstalk_anomaly.delta}) et Gain (g={gain_anomaly.g})")
    
    final_loaders_to_test = []
    for loader in loaders_2_2_parts:
        final_loaders_to_test.extend([loader, loader])

    for i, (loader_part, anomaly) in enumerate(zip(final_loaders_to_test, final_anomaly_definitions)):
        zone_index = (i // 2) + 1
        anomaly_type = anomaly.__class__.__name__
        anomaly_name = f"Zone_2_2_Part_{zone_index}"
        anomaly_log_name = f"{anomaly_name}_{anomaly_type}"
        
        print(f"\n[*] Injection de l'anomalie sur {anomaly_name} : {anomaly_type} ({anomaly.name})")
        X_test_h1 = extract_anomalous_features(loader_part, anomaly)
        
        # Calcul des scores d'anomalies
        depth_scores = DepthCalibrator().load_and_score(X_test_h1, calib_dir)
        coh_scores = CoherenceCalibrator().compute_scores(X_test_h1)
        
        # Sauvegarde
        np.save(out_dir / f"{anomaly_log_name}_Depth_Scores.npy", depth_scores)
        np.save(out_dir / f"{anomaly_log_name}_Coherence_Scores.npy", coh_scores)
        np.save(out_dir / f"X_{anomaly_log_name}_features.npy", X_test_h1)
        print(f"[+] Scores sauvegardés pour l'anomalie : {anomaly_log_name}")