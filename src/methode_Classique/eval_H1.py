import sys
import os
import torch
import datetime
import numpy as np
from pathlib import Path
from tqdm import tqdm

#Ajout des chemins pour importer vos modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "cvnn", "src"))
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared_setup import setup_experiment_env, get_test_loaders, get_shared_anomaly_generator
from feature_extraction import extract_features_from_loader
from H0 import DepthCalibrator
from H1 import Crosstalk


if __name__ == "__main__":
    print("\n[*] Récupération des statistiques depuis la Zone 1...")
    cfg, config_base, _, _ = setup_experiment_env(sys.argv, __file__, force_cpu=True)

    print("\n[*] Chargement des zones 2.1 et 2.2 pour les tests...")
    loader_test_2_1, loaders_2_2_parts = get_test_loaders(config_base, use_cuda=False)
    print(f"   -> Zone 2.1 (Saine)    : {len(loader_test_2_1.dataset)} patchs")
    for i, loader in enumerate(loaders_2_2_parts):
        print(f"   -> Zone 2.2 (Part {i+1}) : {len(loader.dataset)} patchs")

    base_calib_dir = Path("data_calibration")
    if not base_calib_dir.exists() or not list(base_calib_dir.glob("run_*")):
        print("[!] ERREUR: Aucun dossier de calibration H0 trouvé. Lancez H0.py en premier.")
        sys.exit(1)

    #Récupérer le dossier de calibration le plus récent
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
    np.save(out_dir / "X_test_pure_2_1_features.npy", X_test_pure_2_1) #Sauvegarde pour methode_MachineLearning

    print("\n[*] Extraction des caractéristiques sur les données PURES (Zone 2.2 divisée)...")
    X_test_pure_2_2_parts = []
    for i, loader_part in enumerate(loaders_2_2_parts):
        zone_index = i + 1
        anomaly_name = f"Zone_2_2_Part_{zone_index}"
        X_test_pure_part = extract_features_from_loader(loader_part, desc=f"Extraction Pure {anomaly_name}")
        X_test_pure_2_2_parts.append(X_test_pure_part)
        
    #Concaténation pour sauvegarder une Pure_2_2 globale
    X_test_pure_2_2 = np.vstack(X_test_pure_2_2_parts)
    print("\n[*] Calcul et sauvegarde des scores pour l'ensemble des données PURES 2.2...")
    np.save(out_dir / "Pure_2_2_Depth_Scores.npy", DepthCalibrator().load_and_score(X_test_pure_2_2, calib_dir))
    np.save(out_dir / "X_test_pure_2_2_features.npy", X_test_pure_2_2)

    # ==============================================================================
    # B. INJECTION ET ÉVALUATION DES ANOMALIES (H1_test)
    # ==============================================================================
    print("\n[*] Génération des anomalies (Crosstalk et Gain testés distinctement sur chaque sous-zone)...")
    
    #2. Génération des valeurs de delta pour les 3 sous-zones de la Zone 2.2
    delta_generator, anomaly_seed = get_shared_anomaly_generator(cfg)
    delta_values = delta_generator(num_samples=3, seed=anomaly_seed)
    
    final_anomaly_definitions = []
    for i in range(3):
        crosstalk_anomaly = Crosstalk(delta=delta_values[i].item())
        final_anomaly_definitions.append(crosstalk_anomaly)
        print(f"   - Anomalies pour sous-zone {i+1}: Crosstalk (delta={crosstalk_anomaly.delta})")

    anomalies_info = {}
    for i, (loader_part, anomaly) in enumerate(zip(loaders_2_2_parts, final_anomaly_definitions)):
        zone_index = i + 1
        anomaly_type = anomaly.__class__.__name__
        anomaly_name = f"Zone_2_2_Part_{zone_index}"
        anomaly_log_name = f"{anomaly_name}_{anomaly_type}"
        
        anomalies_info[anomaly_log_name] = str(anomaly.delta)

        print(f"\n[*] Injection de l'anomalie sur {anomaly_name} : {anomaly_type} ({anomaly.name})")
        X_test_h1 = extract_features_from_loader(
            loader_part,
            desc=f"Injection {anomaly_type} sur {anomaly_name}",
            anomaly_generator=anomaly
        )
        
        #Calcul des scores d'anomalies
        depth_scores = DepthCalibrator().load_and_score(X_test_h1, calib_dir)
        
        #Sauvegarde
        np.save(out_dir / f"{anomaly_log_name}_Depth_Scores.npy", depth_scores)
        np.save(out_dir / f"X_{anomaly_log_name}_features.npy", X_test_h1)
        print(f"[+] Scores sauvegardés pour l'anomalie : {anomaly_log_name}")

    import json
    with open(out_dir / "anomalies_info.json", "w") as f:
        json.dump(anomalies_info, f, indent=4)