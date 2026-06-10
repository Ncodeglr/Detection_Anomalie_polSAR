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
from H0 import DepthCalibrator

if __name__ == "__main__":
    #1. Setup et chargement
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))
    
    #Résolution absolue du chemin des données par rapport à la racine du projet
    trainpath = Path(cfg["data"]["dataset"]["trainpath"])
    if not trainpath.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        cfg["data"]["dataset"]["trainpath"] = str((repo_root / trainpath).resolve())

    print("\n[*] Chargement des données saines (H0)...")
    loaders_dict = azimut_split(cfg, use_cuda=False)
    train_loader, valid_loader, test_loader = loaders_dict["loader1_splits"]
    
    X_train = extract_features_from_loader(train_loader, desc="Extraction features Train (Zone 1)")
    X_valid = extract_features_from_loader(valid_loader, desc="Extraction features Valid (Zone 1)")
    X_test  = extract_features_from_loader(test_loader, desc="Extraction features Test (Zone 1)")

    #Combinaison des 3 loaders de la Zone 1
    X_zone1 = np.vstack([X_train, X_valid, X_test])
    print(f"[*] Taille totale des données saines (Zone 1) : {X_zone1.shape}")

    #2. Dossier de sortie
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("data_calibration") / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    #Sauvegarde de la matrice des features pour les méthodes Machine Learning
    np.save(out_dir / "X_zone1_features.npy", X_zone1)

    #3. Calibration sur la zone 1 (H0)
    print("\n[*] Exécution du DepthCalibrator (Stahel-Donoho)...")
    depth_calib = DepthCalibrator(dim=X_zone1.shape[1])
    d_scores = depth_calib.calibrate(X_zone1)
    depth_calib.save(out_dir, d_scores)

    print(f"\n[+] Calibration H0 terminée avec succès. Résultats sauvés dans '{out_dir}/'")