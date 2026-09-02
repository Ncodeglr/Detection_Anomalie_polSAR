import sys
import os
import copy
import datetime
import numpy as np
from pathlib import Path

#Ajout des chemins pour importer vos modules cvnn et l'extracteur local
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "cvnn", "src"))
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared_setup import setup_experiment_env
from feature_extraction import extract_features_from_loader
from H0 import DepthCalibrator

if __name__ == "__main__":
    
    print("\n[*] Chargement des données saines (H0)...")
    _, _, _, loaders = setup_experiment_env(sys.argv, __file__, force_cpu=True)
    train_loader, valid_loader, _ = loaders
    
    X_train = extract_features_from_loader(train_loader, desc="Extraction features Train (Zone 1)")
    X_valid = extract_features_from_loader(valid_loader, desc="Extraction features Valid (Zone 1)")

    print(f"[*] Taille Train : {X_train.shape} | Taille Valid : {X_valid.shape}")

    #2. Dossier de sortie
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("data_calibration") / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    #Sauvegarde des matrices des features séparées pour les méthodes Machine Learning
    np.save(out_dir / "X_train_features.npy", X_train)
    np.save(out_dir / "X_valid_features.npy", X_valid)

    #3. Apprentissage (Fit) sur le Train
    print("\n[*] Apprentissage de la normalité (Stahel-Donoho) sur le Train...")
    depth_calib = DepthCalibrator(dim=X_train.shape[1])
    train_scores = depth_calib.calibrate(X_train)
    depth_calib.save(out_dir, train_scores) #Sauvegarde les matrices et Train_Depth_Scores.npy
    print("Shape des scores sur le Train :", train_scores.shape)
    print("Exemples de scores sur le Train :")
    for i in range(5):
        print(f"Score Train [{i}] : {train_scores[i]}")

    #4. Calibration du seuil sur le Valid
    print("\n[*] Évaluation sur le Valid pour calibration future du seuil...")
    valid_scores = depth_calib.score_batched(X_valid)
    print("Shape des scores sur le Valid :", valid_scores.shape)
    print("Exemples de scores sur le Valid :")
    for i in range(5):
        print(f"Score Valid [{i}] : {valid_scores[i]}")
    np.save(out_dir / "Valid_Depth_Scores.npy", valid_scores)

    print(f"\n[+] Calibration H0 terminée avec succès. Résultats sauvés dans '{out_dir}/'")