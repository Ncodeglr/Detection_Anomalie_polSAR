import os
import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from svdd_custom import train_svdd_complex_simple2, score_svdd_batched

def load_latest_features(base_dir: Path, prefix: str):
    """ Trouve le run le plus récent et charge le fichier .npy correspondant. """
    if not base_dir.exists() or not list(base_dir.glob("run_*")):
        print(f"[!] ERREUR: Le dossier {base_dir} est vide. Lancez les scripts de methode_Classique d'abord.")
        sys.exit(1)
        
    latest_run = max([d for d in base_dir.glob("run_*") if d.is_dir()], key=os.path.getmtime)
    
    #Trouver le fichier correspondant au préfixe
    files = list(latest_run.glob(f"{prefix}*.npy"))
    if not files:
        raise FileNotFoundError(f"Fichier {prefix}*.npy non trouvé dans {latest_run}")
        
    return np.load(files[0])

def main():
    print("\n[*] Chargement des matrices de caractéristiques (16D)...")
    calib_base = Path("../methode_Classique/data_calibration")
    test_base = Path("../methode_Classique/test_results")
    
    X_train = load_latest_features(calib_base, "X_train_features")
    X_valid = load_latest_features(calib_base, "X_valid_features")
    X_pure_21 = load_latest_features(test_base, "X_test_pure_2_1_features")
    X_pure_22 = load_latest_features(test_base, "X_test_pure_2_2_features")
    
    print("[*] Normalisation des données (StandardScaler)...")
    scaler = StandardScaler()
    
    #On apprend la normalisation uniquement sur les données saines d'entraînement
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_pure_21 = scaler.transform(X_pure_21)
    X_pure_22 = scaler.transform(X_pure_22)
    
    # --- SOUS-ÉCHANTILLONNAGE POUR L'ENTRAÎNEMENT ---
    # CVXOPT a une complexité cubique O(N^3). Résoudre une matrice 10000x10000 nécessite beaucoup de RAM et de temps. On sélectionne un sous-ensemble aléatoire pour l'entraînement.
    
    N_TRAIN = 5000  # Nombre maximum de patchs pour l'entraînement SVDD
    np.random.seed(42)
    print("Nombre de patchs d'entraînement avant sous-échantillonnage:", X_train.shape[0])
    indices = np.random.choice(X_train.shape[0], min(N_TRAIN, X_train.shape[0]), replace=False)
    X_train_sub = X_train[indices]
    
    print(f"[*] Entraînement du modèle SVDD (N={X_train_sub.shape[0]} patchs)...")
    #On utilise le kernel RBF normal car vos features sont déjà réelles (Partie Réelle, Imaginaire et Spans séparés)
    svdd_model = train_svdd_complex_simple2(X_train_sub, C=0.5, kernel='rbf', gamma=0.5, verbose=True)
    
    print("\n[*] Calcul des scores de distance...")
    #On évalue sur le set de Validation pour trouver le seuil PFA robuste
    print("Type X_train:", type(X_train), "Shape:", X_train.shape)
    print("Type X_valid:", type(X_valid), "Shape:", X_valid.shape)
    print("Type X_pure_21:", type(X_pure_21), "Shape:", X_pure_21.shape)
    print("Type X_pure_22:", type(X_pure_22), "Shape:", X_pure_22.shape)
    
    scores_train = score_svdd_batched(svdd_model, X_train) #Zone 1 train
    scores_valid = score_svdd_batched(svdd_model, X_valid) #Zone 1 valid
    scores_z21 = score_svdd_batched(svdd_model, X_pure_21)
    scores_z22 = score_svdd_batched(svdd_model, X_pure_22) # Ajout pour l'histogramme

    print("\n[*] Résumé des scores calculés :")
    print(f"[*] Scores calculés : Train={len(scores_train)}, Valid={len(scores_valid)}, Zone 2.1={len(scores_z21)}, Zone 2.2={len(scores_z22)}")

    
    #Seuil pour 5% de fausses alarmes (Les distances élevées sont des anomalies)
    pfa = 0.05
    threshold = np.percentile(scores_valid, 100 * (1 - pfa))
    print(f"\n[*] Seuil pour PFA={pfa*100:.1f}% : Threshold = {threshold:.4f}")



    #Dossier de sauvegarde pour les graphiques
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("ml_results") / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- AFFICHAGE DES RÉSULTATS ---
    print("\n" + "-" * 65)
    print(f"🔹 TEST : Support Vector Data Description (SVDD)")
    
    #Zone 2.1 (Vérification Fausse Alarme)
    total = len(scores_z21)
    rejetes = int(np.sum(scores_z21 > threshold))
    acceptes = total - rejetes
    print(f"\n   [Zone 2.1 - Saine]")
    print(f"   ↳ Acceptées : {acceptes:5d} / {total} ({100.0*acceptes/total:6.2f}%) | ✅ Vrais Négatifs")
    print(f"   ↳ Rejetées  : {rejetes:5d} / {total} ({100.0*rejetes/total:6.2f}%) | ❌ Fausses Alarmes")
    
    #Zone 2.2 (Saine globale)
    total_22 = len(scores_z22)
    rejetes_22 = int(np.sum(scores_z22 > threshold))
    acceptes_22 = total_22 - rejetes_22
    print(f"\n   [Zone 2.2 - Saine (Pure Globale)]")
    print(f"   ↳ Acceptées : {acceptes_22:5d} / {total_22} ({100.0*acceptes_22/total_22:6.2f}%) | ✅ Vrais Négatifs")
    print(f"   ↳ Rejetées  : {rejetes_22:5d} / {total_22} ({100.0*rejetes_22/total_22:6.2f}%) | ❌ Fausses Alarmes")

    #Recherche des anomalies dynamiquement dans le dossier de test_results de Methode_Classique pour les évaluer
    latest_test_run = max([d for d in test_base.glob("run_*") if d.is_dir()], key=os.path.getmtime)
    anomaly_files = sorted([f for f in latest_test_run.glob("X_*_features.npy") if "pure" not in f.name])
    print(len(anomaly_files), "Fichiers d'anomalies détectés dans le dossier", latest_test_run)
    print("Anomalies :", [f.name for f in anomaly_files])


    import json
    anomalies_info = {}
    info_path = latest_test_run / "anomalies_info.json"
    if info_path.exists():
        with open(info_path, "r") as f:
            anomalies_info = json.load(f)

    # --- ÉVALUATION DES ANOMALIES ---
    anomaly_scores_dict = {}
    for a_file in anomaly_files:
        anomaly_name = a_file.name.replace("X_", "").replace("_features.npy", "")
        X_anom = np.load(a_file)
        
        #Normaliser aussi les données de test avec le même scaler
        X_anom = scaler.transform(X_anom)
        
        scores_anom = score_svdd_batched(svdd_model, X_anom)
        anomaly_scores_dict[anomaly_name] = scores_anom
        
        total = len(scores_anom)
        rejetes = int(np.sum(scores_anom > threshold))
        acceptes = total - rejetes
        
        # --- Calcul AUC-ROC ---
        y_true = np.concatenate([np.zeros_like(scores_z22), np.ones_like(scores_anom)])
        y_scores_concat = np.concatenate([scores_z22, scores_anom])
        auc_roc = roc_auc_score(y_true, y_scores_concat)

        delta_str = ""
        if anomaly_name in anomalies_info:
            try:
                delta_cplx = complex(anomalies_info[anomaly_name])
                amp = abs(delta_cplx)
                phase_deg = np.angle(delta_cplx, deg=True)
                delta_str = f" | delta: {delta_cplx:.4g} (Amp: {amp:.4f}, Phase: {phase_deg:.1f}°)"
            except ValueError:
                delta_str = f" | delta: {anomalies_info[anomaly_name]}"

        print(f"\n   [Zone 2.2 - Anomalie : {anomaly_name}{delta_str}]")
        print(f"   ↳ Acceptées : {acceptes:5d} / {total} ({100.0*acceptes/total:6.2f}%) | ❌ Faux Négatifs")
        print(f"   ↳ Rejetées  : {rejetes:5d} / {total} ({100.0*rejetes/total:6.2f}%) | 🚨 Vrais Positifs (Détection)")
        print(f"   ↳ AUC-ROC   : {auc_roc:.4f}")
        
    print("-" * 65)

if __name__ == "__main__":
    main()