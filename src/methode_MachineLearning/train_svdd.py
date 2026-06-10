import os
import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Import de votre script SVDD personnalisé
from svdd_custom import train_svdd_complex_simple2, score_svdd_batched

def load_latest_features(base_dir: Path, prefix: str):
    """ Trouve le run le plus récent et charge le fichier .npy correspondant. """
    if not base_dir.exists() or not list(base_dir.glob("run_*")):
        print(f"[!] ERREUR: Le dossier {base_dir} est vide. Lancez les scripts de methode_Classique d'abord.")
        sys.exit(1)
        
    latest_run = max([d for d in base_dir.glob("run_*") if d.is_dir()], key=os.path.getmtime)
    
    # Trouver le fichier correspondant au préfixe
    files = list(latest_run.glob(f"{prefix}*.npy"))
    if not files:
        raise FileNotFoundError(f"Fichier {prefix}*.npy non trouvé dans {latest_run}")
        
    return np.load(files[0])

def main():
    print("\n[*] Chargement des matrices de caractéristiques (16D)...")
    calib_base = Path("../methode_Classique/data_calibration")
    test_base = Path("../methode_Classique/test_results")
    
    X_zone1 = load_latest_features(calib_base, "X_zone1_features")
    X_pure_21 = load_latest_features(test_base, "X_test_pure_2_1")
    X_pure_22 = load_latest_features(test_base, "X_test_pure_2_2")
    
    print("[*] Normalisation des données (StandardScaler)...")
    scaler = StandardScaler()
    # On "apprend" la normalisation uniquement sur les données saines (Zone 1)
    X_zone1 = scaler.fit_transform(X_zone1)
    X_pure_21 = scaler.transform(X_pure_21)
    X_pure_22 = scaler.transform(X_pure_22)
    
    # --- SOUS-ÉCHANTILLONNAGE POUR L'ENTRAÎNEMENT ---
    # CVXOPT a une complexité cubique O(N^3). Résoudre une matrice 10000x10000 
    # nécessite beaucoup de RAM et de temps. On sélectionne un sous-ensemble aléatoire pour l'entraînement.
    N_TRAIN = 2500
    np.random.seed(42)
    indices = np.random.choice(X_zone1.shape[0], min(N_TRAIN, X_zone1.shape[0]), replace=False)
    X_train_sub = X_zone1[indices]
    
    print(f"[*] Entraînement du modèle SVDD (N={X_train_sub.shape[0]} patchs)...")
    # On utilise le kernel RBF normal car vos features sont déjà réelles (Partie Réelle, Imaginaire et Spans séparés)
    svdd_model = train_svdd_complex_simple2(X_train_sub, C=0.5, kernel='rbf', gamma=0.5, verbose=True)
    
    print("\n[*] Calcul des scores de distance...")
    # On évalue sur toute la Zone 1 pour trouver le seuil PFA
    scores_z1 = score_svdd_batched(svdd_model, X_zone1)
    scores_z21 = score_svdd_batched(svdd_model, X_pure_21)
    scores_z22 = score_svdd_batched(svdd_model, X_pure_22) # Ajout pour l'histogramme
    
    # Seuil pour 5% de fausses alarmes (Les distances élevées sont des anomalies)
    pfa = 0.05
    threshold = np.percentile(scores_z1, 100 * (1 - pfa))
    
    # Dossier de sauvegarde pour les graphiques
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("ml_results") / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- AFFICHAGE DES RÉSULTATS ---
    print("\n" + "-" * 65)
    print(f"🔹 TEST : Support Vector Data Description (SVDD)")
    
    # Zone 2.1 (Vérification Fausse Alarme)
    total = len(scores_z21)
    rejetes = int(np.sum(scores_z21 > threshold))
    acceptes = total - rejetes
    print(f"\n   [Zone 2.1 - Saine]")
    print(f"   ↳ Acceptées : {acceptes:5d} / {total} ({100.0*acceptes/total:6.2f}%) | ✅ Vrais Négatifs")
    print(f"   ↳ Rejetées  : {rejetes:5d} / {total} ({100.0*rejetes/total:6.2f}%) | ❌ Fausses Alarmes")
    
    # Recherche des anomalies dynamiquement
    latest_test_run = max([d for d in test_base.glob("run_*") if d.is_dir()], key=os.path.getmtime)
    anomaly_files = sorted([f for f in latest_test_run.glob("X_*_features.npy") if "pure" not in f.name])
    
    import json
    anomalies_info = {}
    info_path = latest_test_run / "anomalies_info.json"
    if info_path.exists():
        with open(info_path, "r") as f:
            anomalies_info = json.load(f)

    anomaly_scores_dict = {}
    for a_file in anomaly_files:
        anomaly_name = a_file.name.replace("X_", "").replace("_features.npy", "")
        X_anom = np.load(a_file)
        
        # Ne pas oublier de normaliser aussi les données de test avec le même scaler !
        X_anom = scaler.transform(X_anom)
        
        scores_anom = score_svdd_batched(svdd_model, X_anom)
        anomaly_scores_dict[anomaly_name] = scores_anom
        
        total = len(scores_anom)
        rejetes = int(np.sum(scores_anom > threshold))
        acceptes = total - rejetes
        
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
        
    print("-" * 65)
    
    # --- GÉNÉRATION DE L'HISTOGRAMME ---
    print("\n[*] Génération de l'histogramme des densités SVDD...")
    plt.figure(figsize=(10, 6))
    kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=50)
    
    plt.hist(scores_z1, label='Zone 1 (Train)', color='blue', **kwargs)
    plt.hist(scores_z21, label='Zone 2.1 (Saine)', color='cyan', **kwargs)
    plt.hist(scores_z22, label='Zone 2.2 (Pure)', color='green', **kwargs)
    
    cmap = plt.get_cmap('tab10')
    for i, (anom_name, anom_scores) in enumerate(anomaly_scores_dict.items()):
        clean_label = anom_name.replace("Zone_2_2_", "")
        plt.hist(anom_scores, label=f'{clean_label}', color=cmap(i % 10), histtype='step', linewidth=2, density=True, bins=50)
        
    plt.axvline(x=threshold, color='black', linestyle='--', linewidth=2, label=f'Seuil (PFA={pfa*100:.0f}%)')
    plt.title("Distribution des scores SVDD (Distances au centre)")
    plt.xlabel("Score d'anomalie (Distance²)")
    plt.ylabel("Densité")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "Distributions_SVDD.png")
    plt.close()
    print(f"[+] Graphique sauvegardé dans '{out_dir}/Distributions_SVDD.png'")

if __name__ == "__main__":
    main()