import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

def get_anomaly_score(scores: np.ndarray, metric_name: str) -> np.ndarray:
    """ 
    Convertit les métriques brutes en 'Scores d'Anomalie' (Plus c'est élevé, plus c'est anormal).
    """
    if metric_name in ["Depth"]:
        return 1.0 - scores  # Profondeur faible = anomalie
    return scores

def print_evaluation_ratios(calib_dir: Path, test_dir: Path, anomalies: list, pfa: float = 0.05):
    """ Calcule et affiche les ratios d'acceptation/rejet sur les Zones 2.1 et 2.2. """
    metrics_map = {"Depth Projection": "Depth"}
    
    anomalies_info = {}
    info_path = test_dir / "anomalies_info.json"
    if info_path.exists():
        with open(info_path, "r") as f:
            anomalies_info = json.load(f)

    for print_name, metric in metrics_map.items():
        # 1. Charger les scores de Validation pour définir le seuil de manière robuste
        scores_valid = np.load(calib_dir / f"Valid_{metric}_Scores.npy")
        y_score_valid = get_anomaly_score(scores_valid, metric)
        
        # Définition du seuil : On garde le percentile correspondant au PFA toléré (ex: 95% acceptés)
        threshold = np.percentile(y_score_valid, 100 * (1 - pfa))
        
        print("-" * 65)
        print(f"🔹 TEST : {print_name} (Classique)")
        
        # --- Évaluation Zone 2.1 (Saine) ---
        scores_z21 = np.load(test_dir / f"Pure_2_1_{metric}_Scores.npy")
        y_score_z21 = get_anomaly_score(scores_z21, metric)
        total = len(y_score_z21)
        rejetes = int(np.sum(y_score_z21 > threshold))
        acceptes = total - rejetes
        print(f"\n   [Zone 2.1 - Saine]")
        print(f"   ↳ Acceptées : {acceptes:5d} / {total} ({100.0*acceptes/total:6.2f}%) | ✅ Vrais Négatifs")
        print(f"   ↳ Rejetées  : {rejetes:5d} / {total} ({100.0*rejetes/total:6.2f}%) | ❌ Fausses Alarmes")
        
        # --- Évaluation Zone 2.2 (Saine globale) ---
        scores_z22 = np.load(test_dir / f"Pure_2_2_{metric}_Scores.npy")
        y_score_z22 = get_anomaly_score(scores_z22, metric)
        total_22 = len(y_score_z22)
        rejetes_22 = int(np.sum(y_score_z22 > threshold))
        acceptes_22 = total_22 - rejetes_22
        print(f"\n   [Zone 2.2 - Saine (Pure Globale)]")
        print(f"   ↳ Acceptées : {acceptes_22:5d} / {total_22} ({100.0*acceptes_22/total_22:6.2f}%) | ✅ Vrais Négatifs")
        print(f"   ↳ Rejetées  : {rejetes_22:5d} / {total_22} ({100.0*rejetes_22/total_22:6.2f}%) | ❌ Fausses Alarmes")

        # --- Évaluation Zone 2.2 (Anomalies) ---
        for anomaly in anomalies:
            scores_h1 = np.load(test_dir / f"{anomaly}_{metric}_Scores.npy")
            y_score_h1 = get_anomaly_score(scores_h1, metric)
            total = len(y_score_h1)
            rejetes = int(np.sum(y_score_h1 > threshold))
            acceptes = total - rejetes
            
            # --- Calcul AUC-ROC ---
            # Classe 0 = Saine (Zone 2.2 pure), Classe 1 = Anomalie (Zone 2.2 corrompue)
            auc_str = "N/A (sklearn non installé)"
            if roc_auc_score is not None:
                y_true = np.concatenate([np.zeros_like(y_score_z22), np.ones_like(y_score_h1)])
                y_scores = np.concatenate([y_score_z22, y_score_h1])
                auc_roc = roc_auc_score(y_true, y_scores)
                auc_str = f"{auc_roc:.4f}"

            delta_str = ""
            if anomaly in anomalies_info:
                try:
                    delta_cplx = complex(anomalies_info[anomaly])
                    amp = abs(delta_cplx)
                    phase_deg = np.angle(delta_cplx, deg=True)
                    delta_str = f" | delta: {delta_cplx:.4g} (Amp: {amp:.4f}, Phase: {phase_deg:.1f}°)"
                except ValueError:
                    delta_str = f" | delta: {anomalies_info[anomaly]}"
                    
            print(f"\n   [Zone 2.2 - Anomalie : {anomaly}{delta_str}]")
            print(f"   ↳ Acceptées : {acceptes:5d} / {total} ({100.0*acceptes/total:6.2f}%) | ❌ Faux Négatifs")
            print(f"   ↳ Rejetées  : {rejetes:5d} / {total} ({100.0*rejetes/total:6.2f}%) | 🚨 Vrais Positifs (Détection)")
            print(f"   ↳ AUC-ROC   : {auc_str}")
    print("-" * 65)

def plot_histograms(metric: str, calib_dir: Path, test_dir: Path, anomalies: list):
    """ Affiche les distributions des scores pour vérifier la stabilité spatiale et l'impact des anomalies. """
    plt.figure(figsize=(10, 6))
    
    # 1. Charger les données
    scores_z1 = np.load(calib_dir / f"Train_{metric}_Scores.npy")
    scores_valid = np.load(calib_dir / f"Valid_{metric}_Scores.npy")
    scores_z21 = np.load(test_dir / f"Pure_2_1_{metric}_Scores.npy")
    scores_z22 = np.load(test_dir / f"Pure_2_2_{metric}_Scores.npy")
    
    # 2. Tracer les zones pures (Saines)
    kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=50)
    plt.hist(scores_z1, label='Zone 1 (Train)', color='blue', **kwargs)
    plt.hist(scores_valid, label='Zone 1 (Valid)', color='purple', **kwargs)
    plt.hist(scores_z21, label='Zone 2.1 (Valid PFA)', color='cyan', **kwargs)
    plt.hist(scores_z22, label='Zone 2.2 (Pure)', color='green', **kwargs)
    
    # 3. Tracer les anomalies (Corrompues)
    cmap = plt.get_cmap('tab10') # Utilisation d'une palette de 10 couleurs distinctes
    for i, anomaly in enumerate(anomalies):
        scores_h1 = np.load(test_dir / f"{anomaly}_{metric}_Scores.npy")
        # Nettoyage du nom pour la légende (ex: Zone_2_2_Part_1_Crosstalk -> Part_1_Crosstalk)
        clean_label = anomaly.replace("Zone_2_2_", "")
        plt.hist(scores_h1, label=f'{clean_label}', color=cmap(i % 10), histtype='step', linewidth=2, density=True, bins=50)
        
    plt.title(f"Distribution de la métrique : {metric}")
    plt.xlabel("Score Brut")
    plt.ylabel("Densité")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Déplace la légende à l'extérieur
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(test_dir / f"Distributions_{metric}.png")
    plt.close()

if __name__ == "__main__":
    base_calib_dir = Path("data_calibration")
    base_test_dir = Path("test_results")
    
    if not base_test_dir.exists() or not list(base_test_dir.glob("run_*")):
        print("[!] ERREUR: Aucun dossier test_results n'existe. Lancez H1.py d'abord.")
        import sys; sys.exit(1)
        
    # Sélection automatique des derniers dossiers générés
    calib_dir = max([d for d in base_calib_dir.glob("run_*") if d.is_dir()], key=os.path.getmtime)
    test_dir = max([d for d in base_test_dir.glob("run_*") if d.is_dir()], key=os.path.getmtime)
    
    print(f"[*] Utilisation du dossier de calibration : {calib_dir}")
    print(f"[*] Utilisation du dossier de test        : {test_dir}")
        
    # Détection automatique des anomalies testées en scannant les fichiers
    anomalies = []
    for f in sorted(test_dir.glob("*_Depth_Scores.npy")):
        name = f.name.replace("_Depth_Scores.npy", "")
        if not name.startswith("Pure"):
            anomalies.append(name)
            
    print(f"[*] Anomalies détectées : {anomalies}")
    
    # 0. Afficher les statistiques de rejet et de détection
    print("\n[*] Évaluation des performances (Seuil calculé sur Zone 1 pour PFA = 5%)...")
    print_evaluation_ratios(calib_dir, test_dir, anomalies, pfa=0.05)

    # 1. Tracer les histogrammes
    print("\n[*] Génération des histogrammes de distribution...")
    for metric in ["Depth"]:
        plot_histograms(metric, calib_dir, test_dir, anomalies)
        
    print(f"\n[+] Terminé ! Les graphiques ont été sauvegardés dans le dossier '{test_dir}/'.")