import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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
    
    for print_name, metric in metrics_map.items():
        # 1. Charger les scores de la Zone 1 pour définir le seuil
        scores_z1 = np.load(calib_dir / f"Train_{metric}_Scores.npy")
        y_score_z1 = get_anomaly_score(scores_z1, metric)
        
        # Définition du seuil : On garde le percentile correspondant au PFA toléré (ex: 95% acceptés)
        threshold = np.percentile(y_score_z1, 100 * (1 - pfa))
        
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
            print(f"\n   [Zone 2.2 - Anomalie : {anomaly}]")
            print(f"   ↳ Acceptées : {acceptes:5d} / {total} ({100.0*acceptes/total:6.2f}%) | ❌ Faux Négatifs")
            print(f"   ↳ Rejetées  : {rejetes:5d} / {total} ({100.0*rejetes/total:6.2f}%) | 🚨 Vrais Positifs (Détection)")
    print("-" * 65)

def plot_histograms(metric: str, calib_dir: Path, test_dir: Path, anomalies: list):
    """ Affiche les distributions des scores pour vérifier la stabilité spatiale et l'impact des anomalies. """
    plt.figure(figsize=(10, 6))
    
    # 1. Charger les données
    scores_z1 = np.load(calib_dir / f"Train_{metric}_Scores.npy")
    scores_z21 = np.load(test_dir / f"Pure_2_1_{metric}_Scores.npy")
    scores_z22 = np.load(test_dir / f"Pure_2_2_{metric}_Scores.npy")
    
    # 2. Tracer les zones pures (Saines)
    kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=50)
    plt.hist(scores_z1, label='Zone 1 (Train)', color='blue', **kwargs)
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
    base_calib_dir = Path("calibration_results")
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