import sys
import os
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import itertools

# Import de vos scripts SVDD et du chargeur de features
from svdd_custom import train_svdd_complex_simple2, score_svdd_batched
from train_svdd import load_latest_features

def main():
    print("\n[*] Lancement de l'optimisation des hyperparamètres SVDD...")
    
    calib_base = Path("../methode_Classique/calibration_results")
    test_base = Path("../methode_Classique/test_results")
    
    X_zone1 = load_latest_features(calib_base, "X_zone1_features")
    X_pure_21 = load_latest_features(test_base, "X_test_pure_2_1")
    
    # Charger dynamiquement toutes les anomalies
    latest_test_run = max([d for d in test_base.glob("run_*") if d.is_dir()], key=os.path.getmtime)
    anomaly_files = [f for f in latest_test_run.glob("X_*_features.npy") if "pure" not in f.name]
    
    anomalies_dict = {}
    for a_file in anomaly_files:
        anomaly_name = a_file.name.replace("X_", "").replace("_features.npy", "")
        anomalies_dict[anomaly_name] = np.load(a_file)
        
    print(f"[*] Anomalies trouvées pour l'évaluation : {list(anomalies_dict.keys())}")

    print("[*] Normalisation des données (StandardScaler)...")
    scaler = StandardScaler()
    X_zone1 = scaler.fit_transform(X_zone1)
    X_pure_21 = scaler.transform(X_pure_21)
    for name in anomalies_dict:
        anomalies_dict[name] = scaler.transform(anomalies_dict[name])
        
    # --- SOUS-ÉCHANTILLONNAGE POUR L'OPTIMISATION ---
    # On réduit N_TRAIN pour que le Grid Search tourne rapidement
    N_TRAIN = 2500
    np.random.seed(42)
    indices = np.random.choice(X_zone1.shape[0], min(N_TRAIN, X_zone1.shape[0]), replace=False)
    X_train_sub = X_zone1[indices]
    
    # --- GRILLE D'HYPERPARAMÈTRES ---
    C_values = [0.05, 0.1, 0.5]
    gamma_values = [0.01, 0.1, 0.5, 1.0, 5.0]
    
    pfa_target = 0.05
    results = []
    best_score = -1.0
    best_params = {}

    print(f"\n[*] Début du Grid Search ({len(C_values) * len(gamma_values)} combinaisons)...")
    print(f"{'C':<6} | {'gamma':<6} | {'FPR (Zone 2.1)':<16} | {'TPR Moyen (Détections)':<20}")
    print("-" * 57)
    
    for C, gamma in itertools.product(C_values, gamma_values):
        try:
            # verbose=False pour ne pas polluer le terminal pendant la boucle
            model = train_svdd_complex_simple2(X_train_sub, C=C, kernel='rbf', gamma=gamma, verbose=False)
            
            # Calcul du seuil strict sur Zone 1 (PFA = 5%)
            scores_z1 = score_svdd_batched(model, X_zone1)
            threshold = np.percentile(scores_z1, 100 * (1 - pfa_target))
            
            # Évaluation du Taux de Fausse Alarme (FPR) sur la Zone 2.1 saine
            scores_z21 = score_svdd_batched(model, X_pure_21)
            fpr_z21 = np.mean(scores_z21 > threshold)
            
            # Évaluation du Taux de Vrais Positifs (TPR) moyen sur les anomalies
            tpr_list = []
            for anom_X in anomalies_dict.values():
                scores_anom = score_svdd_batched(model, anom_X)
                tpr_list.append(np.mean(scores_anom > threshold))
            mean_tpr = np.mean(tpr_list)
            
            print(f"{C:<6.2f} | {gamma:<6.2f} | {fpr_z21:>15.2%} | {mean_tpr:>19.2%}")
            
            results.append({'C': C, 'gamma': gamma, 'FPR': fpr_z21, 'TPR': mean_tpr})
            
            # On cherche à maximiser le TPR (La détection)
            if mean_tpr > best_score:
                best_score = mean_tpr
                best_params = {'C': C, 'gamma': gamma}
                
        except Exception as e:
            print(f"{C:<6.2f} | {gamma:<6.2f} | ERREUR: {str(e)}")

    print("-" * 57)
    print(f"\n[+] Meilleurs hyperparamètres trouvés : C = {best_params['C']}, gamma = {best_params['gamma']}")
    print(f"[+] Avec un Taux de Détection Moyen de  : {best_score:.2%}")
    print("\nVous pouvez maintenant mettre à jour ces valeurs manuellement dans train_svdd.py !")

if __name__ == "__main__":
    main()