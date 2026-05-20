import sys
import os
import json
import copy
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import math

# Ajout des chemins pour importer les modules du projet
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "cvnn", "src"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cvnn.config import load_config
from cvnn.data import azimut_split, get_full_image_dataloader
from cvnn.models import LatentAutoEncoder
from cvnn.visualize import plot_latent_space, plot_reconstructions

from ood_detector import OOD_Detector
from anomalies import Crosstalk, ChannelGainImbalance
from synthetic_parameter_generator import SyntheticParameterGenerator

def evaluate_ood_system(model, loader1_train, loader1_valid, loader2_1, loaders_2_2_parts, anomaly_definitions, pfa_target=0.05, device="cpu", out_dir=Path(".")):
    """
    Entraîne le détecteur sur l'espace latent, calibre les seuils, et évalue les performances globales.
    """
    print("--- Initialisation du Détecteur OoD ---")
    ood_metrics = {"pfa_target": pfa_target}
    
    detector = OOD_Detector(model, device=device)
    
    # 1. Modélisation de l'espace latent normal
    print("1. Calcul de la distribution de Mahalanobis (sur Train Sain)...")
    detector.fit_mahalanobis(loader1_train)
    
    # 2. Calibration
    print(f"2. Calibration du seuil pour une PFA de {pfa_target*100}% (sur Valid Sain)...")
    thresh_recon, thresh_mah = detector.calibrate_thresholds(loader1_valid, pfa=pfa_target)
    print(f"   -> Seuil Reconstruction : {thresh_recon:.4f}")
    print(f"   -> Seuil Mahalanobis    : {thresh_mah:.4f}")
    ood_metrics["threshold_recon"] = float(thresh_recon)
    ood_metrics["threshold_mah"] = float(thresh_mah)
    
    # 3. Test sur les données saines de la zone 2.1 avec vérification de la PFA
    print("\n--- Évaluation sur Zone 2.1 (Pures) ---")
    preds_recon_sain, _, preds_mah_sain, _ = detector.detect(loader2_1)
    print(f"PFA empirique (Recon) : {np.mean(preds_recon_sain)*100:.2f}% (Cible: {pfa_target*100}%)")
    print(f"PFA empirique (Mahal) : {np.mean(preds_mah_sain)*100:.2f}%")
    
    ood_metrics["Zone_2_1_Sain"] = {
        "pfa_recon": float(np.mean(preds_recon_sain)),
        "pfa_mah": float(np.mean(preds_mah_sain))
    }
    
    ood_metrics["Anomalies"] = {}

    for i, (loader_part, anomaly_def) in enumerate(zip(loaders_2_2_parts, anomaly_definitions)):
        zone_index = i // 2 + 1
        anomaly_type = anomaly_def.__class__.__name__
        anomaly_name = f"Zone_2_2_Part_{zone_index}"
        anomaly_log_name = f"{anomaly_name}_{anomaly_type}"

        print(f"\n--- Évaluation sur {anomaly_name} avec anomalie {anomaly_type} ---")

        # A. Get baseline scores on the PURE data for this part
        print("   - Calcul des scores sur les données pures de la sous-zone...")
        preds_recon_pure, scores_recon_pure, preds_mah_pure, scores_mah_pure = detector.detect(loader_part)
        print(f"   -> PFA de référence (Recon): {np.mean(preds_recon_pure)*100:.2f}%")

        # B. Inject anomaly and get anomalous scores
        base_ds = loader_part.dataset
        while hasattr(base_ds, 'dataset') or hasattr(base_ds, 'base_dataset'):
            base_ds = getattr(base_ds, 'dataset', getattr(base_ds, 'base_dataset', base_ds))
        original_transform = getattr(base_ds, 'transform', None)

        if hasattr(original_transform, 'transforms'):
            new_transforms = []
            injected = False
            for t in original_transform.transforms:
                if t.__class__.__name__ == 'LogAmplitude':
                    new_transforms.append(anomaly_def)
                    injected = True
                new_transforms.append(t)
            if not injected:
                new_transforms.append(anomaly_def)
            base_ds.transform = original_transform.__class__(new_transforms)
        else:
            print(f"[!] Attention: transform n'est pas un Compose. L'anomalie sera appliquée en premier.")
            base_ds.transform = torch.nn.Sequential(anomaly_def, original_transform) if original_transform else anomaly_def

        print("   - Calcul des scores sur les données avec anomalie...")
        preds_recon_ano, scores_recon_ano, preds_mah_ano, scores_mah_ano = detector.detect(loader_part)

        # Restore transform
        base_ds.transform = original_transform

        # C. Calculate metrics
        print(f"   -> Taux de Détection (Recon): {np.mean(preds_recon_ano)*100:.2f}%")
        print(f"   -> Taux de Détection (Mahal): {np.mean(preds_mah_ano)*100:.2f}%")

        # Calcul de l'AUC-ROC (Comparaison Z2.2 Sain vs Z2.2 Anomalie pour cette partie)
        with torch.no_grad():
            y_true = np.concatenate([np.zeros_like(scores_recon_pure), np.ones_like(scores_recon_ano)])
            
            auc_recon = roc_auc_score(y_true, np.concatenate([scores_recon_pure, scores_recon_ano]))
            auc_mah = roc_auc_score(y_true, np.concatenate([scores_mah_pure, scores_mah_ano]))
            
            print(f"   -> Score AUC-ROC (Recon) : {auc_recon:.4f} (1.0 = Parfait)")
            print(f"   -> Score AUC-ROC (Mahal) : {auc_mah:.4f}")
            
            ood_metrics["Anomalies"][anomaly_log_name] = {
                "detection_rate_recon": float(np.mean(preds_recon_ano)),
                "detection_rate_mah": float(np.mean(preds_mah_ano)),
                "auc_roc_recon": float(auc_recon),
                "auc_roc_mah": float(auc_mah)
            }
    
        # ---Visualisation des Reconstructions de l'anomalie ---
        model.eval()
        with torch.no_grad():
            for batch in loader_part:
                x_batch = batch[0] if isinstance(batch, (list, tuple)) else batch
                x_batch = x_batch.to(device)
                outputs = model(x_batch)
                break  # On ne prend qu'un seul batch pour générer l'image
                
        fig_recon = plot_reconstructions(
            inputs=x_batch.cpu(),
            outputs=outputs.cpu(),
            dataset_type="polsar",
            num_samples=min(5, x_batch.size(0)),
            show_spectrum=False
        )
        save_path_recon = out_dir / f"reconstructions_Z22_vs_{anomaly_log_name}.png"
        fig_recon.savefig(save_path_recon, bbox_inches="tight", dpi=300)
        plt.close(fig_recon)
        print(f"   [+] Reconstructions sauvegardées : {save_path_recon}")

    metrics_path = out_dir / "ood_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(ood_metrics, f, indent=4)
    print(f"   [+] Métriques OoD sauvegardées : {metrics_path}")

    return detector

def main():
    # 1. Configuration et chargement des données
    repo_root = Path(__file__).resolve().parents[2]
    
    if len(sys.argv) >= 2:
        config_path = sys.argv[1]
    else:
        config_path = str(repo_root / "configs" / "config.yaml")
        
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Résolution absolue du chemin des données par rapport à la racine du projet
    trainpath = Path(config["data"]["dataset"]["trainpath"])
    if not trainpath.is_absolute():
        config["data"]["dataset"]["trainpath"] = str((repo_root / trainpath).resolve())
    
    print("[*] Découpage des données (Zones 1, 2.1 et 2.2)...")
    loaders_dict = azimut_split(config, use_cuda=False)
    loader1_train, loader1_valid, loader1_test = loaders_dict["loader1_splits"]
    loader_2_1, _, _ = loaders_dict["loader2_1_full"]

    # --- Division de la Zone 2.2 en 3 parties sur l'axe du range (colonnes) ---
    print("[*] Division de la Zone 2.2 en 3 sous-zones (range)...")
    dataset_cfg = config["data"]["dataset"]
    crop_cfg = dataset_cfg.get("crop_coordinates", {})
    
    zone2_2_start_row = dataset_cfg.get("azimut_split_x2")
    zone2_2_end_row = crop_cfg.get("end_row_crop")
    zone2_2_start_col = crop_cfg.get("start_col", 0)
    zone2_2_end_col = crop_cfg.get("end_col_crop")

    col_split_points = np.linspace(zone2_2_start_col, zone2_2_end_col, 4, dtype=int) # 4 points pour créer 3 intervalles
    print(f"   - Points de split (colonnes) : {col_split_points}")

    loaders_2_2_parts = []
    for i in range(3):
        cfg_part = copy.deepcopy(config)
        cfg_part["data"]["dataset"]["crop_coordinates"] = {
            "start_row": zone2_2_start_row, "end_row": zone2_2_end_row,
            "start_col": col_split_points[i], "end_col": col_split_points[i+1],
            "max_rows": crop_cfg.get("max_rows"), "max_cols": crop_cfg.get("max_cols") # On conserve les max_rows et max_cols pour éviter de charger des patchs hors limites, même si la zone de crop est plus petite que l'image entière
        }
        # On utilise get_full_image_dataloader pour créer un loader pour cette sous-partie
        loader, _, _ = get_full_image_dataloader(cfg_part, use_cuda=False) #On utlise cette fonction car on est en Mode Inférence/Test (Toutes les données + Ordre spatial conservé) 
        loaders_2_2_parts.append(loader)
        print(f"   - Sous-zone {i+1} créée ({col_split_points[i]} -> {col_split_points[i+1]}) avec {len(loader.dataset)} patchs.")

    # --- Génération des anomalies pour chaque sous-zone (appliquées séparément) ---
    print("[*] Génération des anomalies (Crosstalk et Gain testés distinctement sur chaque sous-zone)...")
    
    # Création des générateurs
    delta_generator = SyntheticParameterGenerator(delta=0.8)
    g_generator = SyntheticParameterGenerator(delta=0.5)
    
    # Utilisation d'une seed fixe pour la reproductibilité entre les méthodes
    anomaly_seed = config.get("anomaly_seed", 1234)
    
    delta_values = delta_generator(seed=anomaly_seed)
    g_values = g_generator(seed=anomaly_seed + 1) # Seed différente pour s'assurer que g et delta ne sont pas identiques

    # On crée une liste de 6 anomalies: [Crosstalk_z1, Gain_z1, Crosstalk_z2, Gain_z2, ...]
    final_anomaly_definitions = []
    for i in range(3):
        # Création sur CPU (sans .to(device)) pour accélérer le traitement du DataLoader
        crosstalk_anomaly = Crosstalk(delta=delta_values[i].item())
        gain_anomaly = ChannelGainImbalance(g=g_values[i].item())
        final_anomaly_definitions.extend([crosstalk_anomaly, gain_anomaly])
        print(f"   - Anomalies pour sous-zone {i+1}: Crosstalk (delta={crosstalk_anomaly.delta:.3f}) et Gain (g={gain_anomaly.g:.3f})")

    # On duplique les loaders pour correspondre: [loader_z1, loader_z1, loader_z2, loader_z2, ...]
    final_loaders_to_test = []
    for loader in loaders_2_2_parts:
        final_loaders_to_test.extend([loader, loader])
    
    
    # 2. Initialisation et chargement du Modèle
    print("[*] Chargement du modèle pré-entraîné...")
    model_cfg = config.get("model", {})
    
    in_channels = config["data"].get("inferred_input_channels", 4)
    input_size = config["data"].get("inferred_input_size", config["data"]["dataset"].get("patch_size", 32))
    
    model = LatentAutoEncoder(
        num_channels=in_channels,
        num_layers=model_cfg.get("num_layers", 3),
        channels_width=model_cfg.get("channels_width", 16),
        input_size=input_size,
        activation=model_cfg.get("activation", "relu"),
        upsampling_layer=model_cfg.get("upsampling_layer", "bilinear"),
        layer_mode=model_cfg.get("layer_mode", "complex"),
        normalization_layer=model_cfg.get("normalization_layer", "batch"),
        residual=model_cfg.get("residual", False),
        num_blocks=model_cfg.get("num_blocks", 1),
        latent_dim=model_cfg.get("latent_dim", 128)
    ).to(device)

    #Chargement dynamique du modèle le plus récent
    results_dir = Path("ml_results")
    if not results_dir.exists():
        print(f"[!] ERREUR: Le dossier '{results_dir}' n'existe pas. Aucun entraînement n'a été lancé.")
        sys.exit(1)

    #Trouve tous les dossiers de run contenant le fichier du modèle
    run_dirs = [d for d in results_dir.iterdir() if d.is_dir() and (d / "best_weights_autoencoder.pt").exists()]

    if not run_dirs:
        print(f"[!] ERREUR: Aucun entraînement valide (avec 'best__weights_autoencoder.pt') trouvé dans '{results_dir}'.")
        print("     Veuillez d'abord lancer le script 'train_autoencoder.py'.")
        sys.exit(1)

    #Sélectionne le dossier de run le plus récent en se basant sur la date de modification
    latest_run_dir = max(run_dirs, key=os.path.getmtime)
    model_path = latest_run_dir / "best_weights_autoencoder.pt"
    print(f"[*] Utilisation du modèle le plus récent trouvé dans : {latest_run_dir.name}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"[+] Poids du modèle chargés depuis {model_path}")

    #3. Lancement de l'évaluation
    evaluate_ood_system(model, loader1_train, loader1_valid, loader_2_1, 
                        final_loaders_to_test, final_anomaly_definitions, pfa_target=0.05, device=device, out_dir=latest_run_dir)

if __name__ == "__main__":
    main()