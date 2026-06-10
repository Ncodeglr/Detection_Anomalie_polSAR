import sys
import os
import json
import copy
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# Ajout des chemins pour importer les modules du projet
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "cvnn", "src"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cvnn.config import load_config
from cvnn.data import get_dataloaders, get_full_image_dataloader
from cvnn.models import UNet
from cvnn.visualize import plot_latent_space

from tensor_ood_detector import Tensor_OOD_Detector
from anomalies import Crosstalk
from synthetic_parameter_generator import SyntheticParameterGenerator

def main():
    print("[*] Début de l'évaluation OoD sur des données non vues (ALOS2-San Francisco)")
    repo_root = Path(__file__).resolve().parents[2]
    
    if len(sys.argv) >= 2:
        config_path = sys.argv[1]
    else:
        config_path = str(repo_root / "configs" / "config_Unet.yaml")
        
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #1. Résolution du chemin absolu de l'image ALOS2
    trainpath = Path(config["data"]["dataset"]["trainpath"])
    if not trainpath.is_absolute():
        config["data"]["dataset"]["trainpath"] = str((repo_root / trainpath).resolve())
        
    print(f"[*] Fichier cible (Doit être l'image complète ALOS2 8080x22608) : {config['data']['dataset']['trainpath']}")

    # 2. Définition des coordonnées originelles de PolSF pour la calibration
    # L'image PolSF a été extraite aux coordonnées (x1=736, y1=2832, x2=3520, y2=7888)
    # PolSFDataset est géré nativement par cvnn comme étant déjà la bonne zone rognée.
    # On retire donc tout crop_coordinates manuel pour éviter un double-crop vide.
    config_polsf = copy.deepcopy(config)
    config_polsf["data"]["dataset"].pop("crop_coordinates", None)
    # On DOIT forcer le calcul des statistiques pour obtenir min_value/max_value pour LogAmplitude
    config_polsf["data"]["recompute_statistics"] = True
    
    print("\n[*] 1. Chargement de la région originelle (PolSF) pour la calibration...")
    train_loader, valid_loader, _ = get_dataloaders(config_polsf, device) #On obtient le train_loader et valid_loader de PolSF

    #3. Chargement du modèle UNet
    print("\n[*] 2. Chargement du modèle UNet pré-entraîné...")
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    in_channels = data_cfg.get("inferred_input_channels", 4)
    input_size = data_cfg.get("inferred_input_size", data_cfg.get("dataset", {}).get("patch_size", 32))
    
    model = UNet(
        num_channels=in_channels,
        num_layers=model_cfg.get("num_layers", 3),
        channels_width=model_cfg.get("channels_width", 16),
        input_size=input_size,
        activation=model_cfg.get("activation", "CRelu"),
        num_classes=model_cfg.get("num_classes", 7),
        num_blocks=model_cfg.get("num_blocks", 1),
        layer_mode=model_cfg.get("layer_mode", "complex"),
        normalization_layer=model_cfg.get("normalization_layer", "batch"),
        downsampling_layer=model_cfg.get("downsampling_layer", "maxpool"),
        upsampling_layer=model_cfg.get("upsampling_layer", "nearest"),
        residual=model_cfg.get("residual", True),
        projection_layer=model_cfg.get("projection_layer", "amplitude")
    ).to(device)

    results_dir = Path("Unet_results")
    if not results_dir.exists():
        print(f"[!] ERREUR: Le dossier '{results_dir}' n'existe pas.")
        sys.exit(1)

    run_dirs = [d for d in results_dir.iterdir() if d.is_dir() and (d / "best_weights_unet.pt").exists()]
    if not run_dirs:
        print("[!] ERREUR: Aucun modèle valide trouvé.")
        sys.exit(1)

    latest_run_dir = max(run_dirs, key=os.path.getmtime)
    model.load_state_dict(torch.load(latest_run_dir / "best_weights_unet.pt", map_location=device))
    print(f"   [+] Poids chargés depuis {latest_run_dir.name}")

    #4. Calibration du détecteur sur la zone originelle
    print("\n[*] 3. Initialisation et Calibration du Détecteur OoD...")
    detector = Tensor_OOD_Detector(model, device=device)
    detector.fit_mahalanobis(train_loader)
    
    pfa_target = 0.05
    thresh_mah = detector.calibrate_thresholds(valid_loader, pfa=pfa_target)
    print(f"   -> Seuil Mahalanobis    : {thresh_mah:.4f}")

    #5. Sélection de régions STRICTEMENT non vues dans l'image ALOS2 (8080 x 22608)
    
    #Région A (Saine) : On prend une région en dessous de PolSF - Pour sortir de la zone de San Francisco, on passe explicitement sur l'image maître ALOS2.
    config_unseen_sain = copy.deepcopy(config_polsf) # Hérite des statistiques fraîchement calculées
    config_unseen_sain["data"]["dataset"]["name"] = "ALOSDataset"
    config_unseen_sain["data"]["recompute_statistics"] = False
    config_unseen_sain["data"]["dataset"]["crop_coordinates"] = {
        "start_row": 4000, "end_row": 6000,     # Lignes 4000 à 6000
        "start_col": 2832, "end_col": 7888      # Mêmes colonnes que PolSF
    }
    
    #Région B (Anomalie) : On prend une région totalement décalée vers la droite
    config_unseen_ano = copy.deepcopy(config_polsf) # Hérite des statistiques fraîchement calculées
    config_unseen_ano["data"]["dataset"]["name"] = "ALOSDataset"
    config_unseen_ano["data"]["recompute_statistics"] = False
    config_unseen_ano["data"]["dataset"]["crop_coordinates"] = {
        "start_row": 4000, "end_row": 6000,
        "start_col": 10000, "end_col": 15000
    }

    print("\n[*] 4. Chargement des régions Non Vues...")
    loader_sain, _, _ = get_full_image_dataloader(config_unseen_sain, use_cuda=False)
    loader_ano, _, _ = get_full_image_dataloader(config_unseen_ano, use_cuda=False)
    
    print(f"   -> Région A (Saine)    : {len(loader_sain.dataset)} patchs")
    print(f"   -> Région B (Anomalie) : {len(loader_ano.dataset)} patchs")

    # 6. Évaluation des Fausse Alarmes sur une région non vue
    print("\n--- Évaluation du Bruit de Fond (Sain) sur Région Non Vue ---")
    preds_mah_sain, scores_mah_sain = detector.detect(loader_sain)
    print(f"PFA empirique (Mahal)    : {np.mean(preds_mah_sain)*100:.2f}% (Idéal proche de {pfa_target*100}%)")

    # 7. Génération et injection du Crosstalk
    print("\n[*] 5. Génération et injection de l'anomalie Crosstalk...")
    delta_generator = SyntheticParameterGenerator(
        mean_db=-22.49, std_dev_amp=0.01, phase_mean_rad=0.0, phase_concentration=1e-5
    )
    #crosstalk_anomaly = Crosstalk(delta=delta_generator(num_samples=1)[0].item())
    #print(f"   - Crosstalk (delta={crosstalk_anomaly.delta:.3f}) injecté sur Région B")

    crosstalk_anomaly = Crosstalk(delta=-0.021256418898701668+0.05765506625175476j)
    print(f"   - Crosstalk (delta={crosstalk_anomaly.delta}) injecté sur Région B")


    # Injection de l'anomalie dans le dataset "Région B"
    base_ds = loader_ano.dataset
    while hasattr(base_ds, 'dataset') or hasattr(base_ds, 'base_dataset'):
        base_ds = getattr(base_ds, 'dataset', getattr(base_ds, 'base_dataset', base_ds))
    original_transform = getattr(base_ds, 'transform', None)

    if hasattr(original_transform, 'transforms'):
        new_transforms = []
        injected = False
        for t in original_transform.transforms:
            if t.__class__.__name__ == 'LogAmplitude':
                new_transforms.append(crosstalk_anomaly)
                injected = True
            new_transforms.append(t)
        if not injected: new_transforms.append(crosstalk_anomaly)
        base_ds.transform = original_transform.__class__(new_transforms)
    else:
        base_ds.transform = torch.nn.Sequential(crosstalk_anomaly, original_transform) if original_transform else crosstalk_anomaly

    # 8. Évaluation sur la région contenant le Crosstalk
    print("\n--- Évaluation du Taux de Détection sur Crosstalk ---")
    preds_mah_ano, scores_mah_ano = detector.detect(loader_ano)

    print(f"   -> Taux de Détection (Mahal)    : {np.mean(preds_mah_ano)*100:.2f}%")

    # Calcul de l'AUC
    y_true = np.concatenate([np.zeros_like(scores_mah_sain), np.ones_like(scores_mah_ano)])
    auc_mah = roc_auc_score(y_true, np.concatenate([scores_mah_sain, scores_mah_ano]))
        
    print(f"   -> Score AUC-ROC (Mahal)        : {auc_mah:.4f}")

    # 9. Visualisation de l'espace latent pour comparer Région Saine (Non vue) et Crosstalk
    print(f"\n[*] 6. Visualisation de l'espace latent (PCA)...")
    latents_clean, latents_ano = [], []
    num_batches_viz = min(1, len(loader_sain), len(loader_ano))
    
    with torch.no_grad():
        for i, batch in enumerate(loader_sain):
            if i >= num_batches_viz: break
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            z_features = detector.get_latent_features(x.to(device))
            latents_clean.append(z_features.cpu())
            
        for i, batch in enumerate(loader_ano):
            if i >= num_batches_viz: break
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            z_features = detector.get_latent_features(x.to(device))
            latents_ano.append(z_features.cpu())

    Z_c, Z_a = torch.cat(latents_clean, dim=0), torch.cat(latents_ano, dim=0)
    Z_all = torch.cat([Z_c, Z_a], dim=0)
    labels_all = np.concatenate([np.zeros(len(Z_c)), np.ones(len(Z_a))])
    
    fig_latent = plot_latent_space(
        latents=Z_all, labels=labels_all, method="pca", 
        classes_names={0: "ALOS2 Sain (Non Vu)", 1: "ALOS2 + Crosstalk"}
    )
    
    save_path_latent = latest_run_dir / "latent_space_alos2_unseen.png"
    fig_latent.savefig(save_path_latent, bbox_inches="tight", dpi=300)
    plt.close(fig_latent)
    print(f"   [+] PCA sauvegardée : {save_path_latent}")

    # Restauration finale
    base_ds.transform = original_transform
    
    ood_metrics = {
        "pfa_target": pfa_target,
        "ALOS2_Unseen_Sain": {
            "pfa_mah": float(np.mean(preds_mah_sain))
        },
        "ALOS2_Unseen_Crosstalk": {
            "detection_rate_mah": float(np.mean(preds_mah_ano)),
            "auc_roc_mah": float(auc_mah)
        }
    }
    
    metrics_path = latest_run_dir / "ood_metrics_alos2_unseen.json"
    with open(metrics_path, "w") as f:
        json.dump(ood_metrics, f, indent=4)
    print(f"   [+] Métriques OoD sauvegardées : {metrics_path}")

if __name__ == "__main__":
    main()