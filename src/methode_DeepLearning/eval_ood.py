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

from shared_setup import setup_experiment_env, get_test_loaders, get_shared_anomaly_generator
from cvnn.models import UNet
from cvnn.visualize import plot_latent_space

from methode_DeepLearning.ood_detector import OOD_Detector
from anomalies import Crosstalk

def main():
    print("[*] Début de l'évaluation OoD sur des données non vues (ALOS2-San Francisco)")
    
    print("\n[*] 1. Chargement de la région originelle (PolSF) pour la calibration...")
    config, config_polsf, device, loaders = setup_experiment_env(sys.argv, __file__, force_cpu=False)
    train_loader, valid_loader, _ = loaders
    
    print(f"[*] Fichier cible (Doit être l'image complète ALOS2 8080x22608) : {config['data']['dataset']['trainpath']}")

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
    detector = OOD_Detector(model, device=device)
    detector.fit_mahalanobis(train_loader)
    
    pfa_target = 0.05
    thresh_mah = detector.calibrate_thresholds(valid_loader, pfa=pfa_target)
    print(f"   -> Seuil Mahalanobis    : {thresh_mah:.4f}")

    #5. Sélection de régions STRICTEMENT non vues dans l'image ALOS2 (8080 x 22608)
    
    print("\n[*] 4. Chargement des régions Non Vues...")
    loader_sain, loaders_ano_parts = get_test_loaders(config_polsf, use_cuda=False)
    print(f"   -> Zone 2.1 (Saine)    : {len(loader_sain.dataset)} patchs")
    for i, loader in enumerate(loaders_ano_parts):
        print(f"   -> Zone 2.2 (Part {i+1}) : {len(loader.dataset)} patchs")

    # 6. Évaluation des Fausse Alarmes sur les zones pures (2.1 et 2.2)
    print("\n" + "-" * 65)
    print("🔹 TEST : Deep Learning (Ensemble Sémantique + Physique)")

    # --- Zone 2.1 ---
    preds_z21, scores_z21 = detector.detect(loader_sain)
    total_21 = len(scores_z21)
    rejetes_21 = int(np.sum(scores_z21 > 1.0))
    acceptes_21 = total_21 - rejetes_21
    
    print(f"\n   [Zone 2.1 - Saine]")
    print(f"   ↳ Acceptées : {acceptes_21:5d} / {total_21} ({100.0*acceptes_21/total_21:6.2f}%) | ✅ Vrais Négatifs")
    print(f"   ↳ Rejetées  : {rejetes_21:5d} / {total_21} ({100.0*rejetes_21/total_21:6.2f}%) | ❌ Fausses Alarmes")

    ood_metrics = {
        "pfa_target": pfa_target,
        "Zone_2_1_Saine": {
            "pfa_mah": float(rejetes_21 / total_21)
        }
    }

    # --- Zone 2.2 (Saine globale) ---
    scores_z22_parts = []
    for loader_part in loaders_ano_parts:
        _, scores_part = detector.detect(loader_part)
        scores_z22_parts.append(scores_part)
    scores_z22 = np.concatenate(scores_z22_parts)
    
    total_22 = len(scores_z22)
    rejetes_22 = int(np.sum(scores_z22 > 1.0))
    acceptes_22 = total_22 - rejetes_22

    print(f"\n   [Zone 2.2 - Saine (Pure Globale)]")
    print(f"   ↳ Acceptées : {acceptes_22:5d} / {total_22} ({100.0*acceptes_22/total_22:6.2f}%) | ✅ Vrais Négatifs")
    print(f"   ↳ Rejetées  : {rejetes_22:5d} / {total_22} ({100.0*rejetes_22/total_22:6.2f}%) | ❌ Fausses Alarmes")
    
    ood_metrics["Zone_2_2_Saine"] = {
        "pfa_mah": float(rejetes_22 / total_22)
    }

    # Preparation de l'espace latent pour la région saine
    latents_clean = []
    num_batches_viz = 1
    with torch.no_grad():
        for i, batch in enumerate(loader_sain):
            if i >= num_batches_viz: break
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            z_features = detector.get_latent_features(x.to(device))
            latents_clean.append(z_features.cpu())

    latents_ano_all = []

    # 7. Génération et injection des 3 Crosstalks
    print("\n[*] 5. Génération et injection des anomalies Crosstalk (3 deltas)...")
    delta_generator, anomaly_seed = get_shared_anomaly_generator(config)
    delta_values = delta_generator(num_samples=3, seed=anomaly_seed)

    for i in range(3):
        loader_ano = loaders_ano_parts[i]
        crosstalk_anomaly = Crosstalk(delta=delta_values[i].item())
        
        delta_cplx = complex(crosstalk_anomaly.delta)
        amp = abs(delta_cplx)
        phase_deg = np.angle(delta_cplx, deg=True)
        delta_str = f" | delta: {delta_cplx:.4g} (Amp: {amp:.4f}, Phase: {phase_deg:.1f}°)"

        # Injection de l'anomalie dans le dataset
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
        preds_mah_ano, scores_mah_ano = detector.detect(loader_ano)
        
        total_ano = len(scores_mah_ano)
        rejetes_ano = int(np.sum(scores_mah_ano > 1.0))
        acceptes_ano = total_ano - rejetes_ano

        # Calcul de l'AUC (comparaison avec la Zone 2.2 Pure)
        y_true = np.concatenate([np.zeros_like(scores_z22), np.ones_like(scores_mah_ano)])
        auc_mah = roc_auc_score(y_true, np.concatenate([scores_z22, scores_mah_ano]))
        
        print(f"\n   [Zone 2.2 - Anomalie : Zone_2_2_Part_{i+1}_Crosstalk{delta_str}]")
        print(f"   ↳ Acceptées : {acceptes_ano:5d} / {total_ano} ({100.0*acceptes_ano/total_ano:6.2f}%) | ❌ Faux Négatifs")
        print(f"   ↳ Rejetées  : {rejetes_ano:5d} / {total_ano} ({100.0*rejetes_ano/total_ano:6.2f}%) | 🚨 Vrais Positifs (Détection)")
        print(f"   ↳ AUC-ROC   : {auc_mah:.4f}")
        
        ood_metrics[f"Zone_2_2_Part_{i+1}_Crosstalk"] = {
            "delta": str(crosstalk_anomaly.delta),
            "detection_rate_mah": float(rejetes_ano / total_ano),
            "auc_roc_mah": float(auc_mah)
        }

        # Collect latent features for visualization
        with torch.no_grad():
            for j, batch in enumerate(loader_ano):
                if j >= num_batches_viz: break
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                z_features = detector.get_latent_features(x.to(device))
                latents_ano_all.append(z_features.cpu())

        # Restauration finale pour la propreté du dataset
        base_ds.transform = original_transform
        
    print("-" * 65)

    # 9. Visualisation de l'espace latent pour comparer Région Saine (Non vue) et Crosstalk
    print(f"\n[*] 6. Visualisation de l'espace latent (PCA)...")
    Z_c, Z_a = torch.cat(latents_clean, dim=0), torch.cat(latents_ano_all, dim=0)
    Z_all = torch.cat([Z_c, Z_a], dim=0)
    labels_all = np.concatenate([np.zeros(len(Z_c)), np.ones(len(Z_a))])
    
    fig_latent = plot_latent_space(
        latents=Z_all, labels=labels_all, method="pca", 
        classes_names={0: "ALOS2 Sain (Non Vu)", 1: "ALOS2 + Crosstalk (Mix 3 zones)"}
    )
    
    save_path_latent = latest_run_dir / "latent_space_alos2_unseen.png"
    fig_latent.savefig(save_path_latent, bbox_inches="tight", dpi=300)
    plt.close(fig_latent)
    print(f"   [+] PCA sauvegardée : {save_path_latent}")
    
    metrics_path = latest_run_dir / "ood_metrics_alos2_unseen.json"
    with open(metrics_path, "w") as f:
        json.dump(ood_metrics, f, indent=4)
    print(f"   [+] Métriques OoD sauvegardées : {metrics_path}")

if __name__ == "__main__":
    main()