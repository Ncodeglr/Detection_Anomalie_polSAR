import sys
import os
import json
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# Ajout des chemins pour importer les modules du projet
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "cvnn", "src"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared_setup import setup_experiment_env, get_test_loaders, get_shared_anomaly_generator
from cvnn.models import AutoEncoder

# Imports spécifiques à SimCLR et à la détection
from simclr_model import RobustComplexSimCLR
from ood_detector import OOD_Detector
from anomalies import Crosstalk

def main():
    print("[*] Début de l'évaluation OoD pour le modèle SimCLR (ALOS2-San Francisco)")
    
    # 1. Chargement de la configuration et des données de la Zone 1 (calibration)
    print("\n[*] 1. Chargement de la configuration SimCLR et des données de calibration...")
    # On utilise le config SimCLR par défaut
    config, config_base, device, loaders = setup_experiment_env(sys.argv, __file__, force_cpu=False, default_config="config_SimCLR.yaml")
    train_loader, valid_loader, _ = loaders
    
    print(f"[*] Fichier de configuration utilisé : {sys.argv[1] if len(sys.argv) > 1 else 'configs/config_SimCLR.yaml'}")

    # 2. Chargement du modèle SimCLR pré-entraîné
    print("\n[*] 2. Chargement du modèle SimCLR pré-entraîné...")
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    
    in_channels = data_cfg.get("inferred_input_channels", 4)
    input_size = data_cfg.get("inferred_input_size", data_cfg.get("dataset", {}).get("patch_size", 64))
    
    # a. Recréer l'encodeur CVNN
    cvnn_encoder = AutoEncoder(
        num_channels=in_channels,
        num_layers=model_cfg.get("num_layers", 3),
        channels_width=model_cfg.get("channels_width", 32),
        input_size=input_size,
        activation=model_cfg.get("activation", "CReLU"),
        upsampling_layer=model_cfg.get("upsampling_layer", "nearest"),
        num_blocks=model_cfg.get("num_blocks", 2),
        layer_mode=model_cfg.get("layer_mode", "complex"),
        normalization_layer=model_cfg.get("normalization_layer", "batch"),
        downsampling_layer=model_cfg.get("downsampling_layer", "maxpool"),
        residual=model_cfg.get("residual", False),
        dropout=model_cfg.get("dropout", 0.0),
        latent_dim=model_cfg.get("latent_dim", 128)
    ).to(device)

    # b. Déduire la dimension de sortie de l'encodeur (comme dans train_simclr.py)
    cvnn_encoder.eval()
    dummy_input = torch.complex(
        torch.randn(1, in_channels, input_size, input_size), 
        torch.randn(1, in_channels, input_size, input_size)
    ).to(device)
    with torch.no_grad():
        if hasattr(cvnn_encoder, "get_latent"):
            dummy_latent = cvnn_encoder.get_latent(dummy_input)
        else:
            dummy_latent = cvnn_encoder(dummy_input)
            if isinstance(dummy_latent, (list, tuple)):
                dummy_latent = dummy_latent[-1]
    complex_channels_out = dummy_latent.shape[1]
    print(f"   [+] Canaux latents extraits automatiquement de l'encodeur : {complex_channels_out}")

    # c. Recréer le modèle SimCLR complet
    model = RobustComplexSimCLR(
        cvnn_encoder=cvnn_encoder, 
        complex_feature_dim=complex_channels_out, 
        projection_dim=128 # La valeur est fixe dans le modèle, mais n'impacte pas l'évaluation sur l'encodeur
    ).to(device)

    # d. Charger les poids
    results_dir = Path("SimCLR_results")
    if not results_dir.exists():
        print(f"[!] ERREUR: Le dossier '{results_dir}' n'existe pas. Avez-vous lancé train_simclr.py ?")
        sys.exit(1)

    run_dirs = [d for d in results_dir.iterdir() if d.is_dir() and (d / "best_weights_simclr_full.pt").exists()]
    if not run_dirs:
        print(f"[!] ERREUR: Aucun modèle SimCLR valide trouvé dans {results_dir}.")
        sys.exit(1)

    latest_run_dir = max(run_dirs, key=os.path.getmtime)
    model.load_state_dict(torch.load(latest_run_dir / "best_weights_simclr_full.pt", map_location=device))
    print(f"   [+] Poids chargés depuis {latest_run_dir.name}")

    # 3. Calibration du détecteur sur l'encodeur du modèle SimCLR
    print("\n[*] 3. Initialisation et Calibration du Détecteur OoD sur l'encodeur...")
    # Le détecteur OoD travaille sur l'espace latent de l'encodeur, pas sur la tête de projection
    detector = OOD_Detector(model.encoder, device=device)
    detector.fit_mahalanobis(train_loader)
    
    pfa_target = 0.05
    thresh_mah = detector.calibrate_thresholds(valid_loader, pfa=pfa_target)
    print(f"   -> Seuil Mahalanobis calibré : {thresh_mah:.4f} (pour PFA={pfa_target*100}%)")

    # 4. Chargement des régions de test non vues
    print("\n[*] 4. Chargement des régions de test Non Vues (Zones 2.1 et 2.2)...")
    loader_sain, loaders_ano_parts = get_test_loaders(config_base, use_cuda=False)
    print(f"   -> Zone 2.1 (Saine)    : {len(loader_sain.dataset)} patchs")
    for i, loader in enumerate(loaders_ano_parts):
        print(f"   -> Zone 2.2 (Part {i+1}) : {len(loader.dataset)} patchs")

    # 5. Évaluation des Fausses Alarmes sur les zones saines non vues
    print("\n" + "-" * 65)
    print("🔹 TEST : SimCLR + Détecteur OoD (Ensemble Sémantique + Physique)")

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
            "pfa_observed": float(rejetes_21 / total_21)
        }
    }

    # --- Zone 2.2 (Saine globale, avant injection) ---
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
        "pfa_observed": float(rejetes_22 / total_22)
    }

    # 6. Génération et injection des anomalies Crosstalk
    print("\n[*] 5. Génération et injection des anomalies Crosstalk (3 niveaux)...")
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

        # On injecte l'anomalie à la fin du pipeline de transformation (après ToTensor)
        if original_transform:
            if hasattr(original_transform, 'transforms'):
                # Si c'est un Compose, on récupère sa liste et on insère l'anomalie à la fin
                new_transforms = list(original_transform.transforms) + [crosstalk_anomaly]
                base_ds.transform = original_transform.__class__(new_transforms)
            else:
                from torchvision.transforms import Compose
                base_ds.transform = Compose([original_transform, crosstalk_anomaly])
        else:
            base_ds.transform = crosstalk_anomaly

        # 7. Évaluation sur la région contenant le Crosstalk
        preds_mah_ano, scores_mah_ano = detector.detect(loader_ano)
        
        total_ano = len(scores_mah_ano)
        rejetes_ano = int(np.sum(scores_mah_ano > 1.0))
        acceptes_ano = total_ano - rejetes_ano

        # Calcul de l'AUC (comparaison avec la Zone 2.2 Pure)
        y_true = np.concatenate([np.zeros_like(scores_z22), np.ones_like(scores_mah_ano)])
        auc_mah = roc_auc_score(y_true, np.concatenate([scores_z22, scores_mah_ano]))
        
        print(f"\n   [Zone 2.2 - Anomalie : Part_{i+1}_Crosstalk{delta_str}]")
        print(f"   ↳ Acceptées : {acceptes_ano:5d} / {total_ano} ({100.0*acceptes_ano/total_ano:6.2f}%) | ❌ Faux Négatifs")
        print(f"   ↳ Rejetées  : {rejetes_ano:5d} / {total_ano} ({100.0*rejetes_ano/total_ano:6.2f}%) | 🚨 Vrais Positifs (Détection)")
        print(f"   ↳ AUC-ROC   : {auc_mah:.4f}")
        
        ood_metrics[f"Zone_2_2_Part_{i+1}_Crosstalk"] = {
            "delta": str(crosstalk_anomaly.delta),
            "detection_rate": float(rejetes_ano / total_ano),
            "auc_roc": float(auc_mah)
        }

        # Restauration de la transformation originale pour ne pas affecter les boucles suivantes
        base_ds.transform = original_transform
        
    print("-" * 65)

    metrics_path = latest_run_dir / "ood_metrics_crosstalk.json"
    with open(metrics_path, "w") as f:
        json.dump(ood_metrics, f, indent=4)
    print(f"   [+] Métriques OoD sauvegardées : {metrics_path}")

if __name__ == "__main__":
    main()