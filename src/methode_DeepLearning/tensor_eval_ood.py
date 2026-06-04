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
from cvnn.data import get_dataloaders
from cvnn.models import UNet
from cvnn.visualize import plot_latent_space

from tensor_ood_detector import Tensor_OOD_Detector
from anomalies import Crosstalk
from synthetic_parameter_generator import SyntheticParameterGenerator

def evaluate_ood_system(model, train_loader, valid_loader, test_loader, anomaly_definitions, pfa_target=0.05, device="cpu", out_dir=Path(".")):
    """
    Entraîne le détecteur sur l'espace latent, calibre les seuils, et évalue les performances globales.
    Intègre également la visualisation de l'espace latent.
    """
    print("--- Initialisation du Détecteur OoD ---")
    ood_metrics = {"pfa_target": pfa_target}
    
    detector = Tensor_OOD_Detector(model, device=device)
    
    #1. Modélisation de l'espace latent normal
    print("1. Calcul de la distribution de Mahalanobis (sur Train Sain)...")
    detector.fit_mahalanobis(train_loader)
    
    #2. Calibration
    print(f"2. Calibration du seuil pour une PFA de {pfa_target*100}% (sur Valid Sain)...")
    thresh_mah = detector.calibrate_thresholds(valid_loader, pfa=pfa_target)
    print(f"   -> Seuil Mahalanobis    : {thresh_mah:.4f}")
    ood_metrics["threshold_mah"] = float(thresh_mah)
    
    #3. Test sur les données saines de test avec vérification de la PFA
    print("\n--- Évaluation sur le Set de Test (Sain) ---")
    preds_mah_sain, scores_mah_sain = detector.detect(test_loader)
    print(f"PFA empirique (Mahal) : {np.mean(preds_mah_sain)*100:.2f}% (Cible: {pfa_target*100}%)")
    
    ood_metrics["Test_Sain"] = {
        "pfa_mah": float(np.mean(preds_mah_sain))
    }
    
    ood_metrics["Anomalies"] = {}

    #4. Injection des anomalies sur le loader de test
    base_ds = test_loader.dataset
    while hasattr(base_ds, 'dataset') or hasattr(base_ds, 'base_dataset'):
        base_ds = getattr(base_ds, 'dataset', getattr(base_ds, 'base_dataset', base_ds))
    original_transform = getattr(base_ds, 'transform', None)

    for anomaly_def in anomaly_definitions:
        anomaly_type = anomaly_def.__class__.__name__
        anomaly_log_name = f"Test_{anomaly_type}"

        print(f"\n--- Évaluation globale avec anomalie {anomaly_type} ---")

        # A. Injection de l'anomalie
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
        preds_mah_ano, scores_mah_ano = detector.detect(test_loader)

        # B. Calcul des métriques de détection
        print(f"   -> Taux de Détection (Mahal): {np.mean(preds_mah_ano)*100:.2f}%")

        # Calcul de l'AUC-ROC (Comparaison Test Sain vs Test Anomalie)
        y_true = np.concatenate([np.zeros_like(scores_mah_sain), np.ones_like(scores_mah_ano)])
            
        auc_mah = roc_auc_score(y_true, np.concatenate([scores_mah_sain, scores_mah_ano]))
            
        print(f"   -> Score AUC-ROC (Mahal)    : {auc_mah:.4f}")
            
        ood_metrics["Anomalies"][anomaly_log_name] = {
            "detection_rate_mah": float(np.mean(preds_mah_ano)),
            "auc_roc_mah": float(auc_mah)
        }
    
        # --- C. Visualisation de l'espace latent ---
        print(f"   - Génération de la visualisation de l'espace latent (PCA) vs {anomaly_type}...")
        latents_clean, latents_ano = [], []
        
        # On restaure temporairement le transform sain pour extraire les latents purs
        base_ds.transform = original_transform
        num_batches_viz = min(5, len(test_loader)) # Prendre quelques batchs pour ne pas saturer la RAM (ex: 5)
        
        with torch.no_grad():
            # Extraire Latent Sain
            for i, batch in enumerate(test_loader):
                if i >= num_batches_viz: break
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                z = detector.extract_latent(x.to(device))
                # Agrégation spatiale pour obtenir 1 point par patch (évite 10 millions de points PCA)
                z_mean = z.mean(dim=(1, 2))
                # Séparation complexe vers réel car PCA(Scikit) ne supporte pas le complexe
                if z_mean.is_complex():
                    z_mean = torch.cat([z_mean.real, z_mean.imag], dim=1)
                latents_clean.append(z_mean.cpu())
                
            # Ré-injection de l'anomalie pour extraire Latent Anomalie
            base_ds.transform = torch.nn.Sequential(anomaly_def, original_transform) if original_transform else anomaly_def
            for i, batch in enumerate(test_loader):
                if i >= num_batches_viz: break
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                z = detector.extract_latent(x.to(device))
                z_mean = z.mean(dim=(1, 2))
                if z_mean.is_complex():
                    z_mean = torch.cat([z_mean.real, z_mean.imag], dim=1)
                latents_ano.append(z_mean.cpu())
        
        # Concaténation et Affichage
        Z_c = torch.cat(latents_clean, dim=0)
        Z_a = torch.cat(latents_ano, dim=0)
        Z_all = torch.cat([Z_c, Z_a], dim=0)
        labels_all = np.concatenate([np.zeros(len(Z_c)), np.ones(len(Z_a))])
        
        fig_latent = plot_latent_space(
            latents=Z_all, 
            labels=labels_all, 
            method="pca", 
            classes_names={0: "Sain (Normal)", 1: f"Anomalie ({anomaly_type})"}
        )
        save_path_latent = out_dir / f"latent_space_pca_{anomaly_log_name}.png"
        fig_latent.savefig(save_path_latent, bbox_inches="tight", dpi=300)
        plt.close(fig_latent)
        print(f"   [+] PCA Espace Latent sauvegardée : {save_path_latent}")
        
    #Restauration finale du transform original
    base_ds.transform = original_transform

    metrics_path = out_dir / "ood_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(ood_metrics, f, indent=4)
    print(f"   [+] Métriques OoD sauvegardées : {metrics_path}")

    return detector

def main():
    #1. Configuration et chargement des données
    repo_root = Path(__file__).resolve().parents[2]
    
    if len(sys.argv) >= 2:
        config_path = sys.argv[1]
    else:
        config_path = str(repo_root / "configs" / "config_Unet.yaml")
        
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #Résolution absolue du chemin des données par rapport à la racine du projet
    trainpath = Path(config["data"]["dataset"]["trainpath"])
    if not trainpath.is_absolute():
        config["data"]["dataset"]["trainpath"] = str((repo_root / trainpath).resolve())
        
    #On s'assure qu'on utilise l'image entière si souhaité ou selon config
    config["data"]["dataset"].pop("crop_coordinates", None)
    
    print("[*] Chargement des données via get_dataloaders...")
    train_loader, valid_loader, test_loader = get_dataloaders(config, device)

    # --- Génération des anomalies ---
    print("[*] Génération des anomalies (Crosstalk)...")
    
    #1. Instanciation du delta pour le Crosstalk (variation d'amplitude et de phase)
    #Pour delta: par exemple une amplitude à -30 dB et une phase concentrée autour de 45°
    delta_generator = SyntheticParameterGenerator(
        mean_db=-15.0,                #Niveau typique de cross-talk cité dans l'article
        std_dev_amp=0.01,             #Légère variation
        phase_mean_rad=0.0,           #Peu importe si le kappa est à 0
        phase_concentration=1e-5      #Kappa = 0 donne une phase aléatoire uniforme (typiques des bruits de couplage)
    )

    #2. Génération des valeurs
    delta_values = delta_generator(num_samples=1, seed=1234)
    crosstalk_anomaly = Crosstalk(delta=delta_values[0].item())
    final_anomaly_definitions = [crosstalk_anomaly]
    print(f"   - Anomalie instanciée: Crosstalk (delta={crosstalk_anomaly.delta:.3f})")
    
    #3. Initialisation et chargement du Modèle UNet
    print("[*] Chargement du modèle UNet pré-entraîné...")
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
        projection_layer=model_cfg.get("projection_layer", "amplitude"),
        projection_config=model_cfg.get("projection_config", None),
        dropout=model_cfg.get("dropout", 0.0),
        gumbel_softmax=model_cfg.get("gumbel_softmax", None)
    ).to(device)

    #Chargement dynamique du modèle UNet le plus récent
    results_dir = Path("Unet_results")
    if not results_dir.exists():
        print(f"[!] ERREUR: Le dossier '{results_dir}' n'existe pas. Aucun entraînement n'a été lancé.")
        sys.exit(1)

    #Trouve tous les dossiers de run contenant le fichier du modèle
    run_dirs = [d for d in results_dir.iterdir() if d.is_dir() and (d / "best_weights_unet.pt").exists()]

    if not run_dirs:
        print(f"[!] ERREUR: Aucun entraînement valide (avec 'best_weights_unet.pt') trouvé dans '{results_dir}'.")
        print("     Veuillez d'abord lancer le script 'train_unet.py'.")
        sys.exit(1)

    #Sélectionne le dossier de run le plus récent en se basant sur la date de modification
    latest_run_dir = max(run_dirs, key=os.path.getmtime)
    model_path = latest_run_dir / "best_weights_unet.pt"
    print(f"[*] Utilisation du modèle le plus récent trouvé dans : {latest_run_dir.name}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"[+] Poids du modèle chargés depuis {model_path}")

    #4. Lancement de l'évaluation
    evaluate_ood_system(model, train_loader, valid_loader, test_loader, 
                        final_anomaly_definitions, pfa_target=0.05, device=device, out_dir=latest_run_dir)

if __name__ == "__main__":
    main()