import sys
import os
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# Ajout des chemins pour importer les modules du projet
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "cvnn", "src"))

from cvnn.config import load_config
from cvnn.data import azimut_split
from cvnn.models import LatentAutoEncoder
from cvnn.visualize import plot_latent_space, plot_reconstructions

from ood_detector import OODDetector
from anomalies import Crosstalk, ChannelGainImbalance

def evaluate_ood_system(model, loader1_train, loader1_valid, loader2_1, loader2_2, pfa_target=0.05, device="cpu", out_dir=Path(".")):
    """
    Entraîne le détecteur sur l'espace latent, calibre les seuils, et évalue les performances globales.
    """
    print("--- Initialisation du Détecteur OoD ---")
    ood_metrics = {"pfa_target": pfa_target}
    
    detector = OODDetector(model, device=device)
    
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
    
    # 4. Évaluation de référence sur la Zone 2.2 (SANS anomalie)
    print("\n--- Évaluation de référence sur Zone 2.2 (Pures) ---")
    preds_recon_22, scores_recon_22, preds_mah_22, scores_mah_22 = detector.detect(loader2_2)
    print(f"PFA empirique Zone 2.2 (Recon) : {np.mean(preds_recon_22)*100:.2f}%")
    print(f"PFA empirique Zone 2.2 (Mahal) : {np.mean(preds_mah_22)*100:.2f}%")
    
    ood_metrics["Zone_2_2_Sain"] = {
        "pfa_recon": float(np.mean(preds_recon_22)),
        "pfa_mah": float(np.mean(preds_mah_22))
    }
    
    # --- AJOUT: Extraction des latents purs pour la visualisation ---
    latents_pure = []
    for batch in loader2_2:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        latents_pure.append(detector._extract_real_latent(x.to(device)).cpu().numpy())
    latents_pure = np.concatenate(latents_pure)

    # 5. Test sur les anomalies de la Zone 2.2
    anomalies_to_test = [
        Crosstalk(delta=0.15).to(device),
        ChannelGainImbalance(g=1.3).to(device)
    ]
    
    # Récupération du dataset de base pour modifier son transform à la volée
    base_ds = loader2_2.dataset
    while hasattr(base_ds, 'dataset') or hasattr(base_ds, 'base_dataset'):
        base_ds = getattr(base_ds, 'dataset', getattr(base_ds, 'base_dataset', base_ds))
        
    # PolSFDataset délègue souvent le transform à son alos_dataset interne
    transform_holder = base_ds
    if not hasattr(transform_holder, 'transform') and hasattr(transform_holder, 'alos_dataset'):
        transform_holder = transform_holder.alos_dataset
        
    original_transform = getattr(transform_holder, 'transform', None)
    
    class AnomalyTransformWrapper:
        def __init__(self, anomaly_gen, device):
            self.anomaly_gen = anomaly_gen
            self.device = device
        def __call__(self, x):
            x_t = x.unsqueeze(0).to(self.device)
            with torch.no_grad():
                x_t = self.anomaly_gen(x_t)
            return x_t.squeeze(0).cpu()

    ood_metrics["Anomalies"] = {}

    for anomaly in anomalies_to_test:
        anomaly_name = anomaly.__class__.__name__
        print(f"\n--- Évaluation sur Zone 2.2 avec {anomaly_name} ---")
        
        # Injection de l'anomalie AVANT LogAmplitude en l'insérant dans le Compose
        if hasattr(original_transform, 'transforms'):
            new_transforms = []
            injected = False
            for t in original_transform.transforms:
                if t.__class__.__name__ == 'LogAmplitude':
                    new_transforms.append(AnomalyTransformWrapper(anomaly, device))
                    injected = True
                new_transforms.append(t)
            if not injected:
                new_transforms.append(AnomalyTransformWrapper(anomaly, device))
            transform_holder.transform = original_transform.__class__(new_transforms)
        else:
            print(f"[!] Erreur: transform n'est pas un Compose. Ignoré.")
            continue
        
        preds_recon_ano, scores_recon_ano, preds_mah_ano, scores_mah_ano = detector.detect(loader2_2)
        
        print(f"Taux de Détection (Recon) : {np.mean(preds_recon_ano)*100:.2f}%")
        print(f"Taux de Détection (Mahal) : {np.mean(preds_mah_ano)*100:.2f}%")
        
        # Calcul de l'AUC-ROC (Comparaison Z2.2 Sain vs Z2.2 Anomalie)
        with torch.no_grad():
            y_true = np.concatenate([np.zeros(len(scores_recon_22)), np.ones(len(scores_recon_ano))])
            
            auc_recon = roc_auc_score(y_true, np.concatenate([scores_recon_22, scores_recon_ano]))
            auc_mah = roc_auc_score(y_true, np.concatenate([scores_mah_22, scores_mah_ano]))
            
            print(f"Score AUC-ROC (Recon) : {auc_recon:.4f} (1.0 = Parfait)")
            print(f"Score AUC-ROC (Mahal) : {auc_mah:.4f}")
            
            ood_metrics["Anomalies"][anomaly_name] = {
                "detection_rate_recon": float(np.mean(preds_recon_ano)),
                "detection_rate_mah": float(np.mean(preds_mah_ano)),
                "auc_roc_recon": float(auc_recon),
                "auc_roc_mah": float(auc_mah)
            }
    
        # --- Extraction des latents anormaux et Création du Graphique ---
        latents_ano = []
        for batch in loader2_2:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            latents_ano.append(detector._extract_real_latent(x.to(device)).cpu().numpy())
        latents_ano = np.concatenate(latents_ano)
        
        all_latents = np.concatenate([latents_pure, latents_ano])
        all_labels = np.concatenate([np.zeros(len(latents_pure)), np.ones(len(latents_ano))])
        
        fig_latent = plot_latent_space(
            latents=all_latents, labels=all_labels, method="pca", 
            classes_names={0: "Zone 2.2 (Saine)", 1: f"Anomalie ({anomaly_name})"}
        )
        save_path = out_dir / f"latent_space_Z22_vs_{anomaly_name}.png"
        fig_latent.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig_latent)
        print(f"   [+] Visualisation sauvegardée : {save_path}")

        # --- AJOUT: Visualisation des Reconstructions de l'anomalie ---
        model.eval()
        with torch.no_grad():
            for batch in loader2_2:
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
        save_path_recon = out_dir / f"reconstructions_Z22_vs_{anomaly_name}.png"
        fig_recon.savefig(save_path_recon, bbox_inches="tight", dpi=300)
        plt.close(fig_recon)
        print(f"   [+] Reconstructions sauvegardées : {save_path_recon}")

    # Restauration du transform original propre
    transform_holder.transform = original_transform
    
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

    if "loader2_1_full" not in loaders_dict or "loader2_2_full" not in loaders_dict:
        print("[!] ERREUR: Les loaders 'loader2_1_full' ou 'loader2_2_full' n'ont pas été trouvés.")
        print("     Veuillez vérifier votre configuration 'azimut_split' dans le fichier config.yaml.")
        sys.exit(1)
    loader_2_1, _, _ = loaders_dict["loader2_1_full"]
    loader_2_2, _, _ = loaders_dict["loader2_2_full"]
    
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
    evaluate_ood_system(model, loader1_train, loader1_valid, loader_2_1, loader_2_2, pfa_target=0.05,device=device, out_dir=latest_run_dir)

if __name__ == "__main__":
    main()