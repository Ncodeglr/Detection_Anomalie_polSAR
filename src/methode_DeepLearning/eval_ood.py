import sys
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import roc_auc_score

# Ajout des chemins pour importer les modules du projet
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "cvnn", "src"))

from cvnn.config import load_config
from cvnn.data import azimut_split
from cvnn.models import LatentAutoEncoder

# Nouveaux imports pour l'OoD
from ood_detector import ComplexOODDetector
from anomalies import Crosstalk, ChannelGainImbalance

def evaluate_ood_system(model, train_loader, valid_loader, test_sain_loader, anomaly_loader, pfa_target=0.05, device="cpu"):
    """
    Entraîne le détecteur sur l'espace latent, calibre les seuils, et évalue les performances globales.
    """
    print("--- Initialisation du Détecteur OoD ---")
    detector = ComplexOODDetector(model, device=device)
    
    # 1. Modélisation de l'espace latent normal
    print("1. Calcul de la distribution de Mahalanobis (sur Train Sain)...")
    detector.fit_mahalanobis(train_loader)
    
    # 2. Calibration
    print(f"2. Calibration du seuil pour une PFA de {pfa_target*100}% (sur Valid Sain)...")
    thresh_recon, thresh_mah = detector.calibrate_thresholds(valid_loader, pfa=pfa_target)
    print(f"   -> Seuil Reconstruction : {thresh_recon:.4f}")
    print(f"   -> Seuil Mahalanobis    : {thresh_mah:.4f}")
    
    # 3. Test sur les données saines (Vérification de la PFA)
    print("\n--- Évaluation sur Zone 2.1 (Pures) ---")
    preds_recon_sain, _, preds_mah_sain, _ = detector.detect(test_sain_loader)
    print(f"PFA empirique (Recon) : {np.mean(preds_recon_sain)*100:.2f}% (Cible: {pfa_target*100}%)")
    print(f"PFA empirique (Mahal) : {np.mean(preds_mah_sain)*100:.2f}%")
    
    # 4. Évaluation de référence sur la Zone 2.2 (SANS anomalie)
    print("\n--- Évaluation de référence sur Zone 2.2 (Pures) ---")
    preds_recon_22, scores_recon_22, preds_mah_22, scores_mah_22 = detector.detect(anomaly_loader)
    print(f"PFA empirique Zone 2.2 (Recon) : {np.mean(preds_recon_22)*100:.2f}%")
    print(f"PFA empirique Zone 2.2 (Mahal) : {np.mean(preds_mah_22)*100:.2f}%")
    
    # 5. Test sur les anomalies de la Zone 2.2
    anomalies_to_test = [
        Crosstalk(delta=0.15).to(device),
        ChannelGainImbalance(g=1.3).to(device)
    ]
    
    # Récupération du dataset de base pour modifier son transform à la volée
    base_ds = anomaly_loader.dataset
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
        
        preds_recon_ano, scores_recon_ano, preds_mah_ano, scores_mah_ano = detector.detect(anomaly_loader)
        
        print(f"Taux de Détection (Recon) : {np.mean(preds_recon_ano)*100:.2f}%")
        print(f"Taux de Détection (Mahal) : {np.mean(preds_mah_ano)*100:.2f}%")
        
        # Calcul de l'AUC-ROC (Comparaison Z2.2 Sain vs Z2.2 Anomalie)
        with torch.no_grad():
            y_true = np.concatenate([np.zeros(len(scores_recon_22)), np.ones(len(scores_recon_ano))])
            
            auc_recon = roc_auc_score(y_true, np.concatenate([scores_recon_22, scores_recon_ano]))
            auc_mah = roc_auc_score(y_true, np.concatenate([scores_mah_22, scores_mah_ano]))
            
            print(f"Score AUC-ROC (Recon) : {auc_recon:.4f} (1.0 = Parfait)")
            print(f"Score AUC-ROC (Mahal) : {auc_mah:.4f}")
    
    # Restauration du transform original propre
    transform_holder.transform = original_transform
    
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
    train_loader, valid_loader, test_sain_loader = loaders_dict["part1_loaders"]

    if "loader2_1" not in loaders_dict or "loader2_2" not in loaders_dict:
        print("[!] ERREUR: Les loaders 'loader2_1' ou 'loader2_2' n'ont pas été trouvés.")
        print("     Veuillez vérifier votre configuration 'azimut_split' dans le fichier config.yaml.")
        sys.exit(1)
    loader_2_1, _, _ = loaders_dict["loader2_1"]
    loader_2_2, _, _ = loaders_dict["loader2_2"]
    
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

    # --- Chargement dynamique du modèle le plus récent ---
    results_dir = Path("ml_results")
    if not results_dir.exists():
        print(f"[!] ERREUR: Le dossier '{results_dir}' n'existe pas. Aucun entraînement n'a été lancé.")
        sys.exit(1)

    # Trouve tous les dossiers de run contenant le fichier du modèle
    run_dirs = [d for d in results_dir.iterdir() if d.is_dir() and (d / "best_autoencoder.pt").exists()]

    if not run_dirs:
        print(f"[!] ERREUR: Aucun entraînement valide (avec 'best_autoencoder.pt') trouvé dans '{results_dir}'.")
        print("     Veuillez d'abord lancer le script 'train_autoencoder.py'.")
        sys.exit(1)

    # Sélectionne le dossier de run le plus récent en se basant sur la date de modification
    latest_run_dir = max(run_dirs, key=os.path.getmtime)
    model_path = latest_run_dir / "best_autoencoder.pt"
    print(f"[*] Utilisation du modèle le plus récent trouvé dans : {latest_run_dir.name}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"[+] Poids du modèle chargés depuis {model_path}")

    # 3. Lancement de l'évaluation
    evaluate_ood_system(model, train_loader, valid_loader, loader_2_1, loader_2_2, device=device)

if __name__ == "__main__":
    main()