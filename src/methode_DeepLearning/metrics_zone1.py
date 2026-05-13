import sys
import os
import pprint
from pathlib import Path
import torch
import numpy as np

# Ajout des chemins pour importer les modules du projet
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "cvnn", "src"))

# --- Contournement (Mock) de l'import manquant sans toucher à cvnn ---
from unittest.mock import MagicMock
sys.modules['cvnn.models.feature_extractors'] = MagicMock()

from cvnn.config import load_config
from cvnn.data import azimut_split
from cvnn.models import LatentAutoEncoder
from cvnn.evaluate import evaluate
from cvnn.inference import reconstruct_full_image
from cvnn.data_processing import revert_transforms
from cvnn.metrics_registry import (
    compute_h_alpha_metrics,
    compute_cameron_metrics,
    compute_reconstruction_errors,
)

def main():
    print("[*] Démarrage de l'évaluation des métriques (Zone 1)...")
    repo_root = Path(__file__).resolve().parents[2]
    
    if len(sys.argv) >= 2:
        config_path = sys.argv[1]
    else:
        config_path = str(repo_root / "configs" / "config.yaml")
        
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Résolution absolue du chemin des données
    trainpath = Path(config["data"]["dataset"]["trainpath"])
    if not trainpath.is_absolute():
        config["data"]["dataset"]["trainpath"] = str((repo_root / trainpath).resolve())

    # 1. Chargement des données via azimut_split
    print("[*] Chargement des données Test de la Zone 1 pour l'évaluation des métriques")
    loaders_dict = azimut_split(config, use_cuda=torch.cuda.is_available())
    _, _ , test_loader1 = loaders_dict["loader1_splits"]
    loaders_part1_full, n_rows_1_full, n_cols_1_full = loaders_dict["loader1_full"]
    
    # 2. Chargement du modèle
    print("[*] Chargement du modèle pré-entraîné...")
    model_cfg = config.get("model", {})
    in_channels = config["data"].get("inferred_input_channels", 4)
    input_size = config["data"].get("inferred_input_size", config["data"]["dataset"].get("patch_size", 16))
    
    model = LatentAutoEncoder(
        num_channels=in_channels,
        num_layers=model_cfg.get("num_layers", 3),
        channels_width=model_cfg.get("channels_width", 16),
        input_size=input_size,
        activation=model_cfg.get("activation", "relu"),
        upsampling_layer=model_cfg.get("upsampling_layer", "conv_transpose"),
        layer_mode=model_cfg.get("layer_mode", "complex"),
        normalization_layer=model_cfg.get("normalization_layer", "batch"),
        residual=model_cfg.get("residual", False),
        num_blocks=model_cfg.get("num_blocks", 1),
        latent_dim=model_cfg.get("latent_dim", 128)
    ).to(device)

    # Recherche des poids
    results_dir = Path("ml_results")
    run_dirs = [d for d in results_dir.iterdir() if d.is_dir() and (d / "best_weights_autoencoder.pt").exists()]
    if not run_dirs:
        print(f"[!] ERREUR: Aucun modèle trouvé dans {results_dir}.")
        sys.exit(1)
        
    latest_run_dir = max(run_dirs, key=os.path.getmtime)
    model_path = latest_run_dir / "best_autoencoder.pt"
    if not model_path.exists():
        model_path = latest_run_dir / "best_weights_autoencoder.pt"
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"   [+] Poids chargés depuis {model_path}")

    # 3. Évaluation sur les patchs (PSNR, SSIM, MSE, etc.)
    # CORRECTION : Passage des arguments exactement comme dans experiment.py
    print("\n[*] Évaluation des métriques sur les patchs (loader1_test)...")
    metrics = evaluate(
        task=config.get("task", "reconstruction"),
        model=model,
        test_loader=test_loader1,
        max_samples=config.get("evaluation", {}).get("max_samples", 10000),
        check_invariance=config.get("evaluation", {}).get("check_invariance", False),
        layer_mode=model_cfg.get("layer_mode", "complex"),
        dataset_name=config.get("data", {}).get("dataset", {}).get("name", "PolSFDataset"),
        real_pipeline_type=config.get("data", {}).get("real_pipeline_type"),
        device=device,
        domain=config.get("evaluation", {}).get("domain", "spatial")
    )
    pprint.pprint(metrics.get("metrics", metrics))

    # 4. Évaluation Physique sur l'image complète
    if config["data"].get("supports_full_image_reconstruction"):
        print(f"\n[*] Reconstruction de l'image complète Zone 1 ({n_rows_1_full}x{n_cols_1_full} patchs)...")
        original_img, recon_img = reconstruct_full_image(
            model=model, 
            full_loader=loaders_part1_full, 
            config=config,
            nsamples_per_rows=n_rows_1_full, 
            nsamples_per_cols=n_cols_1_full, 
            device=device
        )

        original_img = revert_transforms(original_img, config)
        recon_img = revert_transforms(recon_img, config)

        print("\n[*] Calcul des erreurs de reconstruction physiques...")
        recon_errors = compute_reconstruction_errors(original=original_img, reconstructed=recon_img, cfg=config)
        
        if "amplitude_difference" in recon_errors and recon_errors["amplitude_difference"].size > 0:
            mean_amp_diff = np.mean(np.abs(recon_errors["amplitude_difference"]))
            print(f"   -> Différence d'amplitude (MAE) : {mean_amp_diff:.4f}")
        if "angular_distance" in recon_errors and recon_errors["angular_distance"].size > 0:
            mean_ang_dist = np.mean(np.abs(recon_errors["angular_distance"]))
            print(f"   -> Distance angulaire moyenne   : {np.degrees(mean_ang_dist):.2f}°")

        # CORRECTION : Ajout de la vérification du type de données (PolSAR) avant de calculer H-Alpha et Cameron
        if config["data"].get("type", "").lower() == "polsar":
            print("\n[*] Calcul des métriques H-Alpha et Cameron...")
            h_alpha_metrics = compute_h_alpha_metrics(image1=original_img, image2=recon_img)
            print(f"   -> Accuracy H-Alpha : {h_alpha_metrics.get('accuracy', 0)*100:.2f}%")
            print(f"   -> Kappa H-Alpha    : {h_alpha_metrics.get('cohen_kappa', 0):.4f}")
            
            cameron_metrics = compute_cameron_metrics(image1=original_img, image2=recon_img)
            print(f"   -> Accuracy Cameron : {cameron_metrics.get('accuracy', 0)*100:.2f}%")
            print(f"   -> Kappa Cameron    : {cameron_metrics.get('cohen_kappa', 0):.4f}")
    else:
        print("\n[!] La configuration n'autorise pas la reconstruction de l'image complète (supports_full_image_reconstruction=False).")

if __name__ == "__main__":
    main()