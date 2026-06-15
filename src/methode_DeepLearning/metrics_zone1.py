import sys
import os
import pprint
import json
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

# Ajout des chemins pour importer les modules du projet
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "cvnn", "src"))

# --- Contournement (Mock) de l'import manquant sans toucher à cvnn ---
from unittest.mock import MagicMock
sys.modules['cvnn.models.feature_extractors'] = MagicMock()

from cvnn.config import load_config
from cvnn.data import azimut_split, get_dataset_split_indices
from cvnn.models import LatentAutoEncoder
from cvnn.evaluate import evaluate
from cvnn.inference import reconstruct_full_image, inference_on_dataloader
from cvnn.data_processing import revert_transforms
from cvnn.metrics_registry import (
    compute_h_alpha_metrics,
    compute_cameron_metrics,
    compute_reconstruction_errors,
)
from cvnn.visualize import (
    create_dataset_split_mask,
    plot_dataset_split_mask,
    plot_reconstructions,
    plot_pauli_decomposition,
    plot_krogager_decomposition,
    plot_cameron_decomposition,
    plot_h_alpha_decomposition,
    plot_h_alpha_plane,
    plot_reconstruction_error_analysis,
    plot_classification_metrics,
)

class NumpyEncoder(json.JSONEncoder):
    """Encodeur personnalisé pour pouvoir sauvegarder des tableaux Numpy en JSON."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)


# =====================================================================
# PARTIE 1 : CALCUL DES MÉTRIQUES
# =====================================================================
def compute_and_save_metrics(model, test_loader, full_loader, n_rows, n_cols, config, device, run_dir):
    """Calcule toutes les métriques (patchs et image complète) et les sauvegarde."""
    print("\n" + "="*50)
    print("[*] ÉTAPE 1 : CALCUL DES MÉTRIQUES")
    print("="*50)
    
    # 1. Évaluation sur les patchs
    print("\n[*] Évaluation sur les patchs (loader1_test)...")
    patch_metrics = evaluate(
        task=config.get("task", "reconstruction"),
        model=model,
        test_loader=test_loader,
        max_samples=config.get("evaluation", {}).get("max_samples", int(1e9)),
        check_invariance=config.get("evaluation", {}).get("check_invariance", False),
        layer_mode=config.get("model", {}).get("layer_mode", "complex"),
        dataset_name=config.get("data", {}).get("dataset", {}).get("name"),
        real_pipeline_type=config.get("data", {}).get("real_pipeline_type"),
        device=device,
        domain=config.get("evaluation", {}).get("domain", "spatial")
    )
    pprint.pprint(patch_metrics.get("metrics", patch_metrics))
    metrics_to_save = {"patch_metrics": patch_metrics.get("metrics", patch_metrics)}

    # Initialisation des variables de retour
    original_img, recon_img = None, None
    recon_errors, h_alpha_metrics, cameron_metrics = {}, {}, {}

    # 2. Évaluation Physique sur l'image complète
    if config["data"].get("supports_full_image_reconstruction"):
        print(f"\n[*] Reconstruction de l'image complète ({n_rows}x{n_cols} patchs)...")
        original_img, recon_img = reconstruct_full_image(
            model=model, 
            full_loader=full_loader, 
            config=config,
            nsamples_per_rows=n_rows, 
            nsamples_per_cols=n_cols, 
            device=device
        )

        original_img = revert_transforms(original_img, config)
        recon_img = revert_transforms(recon_img, config)

        
        print("\n[*] Calcul des erreurs de reconstruction physiques...")
        recon_errors = compute_reconstruction_errors(original=original_img, reconstructed=recon_img, cfg=config)
        
        if "amplitude_difference" in recon_errors and recon_errors["amplitude_difference"].size > 0:
            mean_amp_diff = np.mean(np.abs(recon_errors["amplitude_difference"]))
            print(f"   -> Différence d'amplitude (MAE) : {mean_amp_diff:.4f}")
            metrics_to_save["mean_amplitude_difference"] = mean_amp_diff
            
        if "angular_distance" in recon_errors and recon_errors["angular_distance"].size > 0:
            mean_ang_dist = np.mean(np.abs(recon_errors["angular_distance"]))
            print(f"   -> Distance angulaire moyenne   : {np.degrees(mean_ang_dist):.2f}°")
            metrics_to_save["mean_angular_distance_degrees"] = np.degrees(mean_ang_dist)

        # Calculs PolSAR spécifiques
        if config["data"].get("type", "").lower() == "polsar":
            print("\n[*] Calcul des métriques H-Alpha et Cameron...")
            h_alpha_metrics = compute_h_alpha_metrics(image1=original_img, image2=recon_img)
            print(f"   -> Accuracy H-Alpha : {h_alpha_metrics.get('accuracy', 0)*100:.2f}%")
            metrics_to_save["h_alpha_metrics"] = h_alpha_metrics
            
            cameron_metrics = compute_cameron_metrics(image1=original_img, image2=recon_img)
            print(f"   -> Accuracy Cameron : {cameron_metrics.get('accuracy', 0)*100:.2f}%")
            metrics_to_save["cameron_metrics"] = cameron_metrics
    else:
        print("\n[!] Reconstruction de l'image complète non supportée par la configuration.")

    # Sauvegarde JSON
    metrics_file = run_dir / "evaluation_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics_to_save, f, indent=4, cls=NumpyEncoder)
    print(f"\n[*] Métriques sauvegardées dans : {metrics_file}")

    return original_img, recon_img, recon_errors, h_alpha_metrics, cameron_metrics


# =====================================================================
# PARTIE 2 : VISUALISATIONS
# =====================================================================
def generate_visualizations(model, test_loader, full_loader, n_rows, n_cols, original_img, recon_img, recon_errors, h_alpha_metrics, cameron_metrics, config, device, run_dir):
    """Génère et sauvegarde tous les graphiques."""
    print("\n" + "="*50)
    print("[*] ÉTAPE 2 : GÉNÉRATION DES VISUALISATIONS")
    print("="*50)
    
    vis_dir = run_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # 1. Reconstructions sur les patchs
    print("   -> Sauvegarde de reconstructions_patches.png")
    inputs, outputs, _, _, _, _ = inference_on_dataloader(model=model, data_loader=test_loader, device=device)
    inputs = revert_transforms(inputs, config)
    outputs = revert_transforms(outputs, config)
    fig = plot_reconstructions(
        inputs=inputs, outputs=outputs,
        dataset_type=config["data"].get("type", "polsar").lower(),
        num_samples=5, show_spectrum=False,
    )
    fig.savefig(vis_dir / "reconstructions_patches.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. Visualisations de l'image complète
    if config["data"].get("supports_full_image_reconstruction") and original_img is not None:
        
        # Erreurs de reconstruction
        print("   -> Sauvegarde de reconstruction_error_analysis.png")
        fig = plot_reconstruction_error_analysis(recon_errors)
        fig.savefig(vis_dir / "reconstruction_error_analysis.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Carte de split
        try:
            print("   -> Sauvegarde de dataset_split_visualization.png")
            train_idx, valid_idx, test_idx = get_dataset_split_indices(config)
            mask = create_dataset_split_mask(
                cfg=config, full_loader=full_loader,
                train_indices=train_idx, valid_indices=valid_idx, test_indices=test_idx,
                nsamples_per_cols=n_cols, nsamples_per_rows=n_rows,
            )
            fig = plot_dataset_split_mask(
                mask=mask, patch_size=config["data"]["dataset"].get("patch_size", 16),
                train_indices=train_idx, valid_indices=valid_idx, test_indices=test_idx,
            )
            fig.savefig(vis_dir / "dataset_split_visualization.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"   [!] Impossible de générer la carte de split : {e}")

        # Décompositions PolSAR
        if config["data"].get("type", "").lower() == "polsar":
            
            print("   [*] Correction de l'orientation pour l'affichage (retournement vertical)...")
            original_img_vis = original_img
            recon_img_vis = recon_img
            # ------------------------------------

            print(f"   [*] Dimensions de l'image originale    : {original_img_vis.shape}")
            print(f"   [*] Dimensions de l'image reconstruite : {recon_img_vis.shape}")

            print("   -> Sauvegarde de pauli_zone1_complete.png")
            fig = plot_pauli_decomposition(original_img_vis, recon_img_vis)
            fig.savefig(vis_dir / "pauli_zone1_complete.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

            print("   -> Sauvegarde des autres décompositions (Krogager, H-Alpha, Cameron)...")
            fig = plot_krogager_decomposition(original_img_vis, recon_img_vis)
            fig.savefig(vis_dir / "krogager_decomposition.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

            fig = plot_h_alpha_decomposition(original_img_vis, recon_img_vis)
            fig.savefig(vis_dir / "h_alpha_decomposition.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

            # Note: Le plan H-Alpha (nuage de points) n'est pas une image spatiale, 
            # donc on peut lui passer les images normales (non retournées).
            fig = plot_h_alpha_plane(original_img, recon_img)
            fig.savefig(vis_dir / "h_alpha_plane.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

            fig = plot_cameron_decomposition(original_img_vis, recon_img_vis)
            fig.savefig(vis_dir / "cameron_decomposition.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

            if h_alpha_metrics:
                fig = plot_classification_metrics(
                    h_alpha_metrics,
                    class_names={1: "Complex", 2: "Random aniso", 4: "Double refl", 5: "Aniso particles", 6: "Random surf", 7: "Dihedral", 8: "Dipole", 9: "Bragg"}
                )
                fig.savefig(vis_dir / "h_alpha_classification_metrics.png", dpi=300, bbox_inches="tight")
                plt.close(fig)

            if cameron_metrics:
                fig = plot_classification_metrics(
                    cameron_metrics,
                    class_names={1: "Non-reciprocal", 2: "Asymmetric", 3: "Left helix", 4: "Right helix", 5: "Symmetric", 6: "Trihedral", 7: "Dihedral", 8: "Dipole", 9: "Cylinder", 10: "Narrow dihedral", 11: "Quarter-wave"}
                )
                fig.savefig(vis_dir / "cameron_classification_metrics.png", dpi=300, bbox_inches="tight")
                plt.close(fig)

    print(f"\n[*] Visualisations terminées. Sauvegardées dans : {vis_dir}")


# =====================================================================
# PARTIE 3 : FONCTION PRINCIPALE (SETUP)
# =====================================================================
def main():
    print("[*] Initialisation du script...")
    repo_root = Path(__file__).resolve().parents[2]
    config_path = sys.argv[1] if len(sys.argv) >= 2 else str(repo_root / "configs" / "config_Unet.yaml")
        
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Résolution du chemin
    trainpath = Path(config["data"]["dataset"]["trainpath"])
    if not trainpath.is_absolute():
        config["data"]["dataset"]["trainpath"] = str((repo_root / trainpath).resolve())

    # Chargement des données
    loaders_dict = azimut_split(config, use_cuda=torch.cuda.is_available())
    _, _, test_loader1 = loaders_dict["loader1_splits"]
    loaders_part1_full, n_rows_1_full, n_cols_1_full = loaders_dict["loader1_full"]
    
    # Construction du modèle
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

    # Chargement des poids
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
    print(f"[*] Poids chargés depuis {model_path}")

    # --- APPEL DES DEUX GRANDES FONCTIONS ---
    
    # 1. Calculs
    orig_img, recon_img, errors, h_alpha, cameron = compute_and_save_metrics(
        model=model, 
        test_loader=test_loader1, 
        full_loader=loaders_part1_full, 
        n_rows=n_rows_1_full, 
        n_cols=n_cols_1_full, 
        config=config, 
        device=device, 
        run_dir=latest_run_dir
    )

    # 2. Dessins
    generate_visualizations(
        model=model, 
        test_loader=test_loader1, 
        full_loader=loaders_part1_full, 
        n_rows=n_rows_1_full, 
        n_cols=n_cols_1_full, 
        original_img=orig_img, 
        recon_img=recon_img, 
        recon_errors=errors, 
        h_alpha_metrics=h_alpha, 
        cameron_metrics=cameron, 
        config=config, 
        device=device, 
        run_dir=latest_run_dir
    )

if __name__ == "__main__":
    main()
