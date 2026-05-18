import sys
import os
import copy
import torch
import matplotlib.pyplot as plt
import numpy as np

# Ajout du dossier src au chemin pour pouvoir importer cvnn si vous ne l'avez pas installé via pip/poetry
sys.path.append(os.path.join(os.path.dirname(__file__), "cvnn", "src"))

# Ajout des imports pour charger et assembler l'image complète
from cvnn.data import azimut_split, get_full_image_dataloader
from cvnn.config import load_config
from cvnn.inference import _assemble_image
from cvnn.physics import pauli_transform
from cvnn.data_processing import equalize

def visualize_azimut_split(full_image: np.ndarray, x1: int, x2: int):
    """
    Affiche l'image radar globale avec les lignes de découpe.
    """
    print("   [*] Correction de l'orientation pour l'affichage (retournement vertical)...")
    if full_image.ndim == 3:
        full_image_vis = full_image[:, ::-1, :]
        H, W = full_image_vis.shape[1], full_image_vis.shape[2]
    else:
        full_image_vis = full_image[::-1, :]
        H, W = full_image_vis.shape[0], full_image_vis.shape[1]

    # --- CORRECTION DE L'ASPECT RATIO ---
    # On fixe une largeur de base (ex: 12 pouces) et on calcule la hauteur proportionnellement
    base_width = 12
    proportional_height = base_width * (H / W)
    plt.figure(figsize=(base_width, proportional_height))
    # ------------------------------------

    # Transformation Pauli et égalisation
    if full_image_vis.ndim == 3 and full_image_vis.shape[0] in [3, 4]:
        pauli_img = pauli_transform(full_image_vis)
        pauli_img, _ = equalize(pauli_img)
        display_img = pauli_img.transpose(1, 2, 0)
        
        # Ajout de aspect='equal' pour forcer des pixels carrés
        plt.imshow(display_img, origin='lower', aspect='equal')
    else:
        img_eq, _ = equalize(full_image_vis[0:1] if full_image_vis.ndim == 3 else full_image_vis)
        if img_eq.ndim == 3: img_eq = img_eq.transpose(1, 2, 0).squeeze()
        
        # Ajout de aspect='equal' pour forcer des pixels carrés
        plt.imshow(img_eq, cmap='gray', origin='lower', aspect='equal')
    
    # Ajout des lignes de découpe
    plt.axhline(y=H - x1, color='red', linestyle='--', linewidth=3, label=f'Coupe 1 (x1={x1})')
    plt.axhline(y=H - x2, color='white', linestyle='--', linewidth=3, label=f'Coupe 2 (x2={x2})')
    
    # Annotations des zones
    plt.text(100, H - x1/2, 'ZONE 1 (Train/Valid/Test)', color='red', fontsize=14, fontweight='bold', bbox=dict(facecolor='black', alpha=0.5))
    plt.text(100, H - (x1 + x2)/2, 'ZONE 2.1 (Test A)', color='white', fontsize=14, fontweight='bold', bbox=dict(facecolor='black', alpha=0.5))
    plt.text(100, (H - x2)/2, 'ZONE 2.2 (Test B)', color='white', fontsize=14, fontweight='bold', bbox=dict(facecolor='black', alpha=0.5))
    
    plt.legend()
    plt.title("Répartition des zones sur l'image Radar (SAN_FRANCISCO_ALOS2)")
    plt.show()

def main():
    if len(sys.argv) >= 2:
        config_path = sys.argv[1]
    else:
        config_path = "configs/config.yaml"
        print(f"[*] Aucun fichier de configuration spécifié. Utilisation par défaut : {config_path}")

    print(f"[*] Chargement de la configuration depuis {config_path}...")

    if not os.path.exists(config_path):
        print(f"[!] ERREUR : Le fichier '{config_path}' est introuvable.")
        print(f"[!] Veuillez vérifier que vous l'avez bien sauvegardé.")
        sys.exit(1)
        
    cfg = load_config(config_path)
    
    # Désactiver le calcul des statistiques et retirer les transformations 
    # qui en dépendent pour tester rapidement les tailles des tenseurs
    if "data" in cfg:
        cfg["data"]["recompute_statistics"] = False
        if "transforms" in cfg["data"]:
            cfg["data"]["transforms"] = [
                t for t in cfg["data"]["transforms"] 
                if t.get("name", "").lower() not in ["normalize", "complexnorm", "logamplitude", "global_scalar_normalize"]
            ]

    # --- MÉTHODE 1 : VISUALISATION ---
    print("\n[*] Génération de la visualisation des coupes (Méthode 1)...")
    try:
        dataset_cfg = cfg["data"]["dataset"]
        # On tente de récupérer x1 et x2 (valeurs par défaut de secours si non trouvées)
        x1 = dataset_cfg.get("azimut_split_x1", 4000)
        x2 = dataset_cfg.get("azimut_split_x2", 7000)
        
        cfg_temp = copy.deepcopy(cfg)
        # On retire les coordonnées de coupe temporairement pour charger l'image entière
        if "crop_coordinates" in cfg_temp["data"]["dataset"]:
            del cfg_temp["data"]["dataset"]["crop_coordinates"]
            
        print("    -> Chargement des patchs pour reconstruire l'image globale (cela peut prendre quelques secondes)...")
        loader_full, n_rows, n_cols = get_full_image_dataloader(cfg_temp, use_cuda=False)
        
        patches = []
        indices = []
        for batch in loader_full:
            inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
            patches.extend([seg for seg in inputs.cpu().numpy()])
            
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                idx = batch[1] if len(batch) == 2 else batch[2]
                indices.extend(idx.cpu().detach().numpy())
                    
        patch_size = cfg_temp["data"]["dataset"]["patch_size"]
        num_channels = patches[0].shape[0]
        
        print("    -> Assemblage de l'image radar...")
        full_image = _assemble_image(patches, num_channels, n_rows, n_cols, patch_size, indices if len(indices) > 0 else None)
        
        print("    -> L'image va s'ouvrir. Fermez la fenêtre pour continuer l'exécution du script.")
        visualize_azimut_split(full_image, x1, x2)
            
    except Exception as e:
        print(f"    [!] Erreur lors de la visualisation : {e}")
    # ----------------------------------------

    print("\n[*] Initialisation de la découpe azimut...")
    use_cuda = torch.cuda.is_available()
    
    # azimut_split retourne un dictionnaire avec les loaders
    split_results = azimut_split(cfg, use_cuda=use_cuda)
    
    print("\n--- Résultats de la découpe ---")
    for part_name, loaders in split_results.items():
        print(f"\n=== {part_name.upper()} ===")
        
        # Gérer la différence de structure entre la Zone 1 et les Zones 2
        if part_name == "part1_loaders":
            # loaders est un tuple (train_loader, valid_loader, [test_loader])
            names = ["Train", "Validation", "Test"]
            loader_list = zip(names, loaders)
        else:
            # loaders est un tuple (dataloader, nsamples_per_rows, nsamples_per_cols)
            loader_list = [("Full Image Loader", loaders[0])]
            print(f"  [*] Grille : {loaders[1]} lignes x {loaders[2]} colonnes de patchs")
            
        for name, loader in loader_list:
            print(f"\n  [*] Loader : {name}")
            print(f"      Nombre de batchs : {len(loader)}")
            
            # Afficher les infos du premier batch
            for batch in loader:
                inputs = batch[0] if isinstance(batch, (list, tuple)) else batch.get("data", batch.get("inputs")) if isinstance(batch, dict) else batch
                print(f"      Forme (Shape)    : {inputs.shape}")
                print(f"      Type (dtype)     : {inputs.dtype}")
                break

if __name__ == "__main__":
    main()