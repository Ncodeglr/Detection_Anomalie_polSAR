import sys
import os
import copy
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Ajout du dossier src au chemin pour pouvoir importer cvnn si vous ne l'avez pas installé via pip/poetry
sys.path.append(os.path.join(os.path.dirname(__file__), "cvnn", "src"))

# Ajout des imports pour charger et assembler l'image complète
from cvnn.data import get_full_image_dataloader
from cvnn.config import load_config
from cvnn.inference import _assemble_image
from cvnn.physics import pauli_transform
from cvnn.data_processing import equalize

def visualize_vertical_zones(full_image: np.ndarray):
    """
    Affiche l'image radar globale avec nos nouvelles zones rectangulaires empilées verticalement.
    """
    print("   [*] Correction de l'orientation pour l'affichage (retournement vertical)...")
    if full_image.ndim == 3:
        full_image_vis = full_image[:, ::-1, :]
        H, W = full_image_vis.shape[1], full_image_vis.shape[2]
    else:
        full_image_vis = full_image[::-1, :]
        H, W = full_image_vis.shape[0], full_image_vis.shape[1]

    # --- CORRECTION DE L'ASPECT RATIO ---
    base_width = 12
    proportional_height = base_width * (H / W)
    fig, ax = plt.subplots(figsize=(base_width, proportional_height))
    # ------------------------------------

    # Transformation Pauli et égalisation (Ta logique d'affichage)
    if full_image_vis.ndim == 3 and full_image_vis.shape[0] in [3, 4]:
        pauli_img = pauli_transform(full_image_vis)
        pauli_img, _ = equalize(pauli_img)
        display_img = pauli_img.transpose(1, 2, 0)
        
        ax.imshow(display_img, origin='lower', aspect='equal')
    else:
        img_eq, _ = equalize(full_image_vis[0:1] if full_image_vis.ndim == 3 else full_image_vis)
        if img_eq.ndim == 3: img_eq = img_eq.transpose(1, 2, 0).squeeze()
        
        ax.imshow(img_eq, cmap='gray', origin='lower', aspect='equal')
    
    # --- DESSIN DES ZONES ---
    # ATTENTION : Vu que l'image est retournée verticalement (H-y) et qu'on utilise origin='lower',
    # le "bas" de notre rectangle sur le graphe correspond mathématiquement à (H - r_end).
    
    col_start, col_end = 2832, 7888
    width = col_end - col_start

    # 1. ZONE PolSF Originale (Pour référence)
    r_start_polsf, r_end_polsf = 736, 3520
    rect_polsf = patches.Rectangle(
        (col_start, H - r_end_polsf), width, r_end_polsf - r_start_polsf,
        linewidth=2, edgecolor='yellow', facecolor='none', linestyle=':', label='PolSF (Entraînement)'
    )
    ax.add_patch(rect_polsf)
    ax.text(col_start + 150, H - r_start_polsf - 300, 'PolSF\n(Original)', color='yellow', fontsize=12, weight='bold',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

    # 2. ZONE 2.1 (Région Saine Non Vue)
    r_start_z21, r_end_z21 = 4000, 6000
    rect_2_1 = patches.Rectangle(
        (col_start, H - r_end_z21), width, r_end_z21 - r_start_z21,
        linewidth=2, edgecolor='cyan', facecolor='cyan', alpha=0.3, label='Zone 2.1 (Saine)'
    )
    ax.add_patch(rect_2_1)
    ax.text(col_start + 150, H - r_start_z21 - 300, 'ZONE 2.1\n(Saine)', color='cyan', fontsize=12, weight='bold', 
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

    # 3. ZONE 2.2 (Région avec Anomalies)
    r_start_z22, r_end_z22 = 6500, 8500
    rect_2_2 = patches.Rectangle(
        (col_start, H - r_end_z22), width, r_end_z22 - r_start_z22,
        linewidth=2, edgecolor='magenta', facecolor='magenta', alpha=0.2, label='Zone 2.2 (Anomalies)'
    )
    ax.add_patch(rect_2_2)
    
    # Lignes de subdivision de la Zone 2.2 en 3 parties
    row_split_points = np.linspace(r_start_z22, r_end_z22, 4, dtype=int)
    for i in range(3):
        r_start = row_split_points[i]
        r_end = row_split_points[i+1]
        rect_sub = patches.Rectangle(
            (col_start, H - r_end), width, r_end - r_start,
            linewidth=1.5, edgecolor='white', facecolor='none', linestyle='--'
        )
        ax.add_patch(rect_sub)
        ax.text(col_start + 150, H - r_start - 250, f'Sous-zone {i+1}', color='white', fontsize=10, weight='bold',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
    
    plt.legend(loc='upper right')
    plt.title("Répartition Verticale des Zones sur l'image Radar (SAN_FRANCISCO_ALOS2)")
    
    save_path = "radar_zones_vertical_visualization.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"   [+] L'image a été sauvegardée avec succès sous : {save_path}")
    plt.close(fig)

def main():
    if len(sys.argv) >= 2:
        config_path = sys.argv[1]
    else:
        config_path = "configs/config_Unet.yaml"
        print(f"[*] Aucun fichier de configuration spécifié. Utilisation par défaut : {config_path}")

    print(f"[*] Chargement de la configuration depuis {config_path}...")

    if not os.path.exists(config_path):
        print(f"[!] ERREUR : Le fichier '{config_path}' est introuvable.")
        sys.exit(1)
        
    cfg = load_config(config_path)
    
    # Désactiver le calcul des statistiques pour tester rapidement l'affichage
    if "data" in cfg:
        cfg["data"]["recompute_statistics"] = False
        if "transforms" in cfg["data"]:
            cfg["data"]["transforms"] = [
                t for t in cfg["data"]["transforms"] 
                if t.get("name", "").lower() not in ["normalize", "complexnorm", "logamplitude", "global_scalar_normalize"]
            ]

    # --- VISUALISATION ---
    print("\n[*] Génération de la visualisation de la carte (Nouvelle Logique Verticale)...")
    try:
        cfg_temp = copy.deepcopy(cfg)
        if "crop_coordinates" in cfg_temp["data"]["dataset"]:
            del cfg_temp["data"]["dataset"]["crop_coordinates"]
            
        print("    -> Chargement des patchs pour reconstruire l'image globale...")
        loader_full, n_rows, n_cols = get_full_image_dataloader(cfg_temp, use_cuda=False)
        
        patches_list = []
        indices = []
        for batch in loader_full:
            inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
            patches_list.extend([seg for seg in inputs.cpu().numpy()])
            
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                idx = batch[1] if len(batch) == 2 else batch[2]
                indices.extend(idx.cpu().detach().numpy())
                    
        patch_size = cfg_temp["data"]["dataset"]["patch_size"]
        num_channels = patches_list[0].shape[0]
        
        print("    -> Assemblage de l'image radar...")
        full_image = _assemble_image(patches_list, num_channels, n_rows, n_cols, patch_size, indices if len(indices) > 0 else None)
        
        print("    -> Génération et sauvegarde de l'image...")
        # Appel de notre nouvelle fonction
        visualize_vertical_zones(full_image)
            
    except Exception as e:
        print(f"    [!] Erreur lors de la visualisation : {e}")

if __name__ == "__main__":
    main()