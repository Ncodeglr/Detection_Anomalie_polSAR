import sys
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
from pathlib import Path

# Ajout des chemins
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cvnn", "src")))
sys.path.append(os.path.dirname(__file__))

from cvnn.config import load_config
from cvnn.data import get_full_image_dataloader
from cvnn.inference import _assemble_image
from cvnn.physics import pauli_transform
from cvnn.data_processing import equalize

# 📍 Import de la configuration centralisée
from shared_setup import ZONES_CONFIG

def main():
    print("\n[*] Initialisation de la visualisation multi-panneaux polSAR...")
    
    script_file_path = Path(__file__).resolve()
    repo_root = script_file_path.parents[1]
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(repo_root / "configs" / "config_SimCLR.yaml")
    config = load_config(config_path)
    
    # Préparation de la config pour charger l'image entière
    config_full = copy.deepcopy(config)
    config_full["data"]["dataset"]["name"] = "ALOSDataset"
    config_full["data"]["dataset"].pop("crop_coordinates", None) 
    config_full["data"]["recompute_statistics"] = False          
    
    patch_size = config_full["data"]["dataset"].get("patch_size", 64)
    config_full["data"]["dataset"]["patch_stride"] = patch_size
    
    print("[*] Création du DataLoader pour l'image complète...")
    loader_full, n_rows, n_cols = get_full_image_dataloader(config_full, use_cuda=False)
    
    first_sample = loader_full.dataset[0]
    x_sample = first_sample[0] if isinstance(first_sample, (list, tuple)) else first_sample
    num_channels = x_sample.shape[0]
    
    print("[*] Extraction des segments...")
    original_segments = [data[0].cpu().numpy() if isinstance(data, (tuple, list)) else data.cpu().numpy() for data in loader_full]
    original_segments = [item for sublist in original_segments for item in sublist]
        
    print(f"[*] Assemblage des patchs ({n_rows}x{n_cols} patchs de taille {patch_size})...")
    full_complex_image = _assemble_image(original_segments, num_channels, n_rows, n_cols, patch_size) 
    
    print("[*] Transformation de Pauli et égalisation...")
    pauli_img = pauli_transform(full_complex_image)
    if isinstance(pauli_img, (tuple, list)):
        pauli_img = pauli_img[0]
        
    pauli_img_eq = equalize(pauli_img)
    if isinstance(pauli_img_eq, (tuple, list)):
        pauli_img_eq = pauli_img_eq[0]
        
    if isinstance(pauli_img_eq, np.ndarray) and pauli_img_eq.ndim == 3 and pauli_img_eq.shape[0] in [1, 3, 4]:
        pauli_img_eq = pauli_img_eq.transpose(1, 2, 0)
        
    if pauli_img_eq.dtype == np.float32 or pauli_img_eq.dtype == np.float64:
        pauli_img_eq = np.clip(pauli_img_eq, 0, 1)
    
    # ========================================================
    # PRÉPARATION DE LA LISTE DES ZONES À AFFICHER
    # ========================================================
    zones_to_plot = []
    
    # ⚠️ MODIFICATION ICI : On force l'utilisation de la Zone 1 depuis shared_setup
    zones_to_plot.append({
        "coords": ZONES_CONFIG["zone1_train"], "color": '#00ff00', "label": 'Zone 1 (Train)'
    })

    zones_to_plot.append({
        "coords": ZONES_CONFIG["zone2_1_saine"], "color": '#00ccff', "label": 'Zone 2.1 (Valid)'
    })

    z22 = ZONES_CONFIG["zone2_2_anomalies_global"]
    row_split_points = np.linspace(z22["start_row"], z22["end_row"], 4, dtype=int)
    colors_2_2 = ['#ffaa00', '#ff3300', '#cc00ff']
    
    for i in range(3):
        zones_to_plot.append({
            "coords": {
                "start_row": int(row_split_points[i]), 
                "end_row": int(row_split_points[i+1]), 
                "start_col": z22["start_col"], 
                "end_col": z22["end_col"]
            },
            "color": colors_2_2[i], 
            "label": f'Zone 2.2 (Anomalie P{i+1})'
        })

    # ========================================================
    # GÉNÉRATION DE LA FIGURE MULTI-PANNEAUX (Façon Papier)
    # ========================================================
    print("[*] Génération de la carte des zones (Style Publication)...")
    
    # Création de la figure globale
    fig = plt.figure(figsize=(22, 14))
    
    # Configuration de GridSpec : 5 lignes (pour les 5 zones zoomées) et 2 colonnes.
    # width_ratios=[2.5, 1] donne beaucoup plus de largeur à l'image principale.
    num_zones = len(zones_to_plot)
    gs = gridspec.GridSpec(num_zones, 2, width_ratios=[2.5, 1], wspace=0.05, hspace=0.15)
    
    # --- 1. L'IMAGE PRINCIPALE GAUCHE ---
    ax_main = fig.add_subplot(gs[:, 0]) # S'étend sur toutes les lignes de la colonne 0
    ax_main.imshow(pauli_img_eq)
    ax_main.set_title("ALOS2-San Francisco", fontsize=18, pad=15)
    ax_main.axis('off') # On cache les axes pour un rendu plus propre
    
    # --- 2. TRAITEMENT DE CHAQUE ZONE (Boxes + Zooms Droite) ---
    for i, zone in enumerate(zones_to_plot):
        c = zone["coords"]
        color = zone["color"]
        label = zone["label"]
        
        start_row, end_row = c["start_row"], c["end_row"]
        start_col, end_col = c["start_col"], c["end_col"]
        width = end_col - start_col
        height = end_row - start_row
        
        # A. Dessin du rectangle sur l'image principale
        rect = patches.Rectangle(
            (start_col, start_row), width, height,
            linewidth=3, edgecolor=color, facecolor='none',
            path_effects=[pe.Stroke(linewidth=5, foreground='white'), pe.Normal()]
        )
        ax_main.add_patch(rect)
        
        # Ajout du texte juste au-dessus de la boîte sur l'image principale
        ax_main.text(
            start_col, start_row - 40, label, 
            color=color, fontsize=13, fontweight='bold',
            path_effects=[pe.withStroke(linewidth=3, foreground="black")]
        )
        
        # B. Affichage du zoom sur la colonne de droite
        ax_zoom = fig.add_subplot(gs[i, 1])
        
        # Découpage du patch exact dans l'image
        cropped_img = pauli_img_eq[start_row:end_row, start_col:end_col]
        ax_zoom.imshow(cropped_img)
        
        # Esthétique du zoom : Titre coloré et bordure épaisse de la couleur de la zone
        ax_zoom.set_title(label, color=color, fontsize=14, fontweight='bold', pad=10, 
                          path_effects=[pe.withStroke(linewidth=2, foreground="black")])
        
        ax_zoom.axis('on')          # Activer les axes pour afficher la bordure
        ax_zoom.set_xticks([])      # Mais cacher les ticks X
        ax_zoom.set_yticks([])      # Et cacher les ticks Y
        
        # Application de la couleur de bordure (spine)
        for spine in ax_zoom.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(4)
            spine.set_path_effects([pe.Stroke(linewidth=6, foreground='white'), pe.Normal()])

    # Finitions et sauvegarde
    output_path = repo_root / "zones_visualization_publication.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n[+] Succès ! Visualisation sauvegardée sous : {output_path}")

if __name__ == "__main__":
    main()