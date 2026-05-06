import sys
import os
import copy
import torch
import matplotlib.pyplot as plt
import numpy as np

# Ajout du dossier src au chemin pour pouvoir importer cvnn si vous ne l'avez pas installé via pip/poetry
sys.path.append(os.path.join(os.path.dirname(__file__), "cvnn", "src"))

# Ajout des imports _create_dataset et _parse_dataset_config pour charger l'image complète
from cvnn.data import azimut_split, _create_dataset, _parse_dataset_config
from cvnn.config import load_config

def visualize_azimut_split(labels_full_image: np.ndarray, x1: int, x2: int):
    """
    Affiche la carte de vérité terrain globale avec les lignes de découpe.
    """
    plt.figure(figsize=(10, 15))
    
    # Affichage de la carte des labels
    plt.imshow(labels_full_image, cmap='tab20', interpolation='nearest')
    plt.colorbar(label='Classes')
    
    # Ajout des lignes de découpe
    plt.axhline(y=x1, color='red', linestyle='--', linewidth=3, label=f'Coupe 1 (x1={x1})')
    plt.axhline(y=x2, color='white', linestyle='--', linewidth=3, label=f'Coupe 2 (x2={x2})')
    
    # Annotations des zones
    plt.text(100, x1/2, 'ZONE 1 (Train/Valid)', color='red', fontsize=14, fontweight='bold')
    plt.text(100, x1 + (x2-x1)/2, 'ZONE 2.1 (Test A)', color='white', fontsize=14, fontweight='bold')
    plt.text(100, x2 + 500, 'ZONE 2.2 (Test B)', color='white', fontsize=14, fontweight='bold')
    
    plt.legend()
    plt.title("Répartition des classes selon les coupes azimutales")
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
            
        dataset_info = _parse_dataset_config(cfg_temp)
        
        # Chargement du dataset sans transformation pour récupérer uniquement la vérité terrain
        full_dataset = _create_dataset(cfg_temp, transform=None, dataset_config=dataset_info)
        
        # Extraction de la carte des labels en gérant les différents cas possibles
        if hasattr(full_dataset, 'labels'):
            labels_full = full_dataset.labels
        elif hasattr(full_dataset, 'targets'):
            labels_full = full_dataset.targets
        else:
            # Fallback spécifique au PolSFDataset / ALOSDataset
            labels_full = full_dataset.alos_dataset.labels if hasattr(full_dataset, 'alos_dataset') else None
            
        if labels_full is not None:
            # Conversion en numpy array si c'est un tenseur PyTorch
            if torch.is_tensor(labels_full):
                labels_full = labels_full.numpy()
            
            print("    -> L'image va s'ouvrir. Fermez la fenêtre pour continuer l'exécution du script.")
            visualize_azimut_split(labels_full, x1, x2)
        else:
            print("    [!] Impossible de trouver l'attribut des labels dans le dataset pour la visualisation.")
            
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