import sys
import torch

# Ajout du dossier src au chemin pour pouvoir importer cvnn si vous ne l'avez pas installé via pip/poetry
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "cvnn", "src"))

from cvnn.data import azimut_split
from cvnn.config import load_config

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

    print("[*] Initialisation de la découpe azimut...")
    use_cuda = torch.cuda.is_available()
    
    # azimut_split retourne un dictionnaire avec les loaders de la partie 1 et 2
    split_results = azimut_split(cfg, use_cuda=use_cuda)
    
    print("\n--- Résultats de la découpe ---")
    for part_name, loaders in split_results.items():
        print(f"\n=== {part_name.upper()} ===")
        
        # Gérer le cas du full_loader (tuple) vs les dataloaders classiques (tuple/liste de dataloaders)
        if part_name == "part2_full":
            loader_list = [("Full Image", loaders[0])]
        else:
            names = ["Train", "Validation", "Test"]
            loader_list = zip(names, loaders)
            
        for name, loader in loader_list:
            print(f"\n  [*] Loader : {name}")
            print(f"      Nombre de batchs : {len(loader)}")
            for batch in loader:
                inputs = batch[0] if isinstance(batch, (list, tuple)) else batch.get("data", batch.get("inputs")) if isinstance(batch, dict) else batch
                print(f"      Forme (Shape)    : {inputs.shape}")
                print(f"      Type (dtype)     : {inputs.dtype}")
                break

if __name__ == "__main__":
    main()