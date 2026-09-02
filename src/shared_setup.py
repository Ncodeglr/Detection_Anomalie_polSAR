import sys
import copy
import torch
import numpy as np
from pathlib import Path

from cvnn.config import load_config
from cvnn.utils import set_seed
from cvnn.data import get_dataloaders, get_full_image_dataloader
from synthetic_parameter_generator import SyntheticParameterGenerator

# ==========================================
# DÉFINITION CENTRALISÉE DES ZONES (DRY)
# ==========================================
# Nouvelles coordonnées globales pour maximiser les données d'entraînement (ALOS-2)
ZONES_CONFIG = {
    "zone1_train": {
        "start_row": 0, "end_row": 8000, 
        "start_col": 0, "end_col": 8000 
    },
    "zone2_1_saine": {
        "start_row": 8200, "end_row": 15000, 
        "start_col": 0, "end_col": 8000
    },
    "zone2_2_anomalies_global": {
        "start_row": 15200, "end_row": 20000, 
        "start_col": 0, "end_col": 8000 #22602
    }
}

def setup_experiment_env(sys_argv, script_file_path, force_cpu=False, default_config="config.yaml"):
    """
    Initialise l'environnement de l'expérience d'entraînement/validation.
    Force l'utilisation de la Zone 1 globale pour éviter toute erreur de YAML.
    """
    repo_root = Path(script_file_path).resolve().parents[2]
    
    config_path = sys_argv[1] if len(sys_argv) > 1 else str(repo_root / "configs" / default_config)
    config = load_config(config_path)
    set_seed(config.get("seed", 42))
    
    device = torch.device("cpu") if force_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    trainpath = Path(config["data"]["dataset"]["trainpath"])
    if not trainpath.is_absolute():
        config["data"]["dataset"]["trainpath"] = str((repo_root / trainpath).resolve())
        
    config_base = copy.deepcopy(config)
    
    #On force les coordonnées à la définition stricte de la Zone 1
    #Ainsi, peu importe ce qu'il y a dans le YAML, l'entraînement se fera sur la bonne zone.
    config_base["data"]["dataset"]["crop_coordinates"] = ZONES_CONFIG["zone1_train"]
    config_base["data"]["recompute_statistics"] = True
    
    use_cuda = (device.type == "cuda")
    train_loader, valid_loader, test_loader = get_dataloaders(config_base, use_cuda=use_cuda)
    print(f"[*] Chargement des DataLoaders : Train={len(train_loader.dataset)}, Valid={len(valid_loader.dataset)}, Test={len(test_loader.dataset)}")
    
    return config, config_base, device, (train_loader, valid_loader, test_loader)

def get_test_loaders(config_base, use_cuda=False):
    """
    Génère les chargeurs de données pour la Zone 2.1 (Saine) et la Zone 2.2 (3 sous-zones).
    """
    config_test = copy.deepcopy(config_base)
    config_test["data"]["dataset"]["name"] = "ALOSDataset"
    config_test["data"]["recompute_statistics"] = False

    # ---------------------------------------------------------
    # Zone 2.1 : Région Saine Non Vue 
    # ---------------------------------------------------------
    config_unseen_sain = copy.deepcopy(config_test)
    config_unseen_sain["data"]["dataset"]["crop_coordinates"] = ZONES_CONFIG["zone2_1_saine"]
    loader_test_2_1, _, _ = get_full_image_dataloader(config_unseen_sain, use_cuda=use_cuda)
    print("Shape du loader pour la Zone 2.1 (Saine) :", len(loader_test_2_1.dataset))
    print("Type du loader pour la Zone 2.1 (Saine) :", type(loader_test_2_1.dataset))
    
    # ---------------------------------------------------------
    # Zone 2.2 : Région pour Anomalies (3 sous-zones)
    # ---------------------------------------------------------
    z22 = ZONES_CONFIG["zone2_2_anomalies_global"]
    
    #Séparation exacte en 3 intervalles de lignes
    row_split_points = np.linspace(z22["start_row"], z22["end_row"], 4, dtype=int)
    loaders_test_2_2_parts = []
    
    for i in range(3):
        cfg_part = copy.deepcopy(config_test)
        cfg_part["data"]["dataset"]["crop_coordinates"] = {
            "start_row": int(row_split_points[i]), 
            "end_row": int(row_split_points[i+1]),
            "start_col": z22["start_col"], 
            "end_col": z22["end_col"]
        }
        loader, _, _ = get_full_image_dataloader(cfg_part, use_cuda=use_cuda)
        loaders_test_2_2_parts.append(loader)
        print("Shape du loader pour la sous-zone 2.2 part", i+1, ":", len(loader.dataset))
        print("Type du loader pour la sous-zone 2.2 part", i+1, ":", type(loader.dataset))

    return loader_test_2_1, loaders_test_2_2_parts

def get_shared_anomaly_generator(config):
    delta_generator = SyntheticParameterGenerator(
        mean_db=-25, std_dev_amp=0.02, phase_mean_rad=0.0, phase_concentration=1e-5
    )
    anomaly_seed = config.get("anomaly_seed", 1234)
    return delta_generator, anomaly_seed