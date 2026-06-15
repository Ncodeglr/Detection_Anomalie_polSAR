import sys
import copy
import torch
import numpy as np
from pathlib import Path

from cvnn.config import load_config
from cvnn.utils import set_seed
from cvnn.data import get_dataloaders, get_full_image_dataloader
from synthetic_parameter_generator import SyntheticParameterGenerator

def setup_experiment_env(sys_argv, script_file_path, force_cpu=False):
    """
    Initialise l'environnement de l'expérience de manière unifiée :
    - Gère les chemins absolus et la config
    - Fixe la graine aléatoire (reproductibilité)
    - Configure le device (CPU/GPU) selon les besoins de la méthode
    - Charge et calcule les statistiques de la région originelle (PolSF)
    """
    repo_root = Path(script_file_path).resolve().parents[2]
    
    config_path = sys_argv[1] if len(sys_argv) > 1 else str(repo_root / "configs" / "config_Unet.yaml")
    config = load_config(config_path)
    set_seed(config.get("seed", 42))
    
    device = torch.device("cpu") if force_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    trainpath = Path(config["data"]["dataset"]["trainpath"])
    if not trainpath.is_absolute():
        config["data"]["dataset"]["trainpath"] = str((repo_root / trainpath).resolve())
        
    config_base = copy.deepcopy(config)
    config_base["data"]["dataset"].pop("crop_coordinates", None)
    config_base["data"]["recompute_statistics"] = True
    
    use_cuda = (device.type == "cuda")
    train_loader, valid_loader, test_loader = get_dataloaders(config_base, use_cuda=use_cuda)
    
    return config, config_base, device, (train_loader, valid_loader, test_loader)

def get_test_loaders(config_base, use_cuda=False):
    """
    Génère les chargeurs de données pour la Zone 2.1 (Saine) et la Zone 2.2 (3 sous-zones).
    Utilise un empilement vertical pour garantir que toutes les zones restent sur 
    la terre ferme (Péninsule de San Francisco) et sous la limite des 8080 lignes de l'image.
    """
    config_test = copy.deepcopy(config_base)
    config_test["data"]["dataset"]["name"] = "ALOSDataset"
    config_test["data"]["recompute_statistics"] = False

    # ---------------------------------------------------------
    # Zone 2.1 : Région Saine Non Vue 
    # Lignes 3800 à 5800. Parfaitement sur la terre selon l'image.
    # ---------------------------------------------------------
    config_unseen_sain = copy.deepcopy(config_test)
    config_unseen_sain["data"]["dataset"]["crop_coordinates"] = {
        "start_row": 3800, 
        "end_row": 5800,
        "start_col": 2832, 
        "end_col": 7888
    }
    loader_test_2_1, _, _ = get_full_image_dataloader(config_unseen_sain, use_cuda=use_cuda)

    # ---------------------------------------------------------
    # Zone 2.2 : Région pour Anomalies
    # On descend juste en dessous de la zone 2.1 (Lignes 6000 à 8000).
    # On s'arrête à 8000 pour ne pas dépasser la taille max de l'image (8080).
    # ---------------------------------------------------------
    start_row_2_2 = 6000
    end_row_2_2 = 8000
    
    # Séparation exacte en 3 intervalles de lignes (verticalement)
    row_split_points = np.linspace(start_row_2_2, end_row_2_2, 4, dtype=int)
    loaders_2_2_parts = []
    
    for i in range(3):
        cfg_part = copy.deepcopy(config_test)
        cfg_part["data"]["dataset"]["crop_coordinates"] = {
            "start_row": int(row_split_points[i]), 
            "end_row": int(row_split_points[i+1]),
            "start_col": 2832, 
            "end_col": 7888
        }
        loader, _, _ = get_full_image_dataloader(cfg_part, use_cuda=use_cuda)
        loaders_2_2_parts.append(loader)

    return loader_test_2_1, loaders_2_2_parts

    
def get_shared_anomaly_generator(config):
    """
    Garantit que toutes les méthodes génèrent rigoureusement les mêmes anomalies.
    """
    delta_generator = SyntheticParameterGenerator(
        mean_db=-25, std_dev_amp=0.02, phase_mean_rad=0.0, phase_concentration=1e-5
    )
    anomaly_seed = config.get("anomaly_seed", 1234)
    return delta_generator, anomaly_seed