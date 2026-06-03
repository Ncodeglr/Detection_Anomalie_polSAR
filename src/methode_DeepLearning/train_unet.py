import sys
import os
from pathlib import Path
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb

# Ajout des chemins pour importer les modules du projet
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "cvnn", "src"))

from cvnn.config import load_config
from cvnn.utils import set_seed
from cvnn.data import azimut_split
from cvnn.models import UNET
from cvnn.train import setup_loss_optimizer
from cvnn.schedulers import build_schedulers, step_schedulers
from cvnn.models.utils import init_weights_mode_aware
from cvnn.wandb_utils import setup_wandb, log_config_summary, log_metrics, finish_wandb_run

def main():
    print("[*] Démarrage de l'entraînement du UNet Complexe")
    repo_root = Path(__file__).resolve().parents[2] # Chemin absolu vers la racine du projet
    
    #1. Configuration des chemins et paramètres via load_config de CVNN
    if len(sys.argv) >= 2:
        config_path = sys.argv[1]
    else:
        config_path = str(repo_root / "configs" / "config_Unet.yaml")
        print(f"[*] Aucun fichier de configuration spécifié. Utilisation par défaut : {config_path}")

    config = load_config(config_path)
    set_seed(config.get("seed", 42))




