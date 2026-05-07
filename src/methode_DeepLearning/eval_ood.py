import sys
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset

# Ajout des chemins pour importer les modules du projet
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "cvnn", "src"))

from cvnn.config import load_config
from cvnn.data import azimut_split
from cvnn.models import LatentAutoEncoder

# Nouveaux imports pour l'OoD
from ood_detector import evaluate_ood_system
from anomalies import Crosstalk, ChannelGainImbalance

class AnomalyDatasetWrapper(Dataset):
    """
    Enveloppe un Dataset PyTorch normal pour lui appliquer une anomalie à la volée.
    """
    def __init__(self, base_dataset, anomaly_module):
        self.base_dataset = base_dataset
        self.anomaly_module = anomaly_module

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Récupération de la donnée saine
        item = self.base_dataset[idx]
        x = item[0] if isinstance(item, (tuple, list)) else item
        rest = item[1:] if isinstance(item, (tuple, list)) else ()

        # Les classes de anomalies.py attendent une dimension de batch (B, C, H, W)
        x_batch = x.unsqueeze(0) 
        
        with torch.no_grad():
            x_anom_batch = self.anomaly_module(x_batch)
            
        # On retire la dimension de batch après l'application
        x_anom = x_anom_batch.squeeze(0)

        if rest:
            return (x_anom, *rest)
        return x_anom


def main():
    print("[*] Démarrage de l'évaluation OoD...")

    # 1. Configuration
    config_path = sys.argv[1] if len(sys.argv) >= 2 else "configs/config.yaml"
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Chargement des Dataloaders Sains
    print("[*] Chargement des données saines...")
    loaders_dict = azimut_split(config, use_cuda=torch.cuda.is_available())
    train_loader, valid_loader, test_sain_loader = loaders_dict["part1_loaders"]

    # 3. Création du Dataloader d'Anomalies
    print("[*] Génération des données anormales (OoD)...")
    # On utilise le dataset de test sain comme base pour générer des anomalies
    base_test_dataset = test_sain_loader.dataset
    
    # Choix de l'anomalie (vous pouvez tester Crosstalk ou ChannelGainImbalance)
    anomaly_transform = Crosstalk(delta=0.1) # ou ChannelGainImbalance(g=1.05)
    anomaly_transform = anomaly_transform.to(device)
    
    anomaly_dataset = AnomalyDatasetWrapper(base_test_dataset, anomaly_transform)
    anomaly_loader = DataLoader(
        anomaly_dataset, 
        batch_size=test_sain_loader.batch_size, 
        shuffle=False
    )

    # 4. Initialisation et chargement du Modèle
    print("[*] Chargement du modèle pré-entraîné...")
    model_cfg = config.get("model", {})
    
    # Récupération des dimensions comme dans votre script de train
    in_channels = config["data"].get("inferred_input_channels", 4)
    input_size = config["data"].get("inferred_input_size", config["data"]["dataset"].get("patch_size", 32))
    
    # Instanciation avec l'architecture STRICTEMENT identique au train
    model = LatentAutoEncoder(
        num_channels=in_channels,
        num_layers=model_cfg.get("num_layers", 3),
        channels_width=model_cfg.get("channels_width", [16, 32, 64]),
        input_size=input_size,
        activation=model_cfg.get("activation", "relu"),
        upsampling_layer=model_cfg.get("upsampling_layer", "conv_transpose"),
        layer_mode=model_cfg.get("layer_mode", "complex"),
        normalization_layer=model_cfg.get("normalization_layer", "batch_norm"),
        residual=model_cfg.get("residual", False),
        num_blocks=model_cfg.get("num_blocks", 1),
        latent_dim=model_cfg.get("latent_dim", 128)
    ).to(device)

    # Chargement des poids
    model_path = Path("ml_results") / "local_run" / "best_autoencoder.pt"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"   [+] Poids chargés depuis {model_path}")
    else:
        print(f"   [!] ATTENTION : Aucun poids trouvé dans {model_path}. Le modèle n'est pas entraîné !")

if __name__ == "__main__":
    main()