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
from cvnn.data import get_dataloaders
from cvnn.models import UNet
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

    #Résolution absolue du chemin des données par rapport à la racine du projet
    trainpath = Path(config["data"]["dataset"]["trainpath"])
    if not trainpath.is_absolute():
        config["data"]["dataset"]["trainpath"] = str((repo_root / trainpath).resolve())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Matériel détecté : {device}")

    #2. Initialisation de Weights & Biases (via l'API CVNN) 
    wandb_log, run_name = setup_wandb(config)
    log_config_summary(wandb_log, config)

    #3. Chargement Modulaire des Données (Dataloaders)
    print("[*] Chargement des données (Image entière)...")
    # On supprime les coordonnées de découpe héritées du config.yaml global pour utiliser l'image complète
    config["data"]["dataset"].pop("crop_coordinates", None)
    
    train_loader, valid_loader, _  = get_dataloaders(config, device)

    #4. Initialisation du Modèle (UNET de CVNN)
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    
    in_channels = data_cfg.get("inferred_input_channels", 4)
    input_size = data_cfg.get("inferred_input_size", data_cfg.get("dataset", {}).get("patch_size", 32))
    num_classes = data_cfg.get("inferred_num_classes", model_cfg.get("num_classes", 7))
    
    model = UNet(
        num_channels = in_channels,
        num_layers = model_cfg.get("num_layers", 3),
        channels_width = model_cfg.get("channels_width", 16),
        input_size = input_size,
        activation = model_cfg.get("activation", "CReLU"),
        num_classes = num_classes,
        num_blocks = model_cfg.get("num_blocks", 1),
        layer_mode = model_cfg.get("layer_mode", "complex"),
        normalization_layer = model_cfg.get("normalization_layer", "batch"),
        downsampling_layer = model_cfg.get("downsampling_layer", "maxpool"),
        upsampling_layer = model_cfg.get("upsampling_layer", "nearest"),
        residual = model_cfg.get("residual", True),
        projection_layer = model_cfg.get("projection_layer", "amplitude"),
        projection_config = model_cfg.get("projection_config", None),
        dropout = model_cfg.get("dropout", 0.0),
        gumbel_softmax = model_cfg.get("gumbel_softmax", None)
    ).to(device)


    init_weights_mode_aware(model, model_cfg.get("layer_mode", "complex"))

    #5. Setup de la Loss et de l'Optimiseur (repository CVNN)
    loss_fn, optimizer = setup_loss_optimizer(model, config, train_loader.dataset, device)

    # Initialisation du scheduler défini dans config.yaml (ex: ReduceLROnPlateau)
    warmup_scheduler, scheduler = build_schedulers(optimizer, config, len(train_loader))
    
    epochs = config.get("epochs", 50)
    best_val_loss = float('inf')
    
    # Dossier de sauvegarde du meilleur modèle
    save_dir = Path("Unet_results") / (run_name if run_name else "local_run")
    save_dir.mkdir(parents=True, exist_ok=True)
    best_model_name = "best_weights_unet.pt"


    #6. Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [TRAIN]")
        for x_batch, y_batch in loop:
            # Extraction robuste des inputs selon le format retourné par CVNN
            x_batch = x_batch if isinstance(x_batch, torch.Tensor) else x_batch[0]
            y_batch = y_batch if isinstance(y_batch, torch.Tensor) else y_batch[0]
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device, dtype=torch.long)
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            
            _, outputs_projected = outputs if isinstance(outputs, (tuple, list)) and len(outputs) == 2 else (None, outputs)
            loss_output = loss_fn(outputs_projected, y_batch)
            
            #A. Extraction robuste (Gestion repository CVNN)
            loss = loss_output[0] if isinstance(loss_output, tuple) else loss_output # CVNN peut retourner (loss, metrics_dict) ou juste loss
            
            #B. Sécurité anti-divergence (NaN)
            if not torch.isnan(loss):
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        
        #7. Validation Loop
        model.eval()
        val_loss = 0.0
        
        val_loop = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{epochs} [VALID]")
        with torch.no_grad():
            for x_batch, y_batch in val_loop:
                # Extraction robuste des inputs selon le format retourné par CVNN
                x_batch = x_batch if isinstance(x_batch, torch.Tensor) else x_batch[0]
                y_batch = y_batch if isinstance(y_batch, torch.Tensor) else y_batch[0]
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device, dtype=torch.long)
                
                outputs = model(x_batch)
                
                _, outputs_projected = outputs if isinstance(outputs, (tuple, list)) and len(outputs) == 2 else (None, outputs)
                loss_output = loss_fn(outputs_projected, y_batch)
                
                #A. Extraction robuste
                loss = loss_output[0] if isinstance(loss_output, tuple) else loss_output
                
                #B. Sécurité anti-divergence
                if not torch.isnan(loss):
                    val_loss += loss.item()
                    val_loop.set_postfix(loss=loss.item())
                
        avg_val_loss = val_loss / len(valid_loader)
        print(f"➡️ Bilan Epoch {epoch+1} : Train Loss = {avg_train_loss:.6f} | Val Loss = {avg_val_loss:.6f}")
        
        
        step_schedulers(warmup=None, scheduler=scheduler, metric=avg_val_loss) #Mise à jour du learning rate scheduler
        
        # --- 8. Enregistrement des logs sur WandB ---
        log_metrics(wandb_log, {"loss": avg_train_loss}, step=epoch+1, prefix="training")
        log_metrics(wandb_log, {"loss": avg_val_loss}, step=epoch+1, prefix="validation")
        log_metrics(wandb_log, {"learning_rate": optimizer.param_groups[0]["lr"]}, step=epoch+1, prefix="training")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"   [+] Nouveau meilleur modèle ! Sauvegarde dans {save_dir / best_model_name}")
            torch.save(model.state_dict(), save_dir / best_model_name)

    # Fermeture propre de W&B
    finish_wandb_run()
    print("[*] Entraînement terminé.")

if __name__ == "__main__":
    main()
