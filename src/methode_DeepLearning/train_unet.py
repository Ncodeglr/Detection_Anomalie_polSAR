import sys
import os
from pathlib import Path
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

# Ajout des chemins pour importer les modules du projet
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "cvnn", "src"))
# Import du module partagé du projet (comme dans SimCLR)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared_setup import setup_experiment_env

from cvnn.utils import set_seed
from cvnn.models import UNet
from cvnn.train import setup_loss_optimizer
from cvnn.schedulers import build_schedulers, step_schedulers
from cvnn.models.utils import init_weights_mode_aware
from cvnn.wandb_utils import setup_wandb, log_config_summary, log_metrics, finish_wandb_run

def main():
    print("[*] Démarrage de l'entraînement du UNet Complexe")
    
    # 1. & 2. Chargement unifié de la configuration et des données (Même logique que SimCLR)
    print("[*] Configuration de l'environnement et chargement des données...")
    config, config_base, device, loaders = setup_experiment_env(
        sys.argv, 
        __file__, 
        force_cpu=False, 
        default_config="config_Unet.yaml" # Fichier par défaut adapté au UNet
    )
    train_loader, valid_loader, _ = loaders

    # 3. Initialisation de Weights & Biases (via l'API CVNN) 
    wandb_log, run_name = setup_wandb(config)
    log_config_summary(wandb_log, config)

    # 4. Initialisation du Modèle (UNET de CVNN)
    # On utilise config_base comme dans le script SimCLR
    model_cfg = config_base.get("model", {})
    data_cfg = config_base.get("data", {})
    
    in_channels = data_cfg.get("inferred_input_channels", 4)
    input_size = data_cfg.get("inferred_input_size", data_cfg.get("dataset", {}).get("patch_size", 32))
    num_classes = data_cfg.get("inferred_num_classes", model_cfg.get("num_classes", 7))
    
    # =========================================================
    # [+] AJOUT ICI : On force l'injection pour satisfaire CVNN
    # =========================================================
    config_base.setdefault("model", {})
    config_base["model"]["inferred_num_classes"] = num_classes
    # =========================================================
    
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

    # 5. Setup de la Loss et de l'Optimiseur (repository CVNN)
    loss_fn, optimizer = setup_loss_optimizer(model, config_base, train_loader.dataset, device)

    # Initialisation du scheduler défini dans config.yaml
    warmup_scheduler, scheduler = build_schedulers(optimizer, config_base, len(train_loader))
    
    epochs = config_base.get("epochs", 50)
    best_val_loss = float('inf')
    
    # Dossier de sauvegarde du meilleur modèle
    save_dir = Path("Unet_results") / (run_name if run_name else "local_run")
    save_dir.mkdir(parents=True, exist_ok=True)
    best_model_name = "best_weights_unet.pt"

    # 6. Training Loop
    print(f"[*] Début de l'entraînement sur {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [TRAIN]")
        for batch in loop:
            x_batch = batch[0] if isinstance(batch, (list, tuple)) else batch
            x_batch = x_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            
            # Pour une tâche de reconstruction, la cible est l'entrée elle-même.
            loss_output = loss_fn(outputs, x_batch)
            
            loss = loss_output[0] if isinstance(loss_output, tuple) else loss_output
            if not torch.isnan(loss):
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        
        # 7. Validation Loop
        model.eval()
        val_loss = 0.0
        
        val_loop = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{epochs} [VALID]")
        with torch.no_grad():
            for batch in val_loop:
                x_batch = batch[0] if isinstance(batch, (list, tuple)) else batch
                x_batch = x_batch.to(device)
                
                outputs = model(x_batch)
                
                # La cible est l'entrée elle-même
                loss_output = loss_fn(outputs, x_batch)
                
                loss = loss_output[0] if isinstance(loss_output, tuple) else loss_output
                if not torch.isnan(loss):
                    val_loss += loss.item()
                    val_loop.set_postfix(loss=loss.item())
                
        avg_val_loss = val_loss / len(valid_loader)
        print(f"➡️ Bilan Epoch {epoch+1} : Train Loss = {avg_train_loss:.6f} | Val Loss = {avg_val_loss:.6f}")
        
        step_schedulers(warmup=None, scheduler=scheduler, metric=avg_val_loss) # Mise à jour du learning rate scheduler
        
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
    print("\n[+] Entraînement terminé.")

if __name__ == "__main__":
    main()