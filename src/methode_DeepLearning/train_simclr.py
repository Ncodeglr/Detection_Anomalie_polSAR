import sys
import os
import torch
from pathlib import Path
from torch.optim import AdamW
from tqdm import tqdm

# Import de vos modules (Mise à jour du nom de la fonction de perte)
from simclr_core import AdvancedComplexPolSARTransform, RobustComplexNTXentLoss
from simclr_model import RobustComplexSimCLR

# Import depuis la librairie CVNN
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "cvnn", "src"))
# Import du module partagé du projet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from shared_setup import setup_experiment_env
from cvnn.models import AutoEncoder
from cvnn.wandb_utils import setup_wandb, log_config_summary, log_metrics, finish_wandb_run
from cvnn.models.utils import init_weights_mode_aware
from cvnn.schedulers import build_schedulers, step_schedulers

def main():
    print("[*] Lancement de l'apprentissage contrastif (SimCLR PolSAR)...")
    
    # 1. & 2. Chargement unifié de la configuration et des données (Zone 1)
    print("[*] Configuration de l'environnement et chargement des données...")
    config, config_base, device, loaders = setup_experiment_env(sys.argv, __file__, force_cpu=False, default_config="config_SimCLR.yaml")
    train_loader, valid_loader, _ = loaders

    # Initialisation de Weights & Biases
    wandb_log, run_name = setup_wandb(config)
    log_config_summary(wandb_log, config)

    # 3. Initialisation de l'encodeur AutoEncoder CVNN
    print("   -> Initialisation de la classe AutoEncoder CVNN...")
    model_cfg = config_base.get("model", {})
    data_cfg = config_base.get("data", {})
    
    in_channels = data_cfg.get("inferred_input_channels", 4)
    input_size = data_cfg.get("inferred_input_size", data_cfg.get("dataset", {}).get("patch_size", 16))
    
    cvnn_encoder = AutoEncoder(
        num_channels=in_channels,
        num_layers=model_cfg.get("num_layers", 3),
        channels_width=model_cfg.get("channels_width", 32),
        input_size=input_size,
        activation=model_cfg.get("activation", "CReLU"),
        upsampling_layer=model_cfg.get("upsampling_layer", "nearest"),
        num_blocks=model_cfg.get("num_blocks", 2),
        layer_mode=model_cfg.get("layer_mode", "complex"),
        normalization_layer=model_cfg.get("normalization_layer", "batch"),
        downsampling_layer=model_cfg.get("downsampling_layer", "maxpool"),
        residual=model_cfg.get("residual", False),
        dropout=model_cfg.get("dropout", 0.0),
        latent_dim=model_cfg.get("latent_dim", 128)
    ).to(device)
    
    init_weights_mode_aware(cvnn_encoder, model_cfg.get("layer_mode", "complex"))
    
    # 4. Initialisation de l'architecture SimCLR
    print("   -> Déduction dynamique de la dimension latente...")
    cvnn_encoder.eval()
    
    # Création d'un tenseur dummy complexe pour le passage à vide (dry run)
    dummy_input = torch.complex(
        torch.randn(1, in_channels, input_size, input_size), 
        torch.randn(1, in_channels, input_size, input_size)
    ).to(device)
    
    with torch.no_grad():
        if hasattr(cvnn_encoder, "get_latent"):
            dummy_latent = cvnn_encoder.get_latent(dummy_input)
        else:
            dummy_latent = cvnn_encoder(dummy_input)
            if isinstance(dummy_latent, (list, tuple)):
                dummy_latent = dummy_latent[-1]
                
    complex_channels_out = dummy_latent.shape[1]
    print(f"   [+] Canaux latents extraits automatiquement de l'encodeur : {complex_channels_out}")
    
    model = RobustComplexSimCLR(
        cvnn_encoder=cvnn_encoder, 
        complex_feature_dim=complex_channels_out, 
        projection_dim=128
    ).to(device)
    
    # 5. Data Augmentations et Perte
    # Calcul dynamique du crop_size (~75% de l'image)
    crop_s = max(4, int(input_size * 0.75))
    augmenter = AdvancedComplexPolSARTransform(crop_size=(crop_s, crop_s), speckle_std=0.05).to(device)
    
    # [MODIFICATION] Utilisation de la nouvelle perte robuste native
    criterion = RobustComplexNTXentLoss(temperature=0.5).to(device)
    
    # Configuration de l'optimiseur
    optim_params = config_base.get("optim", {}).get("params", {})
    optimizer = AdamW(model.parameters(), lr=optim_params.get("lr", 1e-3), weight_decay=optim_params.get("weight_decay", 1e-4))
    
    # Initialisation du scheduler
    warmup_scheduler, scheduler = build_schedulers(optimizer, config_base, len(train_loader))
    
    epochs = config_base.get("epochs", 50)
    best_val_loss = float('inf')
    
    # Dossier de sauvegarde
    save_dir = Path("SimCLR_results") / (run_name if run_name else "local_run")
    save_dir.mkdir(parents=True, exist_ok=True)
    best_model_name = "best_weights_simclr_full.pt"
    
    print(f"[*] Début de l'entraînement sur {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [TRAIN]")
        
        for batch in loop:
            # Extraire les images (Patchs)
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            
            # Générer les 2 vues augmentées
            with torch.no_grad():
                view_1, view_2 = augmenter(x)
            
            optimizer.zero_grad()
            
            # Propagation pour obtenir les projections z_1 et z_2 (Tenseurs complexes)
            _, z_1 = model(view_1)
            _, z_2 = model(view_2)
            
            # Calcul de la perte Contrastive (Gérée nativement par PyTorch)
            loss = criterion(z_1, z_2)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Phase de Validation ---
        model.eval()
        val_loss = 0.0
        val_loop = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{epochs} [VALID]")
        
        with torch.no_grad():
            for batch in val_loop:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(device)
                
                view_1, view_2 = augmenter(x)
                
                _, z_1 = model(view_1)
                _, z_2 = model(view_2)
                
                loss = criterion(z_1, z_2)
                
                val_loss += loss.item()
                val_loop.set_postfix(loss=loss.item())
                
        avg_val_loss = val_loss / len(valid_loader)
        print(f"➡️ Bilan Epoch {epoch+1} : Train NT-Xent = {avg_train_loss:.4f} | Val NT-Xent = {avg_val_loss:.4f}")
        
        # Mise à jour du learning rate scheduler
        step_schedulers(warmup=warmup_scheduler, scheduler=scheduler, metric=avg_val_loss)
        
        # --- Enregistrement des logs sur WandB ---
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