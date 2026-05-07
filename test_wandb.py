import wandb
import random
import time

def main():
    # 1. Définition des hyperparamètres (Best Practice : toujours les tracker)
    config = {
        "learning_rate": 0.01,
        "epochs": 10,
        "batch_size": 32,
        "architecture": "ResNet-Dummy"
    }

    print("[*] Initialisation de Weights & Biases...")
    
    # 2. Initialisation avec un gestionnaire de contexte (ferme proprement le run à la fin)
    with wandb.init(
        project="test-integration", # Nom de votre projet (dossier principal sur W&B)
        name="premier-test-local",  # Nom spécifique de cette exécution
        config=config,              # Enregistrement des hyperparamètres
        notes="Script de vérification de l'API wandb."
    ) as run:
        
        print(f"[*] Run initialisé ! Vous pouvez suivre en direct ici : {run.get_url()}")
        print("[*] Début de la simulation d'entraînement...")

        # 3. Boucle d'entraînement (simulation)
        for epoch in range(config["epochs"]):
            # Simulation de métriques qui évoluent
            loss = 10.0 / (epoch + 1) + random.uniform(0, 0.5)
            accuracy = 1.0 - (1.0 / (epoch + 1)) + random.uniform(0, 0.05)

            # Log des métriques à chaque époque
            wandb.log({"epoch": epoch + 1, "train_loss": loss, "val_accuracy": accuracy})
            
            print(f"  -> Epoch {epoch+1}/{config['epochs']} | Loss: {loss:.4f} | Accuracy: {accuracy:.4f}")
            time.sleep(0.5) # On simule un temps de calcul

    print("[+] Test terminé avec succès ! Allez vérifier votre tableau de bord.")

if __name__ == "__main__":
    main()