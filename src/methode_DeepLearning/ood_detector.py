import sys
import os
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

# Ajouts pour le main()
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "cvnn", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "methode_Classique"))

from cvnn.config import load_config
from cvnn.data import azimut_split
from cvnn.models import LatentAutoEncoder
from anomalies import Crosstalk, ChannelGainImbalance

class ComplexOODDetector:
    def __init__(self, model, device="cpu"):
        """
        Prend un modèle pré-entraîné (LatentAutoEncoder) pour faire de la détection OoD.
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
        # Paramètres de la distribution latente (Mahalanobis)
        self.mu = None
        self.inv_cov = None
        self.pca = None
        
        # Seuils calibrés
        self.threshold_recon = None
        self.threshold_latent = None

    def _extract_real_latent(self, x):
        """
        Passe la donnée dans get_latent, l'aplatit et sépare le réel et l'imaginaire.
        """
        with torch.no_grad():
            # Extraction du latent via votre fonction
            z = self.model.get_latent(x)
            
            # On aplatit les dimensions spatiales/canaux (B, C, L) -> (B, D)
            B = z.shape[0]
            z_flat = z.view(B, -1)
            
            # Concaténation des parties réelles et imaginaires
            z_real = torch.cat([z_flat.real, z_flat.imag], dim=1)
            
        return z_real

    def fit_mahalanobis(self, train_loader: DataLoader, n_components: int = 64, eps: float = 1e-6):
        """
        Applique une PCA sur l'espace latent pour la robustesse, 
        puis calcule mu et la covariance inverse.
        """
        all_z = []
        
        for batch in train_loader:  
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(self.device)
            z_real = self._extract_real_latent(x)
            all_z.append(z_real.cpu())
            
        all_z = torch.cat(all_z, dim=0) 
        
        # Réduction de dimension avec PCA
        self.pca = PCA(n_components=n_components, random_state=42)
        all_z_pca = self.pca.fit_transform(all_z.numpy())
        all_z_pca = torch.tensor(all_z_pca, dtype=torch.float32)
        
        # Calcul de la moyenne
        self.mu = torch.mean(all_z_pca, dim=0)
        
        # Calcul de la matrice de covariance
        cov = torch.cov(all_z_pca.T)
        
        # Ajout d'une régularisation (shrinkage) pour éviter une matrice singulière
        cov += eps * torch.eye(cov.shape[0])
        
        # Pseudo-inverse pour plus de stabilité
        self.inv_cov = torch.linalg.pinv(cov)

    def compute_scores(self, x):
        """
        Calcule le score de reconstruction ET la distance de Mahalanobis.
        """
        x = x.to(self.device)
        
        with torch.no_grad():
            # 1. Score de reconstruction
            x_hat = self.model(x)
            recon_dist = torch.mean(torch.abs(x - x_hat)**2, dim=list(range(1, x.ndim)))
            
            # 2. Score latent (Mahalanobis)
            z_real = self._extract_real_latent(x).cpu().numpy()
            # Il faut impérativement appliquer la PCA avant de comparer avec mu !
            z_pca = self.pca.transform(z_real)
            z_pca = torch.tensor(z_pca, dtype=torch.float32)
            
            delta = z_pca - self.mu
            mah_dist = torch.sqrt(torch.einsum('bi,ij,bj->b', delta, self.inv_cov, delta))
            
        return recon_dist.cpu().numpy(), mah_dist.numpy()

    def calibrate_thresholds(self, pfa_loader, pfa: float):
        """
        Calibre les seuils avec les données de PFA (Probabilité de Fausse Alarme).
        """
        all_recon = []
        all_mah = []
        
        for batch in pfa_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            recon_dist, mah_dist = self.compute_scores(x)
            all_recon.append(recon_dist)
            all_mah.append(mah_dist)
            
        all_recon = np.concatenate(all_recon)
        all_mah = np.concatenate(all_mah)
        
        # Calibrage aux quantiles
        self.threshold_recon = np.quantile(all_recon, 1 - pfa)
        self.threshold_latent = np.quantile(all_mah, 1 - pfa)
        
        return self.threshold_recon, self.threshold_latent

    def detect(self, test_loader, alpha=0.5, method='max', anomaly_generator=None):
        """
        Combine les deux scores sur un jeu de test pour une décision finale.
        Doit être appelé APRÈS fit_mahalanobis et calibrate_thresholds.
        """
        if self.threshold_recon is None or self.threshold_latent is None:
            raise ValueError("Les seuils doivent être calibrés avant la détection.")

        all_preds = []
        all_combined_scores = []
        
        for batch in test_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(self.device)
            
            if anomaly_generator is not None:
                x = anomaly_generator(x)
                
            # Calcul des scores bruts
            recon_dist, mah_dist = self.compute_scores(x)
            
            # Normalisation par les seuils (échelle commune)
            norm_recon = recon_dist / self.threshold_recon
            norm_mah = mah_dist / self.threshold_latent
            
            # Combinaison
            if method == 'max':
                combined_score = np.maximum(norm_recon, norm_mah)
            elif method == 'sum':
                combined_score = alpha * norm_recon + (1 - alpha) * norm_mah
            else:
                raise ValueError("La méthode doit être 'max' ou 'sum'")
            
            # Décision (1 = Anomalie / OoD, 0 = Normal)
            preds = (combined_score > 1.0).astype(int)
            
            all_preds.append(preds)
            all_combined_scores.append(combined_score)
            
        return np.concatenate(all_preds), np.concatenate(all_combined_scores)


def evaluate_ood_system(model, train_loader, valid_loader, test_sain_loader, anomaly_loader, pfa_target=0.05, method='max', device="cpu"):
    """
    Entraîne le détecteur sur l'espace latent, calibre les seuils, et évalue les performances globales.
    """
    print("--- Initialisation du Détecteur OoD ---")
    detector = ComplexOODDetector(model, device=device)
    
    # 1. Modélisation de l'espace latent normal
    print("1. Calcul de la distribution de Mahalanobis (sur Train Sain)...")
    detector.fit_mahalanobis(train_loader)
    
    # 2. Calibration
    print(f"2. Calibration des seuils pour une PFA de {pfa_target*100}% (sur Valid Sain)...")
    thresh_recon, thresh_mah = detector.calibrate_thresholds(valid_loader, pfa=pfa_target)
    print(f"   -> Seuil Reconstruction : {thresh_recon:.4f}")
    print(f"   -> Seuil Mahalanobis    : {thresh_mah:.4f}")
    
    # 3. Test sur les données saines (Vérification de la PFA)
    print("\n--- Évaluation sur Zone 2.1 (Pures) ---")
    preds_sain, scores_sain = detector.detect(test_sain_loader, method=method)
    pfa_empirique = np.mean(preds_sain) 
    print(f"PFA empirique : {pfa_empirique*100:.2f}% (Cible: {pfa_target*100}%)")
    
    # 4. Test sur les anomalies de la Zone 2.2
    anomalies_to_test = [
        Crosstalk(delta=0.05).to(device),
        ChannelGainImbalance(g=1.029).to(device)
    ]
    
    for anomaly in anomalies_to_test:
        anomaly_name = anomaly.__class__.__name__
        print(f"\n--- Évaluation sur Zone 2.2 avec {anomaly_name} ---")
        
        preds_ano, scores_ano = detector.detect(anomaly_loader, method=method, anomaly_generator=anomaly)
        taux_detection = np.mean(preds_ano) 
        print(f"Taux de Détection (sur {anomaly_name}) : {taux_detection*100:.2f}%")
        
        # Calcul de l'AUC-ROC
        y_true = np.concatenate([np.zeros(len(scores_sain)), np.ones(len(scores_ano))])
        y_scores = np.concatenate([scores_sain, scores_ano])
        auc = roc_auc_score(y_true, y_scores)
        print(f"Score AUC-ROC : {auc:.4f} (1.0 = Parfait)")
    
    return detector

def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("[*] Découpage des données (Zones 1, 2.1 et 2.2)...")
    loaders_dict = azimut_split(config, use_cuda=False)
    train_loader, valid_loader, _ = loaders_dict["part1_loaders"]
    loader_2_1, _, _ = loaders_dict.get("loader2_1")
    loader_2_2, _, _ = loaders_dict.get("loader2_2")
    
    print("[*] Chargement du modèle...")
    model_cfg = config.get("model", {})
    model = LatentAutoEncoder(
        num_channels=config["data"].get("inferred_input_channels", 4),
        num_layers=model_cfg.get("num_layers", 3),
        channels_width=model_cfg.get("channels_width", [16, 32, 64]),
        latent_dim=model_cfg.get("latent_dim", 512),
        layer_mode=model_cfg.get("layer_mode", "complex")
    ).to(device)
    
    weights_path = Path("ml_results") / "local_run" / "best_autoencoder.pt"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    
    # Lancement du test
    evaluate_ood_system(model, train_loader, valid_loader, loader_2_1, loader_2_2, device=device)

if __name__ == "__main__":
    main()