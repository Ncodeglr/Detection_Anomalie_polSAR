import sys
import os
import torch
import numpy as np
from typing import Union
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from torchcvnn.nn.modules import ComplexMSELoss

class OOD_Detector:
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
        
        # Seuil calibré
        self.threshold_recon = None
        self.threshold_latent = None
        
        # Fonction de perte native CVNN pour l'erreur de reconstruction
        self.recon_loss_fn = ComplexMSELoss(reduction='none')

    def _extract_real_latent(self, x):
        """
        Passe la donnée dans get_latent, l'aplatit et sépare le réel et l'imaginaire.
        """
        with torch.no_grad():
            # Extraction du latent via la fonction corrigée du modèle
            z = self.model.get_latent(x)
            
            z_flat = z.view(z.shape[0], -1)
            z_real = torch.cat([z_flat.real, z_flat.imag], dim=1)
            
        return z_real

    def fit_mahalanobis(self, train_loader: DataLoader, n_components: Union[int, float] = 0.95, eps: float = 1e-6):
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
        Calcule l'erreur de reconstruction ET la distance de Mahalanobis.
        """
        x = x.to(self.device)
        
        with torch.no_grad():
            # 1. Score de reconstruction (MSE Complexe de cvnn)
            x_hat = self.model(x)
            recon_dist = self.recon_loss_fn(x_hat, x).mean(dim=list(range(1, x.ndim)))
            
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
        Calibre les seuils avec les données de PFA.
        """
        all_recon = []
        all_mah = []
        
        for batch in pfa_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            recon, mah = self.compute_scores(x)
            all_recon.append(recon)
            all_mah.append(mah)
            
        all_recon = np.concatenate(all_recon)
        all_mah = np.concatenate(all_mah)
        
        #La fonction np.quantile(donnees, quantile) de NumPy cherche la valeur dans le tableau donnees en dessous de laquelle se trouve un certain pourcentage des données. Ici, le quantile demandé est 1 - pfa
        self.threshold_recon = np.quantile(all_recon, 1 - pfa) 
        self.threshold_latent = np.quantile(all_mah, 1 - pfa)
        
        return self.threshold_recon, self.threshold_latent

    def detect(self, test_loader):
        """
        Calcule et retourne les prédictions indépendantes (Reconstruction et Latent).
        Doit être appelé APRÈS fit_mahalanobis et calibrate_thresholds.
        """
        if self.threshold_latent is None or self.threshold_recon is None:
            raise ValueError("Les seuils doivent être calibrés avant la détection.")

        all_preds_recon, all_scores_recon = [], []
        all_preds_mah, all_scores_mah = [], []
        
        for batch in test_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch #On vérifie si la variable batch est une liste ou un tuple. Si c'est le cas, on suppose que les données d'entrée sont dans la première position (index 0) et on les extrait. Sinon, on considère que batch lui-même contient directement les données d'entrée.
            x = x.to(self.device)
                
            # Calcul des scores bruts
            recon, mah = self.compute_scores(x)
            
            # Normalisation par le seuil pour obtenir un score comparable
            score_recon = recon / self.threshold_recon
            score_mah = mah / self.threshold_latent
            
            #La Normalisation par le seuil permet d'obtenir un score adimensionnel où :
            # 1 correspond au seuil de détection. Ainsi, un score supérieur à 1 indique une anomalie détectée, tandis qu'un score inférieur ou égal à 1 indique une observation considérée comme normale.
            # Si S(x) = 0.5 : L'erreur représente 50% de la tolérance maximale. La donnée est saine.
            # Si S(x) = 1.0 : L'erreur atteint le seuil de tolérance. La donnée est à la limite entre sain et anormal.
            # Si S(x) = 1.5 : L'erreur dépasse de 50% le seuil de tolérance. La donnée est considérée comme anormale.
            
            # Décision (S(x) > 1  Anomalie / OoD, S(x) = 0 ou <1 alors Normal)
            all_preds_recon.append((score_recon > 1.0).astype(int))
            all_scores_recon.append(score_recon)
            
            all_preds_mah.append((score_mah > 1.0).astype(int))
            all_scores_mah.append(score_mah)
            
        return (np.concatenate(all_preds_recon), np.concatenate(all_scores_recon),
                np.concatenate(all_preds_mah), np.concatenate(all_scores_mah))
            