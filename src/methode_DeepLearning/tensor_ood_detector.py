import sys
import os
import torch
import numpy as np
from typing import Union
from torch.utils.data import DataLoader
from torchcvnn.nn.modules import ComplexMSELoss

class Tensor_OOD_Detector:
    def __init__(self, model, device="cpu"):
        """
        Prend un modèle pré-entraîné (AutoEncoder) pour faire de la détection OoD spatiale.
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
        # Paramètres de la distribution latente (Mahalanobis)
        self.mu = None
        self.inv_cov = None
        
        # Seuil calibré
        self.threshold_recon = None
        self.threshold_latent = None
        
        # Fonction de perte native CVNN pour l'erreur de reconstruction
        self.recon_loss_fn = ComplexMSELoss(reduction='none')

    def extract_latent(self, x):
        """
        Extrait le tenseur latent natif complexe,
        et permute pour préserver l'espace : [Batch, Hauteur, Largeur, Canaux].
        """
        with torch.no_grad():
            # Extraction du latent via la fonction du modèle
            z = self.model.get_latent(x) #Shape: [B, C, H, W]
            z = z.permute(0, 2, 3, 1) #Permutation pour mettre les canaux à la fin -> Shape: [B, H, W, C]
        return z

    def fit_mahalanobis(self, train_loader: DataLoader, eps: float = 1e-6):
        """
        Calcule la moyenne et la covariance inverse sur l'axe des canaux, 
        en considérant chaque point spatial comme un échantillon.
        """
        all_z = []
        
        for batch in train_loader:  
            x = batch[0] if isinstance(batch, (list, tuple)) else batch 
            x = x.to(self.device)
            z = self.extract_latent(x) # [B, H, W, C] complexe
            z_flat_spatial = z.reshape(-1, z.shape[-1]) #Aplatissement spatial pour accumuler les "pixels" latents: [B*H*W, C]
            all_z.append(z_flat_spatial.cpu())
            
        all_z = torch.cat(all_z, dim=0) #[N_total_pixels, C] matrice 2D complexe
        self.mu = torch.mean(all_z, dim=0) #Vecteur de taille C complexe
        cov = torch.cov(all_z.T) #Calcul de la matrice de covariance Hermitienne (automatique avec PyTorch)
        cov += eps * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)  # Régularisation complexe
        
        # Pseudo-inverse pour plus de stabilité
        self.inv_cov = torch.linalg.pinv(cov)

    def compute_scores(self, x):
        """
        Calcule l'erreur de reconstruction et agrège la distance de Mahalanobis spatiale.
        """
        x = x.to(self.device)
        
        with torch.no_grad():
            # 1. Score de reconstruction (MSE Complexe de cvnn)
            outputs = self.model(x)
            if isinstance(outputs, (tuple, list)):
                x_hat = outputs[0]
            else:
                x_hat = outputs
                
            # Attention : Si x_hat et x n'ont pas le même nombre de canaux,
            # cette ligne crashera ou ne fera pas une MSE pertinente.
            if x_hat.shape == x.shape:
                recon_dist = self.recon_loss_fn(x_hat, x).mean(dim=list(range(1, x.ndim)))
            else:
                # Fallback : on met l'erreur de reconstruction à zéro si le modèle fait de la classification
                recon_dist = torch.zeros(x.shape[0], device=x.device)
            
            # 2. Score latent spatial (Mahalanobis)
            z = self.extract_latent(x) # [B, H, W, C]
            B, H, W, C_dim = z.shape
            
            #Aplatissement spatial par batch pour le calcul tensoriel : Matrice de taille[B*H*W, C] complexe
            delta = z.reshape(-1, C_dim).cpu() - self.mu 
            
            # Calcul Mahalanobis complexe : D^2 = delta^H * inv_cov * delta
            # L'utilisation de delta.conj() est OBLIGATOIRE pour la forme hermitienne
            mah_dist_sq = torch.einsum('ni,ij,nj->n', delta.conj(), self.inv_cov, delta)
            mah_dist_sq = mah_dist_sq.real # La distance est strictement réelle et positive par définition
            
            mah_dist_map = torch.sqrt(mah_dist_sq).view(B, H, W) # Carte d'anomalie [B, H, W]
            
            # Agrégation : Moyenne de l'anomalie sur toute la carte spatiale
            mah_dist = mah_dist_map.mean(dim=(1, 2))
            
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
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(self.device)
                
            recon, mah = self.compute_scores(x)
            
            score_recon = recon / self.threshold_recon
            score_mah = mah / self.threshold_latent
            
            all_preds_recon.append((score_recon > 1.0).astype(int))
            all_scores_recon.append(score_recon)
            
            all_preds_mah.append((score_mah > 1.0).astype(int))
            all_scores_mah.append(score_mah)
            
        return (np.concatenate(all_preds_recon), np.concatenate(all_scores_recon),
                np.concatenate(all_preds_mah), np.concatenate(all_scores_mah))