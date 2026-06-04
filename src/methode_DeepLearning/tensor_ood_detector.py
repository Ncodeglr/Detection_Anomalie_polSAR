import sys
import os
import torch
import numpy as np
from typing import Union
from torch.utils.data import DataLoader

class Tensor_OOD_Detector:
    def __init__(self, model, device="cpu"):
        """
        Détecteur OoD spatial conçu pour les modèles de segmentation (UNet).
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
        #Paramètres de la distribution latente (Mahalanobis)
        self.mu = None
        self.inv_cov = None
        
        #Seuils calibrés
        self.threshold_latent = None

    def extract_latent(self, x):
        """
        Extrait le tenseur latent natif (complexe ou réel),
        et permute pour préserver l'axe spatial : [Batch, Hauteur, Largeur, Canaux].
        Compatible avec la méthode get_latent de UNet et AutoEncoder.
        """
        with torch.no_grad():
            # Extraction du latent via la fonction du modèle
            z = self.model.get_latent(x) # Shape: [B, C, H, W]
            # Permutation pour mettre les canaux à la fin -> Shape: [B, H, W, C]
            z = z.permute(0, 2, 3, 1) 
            
        return z

    def get_latent_features(self, x):
        """
        Extrait et construit un vecteur de caractéristiques robuste par patch.
        Combine la sémantique (Moyenne, Energie, Texture) et la physique PolSAR stricte.
        """
        # --- 1. Caractéristiques Sémantiques (Latent du UNet) ---
        # Le UNet normalise les données (BatchNorm), ce qui peut masquer les anomalies d'amplitude.
        z = self.extract_latent(x) # [B, H, W, C]
        
        z_mean = z.mean(dim=(1, 2)) # [B, C]
        if z.is_complex():
            z_energy = (z.real**2 + z.imag**2).mean(dim=(1, 2)) # [B, C]
            z_features = torch.cat([z_mean.real, z_mean.imag, z_energy], dim=-1)
        else:
            z_energy = (z**2).mean(dim=(1, 2))
            z_features = torch.cat([z_mean, z_energy], dim=-1)
            
        # --- 2. Caractéristiques Physiques (Entrée PolSAR) ---
        # Le Crosstalk modifie les corrélations entre canaux (HH vers HV, etc.).
        # Calculer la matrice de Covariance spatiale (Gram) et le SPAN de l'entrée brute
        # garantit une détection spectaculaire de cette anomalie physique.
        B, C_in, H, W = x.shape
        x_flat = x.view(B, C_in, -1) # [B, C_in, H*W]
        
        if x.is_complex():
            # Matrice de Covariance spatiale (C4)
            gram = torch.bmm(x_flat, x_flat.conj().transpose(1, 2)) / (H * W) # [B, C_in, C_in]
            
            # FEATURE 1 : SPAN (Énergie Totale)
            span = torch.diagonal(gram.real, dim1=1, dim2=2).sum(dim=1, keepdim=True) # [B, 1]

            # FEATURE 2 : Cross-Polarization Ratio (Extrêmement sensible au Crosstalk)
            cross_pol_ratio = torch.zeros_like(span) # Init à zéro pour les cas non-PolSAR (C_in != 4)
            if C_in == 4:
                # On assume l'ordre des canaux : 0:HH, 1:HV, 2:VH, 3:VV
                power_per_channel = torch.diagonal(gram.real, dim1=1, dim2=2) # [B, C_in]
                power_copol = power_per_channel[:, 0] + power_per_channel[:, 3] # Puissance HH + VV
                power_crosspol = power_per_channel[:, 1] + power_per_channel[:, 2] # Puissance HV + VH
                # Le ratio est un indicateur direct de la "fuite" d'énergie vers les canaux croisés
                cross_pol_ratio = (power_crosspol / (power_copol + 1e-8)).unsqueeze(1) # [B, 1]

            phys_features = torch.cat([gram.real.view(B, -1), gram.imag.view(B, -1), span, cross_pol_ratio], dim=-1)
        else:
            gram = torch.bmm(x_flat, x_flat.transpose(1, 2)) / (H * W)
            span = torch.diagonal(gram, dim1=1, dim2=2).sum(dim=1, keepdim=True)
            
            cross_pol_ratio = torch.zeros_like(span)
            if C_in == 4:
                power_per_channel = torch.diagonal(gram, dim1=1, dim2=2)
                power_copol = power_per_channel[:, 0] + power_per_channel[:, 3]
                power_crosspol = power_per_channel[:, 1] + power_per_channel[:, 2]
                cross_pol_ratio = (power_crosspol / (power_copol + 1e-8)).unsqueeze(1)
            
            phys_features = torch.cat([gram.view(B, -1), span, cross_pol_ratio], dim=-1)
            
        # --- 3. Concaténation (Physico-Sémantique) ---
        return torch.cat([z_features, phys_features], dim=-1)

    def fit_mahalanobis(self, train_loader: DataLoader, eps: float = 1e-5):
        """
        Calcule la distribution de Mahalanobis sur les vecteurs augmentés (Moyenne + Energie).
        L'énergie spatiale est vitale pour capter les transferts de puissance induits par le Crosstalk.
        """
        all_z = []
        
        for batch in train_loader:  
            x = batch[0] if isinstance(batch, (list, tuple)) else batch 
            x = x.to(self.device)
            
            with torch.no_grad():
                z_features = self.get_latent_features(x)
            all_z.append(z_features.cpu())
            
        all_z = torch.cat(all_z, dim=0) # [N_patches, Feature_dim] matrice 2D
        
        print(f"   -> Calcul de la covariance sur {all_z.shape[0]} patchs (Dimensions={all_z.shape[1]})...")
        self.mu = torch.mean(all_z, dim=0) # Vecteur de taille Feature_dim
        
        # Calcul de la matrice de covariance (Strictement réelle maintenant)
        cov = torch.cov(all_z.T)
        
        # Régularisation (ajoute eps sur la diagonale)
        cov += eps * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
        
        # Pseudo-inverse pour plus de stabilité
        self.inv_cov = torch.linalg.pinv(cov)

    def compute_scores(self, x):
        """
        Calcule la distance de Mahalanobis spatiale par patch.
        """
        x = x.to(self.device)
        
        with torch.no_grad():
            # Score latent spatial (Mahalanobis)
            z_features = self.get_latent_features(x)
            delta = z_features.cpu() - self.mu 
            
            # Calcul Mahalanobis (delta et inv_cov sont strictement réels)
            mah_dist_sq = torch.einsum('bi,ij,bj->b', delta, self.inv_cov, delta)
            
            mah_dist = torch.sqrt(torch.relu(mah_dist_sq)) # [B]
            
        return mah_dist.numpy()

    def calibrate_thresholds(self, pfa_loader, pfa: float):
        """
        Calibre les seuils avec les données de PFA.
        """
        all_mah = []
        
        for batch in pfa_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            mah = self.compute_scores(x)
            all_mah.append(mah)
            
        all_mah = np.concatenate(all_mah)
        
        self.threshold_latent = np.quantile(all_mah, 1 - pfa)
        
        return self.threshold_latent

    def detect(self, test_loader):
        """
        Calcule et retourne les prédictions (Mahalanobis Latent).
        Doit être appelé APRÈS fit_mahalanobis et calibrate_thresholds.
        """
        if self.threshold_latent is None:
            raise ValueError("Les seuils doivent être calibrés avant la détection.")

        all_preds_mah, all_scores_mah = [], []
        
        for batch in test_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(self.device)
                
            #Calcul des scores bruts
            mah = self.compute_scores(x)
            
            #Normalisation par le seuil pour obtenir un score comparable (avec sécurité) pour eviter la division par zéro
            score_mah = mah / (self.threshold_latent + 1e-12)
            
            #Décision (S(x) > 1  Anomalie / OoD, S(x) = 0 ou <1 alors Normal)
            all_preds_mah.append((score_mah > 1.0).astype(int))
            all_scores_mah.append(score_mah)
            
        return np.concatenate(all_preds_mah), np.concatenate(all_scores_mah)