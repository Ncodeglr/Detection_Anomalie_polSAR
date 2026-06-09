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
        
        # Paramètres de la distribution sémantique (Latent UNet)
        self.mu_z = None
        self.inv_cov_z = None
        self.std_z = None
        self.threshold_z = None
        
        # Paramètres de la distribution physique (PolSAR)
        self.mu_p = None
        self.inv_cov_p = None
        self.std_p = None
        self.threshold_p = None
        
        # Rétro-compatibilité pour les scripts d'évaluation
        self.threshold_latent = None

    def extract_latent(self, x):
        """
        Extrait le tenseur latent et permute pour préserver l'axe spatial : [Batch, Hauteur, Largeur, Canaux].
        Compatible avec la méthode get_latent de UNet et AutoEncoder.
        """
        with torch.no_grad():
            z = self.model.get_latent(x) #Extraction du tenseur latent de dim [B, C, H, W] via la méthode get_latent du modèle 
            z = z.permute(0, 2, 3, 1)  #Permutation pour mettre les canaux à la fin -> Shape: [B, H, W, C]
        return z

    def _get_features(self, x):
        # --- 1. Caractéristiques Sémantiques (Latent du UNet) ---
        z = self.extract_latent(x) #[B, H, W, C]
        
        z_mean = z.mean(dim=(1, 2)) #[B, C]
        
        # Variance spatiale (dispersion autour de la moyenne) pour détecter les instabilités du modèle
        z_centered = z - z_mean.unsqueeze(1).unsqueeze(1)
        z_var = (z_centered.real**2 + z_centered.imag**2).mean(dim=(1, 2)) #[B, C]
        
        z_energy = (z.real**2 + z.imag**2).mean(dim=(1, 2)) #[B, C]
        z_features = torch.cat([z_mean.real, z_mean.imag, z_var, z_energy], dim=-1) #[B, 4 * C]
       
        # --- 2. Caractéristiques Physiques (Entrée PolSAR) ---
        B, C_in, H, W = x.shape
        x_flat = x.view(B, C_in, -1) # [B, C_in, H*W]
        
        gram = torch.bmm(x_flat, x_flat.conj().transpose(1, 2)) / (H * W) # [B, C_in, C_in]
            
        power_per_channel = torch.diagonal(gram.real, dim1=1, dim2=2) # [B, C_in]

        span = power_per_channel.sum(dim=1, keepdim=True) # [B, 1]
        span_db = 10.0 * torch.log10(span + 1e-12)

        power_copol = power_per_channel[:, 0] + power_per_channel[:, 3] # Puissance HH + VV
        power_crosspol = power_per_channel[:, 1] + power_per_channel[:, 2] # Puissance HV + VH
        
        cross_pol_ratio = (power_crosspol / (power_copol + 1e-12)).unsqueeze(1) # [B, 1]
        cross_pol_ratio_db = 10.0 * torch.log10(cross_pol_ratio + 1e-12)

        if C_in == 4:
            # --- NORMALISATION PARFAITE (Variables 100% Bornées et Indépendantes) ---
            # Le problème de Mahalanobis est qu'une seule variable non bornée (ou colinéaire)
            # détruit l'inverse de la matrice de covariance.
            # En fournissant UNIQUEMENT des Cohérences (-1 à 1) et des ratios partiels,
            # l'ellipsoïde appris est extrêmement dense, stable et hyper-sensible.
            
            eps_power = 1e-6
            
            # --- 1. BASE DE PAULI ---
            k1 = (x_flat[:, 0, :] + x_flat[:, 3, :]) / 1.41421356
            k2 = (x_flat[:, 0, :] - x_flat[:, 3, :]) / 1.41421356
            k3 = (x_flat[:, 1, :] + x_flat[:, 2, :]) / 1.41421356
            
            T11 = torch.sum(k1 * k1.conj(), dim=-1).real / (H * W)
            T22 = torch.sum(k2 * k2.conj(), dim=-1).real / (H * W)
            T33 = torch.sum(k3 * k3.conj(), dim=-1).real / (H * W)
            
            T12 = torch.sum(k1 * k2.conj(), dim=-1) / (H * W)
            T13 = torch.sum(k1 * k3.conj(), dim=-1) / (H * W)
            T23 = torch.sum(k2 * k3.conj(), dim=-1) / (H * W)
            
            span_pauli = T11 + T22 + T33 + eps_power
            
            # --- 2. Ratios d'Énergie ---
            r_T11 = (T11 / span_pauli).unsqueeze(1)
            r_T22 = (T22 / span_pauli).unsqueeze(1)
            # On ignore r_T33 car r_T11 + r_T22 + r_T33 = 1 (Évite la Singularité Mathématique)
            
            # --- 3. Cohérences Complexes de Pauli (Le "Graal" du Crosstalk) ---
            # Sous Crosstalk, l'erreur (Delta) se concentre massivement et uniquement sur T13 et T23.
            def pauli_coherence(T_ij, T_ii, T_jj):
                return (T_ij / (torch.sqrt(T_ii * T_jj) + eps_power)).unsqueeze(1)
                
            coh_13 = pauli_coherence(T13, T11, T33) # Explose statistiquement sous l'anomalie
            coh_23 = pauli_coherence(T23, T22, T33)
            
            # --- 4. Réciprocité et Contexte ---
            cov_hv_vh = gram[:, 1, 2].unsqueeze(1)
            p_hv = power_per_channel[:, 1:2]
            p_vh = power_per_channel[:, 2:3]
            coh_hv_vh = cov_hv_vh / (torch.sqrt(p_hv * p_vh) + eps_power)
            imbalance = (p_hv - p_vh) / (p_hv + p_vh + eps_power)
            
            phys_features = torch.cat([
                r_T11, r_T22,
                coh_13.real, coh_13.imag,
                coh_23.real, coh_23.imag,
                coh_hv_vh.real, coh_hv_vh.imag,
                imbalance
            ], dim=-1) # [B, 9] Le vecteur minimal absolu et suffisant pour Mahalanobis
        else:
            eigvals = torch.linalg.eigvalsh(gram).abs()
            eigvals_db = 10.0 * torch.log10(eigvals + 1e-12)
            phys_features = torch.cat([span_db, eigvals_db], dim=-1)

        return z_features, phys_features

    def get_latent_features(self, x):
        """
        Rétro-compatibilité pour les visualisations (PCA) ou scripts existants.
        """
        z_f, p_f = self._get_features(x)
        return torch.cat([z_f, p_f], dim=-1)

    def _robust_fit(self, features, eps=1e-4):
        """Ajustement robuste en 2 passes pour éliminer les valeurs extrêmes naturelles (ex: scatterers urbains forts) qui élargissent l'ellipsoïde."""
        # --- Passe 1 : Fit standard ---
        mu = torch.mean(features, dim=0)
        std = torch.std(features, dim=0)
        
        # Pour garantir qu'une variable à variance quasi-nulle puisse exploser sous anomalie, 
        # mais éviter le crash de l'inversion de matrice, on limite à 1e-5.
        std_safe = torch.clamp(std, min=1e-5)
        
        scaled = (features - mu) / std_safe
        cov = torch.cov(scaled.T) + eps * torch.eye(features.shape[1], dtype=features.dtype, device=features.device)
        inv_cov = torch.linalg.pinv(cov, hermitian=True)
        
        # --- Filtrage des outliers (Top 5% naturel) ---
        dist_sq = torch.einsum('bi,ij,bj->b', scaled, inv_cov, scaled)
        q95 = torch.quantile(dist_sq, 0.95)
        clean_features = features[dist_sq <= q95]
        
        # --- Passe 2 : Fit sur données purifiées ---
        mu_robust = torch.mean(clean_features, dim=0)
        std_robust = torch.std(clean_features, dim=0)
        std_safe_robust = torch.clamp(std_robust, min=1e-5)
        
        scaled_robust = (clean_features - mu_robust) / std_safe_robust
        cov_robust = torch.cov(scaled_robust.T) + eps * torch.eye(clean_features.shape[1], dtype=clean_features.dtype, device=clean_features.device)
        inv_cov_robust = torch.linalg.pinv(cov_robust, hermitian=True)
        
        return mu_robust, std_safe_robust, inv_cov_robust

    def fit_mahalanobis(self, train_loader: DataLoader, eps: float = 1e-4):
        """
        Entraîne un modèle d'Ensemble: Sémantique d'un côté, Physique de l'autre.
        """
        all_z, all_p = [], []
        
        for batch in train_loader:  
            x = batch[0] if isinstance(batch, (list, tuple)) else batch 
            x = x.to(self.device)
            
            with torch.no_grad():
                z_f, p_f = self._get_features(x)
            all_z.append(z_f.cpu())
            all_p.append(p_f.cpu())
            
        all_z = torch.cat(all_z, dim=0)
        all_p = torch.cat(all_p, dim=0)
        
        print(f"   -> Modèle Sémantique ({all_z.shape[1]} dims) et Physique ({all_p.shape[1]} dims)...")
        
        # --- Modèle Sémantique (Z) ---
        self.mu_z, self.std_z, self.inv_cov_z = self._robust_fit(all_z, eps)
        
        # --- Modèle Physique (P) ---
        self.mu_p, self.std_p, self.inv_cov_p = self._robust_fit(all_p, eps)

    def compute_scores(self, x):
        """
        Calcule les deux distances séparément.
        """
        x = x.to(self.device)
        
        with torch.no_grad():
            z_f, p_f = self._get_features(x)
            
            # Mahalanobis Sémantique
            delta_z = (z_f.cpu() - self.mu_z) / self.std_z 
            mah_dist_sq_z = torch.einsum('bi,ij,bj->b', delta_z, self.inv_cov_z, delta_z)
            mah_dist_z = torch.sqrt(torch.relu(mah_dist_sq_z))
            
            # Mahalanobis Physique
            delta_p = (p_f.cpu() - self.mu_p) / self.std_p 
            mah_dist_sq_p = torch.einsum('bi,ij,bj->b', delta_p, self.inv_cov_p, delta_p)
            mah_dist_p = torch.sqrt(torch.relu(mah_dist_sq_p))
            
        return mah_dist_z.numpy(), mah_dist_p.numpy()

    def calibrate_thresholds(self, valid_loader, pfa: float):
        all_mah_z, all_mah_p = [], []
        
        for batch in valid_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            mah_z, mah_p = self.compute_scores(x)
            all_mah_z.append(mah_z)
            all_mah_p.append(mah_p)
            
        self.threshold_z = np.quantile(np.concatenate(all_mah_z), 1 - pfa)
        self.threshold_p = np.quantile(np.concatenate(all_mah_p), 1 - pfa)
        
        # Retourne le max des deux pour l'affichage console dans les autres scripts
        self.threshold_latent = max(self.threshold_z, self.threshold_p)
        return self.threshold_latent

    def detect(self, test_loader):
        if self.threshold_z is None or self.threshold_p is None:
            raise ValueError("Les seuils doivent être calibrés avant la détection.")

        all_preds, all_scores = [], []
        
        for batch in test_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
                
            mah_z, mah_p = self.compute_scores(x)
            
            score_z = mah_z / (self.threshold_z + 1e-12)
            score_p = mah_p / (self.threshold_p + 1e-12)
            
            # DÉTECTEUR ENSEMBLE : Le score final est le MAX. 
            # L'anomalie est signalée si c'est une bizarrerie sémantique OU physique.
            score_final = np.maximum(score_z, score_p)
            
            all_preds.append((score_final > 1.0).astype(int))
            all_scores.append(score_final)
            
        return np.concatenate(all_preds), np.concatenate(all_scores)