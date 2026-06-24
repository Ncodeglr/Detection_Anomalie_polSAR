import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union
from torch.utils.data import DataLoader

class OOD_Detector:
    def __init__(self, model, device="cpu"):
        """
        Détecteur OoD spatial conçu pour les modèles de segmentation (UNet).
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
        # Paramètres de la distribution sémantique (Latent)
        self.mu_z = None
        self.inv_cov_z = None
        self.std_z = None
        self.threshold_z = None
        
        # Paramètres de réduction de dimension (PCA) pour le modèle sémantique
        self.pca_mean_z = None
        self.pca_components_z = None

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
        
        # On capture non seulement la valeur moyenne (la "couleur") mais aussi la variance spatiale (la "texture")
        z_mean = z.mean(dim=(1, 2)) # [B, C], complexe
        # torch.std sur un tenseur complexe retourne un tenseur réel.
        # Pour capturer la variance des deux parties, on les calcule séparément.
        z_std_real = z.real.std(dim=(1, 2)) # [B, C], réel
        z_std_imag = z.imag.std(dim=(1, 2)) # [B, C], réel
        
        z_features = torch.cat([z_mean.real, z_mean.imag, z_std_real, z_std_imag], dim=-1) #[B, 4*C]
       
        # --- 2. Caractéristiques Physiques (Entrée PolSAR) ---
        B, C_in, H, W = x.shape
        x_flat = x.view(B, C_in, -1) # [B, C_in, H*W]
        
        gram = torch.bmm(x_flat, x_flat.conj().transpose(1, 2)) / (H * W) # [B, C_in, C_in]
            
        # --- GÉOMÉTRIE RIEMANNIENNE (Log-Euclidienne) ---
        # Projection de la matrice de Gram SPD dans son espace tangent via le log matriciel.
        # Cela transforme l'espace courbé des covariances en un espace vectoriel plat
        # où la distance de Mahalanobis redevient parfaitement valide, sans ingénierie manuelle.
        
        # Régularisation adaptative selon la puissance (trace) du patch.
        # Empêche un epsilon fixe d'écraser les faibles corrélations de diaphonie (Crosstalk)
        eps = 1e-6
        trace = gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1).real.clamp(min=1e-12)
        eps_matrix = (eps * trace).view(B, 1, 1) * torch.eye(C_in, dtype=gram.dtype, device=gram.device).unsqueeze(0)
        
        gram_reg = gram + eps_matrix
        
        # 1. Décomposition en valeurs propres (pour matrice hermitienne)
        eigvals, eigvecs = torch.linalg.eigh(gram_reg)
        
        # 2. Logarithme matriciel : log(G) = V * log(Lambda) * V^H
        log_eigvals = torch.log(eigvals.clamp(min=1e-12)).to(gram.dtype)
        log_gram = eigvecs @ torch.diag_embed(log_eigvals) @ eigvecs.conj().transpose(1, 2)
        
        # 3. Extraction du vecteur tangent : diagonale (réelle) et hors-diagonale (complexe)
        diag_elems = torch.diagonal(log_gram.real, dim1=-2, dim2=-1)
        triu_idx = torch.triu_indices(C_in, C_in, offset=1)
        off_diag_elems = log_gram[:, triu_idx[0], triu_idx[1]]

        p_features = torch.cat([
            diag_elems,
            off_diag_elems.real,
            off_diag_elems.imag
        ], dim=-1)

        return z_features, p_features

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
        # mais éviter le crash de l'inversion de matrice, on limite à 1e-8 (pas 1e-5 pour les cross-pol).
        std_safe = torch.clamp(std, min=1e-8)
        
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
        std_safe_robust = torch.clamp(std_robust, min=1e-8)
        
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
        
        # --- Modèle Sémantique (Z) ---
        # --- NOUVEAUTÉ : Réduction de Dimension (PCA) du Modèle Sémantique ---
        # Avec 512 dimensions, la distance de Mahalanobis accumule trop de bruit (Fléau de la dimension).
        # Une PCA permet de concentrer l'énergie sur les composantes principales et de stabiliser l'inversion.
        self.pca_mean_z = all_z.mean(dim=0)
        centered_z = all_z - self.pca_mean_z
        
        # Calcul de la matrice de covariance pour la PCA
        cov_z_pca = torch.cov(centered_z.T)
        eigvals_z, eigvecs_z = torch.linalg.eigh(cov_z_pca)
        
        # Tri des valeurs/vecteurs propres par ordre décroissant
        eigvals_z = eigvals_z.flip(dims=[0])
        eigvecs_z = eigvecs_z.flip(dims=[1])
        
        # On conserve les composantes expliquant 95% de la variance, avec un maximum de 64 dimensions
        explained_var = torch.cumsum(eigvals_z.clamp(min=0), dim=0) / eigvals_z.clamp(min=0).sum()
        k_95 = torch.searchsorted(explained_var, 0.95).item() + 1
        k = min(k_95, 64)
        
        self.pca_components_z = eigvecs_z[:, :k]
        all_z_pca = torch.matmul(centered_z, self.pca_components_z)
        print(f"   -> Modèle Sémantique réduit par PCA : {all_z.shape[1]} -> {k} dims")
        self.mu_z, self.std_z, self.inv_cov_z = self._robust_fit(all_z_pca, eps)
        
        # --- Modèle Physique (P) ---
        print(f"   -> Fitting Modèle Physique : {all_p.shape[1]} dims")
        self.mu_p, self.std_p, self.inv_cov_p = self._robust_fit(all_p, eps)

    def compute_scores(self, x):
        """
        Calcule les deux distances séparément.
        """
        x = x.to(self.device)

        with torch.no_grad():
            z_f, p_f = self._get_features(x)
            
            # --- Projection PCA Sémantique ---
            if self.pca_components_z is not None:
                z_f_pca = torch.matmul(z_f.cpu() - self.pca_mean_z, self.pca_components_z)
            else: # Fallback si la PCA n'a pas été fittée
                z_f_pca = z_f.cpu()
            p_f_cpu = p_f.cpu()
            
            # Mahalanobis Sémantique
            delta_z = (z_f_pca - self.mu_z) / self.std_z 
            mah_dist_sq_z = torch.einsum('bi,ij,bj->b', delta_z, self.inv_cov_z, delta_z)
            mah_dist_z = torch.sqrt(torch.relu(mah_dist_sq_z))
            
            # Mahalanobis Physique
            delta_p = (p_f_cpu - self.mu_p) / self.std_p
            mah_dist_sq_p = torch.einsum('bi,ij,bj->b', delta_p, self.inv_cov_p, delta_p)
            mah_dist_p = torch.sqrt(torch.relu(mah_dist_sq_p))
            
        return mah_dist_z.numpy(), mah_dist_p.numpy()

    def calibrate_thresholds(self, valid_loader, pfa: float):
        all_mah_z, all_mah_p = [], []
        
        for batch in valid_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            # Note: compute_scores retourne des numpy arrays
            mah_z, mah_p = self.compute_scores(x.to(self.device))
            all_mah_z.append(mah_z)
            all_mah_p.append(mah_p)
            
        mah_z_arr = np.concatenate(all_mah_z)
        mah_p_arr = np.concatenate(all_mah_p)
        
        # 1. On calcule les seuils initiaux pour chaque expert indépendamment
        thresh_z_init = np.quantile(mah_z_arr, 1 - pfa) + 1e-12
        thresh_p_init = np.quantile(mah_p_arr, 1 - pfa) + 1e-12
        
        # On les stocke pour la normalisation pendant la détection
        self.threshold_z = thresh_z_init
        self.threshold_p = thresh_p_init
        
        # Retourne le max des deux pour l'affichage console dans les autres scripts
        self.threshold_latent = max(self.threshold_z, self.threshold_p)
        return self.threshold_latent

    def detect(self, test_loader):
        if self.threshold_z is None or self.threshold_p is None:
            raise ValueError("Les seuils doivent être calibrés avant la détection.")

        all_preds, all_scores = [], []
        
        for batch in test_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(self.device)
            
            with torch.no_grad():
                # --- 1. Calcul des scores bruts pour les deux experts ---
                mah_z_np, mah_p_np = self.compute_scores(x)

                # --- 2. Normalisation des scores pour les rendre comparables ---
                score_z = mah_z_np / (self.threshold_z + 1e-12)
                score_p = mah_p_np / (self.threshold_p + 1e-12)

                # --- 3. Stratégie de fusion hiérarchique ---
                # On initialise le score final avec le score sémantique.
                score_final = score_z
                
                # On identifie les patchs que l'expert sémantique a jugés "sains" (score < 1.0)
                semantic_pass_mask = score_z < 1.0
                
                # Pour ces patchs uniquement, on remplace leur score par celui de l'expert physique.
                score_final[semantic_pass_mask] = score_p[semantic_pass_mask]
            
            all_preds.append((score_final > 1.0).astype(int))
            all_scores.append(score_final)
            
        return np.concatenate(all_preds), np.concatenate(all_scores)