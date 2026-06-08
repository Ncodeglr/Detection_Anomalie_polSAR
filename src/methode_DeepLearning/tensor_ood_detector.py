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
        Extrait le tenseur latent et permute pour préserver l'axe spatial : [Batch, Hauteur, Largeur, Canaux].
        Compatible avec la méthode get_latent de UNet et AutoEncoder.
        """
        with torch.no_grad():
            z = self.model.get_latent(x) #Extraction du tenseur latent de dim [B, C, H, W] via la méthode get_latent du modèle 
            z = z.permute(0, 2, 3, 1)  #Permutation pour mettre les canaux à la fin -> Shape: [B, H, W, C]
        return z

    def get_latent_features(self, x):
        """
        Extrait et construit un vecteur de caractéristiques robuste par patch.
        Combine la sémantique (Moyenne, Energie, Texture) et la physique PolSAR stricte.
        """
        # --- 1. Caractéristiques Sémantiques (Latent du UNet) ---
        # Le UNet normalise les données (BatchNorm), ce qui peut masquer les anomalies d'amplitude.
        z = self.extract_latent(x) #[B, H, W, C]
        
        #L'objectif de 1. est de créer un vecteur de caractéristiques (z_features) qui regroupe toutes les informations pertinentes d'un canal
        z_mean = z.mean(dim=(1, 2)) #[B, C]
        z_energy = (z.real**2 + z.imag**2).mean(dim=(1, 2)) #[B, C], On obtient une valeur d'énergie par canal pour chaque image du batch
        z_features = torch.cat([z_mean.real, z_mean.imag, z_energy], dim=-1) #[B, 3 * C]
       
            
        # --- 2. Caractéristiques Physiques (Entrée PolSAR) ---
        #Le Crosstalk modifie les corrélations entre canaux (HH vers HV, etc.).
        #Calculer la matrice de Covariance spatiale (Gram) et le SPAN de l'entrée brute garantit une détection spectaculaire de cette anomalie physique.
        B, C_in, H, W = x.shape
        x_flat = x.view(B, C_in, -1) # [B, C_in, H*W]
        
        #Matrice de Covariance spatiale pour chaque image du Batch
        gram = torch.bmm(x_flat, x_flat.conj().transpose(1, 2)) / (H * W) # [B, C_in, C_in]
            
        #FEATURE 1 : SPAN (Énergie Totale) pour chaque image du Batch
        span = torch.diagonal(gram.real, dim1=1, dim2=2).sum(dim=1, keepdim=True) # [B, 1] en ignorant la dimension 0 (le Batch). Pour chaque image du Batch, on regarde la matrice formée par la dimension 1 (lignes) et la dimension 2 (colonnes), et on extrait la diagonale, on ignore la dimension 0 (le Batch). 

        #FEATURE 2 : Cross-Polarization Ratio (Extrêmement sensible au Crosstalk)
        cross_pol_ratio = torch.zeros_like(span) # Init à zéro pour les cas non-PolSAR (C_in != 4)
        
        #On assume l'ordre des canaux : 0:HH, 1:HV, 2:VH, 3:VV
        power_per_channel = torch.diagonal(gram.real, dim1=1, dim2=2) # [B, C_in]
        power_copol = power_per_channel[:, 0] + power_per_channel[:, 3] # Puissance HH + VV
        power_crosspol = power_per_channel[:, 1] + power_per_channel[:, 2] # Puissance HV + VH
        
        #Le ratio est un indicateur direct de la "fuite" d'énergie vers les canaux croisés  
        #Les signaux HV/VH sont très faibles. Si un défaut d'antenne (Crosstalk) fait "fuir" l'énergie puissante de HH/VV vers HV/VH, ce ratio va exploser et l'algorithme identifiera immédiatement l'anomalie.             
        cross_pol_ratio = (power_crosspol / (power_copol + 1e-12)).unsqueeze(1) # [B, 1]

            
        #gram.real.view(B, -1) : On prend la partie réelle de la matrice $4 \times 4$ et on l'aplatit (.view(B, -1)) en un vecteur 1D. On passe de [B, 4, 4] à [B, 16] pour chaque image du batch. Chaque élément de ce vecteur représente une corrélation spatiale entre les canaux (HH-HH, HH-HV, ..., VV-VV). 
        #gram.imag.view(B, -1) : Idem pour la partie imaginaire
        #span : Une mesure par image donc on garde la dimension [B, 1]
        #cross_pol_ratio : Une mesure par image donc on garde la dimension [B, 1]
        phys_features = torch.cat([gram.real.view(B, -1), gram.imag.view(B, -1), span, cross_pol_ratio], dim=-1) # [B, 16 (real) + 16 (imag) + 1 (span) + 1 (cross_pol)] = [B, 34]
            
        # --- 3. Concaténation (Physico-Sémantique) ---
        return torch.cat([z_features, phys_features], dim=-1) #[32, 3*C_in + 34]

    def fit_mahalanobis(self, train_loader: DataLoader, eps: float = 1e-12):
        """
        Calcule la distribution de Mahalanobis sur les vecteurs augmentés (Moyenne + Energie).
        """
        all_z = []
        
        for batch in train_loader:  
            x = batch[0] if isinstance(batch, (list, tuple)) else batch 
            x = x.to(self.device)
            
            with torch.no_grad():
                z_features = self.get_latent_features(x)
            all_z.append(z_features.cpu())
            
        all_z = torch.cat(all_z, dim=0) # [N_patches, Feature_dim = 130 dans notre cas] matrice 2D
        
        print(f"   -> Calcul de la covariance sur {all_z.shape[0]} patchs (Dimensions={all_z.shape[1]})...")
        
        #Calcul de la moyenne
        self.mu = torch.mean(all_z, dim=0) #On calcule la moyenne de chaque caractéristique sur tous les patchs pour obtenir un vecteur de moyenne de dimension [Feature_dim] (130 dans notre cas) qui représente le centre de la distribution normale multivariée dans l'espace des caractéristiques.
        
        #Calcul de la matrice de covariance (Strictement réelle maintenant)
        ##La fonction torch.cov() de PyTorch attend que les variables (les caractéristiques) soient sur les lignes et les observations (les patchs) sur les colonnes. D'ou le faite d'utiliser all_z.T pour transposer la matrice.
        cov = torch.cov(all_z.T) #La matrice de covariance résultante aura une dimension de [Feature_dim, Feature_dim] (130x130 dans notre cas) et contiendra les covariances entre toutes les paires de caractéristiques.
        
        #Régularisation (ajoute eps sur la diagonale) pour éviter les problèmes de singularité et garantir que la matrice de covariance est inversible. 
        cov += eps * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
        
        #Pseudo-inverse de Moore-Penrosepour plus de stabilité
        # -Si une information est redondante (deux caractéristiques qui évoluent exactement de la même façon), le pseudo-inverse va la "lisser" intelligemment au lieu d'exploser.
        # -Si la matrice est parfaitement saine et inversible, pinv donnera exactement le même résultat que inv
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

    def calibrate_thresholds(self, valid_loader, pfa: float):
        """
        Calibre les seuils avec les données de PFA.
        """
        all_mah = []
        
        for batch in valid_loader:
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