import numpy as np
import torch
from tqdm import tqdm

def compute_batched_global_covariance(batch_patches: np.ndarray) -> np.ndarray:
    """ 
    Calcule la matrice de covariance C (4x4) pour chaque image d'un batch. 
    C'est une estimation de covariance globale par moyennage spatial.
    """
    # B : taille du batch
    # channels : Le nombre de canaux polarimétriques (HH, HV, VH, VV)
    # H et W : dimensions spatiales des patches
    B, channels, H, W = batch_patches.shape
    Np = H * W # Nombre de pixels par patch
    
    # k passe d'une forme (B, 4, H, W) à une forme (B, 4, Np)
    k = batch_patches.reshape(B, channels, Np) 
    k_H = np.conjugate(k.transpose(0, 2, 1))
    
    # Pour chaque image du batch, génère une matrice de taille 4x4
    C_batched = (k @ k_H) / Np
    return C_batched

def extract_batched_correlation_features(C_batched: np.ndarray) -> np.ndarray:
    """ Extrait les 16 caractéristiques (12 corrélations + 4 intensités en dB) pour chaque matrice du batch. """
    B = C_batched.shape[0]
    features = []
    
    # 1. Extraction des 12 caractéristiques de corrélation (géométrie)
    for i in range(4):
        for j in range(i): 
            norm_i = C_batched[:, i, i].real
            norm_j = C_batched[:, j, j].real
            
            denom = np.sqrt(norm_i * norm_j) 
            denom[denom == 0] = 1e-10 
            
            gamma_ij = C_batched[:, i, j] / denom  # Coefficient de corrélation
            
            features.append(gamma_ij.real)
            features.append(gamma_ij.imag)
            
    # 2. Ajout des 4 caractéristiques d'intensité (Diagonale de la matrice de covariance en dB)
    for i in range(4):
        intensity_linear = C_batched[:, i, i].real
        intensity_db = 10 * np.log10(np.maximum(intensity_linear, 1e-10))
        features.append(intensity_db)

    return np.stack(features, axis=1)

def extract_features_from_loader(dataloader, desc="Extraction"):
    """ Construit la matrice X (Features) à partir d'un DataLoader. """
    X_list = []

    for batch in tqdm(dataloader, desc=desc):
        # Gestion robuste des formats de batch
        if isinstance(batch, (list, tuple)):
            inputs = batch[0]
        elif isinstance(batch, dict):
            inputs = batch.get("inputs", batch.get("data"))
        else:
            inputs = batch
            
        x_np = inputs.cpu().numpy()
        
        mat_C_batched = compute_batched_global_covariance(x_np)
        features_batched = extract_batched_correlation_features(mat_C_batched)
        
        X_list.append(features_batched)

    return np.vstack(X_list)