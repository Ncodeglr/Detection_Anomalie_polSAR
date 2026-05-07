import torch
import torch.nn as nn
from typing import Union


class Crosstalk(nn.Module):
    """
    Simule la diaphonie (Crosstalk) entre les canaux de polarisation.
    Opère directement sur les vecteurs de diffusion (scattering vectors).
    """
    def __init__(self, delta: float = 0.05):
        super().__init__()
        self.delta = delta
        
        # Optimisation : On calcule la matrice de transformation 1 seule fois
        D_2x2 = torch.tensor([[1.0, delta], [delta, 1.0]], dtype=torch.float32)
        D_4x4 = torch.kron(D_2x2, D_2x2)
        
        self.register_buffer('D_4x4', D_4x4)   # register_buffer permet au tenseur de suivre le modèle sur GPU (.to(device)) sans être considéré comme un paramètre entraînable par l'optimiseur
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 4:
            raise ValueError(f"Le tenseur d'entrée doit avoir 4 canaux, obtenu: {x.shape[1]}")
            
        # On caste la matrice au type d'entrée (important si l'entrée est de type complexe)
        D = self.D_4x4.to(dtype=x.dtype)
        
        # Application de la distorsion via einsum
        # i: canal de sortie, j: canal d'entrée, b: batch, h: hauteur, w: largeur
        return torch.einsum('ij, bjhw -> bihw', D, x)
        
class ChannelGainImbalance(nn.Module):
    """
    Simule un déséquilibre du gain entre les canaux (Channel Gain Imbalance).
    """
    def __init__(self, g: Union[float, complex] = 1.029):
        super().__init__()
        self.g = g
        
        g_tensor = torch.tensor(g)
        
        # Vecteur diagonal: [1.0, g, g, g**2]
        # Reshape en (1, 4, 1, 1) pour le broadcast PyTorch (B, C, H, W)
        D_diag = torch.stack([
            torch.tensor(1.0, dtype=g_tensor.dtype),
            g_tensor,
            g_tensor,
            g_tensor ** 2
        ]).view(1, 4, 1, 1)
        
        self.register_buffer('D_diag', D_diag)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 4:
            raise ValueError(f"Le tenseur d'entrée doit avoir 4 canaux, obtenu: {x.shape[1]}")
        
        D = self.D_diag.to(dtype=x.dtype)  # On caste le tenseur diagonal au type d'entrée
        
        return x * D