import torch
import torch.nn as nn
from typing import Union


class Crosstalk(nn.Module):
    """
    Simule le Crosstalk entre les canaux de polarisation.
    Opère directement sur les vecteurs de diffusion (scattering vectors).
    """
    def __init__(self, delta: Union[complex, torch.Tensor] = 0.05 + 0.0j):
        super().__init__()
        
        # Si delta est un tenseur (ex: torch.tensor(0.05 + 0.01j)), on extrait le scalaire
        if isinstance(delta, torch.Tensor):
            delta_val = delta.item()
        else:
            delta_val = delta
            
        self.delta = delta_val
        
        # Création de la matrice 2x2 en forçant le type complexe
        D_2x2 = torch.tensor([
            [1.0, delta_val], 
            [delta_val, 1.0]
        ], dtype=torch.complex64)
        
        # Produit de Kronecker pour obtenir la matrice 4x4
        D_4x4 = torch.kron(D_2x2, D_2x2)
        
        # Enregistrement du buffer (il suivra le modèle sur GPU et conservera son type complexe)
        self.register_buffer('D_4x4', D_4x4)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Vérification robuste pour 3D ou 4D
        channel_dim = 1 if x.ndim == 4 else 0
        if x.shape[channel_dim] != 4:
            raise ValueError(f"Le tenseur d'entrée doit avoir 4 canaux, obtenu: {x.shape[channel_dim]}")
            
        # Sécurité : On s'assure que l'entrée est complexe pour ne pas perdre la partie imaginaire de la matrice D lors du cast.
        if not x.is_complex():
            raise TypeError(f"Le tenseur d'entrée doit être de type complexe (ex: torch.complex64), obtenu: {x.dtype}")
            
        # On caste la matrice au type d'entrée
        D = self.D_4x4.to(dtype=x.dtype)
        
        # S'adapte nativement au mode Dataset (3D) ou au mode Inférence/Réseau (4D)
        if x.ndim == 3:
            #i : L'axe des lignes de la matrice D (Axe de sortie, taille 4)
            #j : L'axe des colonnes de la matrice D (Axe d'entrée, taille 4)
            #h, w : Les axes spatiaux de l'image d'entrée (taille H, W)
            return torch.einsum('ij, jhw -> ihw', D, x)
        else:
            #La matrice D n'a toujours que ses dimensions i et j
            #Le tenseur x possède maintenant b, j, h, w
            #Le résultat -> bihw conserve le batch b
            return torch.einsum('ij, bjhw -> bihw', D, x)
