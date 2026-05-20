import numpy as np
import torch
from typing import Union


class Crosstalk:
    """
    Simule un Cross-talk entre les canaux, où une partie de l'énergie du canal Horizontal fuit dans le canal Vertical et inversement.
    """
    def __init__(self, delta: Union[complex, float, torch.Tensor] = 0.05 + 0.0j):
        if isinstance(delta, torch.Tensor):
            delta_val = delta.item()
        else:
            delta_val = delta
            
        self.delta = delta_val
        name_val = str(np.round(np.abs(delta_val), 3)).replace(".", "p")
        self.name = f"Crosstalk_Error_d{name_val}"

    def apply_corruption(self, C_batched: np.ndarray) -> np.ndarray:
        D_2x2 = np.array([
            [1.0, self.delta],
            [self.delta, 1.0]
        ], dtype=complex)

        D_4x4 = np.kron(D_2x2, D_2x2)
        D_4x4_H = np.conjugate(D_4x4.T)

        C_corrupted = np.einsum('ij,njk,kl->nil', D_4x4, C_batched, D_4x4_H)
        return C_corrupted  

class ChannelGainImbalance:
    """
    Simule un déséquilibre du gain entre les canaux (Channel Gain Imbalance).
    """
    def __init__(self, g: Union[complex, float, torch.Tensor] = 1.029 + 0.0j):
        if isinstance(g, torch.Tensor):
            g_val = g.item()
        else:
            g_val = g
            
        self.g = g_val
        name_val = str(np.round(np.abs(g_val), 3)).replace(".", "p")
        self.name = f"ChannelGain_Error_g{name_val}"

    def apply_corruption(self, C_batched: np.ndarray) -> np.ndarray:
        D_4x4 = np.diag([1.0, self.g, self.g, self.g**2]).astype(complex)
        D_4x4_H = np.conjugate(D_4x4.T)
        C_corrupted = np.einsum('ij,njk,kl->nil', D_4x4, C_batched, D_4x4_H)
        return C_corrupted
    
