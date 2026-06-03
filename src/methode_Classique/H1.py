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