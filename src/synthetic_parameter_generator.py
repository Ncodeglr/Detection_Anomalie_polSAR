import torch 
import math
from typing import Optional, List

# On tire le delta d'une distribution elliptique non circulaire pour simuler des anomalies plus réalistes

class SyntheticParameterGenerator:
    """
    Générateur de paramètres synthétiques complexes pour les anomalies (Crosstalk, Gain).
    """
    def __init__(self, delta: float, angle_rad: float = math.pi / 4):
        self.var_real = (1.0 + delta) / 2.0
        self.var_imag = (1.0 - delta) / 2.0
        self.rotation_factor = torch.exp(torch.tensor(1j * angle_rad))
        
    def __call__(self, scales: List[float] = [1.0, 0.1, 0.6], seed: Optional[int] = None) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)
            
        # 1. Génération de l'ellipse de base
        real_part = torch.randn(1) * math.sqrt(self.var_real)
        imag_part = torch.randn(1) * math.sqrt(self.var_imag)
        base_complex = torch.complex(real_part, imag_part)
        
        # 2. Génération des variantes dynamiquement
        variantes = []
        for scale in scales:
            if scale == 1.0:
                val = base_complex
            else:
                # Relation linéaire en amplitude, avec la phase randomisée
                val = torch.complex(base_complex.real * scale, torch.randn(1) * math.sqrt(self.var_imag))
            
            # Rotation dans le plan complexe
            val *= self.rotation_factor
            variantes.append(val)
            
        # 3. Empilement des données dans un tenseur final
        return torch.cat(variantes, dim=0)