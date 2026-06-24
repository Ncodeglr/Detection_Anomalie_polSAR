import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcvnn.nn import modReLU

class ComplexLinear(nn.Module):
    """Couche Linéaire native pour nombres complexes"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc_real = nn.Linear(in_features, out_features)
        self.fc_imag = nn.Linear(in_features, out_features)

    def forward(self, z):
        # z est un tenseur complexe de shape [B, in_features]
        real_out = self.fc_real(z.real) - self.fc_imag(z.imag)
        imag_out = self.fc_real(z.imag) + self.fc_imag(z.real)
        return torch.complex(real_out, imag_out)


class ComplexBatchNorm1d(nn.Module):
    """Batch Normalization complexe (Évite le Mode Collapse dans SimCLR)"""
    def __init__(self, num_features):
        super().__init__()
        self.bn_real = nn.BatchNorm1d(num_features)
        self.bn_imag = nn.BatchNorm1d(num_features)

    def forward(self, z):
        return torch.complex(self.bn_real(z.real), self.bn_imag(z.imag))


class RobustComplexSimCLR(nn.Module):
    def __init__(self, cvnn_encoder, complex_feature_dim, projection_dim=128):
        """
        complex_feature_dim: Nombre de 'out_channels' de la dernière couche de l'encodeur.
        projection_dim: Dimension de l'espace latent pour la loss contrastive (ex: 128).
        """
        super().__init__()
        self.encoder = cvnn_encoder
        
        # Tête de projection complexe avec BatchNorm et modReLU (Crucial pour SimCLR)
        self.projector = nn.Sequential(
            ComplexLinear(complex_feature_dim, complex_feature_dim),
            ComplexBatchNorm1d(complex_feature_dim),
            modReLU(),
            ComplexLinear(complex_feature_dim, projection_dim),
            ComplexBatchNorm1d(projection_dim)
        )

    def forward(self, x):
        # 1. Extraction des caractéristiques complexes (Bypass du classifieur de l'encodeur)
        if hasattr(self.encoder, "get_latent"):
            h_complex = self.encoder.get_latent(x)
        else:
            h_complex = self.encoder(x)
            
        if isinstance(h_complex, (list, tuple)):
            h_complex = h_complex[-1] # [B, C, H, W]
            
        # 2. Global Average Pooling Complexe
        # On réduit les dimensions spatiales (H, W) pour obtenir un vecteur 1D complexe par image.
        if h_complex.ndim == 4:
            h_pooled = h_complex.mean(dim=(2, 3)) # Résultat: [B, C]
        else:
            h_pooled = h_complex
        
        # 3. Passage dans le MLP complexe (Projector)
        z_complex = self.projector(h_pooled) # Résultat: [B, projection_dim]
        
        # h_pooled sera utilisé pour les tâches aval (fine-tuning, linear probing)
        # z_complex sera utilisé uniquement pour le calcul de la loss contrastive
        return h_pooled, z_complex

    def get_loss_format(self, z_complex):
        """
        Prépare le tenseur complexe pour la fonction de similarité cosinus de PyTorch.
        Concatène la partie réelle et imaginaire pour obtenir un tenseur purement réel.
        """
        # Résultat: [B, projection_dim * 2] (Format réel, compatible avec NT-Xent)
        return torch.cat([z_complex.real, z_complex.imag], dim=-1)