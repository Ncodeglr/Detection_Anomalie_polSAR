import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ComplexPolSARTransform(nn.Module):
    def __init__(self, crop_size):
        super().__init__()
        self.crop_size = crop_size

    def apply_transform(self, x):
        B, C, H, W = x.shape
        th, tw = self.crop_size
        
        # 1. & 2. Random Crop et Flip Spatiaux INDÉPENDANTS par image
        # (Évite que tout le batch subisse exactement la même transformation)
        x_aug = torch.empty(B, C, th, tw, dtype=x.dtype, device=x.device)
        
        i_coords = torch.randint(0, max(1, H - th + 1), size=(B,))
        j_coords = torch.randint(0, max(1, W - tw + 1), size=(B,))
        flip_mask = torch.rand(B) > 0.5
        
        for b in range(B):
            i, j = i_coords[b].item(), j_coords[b].item()
            patch = x[b, :, i:i+th, j:j+tw]
            if flip_mask[b]:
                patch = torch.flip(patch, dims=[2]) # L'Azimut (W) devient la dimension 2 sur le patch (C, H, W)
            x_aug[b] = patch

        return x_aug

    def forward(self, x):
        view_1 = self.apply_transform(x)
        view_2 = self.apply_transform(x)
        return view_1, view_2


class AdvancedComplexPolSARTransform(nn.Module):
    """
    Augmentations spécifiques pour données Radar PolSAR à valeurs complexes.
    Oblige l'encodeur SimCLR à devenir invariant au bruit de Speckle et à la phase absolue,
    le forçant ainsi à apprendre les véritables structures et signatures polarimétriques.
    """
    def __init__(self, crop_size, speckle_std=0.05):
        super().__init__()
        self.crop_size = crop_size
        self.speckle_std = speckle_std

    def apply_transform(self, x):
        B, C, H, W = x.shape
        th, tw = self.crop_size
        
        x_aug = torch.empty(B, C, th, tw, dtype=x.dtype, device=x.device)
        
        i_coords = torch.randint(0, max(1, H - th + 1), size=(B,))
        j_coords = torch.randint(0, max(1, W - tw + 1), size=(B,))
        flip_mask = torch.rand(B) > 0.5
        
        for b in range(B):
            i, j = i_coords[b].item(), j_coords[b].item()
            patch = x[b, :, i:i+th, j:j+tw]
            if flip_mask[b]:
                patch = torch.flip(patch, dims=[2])
            x_aug[b] = patch

        # 1. Invariance à la Phase Absolue (Shift aléatoire global par image)
        random_phase = torch.rand(B, 1, 1, 1, device=x.device) * 2 * math.pi
        phase_shift = torch.polar(torch.ones_like(random_phase), random_phase)
        x_aug = x_aug * phase_shift

        # 2. Invariance au Speckle (Bruit Multiplicatif Complexe)
        if self.speckle_std > 0:
            noise_real = 1.0 + torch.randn_like(x_aug.real) * (self.speckle_std / 1.4142)
            noise_imag = torch.randn_like(x_aug.imag) * (self.speckle_std / 1.4142)
            x_aug = x_aug * torch.complex(noise_real, noise_imag)

        return x_aug

    def forward(self, x):
        view_1 = self.apply_transform(x)
        view_2 = self.apply_transform(x)
        return view_1, view_2

class RobustComplexNTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        z_i, z_j : Tenseurs complexes de dimensions [B, D]
        """
        batch_size = z_i.shape[0]

        # 1.Normalisation L2 - Diviser un tenseur complexe par un tenseur réel est nativement supporté.
        norm_i = torch.linalg.vector_norm(z_i, ord=2, dim=-1, keepdim=True).clamp_min(1e-8)
        norm_j = torch.linalg.vector_norm(z_j, ord=2, dim=-1, keepdim=True).clamp_min(1e-8)
        
        z_i = z_i / norm_i
        z_j = z_j / norm_j
        
        # 2.Concaténation pour la matrice globale [2B, D]
        z = torch.cat([z_i, z_j], dim=0) 
        
        # 3.Matrice de Similarité Globale via Produit Hermitien - torch.abs() extrait la magnitude (module) pour garantir l'invariance à la phase.
        sim_matrix = torch.abs(torch.matmul(z, z.conj().T)) / self.temperature
        
        # 4. Extraction des similarités positives
        # Les paires positives se trouvent sur les diagonales décalées de "batch_size"
        sim_ij = torch.diag(sim_matrix, batch_size)  # Diagonale du bloc supérieur droit
        sim_ji = torch.diag(sim_matrix, -batch_size) # Diagonale du bloc inférieur gauche
        positives = torch.cat([sim_ij, sim_ji], dim=0) # [2B]
        
        # 5. Masquage de la "Self-Similarity" (la diagonale principale)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix.masked_fill_(mask, -9e15) # Remplace par une grande valeur négative
        
        # 6. Calcul de la Perte avec LogSumExp (Stabilité numérique garantie)
        # Formule : -positives + log(sum(exp(sim_matrix)))
        lse = torch.logsumexp(sim_matrix, dim=1) # [2B]
        loss = -positives + lse
        
        return loss.mean()