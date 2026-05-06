
import sys
import os
import torch
import numpy as np

# Ajout de cvnn/src au chemin pour pouvoir importer vos modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "cvnn", "src"))

from cvnn.config import load_config
from cvnn.utils import set_seed
from cvnn.data import get_dataloaders, azimut_split

def loader_to_numpy(loader, complex_handling="concat"):
    """
    Extrait les données d'un DataLoader PyTorch et les convertit
    en tableaux NumPy (X, y) pour les méthodes classiques.
    
    Args:
        loader: Le DataLoader PyTorch
        complex_handling: 'concat' (concatène réel et imaginaire sur l'axe des canaux) 
                          ou 'abs' (prend uniquement le module d'amplitude).
    """
    X_list, y_list = [], []
    
    for batch in loader:
        # Gestion robuste des formats de batch PyTorch (Tuple 1, 2 ou 3 éléments, dict, etc.)
        if isinstance(batch, (list, tuple)):
            inputs = batch[0]
            # On prend le 2e élément (target) s'il existe, et on ignore les indices s'il y a un 3e élément
            targets = batch[1] if len(batch) > 1 else None
        elif isinstance(batch, dict):
            inputs = batch.get("inputs", batch.get("data"))
            targets = batch.get("targets", batch.get("labels"))
        else:
            inputs = batch
            targets = None
            
        # Passage sur CPU et conversion NumPy
        inputs_np = inputs.cpu().numpy()
        
        # Traitement des données complexes (Scikit-Learn ne gère pas les complexes)
        if np.iscomplexobj(inputs_np):
            if complex_handling == "concat":
                # (B, C, H, W) -> (B, 2*C, H, W)
                inputs_np = np.concatenate([np.real(inputs_np), np.imag(inputs_np)], axis=1)
            elif complex_handling == "abs":
                inputs_np = np.abs(inputs_np)
                
        X_list.append(inputs_np)
        if targets is not None:
            y_list.append(targets.cpu().numpy())
            
    X = np.concatenate(X_list, axis=0) if X_list else np.array([])
    y = np.concatenate(y_list, axis=0) if y_list else None
    
    # Aplatissement spatial pour les modèles classiques (ex: B, C, H, W -> B, C*H*W)
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
        
    return X, y

def get_classical_data(config_path, use_azimut_split=False, complex_handling="concat"):
    """
    Charge les données en respectant scrupuleusement les splits du Deep Learning.
    """
    cfg = load_config(config_path)
    
    # 1. Figer l'aléatoire exactement comme dans BaseExperiment
    seed = cfg.get("seed", 42)
    set_seed(seed)
    
    # 2. Récupérer les loaders
    if use_azimut_split:
        loaders_dict = azimut_split(cfg, use_cuda=False)
        
        # Convertir automatiquement les loaders de la zone 1 (Train/Valid/Test) en NumPy
        if "part1_loaders" in loaders_dict:
            loaders_dict["part1_numpy"] = [
                loader_to_numpy(l, complex_handling) for l in loaders_dict["part1_loaders"]
            ]
            
        return loaders_dict
    else:
        loaders = get_dataloaders(cfg, use_cuda=False)
        return [loader_to_numpy(l, complex_handling) for l in loaders]
