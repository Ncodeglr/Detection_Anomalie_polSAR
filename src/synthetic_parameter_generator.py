import torch 
import math
from typing import Optional
from torch.distributions.von_mises import VonMises

class SyntheticParameterGenerator:
    """
    Générateur de paramètres synthétiques.
    L'amplitude suit une loi normale centrée sur une valeur en dB.
    La phase suit une distribution de Von Mises.
    """
    def __init__(self, 
                 mean_db: float = -22.49, 
                 std_dev_amp: float = 0.001,
                 phase_mean_rad: float = 0,  # loc (ex: 45 degrés)
                 phase_concentration: float = 1e-5):    # kappa
        
        # --- Paramètres Amplitude ---
        self.mean_db = mean_db
        self.mean_linear = 10 ** (self.mean_db / 20.0) #Conversion pour ne pas être en dB
        self.std_dev_amp = std_dev_amp
        
        # --- Paramètres Phase (Von Mises) - PyTorch requiert que les paramètres de la distribution soient des tenseurs
        self.phase_loc = torch.tensor([phase_mean_rad])
        self.phase_kappa = torch.tensor([phase_concentration])
        
    def __call__(self, num_samples: int = 3, seed: Optional[int] = None) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)
            
        # 1. Tirage des amplitudes (Loi Normale)
        amplitudes = torch.randn(num_samples) * self.std_dev_amp + self.mean_linear
        amplitudes = torch.clamp(amplitudes, min=0.0) # On s'assure que les amplitudes restent positives (physiquement cohérent)
        
        # 2. Tirage des phases (Distribution de Von Mises)
        von_mises_dist = VonMises(self.phase_loc, self.phase_kappa)
        # sample() renvoie une shape [num_samples, 1] à cause des tenseurs d'init, on utilise squeeze()
        phases = von_mises_dist.sample((num_samples,)).squeeze()
        
        # 3. Création du tenseur complexe à partir des coordonnées polaires
        return torch.polar(amplitudes, phases)

if __name__ == "__main__":
    # 1. Instanciation
    generator = SyntheticParameterGenerator(
        mean_db=-15.0, 
        std_dev_amp=0.01,
        phase_mean_rad=0.0,
        phase_concentration=1e-5
    )
    
    # 2. Génération de 50 valeurs pour bien voir l'effet de groupe de Von Mises
    num_valeurs = 3
    complex_tensor = generator(num_samples=num_valeurs, seed=1234)
    
    # 3. Calcul de l'amplitude et de la phase
    amplitudes = torch.abs(complex_tensor)
    phases_rad = torch.angle(complex_tensor)
    phases_deg = torch.rad2deg(phases_rad)

    # 4. Affichage dans la console des 3 premières valeurs
    print(f"--- Objectif : Amplitude de -30 dB et Phase autour de 45° ---")
    print("\nDétail des 3 premiers échantillons :")
    for i in range(3):
        amp = amplitudes[i].item()
        p_deg = phases_deg[i].item()
        p_rad = phases_rad[i].item()
        real_part = complex_tensor[i].real.item()
        imag_part = complex_tensor[i].imag.item()
        print(f"Échantillon {i+1} : Amplitude = {amp:.4f} | Phase = {p_deg:>7.2f}° ({p_rad:>6.3f} rad) | Réel = {real_part:>7.4f} | Imag = {imag_part:>7.4f}")

    