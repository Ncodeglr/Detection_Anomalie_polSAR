import torch
from torch.utils.data import DataLoader
import numpy as np

from AE.src.models.cnets import AE
from AE.src.models.train import train
from AE.src.data.datasets import NumpyTensorDataset

from tqdm import tqdm


class AEDetector:

    def __init__(self, config, device="pu", variant="complex"):
        self.config = config
        self.device = device
        self.variant = variant

        self.batch_size = config["ae"]["batch_size"]
        self.lr = config["ae"]["learning_rate"]
        self.epochs = config["ae"]["epochs"]

        self.model = AE(input_channels=1,feature_sizes=[3,4,5],latent_dim=16,input_size=16,
        encoder_activation='modrelu',decoder_activation='modrelu',final_activation='',norm_type='none').to(self.device)

        self.cfg_train = {
            "train": {
                "epochs": self.epochs,
                "lr": self.lr,
                "batch_size": self.batch_size,
            },
            "misc": {
                "device": device,
                "num_workers": 2
            }
        }

    def _to_loader(self, s, labels=None, shuffle=False):
        if labels is None:
          ds = NumpyTensorDataset(data=s)    
        else:
            ds = NumpyTensorDataset(data=s, labels= labels)
        return DataLoader(ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=0, drop_last=False)

    def fit(self, s_train):
        loader = self._to_loader(s_train,shuffle=False)
        train_losses = train(
        self.model, loader, self.cfg_train)

    def calibrate_threshold(self, s_pfa, pfa):
        loader = self._to_loader(s_pfa, shuffle=False)
        return self._calibrate_from_loader(loader, pfa)

    def _calibrate_from_loader(self, loader, pfa):
        self.model.eval()
        all_distances = []

        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                x_hat = self.model(x)
                dist = torch.mean(torch.abs(x - x_hat)**2, dim=(1,2))
                all_distances.append(dist.cpu().numpy())

        all_distances = np.concatenate(all_distances)
        return np.quantile(all_distances, 1 - pfa)

    def detect(self, s_test, labels, threshold):
        loader = self._to_loader(s_test, labels, shuffle=False)
        return self._detect_loader(loader, threshold)

    def _detect_loader(self, loader, threshold):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for x,_ in loader:
                x = x.to(self.device)
                x_hat = self.model(x)
                dist = torch.mean(torch.abs(x - x_hat)**2, dim=(1,2))
                pred = (dist > threshold).long()
                pred = pred.cpu().numpy()
                predictions.append(pred)
        return np.concatenate(predictions, axis=0)

def run_AE(cfg, data, device, variant="complex"):

    s_train_fft = data["s_train_fft"]
    s_pfa_fft = data["s_pfa_fft"]
    test_data = data["test_data"]
    pfa = cfg["simulation"]["pfa"]

    detector = AEDetector(cfg, device, variant=variant)

    if variant == "complex":
        train_input = s_train_fft[:, :, 0]  # si ton réseau attend 1 canal
        pfa_input   = s_pfa_fft[:, :, 0]
    elif variant == "real":
        train_input = np.abs(s_train_fft[:, :, 0])
        pfa_input   = np.abs(s_pfa_fft[:, :, 0])
    detector.fit(train_input)
    threshold = detector.calibrate_threshold(pfa_input, pfa)

    results = []

    for (snr, doppler), (_, s_test_fft, labels) in test_data.items():

        if variant == "complex":
            test_input = s_test_fft[:, :, 0]
        elif variant == "real":
            test_input = np.abs(s_test_fft[:, :, 0])
        detected = detector.detect(
            test_input,
            labels,
            threshold
        )

        results.append({
            "SNR": snr,
            "doppler": doppler,
            "detected": detected[labels == 1].mean(),
            "pfa": detected[labels == 0].mean()
        })

    return results

def run_latent(cfg, data, device, variant="complex"):

    s_train_fft = data["s_train_fft"]
    s_pfa_fft = data["s_pfa_fft"]
    test_data = data["test_data"]
    pfa = cfg["simulation"]["pfa"]

    detector = AEDetector(cfg, device, variant=variant)

    if variant == "complex":
        train_input = s_train_fft[:, :, 0]  # si ton réseau attend 1 canal
        pfa_input   = s_pfa_fft[:, :, 0]
    elif variant == "real":
        train_input = np.abs(s_train_fft[:, :, 0])
        pfa_input   = np.abs(s_pfa_fft[:, :, 0])

    detector.fit(train_input)
    threshold = detector.calibrate_threshold(pfa_input, pfa)

    for (snr, doppler), (_, s_test_fft, labels) in test_data.items():

        if variant == "complex":
            test_input = s_test_fft[:, :, 0]
        elif variant == "real":
            test_input = np.abs(s_test_fft[:, :, 0])

    loader = detector._to_loader(test_input, labels=None, shuffle=False)
    return detector, loader, labels
