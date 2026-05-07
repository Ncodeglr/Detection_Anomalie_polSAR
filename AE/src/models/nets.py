

from .layers import ConvBlock
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt



class EncoderAE(nn.Module):
    def __init__(self, input_channels, feature_sizes, latent_dim, input_size=16, 
                 activation='relu', norm_type='batch', dropout_rate=0.0):
        super().__init__()
        
        self.input_size = input_size
        layers = []
        current_size = input_size
        
        layers.append(ConvBlock(input_channels, feature_sizes[0],
                                activation=activation, norm_type=norm_type))
        
        for i in range(len(feature_sizes) - 1):
            layers.append(ConvBlock(feature_sizes[i], feature_sizes[i + 1],
                                    activation=activation, norm_type=norm_type))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.MaxPool1d(2, 2))
            current_size //= 2
        
        self.features = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        
        self.final_features = feature_sizes[-1] * current_size
        
        self.fc = nn.Linear(self.final_features, latent_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        z = self.fc(x)
        return z
    

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_channels, feature_sizes, initial_size=None, 
                 activation='relu', final_activation='', norm_type='none'):
        super(Decoder, self).__init__()
        
        # Si initial_size n'est pas spécifié, calculer une valeur raisonnable
        if initial_size is None:
            # Calculer initial_size en fonction du nombre de couches d'upsampling
            n_upsample = len(feature_sizes) - 1
            self.initial_size = 2 ** n_upsample
        else:
            self.initial_size = initial_size
        
        self.feature_sizes = feature_sizes[::-1]  # Inversion du feature vector
        
        # Couche linéaire avec dimension adaptative
        self.fc = nn.Linear(latent_dim, self.feature_sizes[0] * self.initial_size)
        
        layers = []
        for i in range(len(self.feature_sizes) - 1):
            layers.append(
                ConvBlock(
                    self.feature_sizes[i], 
                    self.feature_sizes[i + 1], 
                    conv_type='deconv', 
                    upsample=True,
                    activation=activation,
                    norm_type=norm_type
                )
            )
        
        # Dernière couche avec activation spécifiée
        layers.append(
            ConvBlock(
                self.feature_sizes[-1], 
                output_channels, 
                conv_type='deconv', 
                activation=final_activation,
                upsample=False,
                norm_type='none'  # Pas de normalisation pour la dernière couche
            )
        )
        
        self.deconv = nn.Sequential(*layers)
    
    def forward(self, z):
        z = self.fc(z)
        z = z.view(-1, self.feature_sizes[0], self.initial_size)
        reconstruction = self.deconv(z)
        return reconstruction
    
class AE(nn.Module):
    def __init__(self, input_channels, feature_sizes, latent_dim, input_size=16,
                encoder_activation='relu', decoder_activation='relu', 
                final_activation='', norm_type='none', dropout_rate=0.0):
        super().__init__()
        
        self.encoder = EncoderAE(
            input_channels,
            feature_sizes,
            latent_dim,
            input_size=input_size,
            activation=encoder_activation,
            norm_type=norm_type,
            dropout_rate=dropout_rate
        )
        
        self.decoder = Decoder(
            latent_dim,
            input_channels,
            feature_sizes,
            activation=decoder_activation,
            final_activation=final_activation,
            norm_type=norm_type
        )
    
    def forward(self, x):
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)