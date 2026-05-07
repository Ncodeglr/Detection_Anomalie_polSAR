import torch
import torch.nn as nn
from torchcvnn.nn.modules.upsampling import Upsample
from torchcvnn.nn.modules.activation import CReLU, CPReLU, modReLU
from .cpool1d import MaxPool1d
from .convtrans1d import ConvTranspose1d

class ComplexConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 activation='modrelu', conv_type='conv', upsample=False, norm_type='none', dropout_rate=0.0):
        super(ComplexConvBlock, self).__init__()

        # Configuration de la convolution
        if conv_type == 'conv':
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dtype=torch.complex64)
        elif conv_type == 'deconv':
            if upsample:
                self.conv = nn.Sequential(
                    Upsample(mode='linear', scale_factor=2),
                    nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding, dtype=torch.complex64)
                )
            else:
                self.conv = ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding=stride-1)
        else:
            raise ValueError("conv_type must be 'conv' or 'deconv'")

        # Normalisation pour les nombres complexes
        # Note: Dans un environnement complexe, la normalisation doit être adaptée
        self.use_norm = norm_type != 'none'
        if self.use_norm:
            self.norm_real = nn.BatchNorm1d(out_channels)
            self.norm_imag = nn.BatchNorm1d(out_channels)

        # Activation complexe
        if activation == 'crelu':
            self.activation = CReLU()  # Complex ReLU
        elif activation == 'modrelu':
            self.activation = modReLU()  # Complex modReLU
        elif activation == 'cleakyrelu':
            self.activation = CPReLU()  # Complex PReLU
        elif activation == '':
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        # Dropout complexe
        self.use_dropout = dropout_rate > 0
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv(x)
        
        # Normalisation appliquée séparément sur parties réelle et imaginaire
        if self.use_norm:
            x_real = self.norm_real(x.real)
            x_imag = self.norm_imag(x.imag)
            x = torch.complex(x_real, x_imag)
        
        x = self.activation(x)
        
        # Dropout appliqué identiquement sur parties réelle et imaginaire
        if self.use_dropout:
            # Créer un masque de dropout (les mêmes éléments sont mis à zéro dans 
            # les parties réelle et imaginaire)
            mask = torch.ones_like(x.real)
            mask = self.dropout(mask)
            x = x * mask.unsqueeze(-1)
            
        return x

class ComplexEncoder(nn.Module):
    def __init__(self, input_channels, feature_sizes, latent_dim, input_size=16, 
                 activation='modrelu', norm_type='none', dropout_rate=0.0):
        super(ComplexEncoder, self).__init__()
        
        self.input_size = input_size
        current_size = input_size
        layers = []

        # Construction des couches de convolution
        layers.append(ComplexConvBlock(
            input_channels, 
            feature_sizes[0], 
            activation=activation, 
            norm_type=norm_type, 
            dropout_rate=dropout_rate
        ))
        
        for i in range(len(feature_sizes) - 1):
            layers.append(ComplexConvBlock(
                feature_sizes[i], 
                feature_sizes[i + 1], 
                activation=activation, 
                norm_type=norm_type,
                dropout_rate=dropout_rate
            ))
            layers.append(MaxPool1d(kernel_size=2, stride=2))  # Complex max pooling
            current_size //= 2

        self.features = nn.Sequential(*layers)
        
        # Calcul automatique de la dimension d'entrée pour les couches linéaires
        self.flatten = nn.Flatten()
        self.final_features = feature_sizes[-1] * current_size
        
        #self.fc = nn.Linear(self.final_features, latent_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return x

class ComplexDecoder(nn.Module):
    def __init__(self, latent_dim, output_channels, feature_sizes, initial_size=None,
                 activation='modrelu', final_activation='', norm_type='none', dropout_rate=0.0):
        super(ComplexDecoder, self).__init__()
        
        # Si initial_size n'est pas spécifié, calculer une valeur raisonnable
        if initial_size is None:
            # Calculer initial_size en fonction du nombre de couches d'upsampling
            n_upsample = len(feature_sizes) - 1
            self.initial_size = 2 ** n_upsample
        else:
            self.initial_size = initial_size
            
        self.feature_sizes = feature_sizes[::-1]  # Inversion du feature vector
        
        # Couche linéaire avec dimension adaptative
        self.fc = nn.Linear(latent_dim, self.feature_sizes[0] * self.initial_size, dtype=torch.complex64)

        layers = []
        for i in range(len(self.feature_sizes) - 1):
            layers.append(
                ComplexConvBlock(
                    self.feature_sizes[i], 
                    self.feature_sizes[i + 1], 
                    conv_type='deconv', 
                    upsample=True,
                    activation=activation,
                    norm_type=norm_type,
                    dropout_rate=dropout_rate
                )
            )

        # Dernière couche avec activation spécifiée
        layers.append(
            ComplexConvBlock(
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
        #z = self.fc(z)
        z = z.view(-1, self.feature_sizes[0], self.initial_size)
        reconstruction = self.deconv(z)
        return reconstruction
    
class AE(nn.Module):
    def __init__(self, input_channels, feature_sizes, latent_dim, input_size=16,
                encoder_activation='modrelu', decoder_activation='modrelu', 
                final_activation='', norm_type='none', dropout_rate=0.0):
        super().__init__()
        
        self.encoder = ComplexEncoder(
            input_channels,
            feature_sizes,
            latent_dim,
            input_size=input_size,
            activation=encoder_activation,
            norm_type=norm_type,
            dropout_rate=dropout_rate
        )
        
        self.decoder = ComplexDecoder(
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
