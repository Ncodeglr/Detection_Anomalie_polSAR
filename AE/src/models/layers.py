import torch
from torch import nn
import torch.nn.functional as F



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 activation='relu', conv_type='conv', upsample=False, norm_type='batch'):
        super(ConvBlock, self).__init__()
        
        # Sélection du type de convolution
        # Utilisation de convolutions 1D
        if conv_type == 'conv':
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        elif conv_type == 'deconv':
            if upsample:
                self.conv = nn.Sequential(
                    nn.Upsample(mode='linear', scale_factor=2), # mode='linear' pour 1D
                    nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding)
                )
            else:
                self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding=stride-1)
        else:
            raise ValueError("conv_type must be 'conv' or 'deconv'")
        
        # Sélection du type de normalisation
        if norm_type == 'batch':
            self.bn = nn.BatchNorm1d(out_channels)
        elif norm_type == 'instance':
            self.bn = nn.InstanceNorm1d(out_channels)
        elif norm_type == 'layer':
            self.bn = nn.GroupNorm(1, out_channels)  # Layer norm en 1D
        elif norm_type == 'none':
            self.bn = nn.Identity()
        else:
            raise ValueError(f"Type de normalisation non reconnu: {norm_type}")
        
        # Sélection de la fonction d'activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == '':
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Fonction d'activation non reconnue: {activation}")
        
        self.upsample = upsample
    
    def forward(self, x):
        # Option d'upsampling dans le forward pour plus de flexibilité
        if self.upsample and not isinstance(self.conv, nn.Sequential):
            x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=False)
        
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        
        return x


# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation='relu', conv_type='conv', upsample=False):
#         super(ConvBlock, self).__init__()

#         # Utilisation de convolutions 1D
#         if conv_type == 'conv':
#             self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
#         elif conv_type == 'deconv':
#             if upsample:
#                 self.conv = nn.Sequential(
#                     nn.Upsample(mode='linear', scale_factor=2), # mode='linear' pour 1D
#                     nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding)
#                 )
#             else:
#                 self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding=stride-1)
#         else:
#             raise ValueError("conv_type must be 'conv' or 'deconv'")

#         self.bn = nn.BatchNorm1d(out_channels)  # BatchNorm1d pour les données 1D

#         if activation == 'relu':
#             self.activation = nn.ReLU()
#         elif activation == 'leakyrelu':
#             self.activation = nn.LeakyReLU(0.2)
#         elif activation == '':
#             self.activation = nn.Identity()
#         else:
#             raise ValueError("Unsupported activation function")
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.activation(x)
#         return x
