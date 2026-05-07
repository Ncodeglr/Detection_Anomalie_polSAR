import torch
import torch.nn as nn
from torch.nn.common_types import _size_1_t


class ConvTranspose1d(nn.Module):
    """
    Implementation of torch.nn.ConvTranspose1d for complex numbers.
    Apply ConvTranspose1d on real and imaginary parts of the complex number.

    The parameters are the same as from the upstream pytorch layer:

    Arguments:
        in_channels (int): Number of channels in the input signal
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): ``dilation * (kernel_size - 1) - padding`` zero-padding will be added to both sides of each dimension in the input. Default: 0
        output_padding (int or tuple, optional): Additional size added to one side of each dimension in the output shape. Default: 0
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        padding_mode (str, optional): ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        output_padding: _size_1_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_1_t = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        self.m_real = torch.nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
            device,
            dtype,
        )

        self.m_imag = torch.nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
            device,
            dtype,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass, applying real valued ConvTranspose1d
        independently on both the real and imaginary parts of the input.
        """
        if not z.is_complex():
            raise ValueError("Input tensor must be a complex tensor.")

        real_out = self.m_real(z.real) - self.m_imag(z.imag)
        imag_out = self.m_real(z.imag) + self.m_imag(z.real)

        return torch.view_as_complex(torch.stack((real_out, imag_out), dim=-1))
