import torch
import torch.nn as nn
from typing import Optional
from torch.nn.common_types import _size_1_t


class MaxPool1d(nn.Module):
    r"""
    Applies a 1D max pooling on the magnitude of the input complex signal.
    
    Internally, it relies on torch.nn.MaxPool1d.

    Arguments:
        kernel_size: Size of the pooling window
        stride: Stride of the pooling window
        padding: Implicit zero padding to be added on both sides
        dilation: A parameter that controls the stride of elements in the window
        ceil_mode: When True, will use `ceil` instead of `floor` to compute the output shape
        return_indices: If True, will return the max indices along with the outputs
    """

    def __init__(
        self,
        kernel_size: _size_1_t,
        stride: Optional[_size_1_t] = None,
        padding: _size_1_t = 0,
        dilation: _size_1_t = 1,
        ceil_mode: bool = False,
        return_indices: bool = False,
    ) -> None:
        super().__init__()
        self.return_indices = return_indices
        # We set return_indices=True internally to get the indices of max values.
        # If the user sets return_indices=False, we won't return them but still need them to select values.
        self.m = nn.MaxPool1d(
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode=ceil_mode,
            return_indices=True,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z is complex: shape (N, C, L)
        # We apply max pooling on |z|
        abs_z = torch.abs(z)
        abs_out, indices = self.m(abs_z)  # abs_out: (N, C, L_out), indices: (N, C, L_out)

        # Using the indices, we gather the corresponding complex values
        # from z along the last dimension
        out_z = torch.gather(z, dim=-1, index=indices)  # (N, C, L_out)

        if self.return_indices:
            return out_z, indices
        else:
            return out_z


class AvgPool1d(nn.Module):
    r"""
    Applies a 1D average pooling to the real and imaginary parts of the input separately,
    and returns a complex tensor.

    Internally, it relies on torch.nn.AvgPool1d.

    Arguments:
        kernel_size: Size of the window to take an average over
        stride: Stride of the window
        padding: Implicit zero padding on both sides
        ceil_mode: When True, will use `ceil` instead of `floor` to compute the output shape
        count_include_pad: Whether to include the zero-padding in the averaging calculation
        divisor_override: If specified, it will be used as divisor, otherwise the size of the pooling region is used
    """

    def __init__(
        self,
        kernel_size: _size_1_t,
        stride: Optional[_size_1_t] = None,
        padding: _size_1_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.m = nn.AvgPool1d(
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Split real and imaginary parts
        real_out = self.m(z.real)
        imag_out = self.m(z.imag)
        # Recombine into complex
        return torch.complex(real_out, imag_out)
