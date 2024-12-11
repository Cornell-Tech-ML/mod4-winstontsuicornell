from typing import Tuple
from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling.

    Args:
    ----
        input: Tensor of shape (batch, channel, height, width)
        kernel: Tuple (kernel_height, kernel_width)

    Returns:
    -------
        Tuple containing:
        - Tensor reshaped for pooling
        - new_height
        - new_width

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0, "Height must be divisible by kernel height"
    assert width % kw == 0, "Width must be divisible by kernel width"

    new_height, new_width = height // kh, width // kw

    # Reshape and permute for pooling
    reshaped = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)
    tiled = (
        reshaped.permute(0, 1, 2, 4, 3, 5)
        .contiguous()
        .view(batch, channel, new_height, new_width, kh * kw)
    )

    return tiled, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Perform average pooling on a 2D input tensor.

    Args:
    ----
        input: Tensor of shape (batch, channel, height, width)
        kernel: Tuple (kernel_height, kernel_width)

    Returns:
    -------
        Tensor of shape (batch, channel, new_height, new_width)

    """
    tiled, new_height, new_width = tile(input, kernel)
    return tiled.mean(dim=-1).view(
        input.shape[0], input.shape[1], new_height, new_width
    )


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor."""
    max_vals = max_reduce(input, dim)
    return input == max_vals


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for max."""
        ctx.save_for_backward(input, dim)
        return max_reduce(input, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for max."""
        input, dim = ctx.saved_values
        return grad_output * argmax(input, int(dim.item())), 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction along a specified dimension.

    Args:
    ----
        input (Tensor): The input tensor.
        dim (Optional[int]): The dimension to compute the max over. If None, computes the max over all elements.

    Returns:
    -------
        Tensor: The maximum values along the specified dimension.

    """
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax over the specified dimension."""
    max_vals = max_reduce(input, dim)
    shifted_input = input - max_vals  # Numerical stability
    exp_vals = shifted_input.exp()
    return exp_vals / exp_vals.sum(dim=dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax over the specified dimension."""
    max_vals = max_reduce(input, dim)
    shifted_input = input - max_vals  # Numerical stability
    log_sum_exp = shifted_input.exp().sum(dim=dim).log()
    return shifted_input - log_sum_exp


def maxpool2d(inputtensor: Tensor, pool_size: Tuple[int, int]) -> Tensor:
    """Perform max pooling on a 2D input tensor."""
    # Unpack the input tensor dimensions
    batch_size, num_channels, img_height, img_width = inputtensor.shape

    # Tile the input tensor according to the pool size
    tiled_input, pooled_height, pooled_width = tile(inputtensor, pool_size)

    # Apply the max pooling operation
    tiled_input = max(tiled_input, 4)

    # Reshape to the expected output dimensions
    return tiled_input.view(batch_size, num_channels, pooled_height, pooled_width)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Apply dropout to a tensor."""
    if ignore:
        return input
    else:
        return input * (rand(input.shape) > rate)

    # if ignore:
    #     return input
    # mask = (rand(input.shape) > rate).float()
    # return input * mask / (1.0 - rate)
