"""Custom layers implementation."""
from collections import OrderedDict

import torch.nn as nn

from .utils import Squeeze, get_device

DEVICE = get_device()


def collapsing_layer(inputs, act_fn=nn.ReLU()):
    """Helper layer that collapses the last dimension of inputs.

    E.g. if cell line data includes CNV, inputs can be `[bs, 2048, 5]`
        and the outputs would be `[bs, 2048]`.

    Args:
        inputs (torch.Tensor): of shape
            `[batch_size, *feature_sizes, hidden_size]`
        act_fn (callable): Nonlinearity to be used for collapsing.

    Returns:
        torch.Tensor: Collapsed input of shape `[batch_size, *feature_sizes]`
    """
    collapse = nn.Sequential(
        OrderedDict(
            [
                ('dense', nn.Linear(inputs.shape[-1], 1)), ('act_fn', act_fn),
                ('squeeze', Squeeze())
            ]
        )
    ).to(DEVICE)
    return collapse(inputs)


def apply_dense_attention_layer(inputs, return_alphas=False):
    """Attention mechanism layer for dense inputs.

    Args:
        inputs (torch.Tensor): Data input either of shape
            `[batch_size, feature_size]` or
            `[batch_size, feature_size, hidden_size]`.
        return_alphas (bool): Whether to return attention coefficients variable
            along with layer's output. Used for visualization purpose.
    Returns:
        torch.Tensor or tuple(torch.Tensor, torch.Tensor):
            The tuple (outputs, alphas) if `return_alphas`
            else -by default- only outputs.
            Outputs are of shape `[batch_size, feature_size]`.
            Alphas are of shape `[batch_size, feature_size]`.
    """

    # If inputs have a hidden dimension, collapse them into a scalar
    inputs = collapsing_layer(inputs) if len(inputs.shape) == 3 else inputs
    assert len(inputs.shape) == 2

    attention_layer = nn.Sequential(
        OrderedDict(
            [
                ('dense', nn.Linear(inputs.shape[1], inputs.shape[1])),
                ('softmax', nn.Softmax())
            ]
        )
    ).to(DEVICE)
    alphas = attention_layer(inputs)
    output = alphas * inputs

    return (output, alphas) if return_alphas else output
