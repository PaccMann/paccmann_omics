import torch.nn as nn
import torch.nn.functional as F
from ..utils.hyperparams import ACTIVATION_FN_FACTORY
from .decoder import Decoder


class DenseDecoder(Decoder):
    """Stacking dense layers for decoding."""

    def __init__(self, params, *args, **kwargs):
        """Constructor.

        Args:
            params (dict): A dictionary containing the parameter to built the
                dense Decoder.
                TODO params should become actual arguments (use **params).

        Items in params:
            hidden_sizes_decoder (list): Number and sizes of hidden layers.
            latent_size (int): Size of latent mean and variance.
            input_size (int): Size of output to match input space.
            activation_fn (string, optional): Activation function used in all
                layers for specification in ACTIVATION_FN_FACTORY.
                Default 'relu'.
            batch_norm (bool, optional): Whether batch normalization is
                applied. Default True.
            dropout (float, optional): Dropout probability in all
                except parametric layer. Default 0.0.
            *args, **kwargs: positional and keyword arguments are ignored.
        """
        super(DenseDecoder, self).__init__()

        self.hidden_sizes = params['hidden_sizes_decoder']
        self.latent_size = params['latent_size']
        self.output_size = params['input_size']
        self.activation_fn = ACTIVATION_FN_FACTORY[
            params.get('activation_fn', 'relu')]
        self.dropout = (
            [params.get('dropout', 0.0)] * len(self.hidden_sizes)
            if isinstance(params.get('dropout', 0.0), float) else
            params.get('dropout', 0.0)
        )
        self.params = params
        self._assertion_tests()

        # Stack dense layers
        num_units = [self.latent_size] + self.hidden_sizes + [self.output_size]
        ops = []
        for index in range(1, len(num_units) - 1):
            ops.append(nn.Linear(num_units[index - 1], num_units[index]))
            if params.get('batch_norm', True):
                ops.append(nn.BatchNorm1d(num_units[index]))
            ops.append(self.activation_fn)
            if self.dropout[index - 1] > 0.0:
                ops.append(nn.Dropout(p=self.dropout[index - 1]))
        # Last layer does not use dropout and
        # may require sigmoidal if data is normalized.
        ops.append(nn.Linear(num_units[-2], num_units[-1]))
        if self.params.get('input_normalized', False):
            ops.append(F.sigmoid)

        self.decoding = nn.Sequential(*ops)

    def forward(self, latent_z):
        """Decodes a latent space representation to an input-spaced rep.

        Args:
            latent_z (torch.Tensor): the sampled latent representation used
                for decoding of shape `[batch_size,latent_size]`.

        Returns:
            torch.Tensor: the reconstructed input obtained through the decoder.
        """

        return self.decoding(latent_z)

    def _assertion_tests(self):
        """Checks size issues.

        Hidden sizes should be monotonic data expansions.
        """

        assert self.hidden_sizes == sorted(
            self.hidden_sizes
        ), "Hidden sizes not monotonic."

        assert self.latent_size < self.hidden_sizes[
            0], "First hidden size no expansion."

        assert len(self.dropout) == len(
            self.hidden_sizes
        ), "Unequal dropout/hidden lengths."
