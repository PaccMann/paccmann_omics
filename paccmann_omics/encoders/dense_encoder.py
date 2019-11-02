import torch.nn as nn
from ..utils.hyperparams import ACTIVATION_FN_FACTORY
from .encoder import Encoder


class DenseEncoder(Encoder):
    """Stacking dense layers for encoding."""

    def __init__(self, params, *args, **kwargs):
        """Constructor.

        Args:
            params (dict): A dictionary containing the parameter to built the
                dense Decoder.
                TODO params should become actual arguments (use **params).

        Items in params:
            hidden_sizes_encoder (list): Number and sizes of hidden layers.
            latent_size (int): Size of latent mean and variance.
            input_size (int): Number of features in input data to match output.
            activation_fn (string, optional): Activation function used in all
                layers for specification in ACTIVATION_FN_FACTORY.
                Default 'relu'.
            batch_norm (bool, optional): Whether batch normalization is
                applied. Default True.
            dropout (float, optional): Dropout probability in all
                except parametric layer. Default 0.0.
            *args, **kwargs: positional and keyword arguments are ignored.
        """
        super(DenseEncoder, self).__init__(*args, **kwargs)

        self.input_size = params['input_size']
        self.hidden_sizes = params['hidden_sizes_encoder']
        self.latent_size = params['latent_size']
        self.activation_fn = ACTIVATION_FN_FACTORY[
            params.get('activation_fn', 'relu')]
        self.dropout = (
            [params.get('dropout', 0.0)] * len(self.hidden_sizes)
            if isinstance(params.get('dropout', 0.0), float) else
            params.get('dropout', 0.0)
        )
        self._assertion_tests()

        # Stack dense layers
        num_units = [self.input_size] + self.hidden_sizes
        ops = []
        for index in range(1, len(num_units)):
            ops.append(nn.Linear(num_units[index - 1], num_units[index]))
            if params.get('batch_norm', True):
                ops.append(nn.BatchNorm1d(num_units[index]))
            ops.append(self.activation_fn)
            if self.dropout[index - 1] > 0.0:
                ops.append(nn.Dropout(p=self.dropout[index - 1]))

        self.encoding = nn.Sequential(*ops)

        self.encoding_to_mu = nn.Linear(
            self.hidden_sizes[-1], self.latent_size
        )
        self.encoding_to_logvar = nn.Linear(
            self.hidden_sizes[-1], self.latent_size
        )

    def forward(self, data):
        """Projects an input into the latent space.

        Args:
            data (torch.Tensor): Matrix of shape
                `[batch_size, self.input_size]`.

        Returns:
            (torch.Tensor, torch.Tensor): mu, logvar

            mu (torch.Tensor): Latent means of shape
                `[batch_size, self.latent_size]`.
            logvar (torch.Tensor): Latent log-std of shape
                `[batch_size, self.latent_size]`.
        """

        projection = self.encoding(data)
        return (
            self.encoding_to_mu(projection),
            self.encoding_to_logvar(projection)
        )

    def _assertion_tests(self):
        """Checks size issues.

        Hidden sizes should be monotonic data compressions.
        """
        assert self.hidden_sizes == sorted(
            self.hidden_sizes, reverse=True
        ), "Hidden sizes not monotonic."
        assert self.latent_size < self.hidden_sizes[
            -1], "Latent size not smaller than last hidden size."
        assert self.input_size >= self.hidden_sizes[
            0], "No compression in first hidden layer."
        assert len(self.dropout) == len(
            self.hidden_sizes
        ), "Dropout lengths does not match hidden sizes."
