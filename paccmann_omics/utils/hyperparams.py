"""Factories to ingest string model parameters."""
import torch
import torch.optim as optim
import torch.nn as nn
from .utils import gaussian_mixture, kl_divergence_loss

# LSTM(10, 20, 2) -> input has 10 features, 20 hidden size and 2 layers.
# NOTE: Make sure to set batch_first=True. Optionally set bidirectional=True
RNN_CELL_FACTORY = {'lstm': nn.LSTM, 'gru': nn.GRU}

OPTIMIZER_FACTORY = {
    'Adadelta': optim.Adadelta,
    'Adagrad': optim.Adagrad,
    'Adam': optim.Adam,
    'Adamax': optim.Adamax,
    'RMSprop': optim.RMSprop,
    'SGD': optim.SGD
}

ACTIVATION_FN_FACTORY = {
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'tanh': nn.Tanh(),
    'lrelu': nn.LeakyReLU(),
    'elu': nn.ELU(),
    'celu': nn.CELU()
}
LOSS_FN_FACTORY = {
    'mse': nn.MSELoss(reduction='sum'),
    'l1': nn.L1Loss(),
    'binary_cross_entropy': nn.BCELoss(),
    'kld': kl_divergence_loss
}

AAE_DISTRIBUTION_FACTORY = {
    'Gaussian': torch.randn,
    'Uniform': torch.rand,
    'Gaussian_Mixture': gaussian_mixture
}
