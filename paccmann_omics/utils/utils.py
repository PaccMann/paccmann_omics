import random
from math import cos, sin
from time import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions.normal import Normal


def gaussian_mixture(batchsize, ndim, num_labels=8):
    """Generate gaussian mixture data.

    Reference: Makhzani, A., et al. "Adversarial autoencoders." (2015).

    Args:
        batchsize (int)
        ndim (int): Dimensionality of latent space/each Gaussian.
        num_labels (int, optional): Number of mixed Gaussians. Defaults to 8.

    Raises:
        Exception: ndim is not a multiple of 2.

    Returns:
        torch.Tensor: samples
    """
    if ndim % 2 != 0:
        raise Exception("ndim must be a multiple of 2.")

    def sample(x, y, label, num_labels):
        shift = 1.4
        r = 2.0 * np.pi / float(num_labels) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2, ))

    x_var = 0.5
    y_var = 0.05
    x = np.random.normal(0, x_var, (batchsize, ndim // 2))
    y = np.random.normal(0, y_var, (batchsize, ndim // 2))
    z = np.empty((batchsize, ndim), dtype=np.float32)
    for batch in range(batchsize):
        for zi in range(ndim // 2):
            z[batch, zi * 2:zi * 2 + 2] = sample(
                x[batch, zi], y[batch, zi], random.randint(0, num_labels - 1),
                num_labels
            )
    return torch.Tensor(z)


def pearsonr(x, y):
    """Compute Pearson correlation.

    Args:
        x (torch.Tensor): 1D vector
        y (torch.Tensor): 1D vector of the same size as y.

    Raises:
        TypeError: not torch.Tensors.
        ValueError: not same shape or at least length 2.

    Returns:
        Pearson correlation coefficient.
    """
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError('Function expects torch Tensors.')

    if len(x.shape) > 1 or len(y.shape) > 1:
        raise ValueError(' x and y must be 1D Tensors.')

    if len(x) != len(y):
        raise ValueError('x and y must have the same length.')

    if len(x) < 2:
        raise ValueError('x and y must have length at least 2.')

    # If an input is constant, the correlation coefficient is not defined.
    if bool((x == x[0]).all()) or bool((y == y[0]).all()):
        raise ValueError("Constant input, r is not defined.")

    mx = x - torch.mean(x)
    my = y - torch.mean(y)
    cost = (
        torch.sum(mx * my) /
        (torch.sqrt(torch.sum(mx**2)) * torch.sqrt(torch.sum(my**2)))
    )
    return torch.clamp(cost, min=-1.0, max=1.0)


def correlation_coefficient_loss(labels, predictions):
    """Compute loss based on Pearson correlation.

    Args:
        labels (torch.Tensor): reference values
        predictions (torch.Tensor): predicted values

    Returns:
        torch.Tensor: A loss that when minimized forces high squared correlation coefficient:
        \$1 - r(labels, predictions)^2\$  # noqa
    """
    return 1 - pearsonr(labels, predictions)**2


def mse_cc_loss(labels, predictions):
    """Compute loss based on MSE and Pearson correlation.

    The main assumption is that MSE lies in [0,1] range, i.e.: range is
    comparable with Pearson correlation-based loss.

    Args:
        labels (torch.Tensor): reference values
        predictions (torch.Tensor): predicted values

    Returns:
        torch.Tensor: A loss that computes the following:
        \$mse(labels, predictions) + 1 - r(labels, predictions)^2\$  # noqa
    """
    mse_loss_fn = nn.MSELoss()
    mse_loss = mse_loss_fn(predictions, labels)
    cc_loss = correlation_coefficient_loss(labels, predictions)
    return mse_loss + cc_loss


def kl_divergence_loss(mu, logvar):
    """KL Divergence loss from VAE paper.

    Reference:
        Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014

    Args:
        mu (torch.Tensor): Encoder output of means of shape
            `[batch_size, input_size]`.
        logvar (torch.Tensor): Encoder output of logvariances of shape
            `[batch_size, input_size]`.
    Returns:
        The KL Divergence of the thus specified distribution and a unit
        Gaussian.
    """
    # Increase precision (numerical underflow caused negative KLD).
    mu = mu.double().to(get_device())
    logvar = logvar.double().to(get_device())
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def to_np(x):
    return x.data.cpu().numpy()


def to_var(x):
    return Variable(x).cuda() if cuda() else Variable(x)


def cuda():
    return torch.cuda.is_available()


def get_device():
    return torch.device("cuda" if cuda() else "cpu")


def augment(x, dropout=0., sigma=0.):
    """Performs augmentation on the input data batch x.

    Args:
        x (torch.Tensor): Input of shape `[batch_size, input size]`.
        dropout (float, optional): Probability for each input value to be 0.
            Defaults to 0.
        sigma (float, optional): Variance of added gaussian noise to x
            (x' = x + N(0,sigma). Defaults to 0.

    Returns:
        torch.Tensor: Augmented data
    """
    f = nn.Dropout(p=dropout, inplace=True)
    return f(x).add_(Normal(0, sigma).sample(x.shape).to())


def attention_list_to_matrix(coding_tuple, dim=2):
    """[summary]

    Args:
        coding_tuple (list((torch.Tensor, torch.Tensor))): iterable of
            (outputs, att_weights) tuples coming from the attention function
        dim (int, optional): The dimension along which expansion takes place to
            concatenate the attention weights. Defaults to 2.

    Returns:
        (torch.Tensor, torch.Tensor): raw_coeff, coeff

        raw_coeff: with the attention weights of all multiheads and
            convolutional kernel sizes concatenated along the given dimension,
            by default the last dimension.
        coeff: where the dimension is collapsed by averaging.
    """
    raw_coeff = torch.cat(
        [torch.unsqueeze(tpl[1], 2) for tpl in coding_tuple], dim=dim
    )
    return raw_coeff, torch.mean(raw_coeff, dim=dim)


class Squeeze(nn.Module):
    """Squeeze wrapper for nn.Sequential."""

    def forward(self, data):
        return torch.squeeze(data)


class Unsqueeze(nn.Module):
    """Unsqueeze wrapper for nn.Sequential."""

    def __init__(self, axis):
        super(Unsqueeze, self).__init__()
        self.axis = axis

    def forward(self, data):
        return torch.unsqueeze(data, self.axis)


class VAETracker():
    """Class to track and log performance of a VAE."""

    def __init__(
        self, logger, params, train_loader, val_loader, latent_size, epochs
    ):

        self.logger = logger
        self.params = params
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.latent_size = latent_size
        self.epochs = epochs

        self.best_dict = {
            'min_kl': 1e4,
            'min_rec': 1e4,
            'min_sum': 1e5,
            'min_kl_rec': 1e4,
            'min_rec_kl': 1e4,
            'min_sum_rec': 1e4,
            'min_sum_kl': 1e4,
            'ep_kl': 0,
            'ep_rec': 0,
            'ep_sum': 0
        }

    def new_train_epoch(self, epoch):
        self.train_loss = 0
        self.train_rec = 0
        self.train_kl = 0
        self.time = time()
        self.epoch = epoch

    def new_val_epoch(self, epoch):
        self.val_loss = 0
        self.val_rec = 0
        self.val_kl = 0
        self.time = time()
        self.epoch = epoch

        return None

    def update_train_batch(self, loss, rec, kld):
        self.train_loss += loss.item()
        self.train_rec += rec.item()
        self.train_kl += kld.item()

    def update_val_batch(self, loss, rec, kld):
        self.val_loss += loss.item()
        self.val_rec += rec.item()
        self.val_kl += kld.item()

    def average_losses(self, loss, rec, kl, loader):
        """Maps losses to a meaningful range.

        Args:
            loss (float): Scalar loss summed over all samples of loader.
            rec (float): Scalar reconstruction error summed over all samples of
                loader AND all dimensions (loss uses reduction='sum').
            kl (float): Scalar KL divergence summed over all samples of loader
                AND all latent dim (loss uses torch.sum not torch.mean).
            loader (torch.utils.data.dataloader.DataLoader): A data loader of
                either train or test data.

        Returns:
            ((float, float, float)): transformed loss, rec, kl

            loss: Loss per sample.
            rec: Average reconstruction error per feature (e.g. gene).
            kl: Average KL divergence per latent dimension.
        """
        return (
            loss / (len(loader.dataset)),
            rec / (len(loader.dataset) * next(iter(loader))[0].shape[-1]),
            kl / (len(loader.dataset) * self.latent_size)
        )

    def logg_train_epoch(self):
        self.train_loss_a, self.train_rec_a, self.train_kl_a = (
            self.average_losses(
                self.train_loss, self.train_rec, self.train_kl,
                self.train_loader
            )
        )

        self.logger.info(
            "\t **** TRAINING ****   "
            "Epoch [{0}/{1}], loss: {2:.1f}, rec: {3:.6f}, KLD: {4:.6f} "
            "It took {5:.1f} secs.".format(
                self.epoch + 1, self.epochs, self.train_loss_a,
                self.train_rec_a, self.train_kl_a,
                time() - self.time
            )
        )

    def logg_val_epoch(self):
        self.val_loss_a, self.val_rec_a, self.val_kl_a = self.average_losses(
            self.val_loss, self.val_rec, self.val_kl, self.val_loader
        )
        self.logger.info(
            "\t **** VALIDATION **** "
            "Epoch [{0}/{1}], loss: {2:.1f}, rec: {3:.6f}, KLD: {4:.6f}. ".
            format(
                self.epoch + 1, self.epochs, self.val_loss_a, self.val_rec_a,
                self.val_kl_a
            )
        )

    def save(self, encoder, decoder, model, metric, typ, val=None):
        encoder.save(
            self.params['save_top_model'].format(typ, metric, 'encoder')
        )
        decoder.save(
            self.params['save_top_model'].format(typ, metric, 'decoder')
        )
        model.save(self.params['save_top_model'].format(typ, metric, 'vae'))
        if typ == 'best':
            self.logger.info(
                "\t New best performance in '{0}'"
                " with value : {1:.7f} in epoch: {2}".format(
                    metric, val, self.epoch
                )
            )

    def check_to_save(self, encoder, decoder, model):
        if self.val_rec_a < self.best_dict['min_rec']:
            self.best_dict['min_rec'] = self.val_rec_a
            self.best_dict['min_rec_kl'] = self.val_kl_a
            self.save(encoder, decoder, model, 'rec', 'best', self.val_rec_a)
            self.ep_rec = self.epoch
        if self.val_kl_a < self.best_dict['min_kl']:
            self.best_dict['min_kl'] = self.val_kl_a
            self.best_dict['min_kl_rec'] = self.val_rec_a
            self.save(encoder, decoder, model, 'kl', 'best', self.val_kl_a)
            self.ep_kl = self.epoch
        if self.val_rec_a + self.val_kl_a < self.best_dict['min_sum']:
            self.best_dict['min_sum'] = self.val_rec_a + self.val_kl_a
            self.best_dict['min_sum_rec'] = self.val_rec_a
            self.best_dict['min_sum_kl'] = self.val_kl_a
            self.save(
                encoder, decoder, model, 'both', 'best',
                self.best_dict['min_sum']
            )
            self.ep_sum = self.epoch
        if (self.epoch + 1) % self.params.get('save_model') == 0:
            self.save(encoder, decoder, model, 'epoch', str(self.epoch))

    def final_log(self):
        self.logger.info(
            " Overall best performances are: \n \t "
            "'Reconstruction error' = {0:.4f} in epoch {1} \t (KL was {2:4f})"
            "\n \t 'KL Divergence' = {3:.4f} in epoch {4} \t (rec was {5:2f})"
            "\n \t 'Summed error' = {6:.4f} in epoch {7} \t (KL was {8:4f}, "
            " rec was {9:4f}).".format(
                self.best_dict['min_rec'], self.best_dict['ep_rec'],
                self.best_dict['min_rec_kl'], self.best_dict['min_kl'],
                self.best_dict['ep_kl'], self.best_dict['min_kl_rec'],
                self.best_dict['min_sum'], self.best_dict['ep_sum'],
                self.best_dict['min_sum_kl'], self.best_dict['min_sum_rec']
            )
        )
