import torch
import torch.nn as nn
from ..utils.hyperparams import LOSS_FN_FACTORY
from ..utils.utils import get_device


class VAE(nn.Module):
    """Variational Auto-Encoder (VAE)"""

    def __init__(self, params, encoder, decoder):
        """
        This class specifies a Variational Auto-Encoder (VAE) which
            can be instantiated with different encoder or decoder
            objects.

        Args:
            params (dict): A dict with the model parameters (<dict>).
            encoder (Encoder): An encoder object.
            decoder (Decoder): A decoder object.

         NOTE: This VAE class assumes that:
            1) The latent space should follow a multivariate unit Gaussian.
            2) The encoder strives to learn mean and log-variance of the
                latent space.
        """
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._assertion_tests()
        self.reconstruction_loss = LOSS_FN_FACTORY[
            params.get('reconstruction_loss', 'mse')]
        self.kld_loss = LOSS_FN_FACTORY[params.get('kld_loss', 'kld')]

    def encode(self, data):
        """ VAE encoding
        Args:
            data (torch.Tensor): The input of shape `[batch_size, input_size]`.

        Returns:
            (torch.Tensor, torch.Tensor): mu, logvar

            The latent means mu of shape `[bs, latent_size]`.
            Latent log variances logvar of shape `[bs, latent_size]`.
        """
        return self.encoder(data)

    def reparameterize(self, mu, logvar):
        """Applies reparametrization trick to obtain sample from latent space.

        Args:
            mu (torch.Tensor): The latent means of shape `[bs, latent_size]`.
            logvar (torch.Tensor) : Latent log variances ofshape
                `[bs, latent_size]`.
        Returns:
            torch.Tensor: Sampled Z from the latent distribution.
        """
        return torch.randn_like(mu).mul_(torch.exp(0.5 * logvar)).add_(mu)

    def decode(self, latent_z):
        """ VAE Decoding

        Args:
            latent_z (torch.Tensor): Sampled Z from the latent distribution.

        Returns:
            torch.Tensor: A (realistic) sample decoded from the latent
                representation of length input_size.
        """
        return self.decoder(latent_z)

    def forward(self, data):
        """ The Forward Function passing data through the entire VAE.

        Args:
            data (torch.Tensor): Input data of shape
                `[batch_size, input_size]`.
        Returns:
            (torch.Tensor): A (realistic) sample decoded from the latent
                representation of length  input_size]`. Ideally data == sample.
        """
        self.mu, self.logvar = self.encoder(data)
        latent_z = self.reparameterize(self.mu, self.logvar)
        sample = self.decoder(latent_z)
        return sample

    def joint_loss(self, outputs, targets, alpha=0.5, beta=1.):
        """Loss Function from VAE paper.

        Reference:
            Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014

        Args:
            outputs (torch.Tensor): The decoder output of shape
                `[batch_size, input_size]`.
            targets (torch.Tensor): The encoder input of shape
                `[batch_size, input_size]`.
            alpha (float): Weighting of the 2 losses. Alpha in range [0, 1].
                Defaults to 0.5.
            beta (float): Scaling of the KLD in range [1., 100.] according to
                beta-VAE paper. Defaults to 1.0.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): joint_loss, rec_loss, kld_loss  # noqa

            The VAE joint loss is a weighted combination of the
            reconstruction loss (e.g. L1, MSE) and the KL divergence of a
            multivariate unit Gaussian and the latent space representation.

            Reconstruciton loss is summed across input size and KL-Div is
            averaged across latent space.

            This comes from the fact that L2 norm is feature normalized
            and KL is z-dim normalized, s.t. alpha can be tuned for
            varying X, Z dimensions.
        """
        rec_loss = self.reconstruction_loss(outputs, targets)
        rec_loss = rec_loss.double().to(get_device())
        kld_loss = self.kld_loss(self.mu, self.logvar)
        joint_loss = alpha * rec_loss + (1 - alpha) * beta * kld_loss
        return joint_loss, rec_loss, kld_loss

    def _assertion_tests(self):
        pass

    def load(self, path, *args, **kwargs):
        """Load model from path."""
        weights = torch.load(path, *args, **kwargs)
        self.load_state_dict(weights)

    def save(self, path, *args, **kwargs):
        """Save model to path."""
        torch.save(self.state_dict(), path, *args, **kwargs)
