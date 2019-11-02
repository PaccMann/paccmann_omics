import torch
import torch.nn as nn
from ..utils.hyperparams import ACTIVATION_FN_FACTORY


class DenseDiscriminator(nn.Module):

    def __init__(self, params):
        """
        This class specifies the discriminator of an AAE.
        It can be trained to distinguish real samples from a target distr.
        (e.g. Gaussian, Uniform, Gaussian Mixture ...) from fake samples
        constructed through the generator.

        Args:
            params (dict): A dict with the model parameters.

        """
        super(DenseDiscriminator, self).__init__()

        # Retrieve discriminator architecture
        self.disc_hidden_sizes = params['discriminator_hidden_sizes']
        self.input_size = params['input_size']
        self.disc_activation_fn = ACTIVATION_FN_FACTORY[
            params.get('discriminator_activation_fn', 'relu')]
        self.disc_dropout = (
            [params.get('discriminator_dropout', 0.0)] *
            len(self.hidden_sizes)
            if isinstance(params.get('discriminator_dropout', 0.0),
                          int) else params.get('discriminator_dropout', 0.0)
        )
        self._assertion_tests

        # Build discriminator
        num_units = [self.input_size] + self.disc_hidden_sizes
        ops = []
        for index in range(1, len(num_units)):
            ops.append(nn.Linear(num_units[index - 1], num_units[index]))
            ops.append(self.disc_activation_fn)
            if self.disc_dropout[index - 1] > 0.0:
                ops.append(nn.Dropout(p=self.disc_dropout[index - 1]))

        ops.append(nn.Linear(num_units[-1], 1))
        ops.append(nn.Sigmoid())
        self.discriminator = nn.Sequential(*ops)

    def forward(self, x):
        """The discriminator aiming to classify true and fake latent samples.

        Args:
            data (torch.Tensor) : Input data of shape batch_size x input_size.

        Returns:
            torch.Tensor: Logits, i.e. for each score a probability of being
                from the real target distribution p(z)
                of shape `[batch_size, 1]`.
        """
        return self.discriminator(x)

    def loss(self, real, fake):
        """
        The discriminator loss is fixed to be the binary cross entropy of the
        real and fake samples.

        Args:
            real (torch.Tensor): Discriminator logits for target distribution
                samples. Vector of length `batch_size`.
            fake (torch.Tensor): Discriminator logits for generator samples
                (ideally 0.0). Vector of length `batch_size`.

        Returns:
            torch.Tensor: binary_cross_entropy(real, fake)
        """
        return -torch.mean(torch.cat([torch.log(real), torch.log(1 - fake)]))

    def _assertion_tests(self):
        pass

    def load(self, path):
        """Load model from path."""
        weights = torch.load(path)
        self.load_state_dict(weights)

    def save(self, path):
        """Save model to path."""
        torch.save(self.state_dict(), path)
