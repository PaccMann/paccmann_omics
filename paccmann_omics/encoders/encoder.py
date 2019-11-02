import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    This meta encoder defines loading, saving and related operations which
    should be shared across all encoders.
    Hence, all other encoders should inherit from this meta encoder.
    """

    def __init__(self):
        """ Constructor."""
        super(Encoder, self).__init__()

    def load(self, path, *args, **kwargs):
        """ Load model from path."""
        weights = torch.load(path, *args, **kwargs)
        self.load_state_dict(weights)

    def save(self, path, *args, **kwargs):
        """Save model to path."""
        torch.save(self.state_dict(), path, *args, **kwargs)
