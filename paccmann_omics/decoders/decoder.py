import torch.nn as nn
import torch


class Decoder(nn.Module):
    """
    This meta decoder defines loading, saving and related operations which
    should be shared across all decoder.
    Hence, all other decoder should inherit from this meta decoder.
    """

    def __init__(self):
        """Constructor."""
        super(Decoder, self).__init__()

    def load(self, path, *args, **kwargs):
        """Load model from path."""
        weights = torch.load(path, *args, **kwargs)
        self.load_state_dict(weights)

    def save(self, path, *args, **kwargs):
        """Save model to path."""
        torch.save(self.state_dict(), path, *args, **kwargs)
