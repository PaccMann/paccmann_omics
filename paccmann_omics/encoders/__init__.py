from .dense_encoder import DenseEncoder
from .attention_encoder import AttentionEncoder

ENCODER_FACTORY = {'dense': DenseEncoder, 'attention': AttentionEncoder}
