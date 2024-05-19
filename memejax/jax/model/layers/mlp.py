import flax.linen as nn
from jax import Array

from memejax.jax.model.blocks.cfg import ActivationKind, ModelCfg


class MlpLayer(nn.Module):
    features: int
    dropout: float
    layer_norm: bool
    activation: ActivationKind

    @staticmethod
    def from_cfg(cfg: ModelCfg, features: int) -> "MlpLayer":
        return MlpLayer(
            features=features,
            dropout=cfg.dropout,
            layer_norm=cfg.layer_norm,
            activation=cfg.activation,
        )

    @nn.compact
    def __call__(self, x: Array, train: bool) -> Array:
        x = nn.Dense(features=self.features)(x)
        if self.layer_norm:
            x = nn.LayerNorm()(x)
        activation = self.activation.get_func()
        x = activation(x)
        x = nn.Dropout(rate=self.dropout, deterministic=not train)(x)
        return x
