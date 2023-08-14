from typing import Sequence, Tuple

import jax
import jax.numpy as jnp

import distrax
import flax.linen as nn
from jax_rl.networks.common import MLP, MetricMLP


class DetModel(nn.Module):
    hidden_dims: Sequence[int]
    obs_dim: int

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        inputs = jnp.concatenate([observations, actions], -1)
        outputs = MLP((*self.hidden_dims, self.obs_dim + 1))(inputs)
        return outputs[:, :-1], outputs[:, -1]

class MetricModel(nn.Module):
    hidden_dims: Sequence[int]
    out_dim: int
    conditional: bool = False

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        inputs = jnp.concatenate([observations, actions], -1)
        if not self.conditional:
            inputs = jnp.ones_like(inputs)
        outputs = MLP((*self.hidden_dims, self.out_dim))(inputs)
        diag_vmap = jax.vmap(jnp.diag, 0)
        outputs = diag_vmap(outputs)        
        return outputs
