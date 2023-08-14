import flax
from jax_rl.networks.common import PRNGKey, TrainState


@flax.struct.dataclass
class ModelActorCriticTemp:
    model: TrainState
    actor: TrainState
    critic: TrainState
    target_critic: TrainState
    temp: TrainState
    rng: PRNGKey


@flax.struct.dataclass
class MetricModelActorCriticTemp:
    metric: TrainState
    model: TrainState
    actor: TrainState
    critic: TrainState
    target_critic: TrainState
    temp: TrainState
    rng: PRNGKey
