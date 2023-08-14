import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.algo = 'metric'
    config.exp = ''
    config.seed = 42
    config.dim_distract = 0

    config.model_lr = 3e-4
    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4
    config.metric_lr = 5e-4

    config.hidden_dims = (256, 256)
    config.model_hidden_dim = 64  # for the misspecification experiments

    config.discount = 0.99

    config.outer_steps = 1
    config.inner_steps = 1
    config.dynamic_train_steps = 2

    config.cg_iters = 50

    config.tau = 0.005
    config.target_update_period = 1

    config.init_temperature = 1.0
    config.target_entropy = None

    config.replay_buffer_size = None

    config.use_wandb =  True
    config.wandb_project=  'metric-learning'
    config.wandb_entity=  'theshank'
    config.save_video=  False
    config.save_agent= False

    return config
