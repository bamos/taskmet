import os
import random
import jax
import numpy as np
import tqdm
from absl import app, flags
from omegaconf import OmegaConf

from jax_rl.agents import OMDLearner, MetricLearner
from jax_rl.datasets import ReplayBuffer
from jax_rl.evaluation import evaluate
from jax_rl.utils import make_env
from jax_rl.logger import Logger
from ml_collections import config_flags

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'HalfCheetah-v2', 'Environment name.')
flags.DEFINE_string('save_dir', './exp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(1e4),
                     'Number of training steps to start training.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('no_jit', False, 'Disable JIT (will make code slower).')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')

# config_flags.DEFINE_config_file(
#     'config',
#     'configs/omd_default.py',
#     'File path to the training hyperparameter configuration.',
#     lock_config=False)

config_flags.DEFINE_config_file(
    'config',
    'configs/taskmet_default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)

def main(_):
    if FLAGS.save_dir == 'exp':
        FLAGS.save_dir = os.path.join(FLAGS.save_dir, FLAGS.config.exp, FLAGS.config.algo, str(FLAGS.config.seed))
    
    jax.config.update('jax_disable_jit', FLAGS.no_jit) 
    cfg = OmegaConf.create(dict(FLAGS.config))
    logger = Logger(FLAGS.save_dir, cfg)
    # summary_writer = SummaryWriter(
    #     os.path.join(FLAGS.save_dir, 'tb', str(FLAGS.config.seed)))

    if FLAGS.save_video:
        video_train_folder = os.path.join(FLAGS.save_dir, 'video', 'train')
        video_eval_folder = os.path.join(FLAGS.save_dir, 'video', 'eval')
    else:
        video_train_folder = None
        video_eval_folder = None

    env = make_env(FLAGS.env_name,
                   FLAGS.config.seed,
                   video_train_folder,
                   dim_distract=FLAGS.config.dim_distract)
    eval_env = make_env(FLAGS.env_name,
                        FLAGS.config.seed + 42,
                        video_eval_folder,
                        dim_distract=FLAGS.config.dim_distract)

    np.random.seed(FLAGS.config.seed)
    random.seed(FLAGS.config.seed)

    kwargs = dict(FLAGS.config)
    algo = kwargs.pop('algo')
    replay_buffer_size = kwargs.pop('replay_buffer_size')
    outer_steps = kwargs.pop('outer_steps')

    if algo == 'omd' or algo == 'mle':
        # MLE is implemented within
        agent = OMDLearner(FLAGS.config.seed,
                           env.observation_space.sample()[np.newaxis],
                           env.action_space.sample()[np.newaxis],
                           **kwargs,
                           algo=algo)
    elif algo == 'metric':
        agent = MetricLearner(env.observation_space.sample()[np.newaxis],
                                env.action_space.sample()[np.newaxis],
                                **kwargs,
                                algo="taskmet")
    else:
        raise NotImplementedError()

    action_dim = env.action_space.shape[0]
    replay_buffer = ReplayBuffer(env.observation_space, action_dim,
                                 replay_buffer_size or FLAGS.max_steps)

    eval_returns = []
    observation, done = env.reset(), False
    episode_return = 0.0
    train_info = {}
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        train_info.update({"step": i})
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
        
        next_observation, reward, done, info = env.step(action)
        episode_return += reward
        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(observation, action, reward, mask,
                             next_observation)
        observation = next_observation

        if done:
            observation, done = env.reset(), False
            train_info.update({"episode_reward": episode_return})
            episode_return = 0.0
            # for k, v in info['episode'].items():
                # summary_writer.add_scalar(f'training/{k}', v,
                #                           info['total']['timesteps'])

        if i >= FLAGS.start_training:
            for _ in range(outer_steps):
                batch = replay_buffer.sample(FLAGS.batch_size)
                update_info = agent.update(batch)

            if i % FLAGS.log_interval == 0:
                # logger.log(update_info, "train")
                pass
                # for k, v in update_info.items():
                #     summary_writer.add_scalar(f'training/{k}', v, i)
                # summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, eval_env, FLAGS.eval_episodes)
            eval_stats.update({"step": i})
            # logger.log(eval_stats, "eval")


            eval_returns.append(
                (info['total']['timesteps'], eval_stats['return']))
            np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.config.seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])


if __name__ == '__main__':
    app.run(main)
