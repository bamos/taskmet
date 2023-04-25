import os
import sys
import yaml
import gym
import numpy as np
import time
import pickle
from absl import app
from absl import flags
import hydra
from omegaconf import OmegaConf
import re
from pathlib import Path

import jax
from jax.config import config
config.update("jax_enable_x64", True)

from logger import Logger
from omd import Agent
from replay_buffer import ReplayBuffer
from utils import *

FLAGS = flags.FLAGS
flags.DEFINE_string('exp', 'default', 'Custom description string added to out_dir')
flags.DEFINE_string('out_dir', 'exp', 'Directory for output files')
flags.DEFINE_string('data_path', 'data/buf.pkl', 'Path to save buffer')
flags.DEFINE_string('agent_path', 'data/agent.pkl', 'Path to save agent')
flags.DEFINE_string('env_name', 'CartPole-v1', 'Gym environment id')
flags.DEFINE_integer('seed', 0, 'Random seed')
flags.DEFINE_integer('num_train_steps', 200000, 'Env steps num', lower_bound=0)
flags.DEFINE_integer('hidden_dim', 32, 'Size of hidden layers', lower_bound=1)
flags.DEFINE_integer('model_hidden_dim', 32, 'Model-specific', lower_bound=1)
flags.DEFINE_integer('batch_size', 256, 'Mini-batch samples', lower_bound=1)
flags.DEFINE_integer('init_steps', 1000, 'Steps before training', lower_bound=0)
flags.DEFINE_integer('eval_frequency', 1000, 'Agent evaluation', lower_bound=1)
flags.DEFINE_integer('log_frequency', 1000, 'Logging frequency', lower_bound=1)
flags.DEFINE_integer('num_Q_steps', 1, 'Inner loop steps num', lower_bound=1)
flags.DEFINE_integer('num_T_steps', 1, 'Model steps per update', lower_bound=1)
flags.DEFINE_integer('num_ensemble_vep', 5, 'Value functions #', lower_bound=1)
flags.DEFINE_float('eps', 0.1, 'Random action probability')
flags.DEFINE_float('discount', 0.99, 'Sum of rewards discount factor')
flags.DEFINE_float('lr', 1e-3, '(Outer loop) learning rate')
flags.DEFINE_float('inner_lr', 3e-4, 'Inner loop learning rate')
flags.DEFINE_float('metric_lr', 1e-3, 'Model learning rate')
flags.DEFINE_float('alpha', 0.01, 'Temperature')
flags.DEFINE_float('tau', 0.01, 'Target network update coefficient')
flags.DEFINE_boolean('save_buf', False, 'Save collected buffer with data')
flags.DEFINE_boolean('save_agent', False, 'Save the agent after training')
flags.DEFINE_boolean('hard', False, 'max vs logsumexp for Q learning')
flags.DEFINE_boolean('no_learn_reward', False, 'Use trainable or true rewards')
flags.DEFINE_boolean('prob_model', False, 'Gaussian vs deterministic next obs')
flags.DEFINE_boolean('no_warm', False, 'Not use previous Q* in the inner loop')
flags.DEFINE_boolean('warm_opt', False, 'Reuse inner loop optimizer statistics')
flags.DEFINE_boolean('no_double', False, 'Not use Double Q Learning')
flags.DEFINE_boolean('with_inv_jac', False, 
  'Replaces inverse Jacobian in implicit gradient with identity matrix')
flags.DEFINE_enum('agent_type', 'omd', ['omd', 'mle', 'vep', 'metric'], 'Agent type')
flags.DEFINE_integer('dim_distract', 0, 'Number of distracting states.')
flags.DEFINE_string('use_wandb', 'False', 'Use wandb for logging')
flags.DEFINE_string('wandb_project', 'none', 'Wandb project name')
flags.DEFINE_string('wandb_entity', 'none', 'Wandb entity name')


def main_function(cfg):
  wrapper(cfg)
  env = gym.make(FLAGS.env_name)
  max_episode_steps = env._max_episode_steps
  eval_env = gym.make(FLAGS.env_name)
  if FLAGS.dim_distract > 0:
    print("Adding distractors")
    env  = AddDistractors(env, dim_distract=FLAGS.dim_distract)
    eval_env  = AddDistractors(eval_env, dim_distract=FLAGS.dim_distract)

  for e in [eval_env, env]:
    e.seed(FLAGS.seed)
    e.action_space.seed(FLAGS.seed)
    e.observation_space.seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  rngs = hk.PRNGSequence(FLAGS.seed)
  
  agent = Agent(env.observation_space, env.action_space)
  replay_buffer = ReplayBuffer(env.observation_space.shape, FLAGS.num_train_steps)

  # FLAGS.out_dir = os.path.join(FLAGS.out_dir, time.strftime("%d%H%M"))
  FLAGS.out_dir = os.path.join(FLAGS.out_dir, FLAGS.exp, FLAGS.agent_type, str(FLAGS.seed))
  os.makedirs(FLAGS.out_dir, exist_ok=True)
  logger = Logger(Path(FLAGS.out_dir), cfg)
  
  step, episode = 0, 0
  episode_return, episode_step = 0, 0
  done = False
  obs = env.reset()
  action = agent.act(agent.params_Q, obs, next(rngs))
  start_time = time.time()
  common_metrics = {
    "episode": episode,
    "step": step,
    "total_time": time.time() - start_time,
  }
  print("Beginning of training")
  while step < FLAGS.num_train_steps:
    common_metrics.update({
      "episode": episode,
      "step": step,
      "total_time": time.time() - start_time,
    })
    # evaluate agent periodically
    if step % FLAGS.eval_frequency == 0:
      common_metrics["episode_reward"] = evaluate(agent, eval_env, next(rngs))
      logger.log(common_metrics, "eval")
      # print(agent.params_metric)

    # with epsilon exploration
    action = env.action_space.sample() if (np.random.rand() < FLAGS.eps or step < FLAGS.init_steps) else action.item()
    next_obs, reward, done, _ = env.step(action)

    done = float(done)
    # allow infinite bootstrap
    done_no_max = 0 if episode_step + 1 == max_episode_steps else done
    episode_return += reward

    replay_buffer.add(obs, action, reward, next_obs, done, done_no_max)
    
    obs = next_obs
    episode_step += 1
    step += 1
    
    if done:
      common_metrics["episode_reward"] = episode_return
      obs = env.reset()
      done = False
      episode_return = 0
      episode_step = 0
      episode += 1

    action = agent.act(agent.params_Q, obs, next(rngs))
        
    losses_dict = {}
    if step >= FLAGS.init_steps:
      losses_dict = agent.update(replay_buffer)
      common_metrics["step"] = step
      losses_dict.update(common_metrics)

    if step % FLAGS.log_frequency == 0:
      logger.log(losses_dict, "train")
  
  # final eval after training is done
  common_metrics["episode_reward"] = evaluate(agent, eval_env, next(rngs))
  logger.log(common_metrics, "eval")


  print("Done in {:.1f} minutes".format((time.time() - start_time)/60))

def save_config(FLAGS):
  config = FLAGS.flag_values_dict()
  # save in a yaml file
  with open(os.path.join(FLAGS.out_dir, 'config.yaml'), 'w') as f:
    yaml.dump(config, f, default_flow_style=False)


def wrapper(cfg):
  FLAGS = flags.FLAGS
  FLAGS(sys.argv)
  for k, v in cfg.items():
    if k in FLAGS:
      FLAGS[k].value = v
  return 


if __name__ == '__main__':
  def parse_cfg(cfg_path: str) -> OmegaConf:
    """Parses a config file and returns an OmegaConf object."""
    base = OmegaConf.load(cfg_path / 'default.yaml')
    cli = OmegaConf.from_cli()
    for k,v in cli.items():
      if v == None:
        cli[k] = True
    base.merge_with(cli)
    return base
  
  main_function(parse_cfg(Path().cwd() / "config"))
  # app.run(main_function)
