import os
import sys
import yaml
import gym
import numpy as np
import time
import cloudpickle as pkl
from absl import app
from absl import flags
from pathlib import Path

import jax
from jax.config import config
config.update("jax_enable_x64", True)

from logger import Logger
from omd import Agent
from replay_buffer import ReplayBuffer
from utils import *

class Workspace(object):
  def __init__(self, cfg):
    self.cfg = cfg
    wrapper(cfg)
    self.env = gym.make(FLAGS.env_name)
    self.max_episode_steps = self.env._max_episode_steps
    self.eval_env = gym.make(FLAGS.env_name)
    if FLAGS.dim_distract > 0:
      print("Adding {} distractors".format(FLAGS.dim_distract))
      self.env  = AddDistractors(self.env, dim_distract=FLAGS.dim_distract)
      self.eval_env  = AddDistractors(self.eval_env, dim_distract=FLAGS.dim_distract)

    for e in [self.eval_env, self.env]:
      e.seed(FLAGS.seed)
      e.action_space.seed(FLAGS.seed)
      e.observation_space.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    self.rngs = hk.PRNGSequence(FLAGS.seed)
    
    self.agent = Agent(self.env.observation_space, self.env.action_space)
    self.replay_buffer = ReplayBuffer(self.env.observation_space.shape, FLAGS.num_train_steps)

    # FLAGS.out_dir = os.path.join(FLAGS.out_dir, time.strftime("%d%H%M"))
    FLAGS.out_dir = os.path.join(FLAGS.out_dir, FLAGS.exp, FLAGS.agent_type, str(FLAGS.seed))
    os.makedirs(FLAGS.out_dir, exist_ok=True)
    self.logger = Logger(Path(FLAGS.out_dir), cfg)
    self.best_step = 0
    self.best_return = -np.inf
    self.step, self.episode = 0, 0
  
  def run(self):
    episode_return, episode_step = 0, 0
    done = False
    obs = self.env.reset()
    action = self.agent.act(self.agent.params_Q, obs, next(self.rngs))
    start_time = time.time()
    train_metrics = {
      "episode": self.episode,
      "step": self.step,
      "total_time": time.time() - start_time,
    }
    print("Beginning of training")
    while self.step < FLAGS.num_train_steps:
      train_metrics.update({
        "episode": self.episode,
        "step": self.step,
        "total_time": time.time() - start_time,
      })
      # evaluate self.agent periodically
      if self.step % FLAGS.eval_frequency == 0:
        eval_metrics = {
          "episode": self.episode,
          "step": self.step,
          "total_time": time.time() - start_time,
        }
        eval_metrics["episode_reward"] = evaluate(self.agent, self.eval_env, next(self.rngs))
        self.logger.log(eval_metrics, "eval")
        if eval_metrics["episode_reward"] >= self.best_return:
          self.best_return = eval_metrics["episode_reward"]
          self.best_step = self.step
          self.save("best")

      # with epsilon exploration
      action = self.env.action_space.sample() if (np.random.rand() < FLAGS.eps or self.step < FLAGS.init_steps) else action.item()
      next_obs, reward, done, _ = self.env.step(action)

      done = float(done)
      # allow infinite bootstrap
      done_no_max = 0 if episode_step + 1 == self.max_episode_steps else done
      episode_return += reward

      self.replay_buffer.add(obs, action, reward, next_obs, done, done_no_max)
      
      obs = next_obs
      episode_step += 1
      self.step += 1
      
      if done:
        train_metrics["episode_reward"] = episode_return
        obs = self.env.reset()
        done = False
        episode_return = 0
        episode_step = 0
        self.episode += 1

      action = self.agent.act(self.agent.params_Q, obs, next(self.rngs))
          
      train_metrics["step"] = self.step
      losses_dict = {}
      if self.step >= FLAGS.init_steps:
        update_metric = True if self.step>=FLAGS.metric_warmup_steps else False
        losses_dict = self.agent.update(self.replay_buffer, update_metric=update_metric)
      train_metrics.update(losses_dict)

      if self.step % FLAGS.log_frequency == 0:
        # print(self.agent.getmetric())
        self.logger.log(train_metrics, "train")
        self.save("{}".format(self.step))
    
    # final eval after training is done
    eval_metrics["episode_reward"] = evaluate(self.agent, self.eval_env, next(self.rngs))
    self.logger.log(eval_metrics, "eval")

    print("Done in {:.1f} minutes".format((time.time() - start_time)/60))

  def save(self, tag="latest"):
    # self.agent.save(os.path.join(FLAGS.out_dir, f"{tag}.pkl"))
    path = os.path.join(FLAGS.out_dir, f"{tag}.pkl")
    with open(path, "wb") as f:
      pkl.dump(self, f)

  # def load(self, tag="latest"):
  #   self.agent.load(os.path.join(FLAGS.out_dir, f"{tag}.pkl"))

  @classmethod
  def load(cls, path):
    with open(path, "rb") as f:
      return pkl.load(f)

def wrapper(cfg):
  FLAGS = flags.FLAGS
  FLAGS(sys.argv)
  for k, v in cfg.items():
    if k in FLAGS:
      FLAGS[k].value = v
  return 



