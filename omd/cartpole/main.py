import os
import sys
import yaml
from pathlib import Path
from omegaconf import OmegaConf
from absl import app
from absl import flags
from train import Workspace

FLAGS = flags.FLAGS
flags.DEFINE_string('exp', 'default', 'Custom description string added to out_dir')
flags.DEFINE_string('out_dir', 'exp', 'Directory for output files')
flags.DEFINE_string('data_path', 'data/buf.pkl', 'Path to save buffer')
flags.DEFINE_string('agent_path', 'data/agent.pkl', 'Path to save agent')
flags.DEFINE_string('env_name', 'CartPole-v1', 'Gym environment id')
flags.DEFINE_string('use_wandb', 'False', 'Use wandb for logging')
flags.DEFINE_string('wandb_project', 'none', 'Wandb project name')
flags.DEFINE_string('wandb_entity', 'none', 'Wandb entity name')
flags.DEFINE_integer('seed', 0, 'Random seed')
flags.DEFINE_integer('num_train_steps', 200000, 'Env steps num', lower_bound=0)
flags.DEFINE_integer('cg_iters', 100, 'CG iters for getting implsicit derivative', lower_bound=1)
flags.DEFINE_integer('hidden_dim', 32, 'Size of hidden layers', lower_bound=1)
flags.DEFINE_integer('model_hidden_dim', 32, 'Model-specific', lower_bound=1)
flags.DEFINE_integer('dim_distract', 0, 'Number of distracting states.')
flags.DEFINE_integer('batch_size', 256, 'Mini-batch samples', lower_bound=1)
flags.DEFINE_integer('init_steps', 1000, 'Steps before training', lower_bound=0)
flags.DEFINE_integer('metric_warmup_steps', 1000, 'Steps before updating the metric', lower_bound=0)
flags.DEFINE_integer('eval_frequency', 1000, 'Agent evaluation', lower_bound=1)
flags.DEFINE_integer('log_frequency', 1000, 'Logging frequency', lower_bound=1)
flags.DEFINE_integer('model_log_frequency', 10000, 'Model logging frequency', lower_bound=1)
flags.DEFINE_integer('num_Q_steps', 1, 'Inner loop steps num', lower_bound=1)
flags.DEFINE_integer('num_T_steps', 1, 'Model steps per update', lower_bound=1)
flags.DEFINE_integer('num_ensemble_vep', 5, 'Value functions #', lower_bound=1)
flags.DEFINE_float('eps', 0.1, 'Random action probability')
flags.DEFINE_float('discount', 0.99, 'Sum of rewards discount factor')
flags.DEFINE_float('lr', 1e-3, '(Outer loop) learning rate')
flags.DEFINE_float('inner_lr', 3e-4, 'Inner loop learning rate')
flags.DEFINE_float('metric_lr', 1e-3, 'Model learning rate')
flags.DEFINE_float('weight_decay', 0.0, 'Weight decay for metric parameters')
flags.DEFINE_float('regularization_coeff', 0.0, 'Regularization coefficient for metric parameters')
flags.DEFINE_float('alpha', 0.01, 'Temperature')
flags.DEFINE_float('tau', 0.01, 'Target network update coefficient')
flags.DEFINE_boolean('q_grad_clip', False, 'Target network update coefficient')
flags.DEFINE_boolean('metric_grad_clip', False, 'Target network update coefficient')
flags.DEFINE_boolean('metric_conditional', True, 'If the metric is conditioned on the state')
flags.DEFINE_boolean('diag_metric', False, 'If the metric is diagonal')
flags.DEFINE_boolean('full_network', True, 'If diag metric, then if just use whole network or diag parameters')
flags.DEFINE_boolean('lower_tri_matrix', True, 'If lower triangular matrix to parameterize metric')
flags.DEFINE_boolean('save_buf', False, 'Save collected buffer with data')
flags.DEFINE_boolean('save_agent', False, 'Save the agent after training')
flags.DEFINE_boolean('hard', False, 'max vs logsumexp for Q learning')
flags.DEFINE_boolean('no_learn_reward', False, 'Use trainable or true rewards')
flags.DEFINE_boolean('prob_model', False, 'Gaussian vs deterministic next obs')
flags.DEFINE_boolean('no_warm', False, 'Not use previous Q* in the inner loop')
flags.DEFINE_boolean('warm_opt', False, 'Reuse inner loop optimizer statistics')
flags.DEFINE_boolean('no_double', False, 'Not use Double Q Learning')
flags.DEFINE_boolean('with_inv_jac', False, 
  'Replaces inverse Jacobian in implicit gradient with identity matrix for Q function')
flags.DEFINE_boolean('with_inv_jac_model', False, 
  'Replaces inverse Jacobian in implicit gradient with identity matrix for the model')
flags.DEFINE_enum('agent_type', 'omd', ['omd', 'mle', 'vep', 'metric'], 'Agent type')
flags.DEFINE_enum('metric_activation', 'normalize', ['normalize', 'sigmoid', 'softplus'] ,'Activation to be used for the metric')

def main_function(cfg):
  workspace = Workspace(cfg)
  workspace.run()

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