import numpy as np
from functools import partial
from collections import namedtuple
import gym
import numpy as np
from gym.spaces import Box

import jax
import jax.numpy as jnp
import optax
from jax.scipy.sparse.linalg import cg
from jax.lax import stop_gradient

import haiku as hk
from absl import flags
FLAGS = flags.FLAGS

class AddDistractors(gym.ObservationWrapper):
    def __init__(self, env, dim_distract=0):
        super().__init__(env)

        assert isinstance(self.observation_space, Box)

        self.dim_distract = dim_distract

        obs_space = self.observation_space

        self.observation_space = Box(
            np.concatenate((obs_space.low, [-np.inf] * dim_distract)),
            np.concatenate((obs_space.high, [np.inf] * dim_distract)),
            (obs_space.shape[0] + dim_distract, ))

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.concatenate(
            (observation, np.random.randn(self.dim_distract)))

def evaluate(agent, eval_env, rng, num_eval_episodes=10):
  average_episode_reward = 0
  for episode in range(num_eval_episodes):
    obs = eval_env.reset()
    done = False
    episode_reward = 0
    while not done:
      rng, _ = jax.random.split(rng)
      action = agent.act(agent.params_Q, obs, rng).item()
      obs, reward, done, _ = eval_env.step(action)
      episode_reward += reward
    average_episode_reward += episode_reward
  average_episode_reward /= num_eval_episodes
  return average_episode_reward

@partial(jax.custom_vjp, nondiff_argnums=(0, 5))
def root_solve(param_func, init_xs, params, replay, rng, solvers):
  # to mimic two_phase_solve API
  fwd_solver = solvers[0]
  return fwd_solver(param_func, init_xs, params, replay, rng)

def root_solve_fwd(param_func, init_xs, params, replay, rng, solvers):
  sol = root_solve(param_func, init_xs, params, replay, rng, solvers)
  tpQ = jax.lax.stop_gradient(sol.target_params_Q)
  return sol, (sol.params_Q, params, replay, rng, tpQ)

def root_solve_bwd(param_func, solvers, res, g):
  pQ, params, replay, rng, tpQ = res
  _, vdp_fun = jax.vjp(lambda y: param_func(y, pQ, replay, rng, tpQ), params)
  g_main = g[0] if isinstance(g, tuple) else g
  if FLAGS.with_inv_jac:
    # _, vds_fun = jax.vjp(lambda x: param_func(params, x), pQ)
    # (J)^-1 -> (J+cI)^-1
    _, vds_fun = jax.vjp(lambda x: jax.tree_multimap(
      lambda y,z: y + 1e-5*z, param_func(params, x, replay, rng, tpQ), x), pQ)
    vdsinv = cg(lambda z: vds_fun(z)[0], g_main, maxiter=FLAGS.cg_iters)[0]
    vdp = vdp_fun(vdsinv)[0]
  else:
    vdp = vdp_fun(g_main)[0]
  z_sol, z_replay, z_rng = jax.tree_map(jnp.zeros_like, (pQ, replay, rng))
  return z_sol, jax.tree_map(lambda x: -x, vdp), z_replay, z_rng

root_solve.defvjp(root_solve_fwd, root_solve_bwd)

@partial(jax.custom_vjp, nondiff_argnums=(0, 5))
def dynamics_root_solve(param_func, init_xs, params, replay, rng, solvers):
  # to mimic two_phase_solve API
  fwd_solver = solvers[0]
  return fwd_solver(param_func, init_xs, params, replay, rng)

def dynamics_root_solve_fwd(param_func, init_xs, params, replay, rng, solvers):
  sol = dynamics_root_solve(param_func, init_xs, params, replay, rng, solvers)
  return sol, (sol.params_T, params, replay, rng)

def dynamics_root_solve_bwd(param_func, solvers, res, g):
  pT, params, replay, rng = res
  _, vdp_fun = jax.vjp(lambda y: param_func(y, pT, replay, rng), params)
  g_main = g[0] if isinstance(g, tuple) else g
  if FLAGS.with_inv_jac_model:
    # assert False # (use jaxopt in omd.py now)
    # _, vds_fun = jax.vjp(lambda x: param_func(params, x), pT)
    # (J)^-1 -> (J+cI)^-1
    _, vds_fun = jax.vjp(lambda x: jax.tree_multimap(
      lambda y,z: y + 1e-5*z, param_func(params, x, replay, rng), x), pT)
    vdsinv = cg(lambda z: vds_fun(z)[0], g_main, maxiter=FLAGS.cg_iters)[0]
    vdp = vdp_fun(vdsinv)[0]
  else:
    vdp = vdp_fun(g_main)[0]
  z_sol, z_replay, z_rng = jax.tree_map(jnp.zeros_like, (pT, replay, rng))
  return z_sol, jax.tree_map(lambda x: -x, vdp), z_replay, z_rng

dynamics_root_solve.defvjp(dynamics_root_solve_fwd, dynamics_root_solve_bwd)

def add_dict(d, k, v):
  if not isinstance(v, list):
    v = [v]
  if k in d:
    d[k].extend(v)
  else:
    d[k] = v

@jax.jit
def soft_update_params(tau, params, target_params):
  return jax.tree_multimap(
    lambda p, tp: tau * p + (1 - tau) * tp, 
    params, target_params)

@jax.jit
def tree_norm(tree):
  return jnp.sqrt(sum((x**2).sum() for x in jax.tree_leaves(tree)))
  
@partial(jax.jit, static_argnums=(1,))
def fill_lower_tri(v, dim):
    # we can use jax.ensure_compile_time_eval + jnp.tri to do mask indexing
    # but best practice is use numpy for static variable
    # and jnp.tril_indices is just a wrapper around np.tril_indices
    idx = np.tril_indices(dim)
    return jnp.zeros((dim, dim), dtype=v.dtype).at[idx].set(v)

def net_fn(net_type, dims, x):
  obs_dim, action_dim, hidden_dim = dims
  activation = jax.nn.relu
  init = hk.initializers.Orthogonal(scale=jnp.sqrt(2.0))
  layers = [
    hk.Linear(hidden_dim, w_init=init), activation,
    hk.Linear(hidden_dim, w_init=init), activation,
  ]
  final_init = hk.initializers.Orthogonal(scale=1e-2)
  # T -- model, Q and V -- value functions
  if FLAGS.agent_type == 'vep':
    if net_type == 'V':
      ensemble = []
      for i in range(FLAGS.num_ensemble_vep):
        layers = [
          hk.Linear(hidden_dim, w_init=init), activation,
          hk.Linear(hidden_dim, w_init=init), activation,
          hk.Linear(1, w_init=final_init)
        ]
        mlp = hk.Sequential(layers)
        ensemble.append(mlp(x))
      return ensemble
  if net_type == 'V':
    layers += [hk.Linear(1, w_init=final_init)] 
  elif net_type == 'Q':
    layers += [hk.Linear(action_dim, w_init=final_init)]
  elif net_type == 'T':
    out_dim = 2 * obs_dim if FLAGS.prob_model else obs_dim
    layers += [hk.Linear(out_dim, w_init=final_init)]
  elif net_type == 'metric':
    if FLAGS.diag_metric: 
      out_dim = obs_dim
    else:
      out_dim = obs_dim*(obs_dim+1)//2 if FLAGS.lower_tri_matrix else obs_dim**2
    if FLAGS.full_network:
      layers = [hk.Linear(hidden_dim, w_init=init), activation,
        hk.Linear(out_dim, w_init=final_init, b_init=hk.initializers.Constant(1.0))]
    else:
      layers = hk.get_parameter('metric', shape=(out_dim,), init=hk.initializers.Constant(1.0)) if FLAGS.diag_metric else [hk.Linear(out_dim, with_bias=False, w_init=hk.initializers.Constant(1.0))]
    if not FLAGS.lower_tri_matrix and not FLAGS.diag_metric:
      layers += [hk.Reshape((obs_dim, obs_dim))]

  if net_type == 'Q' and not FLAGS.no_double:
    layers2 = [
      hk.Linear(hidden_dim, w_init=init), activation,
      hk.Linear(hidden_dim, w_init=init), activation,
      hk.Linear(action_dim, w_init=final_init)
    ]
    mlp1, mlp2 = hk.Sequential(layers), hk.Sequential(layers2)
    return mlp1(x), mlp2(x)
  elif net_type == 'T' and not FLAGS.no_learn_reward:
    layers2 = [
      hk.Linear(hidden_dim, w_init=init), activation,
      hk.Linear(hidden_dim, w_init=init), activation,
      hk.Linear(1, w_init=final_init)
    ]
    mlp1, mlp2 = hk.Sequential(layers), hk.Sequential(layers2)
    return mlp1(x), mlp2(x)
  
  elif net_type == 'metric':        
    if not FLAGS.metric_conditional:
      x = jnp.ones_like(x)

    if FLAGS.full_network:
      metric = hk.Sequential(layers)(x)
    else:
      metric = hk.Sequential(layers)(x) if not FLAGS.diag_metric else layers*x

    if FLAGS.lower_tri_matrix and not FLAGS.diag_metric:
      vmap_fill_lower_tri = jax.vmap(fill_lower_tri, (0, None), 0)
      metric = vmap_fill_lower_tri(metric, obs_dim)
    
    diag_vmap = jax.vmap(jnp.diag, 0)
    if FLAGS.diag_metric:
      metric = jax.nn.softplus(metric)
      metric = diag_vmap(metric)
    else:
      # metric = metric@metric.transpose((0, 2, 1)) # when metric is covariance
      metric = metric.transpose((0, 2, 1))@metric # when metric is inverse of covariance

    if FLAGS.metric_activation == 'normalize':
      diag_metric = diag_vmap(metric)
      # print(diag_metric.shape)
      metric_norm = jnp.linalg.norm(diag_metric, axis=1,keepdims=True)
      metric_norm = metric_norm[...,None] # to make metric_norm broadcastable i.e (batch_size, 1, 1)
      metric = metric/metric_norm*np.sqrt(obs_dim) # because in diag_metric, we are directly using metric as variance instread of std dev as in not_diag variant
      # metric = metric/(metric_norm**2)*obs_dim # normalize the standard deviation to standard deviation of mse metric
    # print(metric)
    # metric = jnp.linalg.inv(metric) # when we assume that we initialized metric as covariance matrix, then we need to take inverse as that's what we use in our formulation
    return metric
  else:
    mlp = hk.Sequential(layers)
    return mlp(x)

def init_net_opt(net_type, dims):
  net = hk.without_apply_rng(hk.transform(partial(net_fn, net_type, dims)))
  if net_type == 'Q':
    if FLAGS.q_grad_clip:
      print('Using gradient clipping for Q')
      opt = optax.chain(optax.clip_by_global_norm(FLAGS.q_grad_clip), 
                        optax.adam(FLAGS.inner_lr))
    else:
      opt = optax.adam(FLAGS.inner_lr)
  elif net_type == 'metric':
    if FLAGS.metric_grad_clip:
      print('Using gradient clipping for metric')
      if FLAGS.weight_decay == 0.0:
        opt = optax.chain(optax.clip_by_global_norm(FLAGS.metric_grad_clip), 
                          optax.adam(FLAGS.metric_lr))
      else:
        opt = optax.chain(optax.clip_by_global_norm(FLAGS.metric_grad_clip), 
                          optax.adamw(FLAGS.metric_lr, weight_decay=FLAGS.weight_decay))
    else:
      if FLAGS.weight_decay == 0.0:
        opt = optax.adam(FLAGS.metric_lr)
      else:
        opt = optax.adamw(FLAGS.metric_lr, weight_decay=FLAGS.weight_decay)
  else:
    opt = optax.adam(FLAGS.lr)
  Model = namedtuple(net_type, 'net opt')
  return Model(net, opt)
