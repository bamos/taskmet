from collections.abc import Callable
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from absl import flags

import jaxopt

from jax_rl.agents.actor_critic_temp import ModelActorCriticTemp, MetricModelActorCriticTemp
from jax_rl.agents.omd import actor, critic, temperature
from jax_rl.datasets import Batch
from jax_rl.networks.common import InfoDict, Params, TrainState

FLAGS = flags.FLAGS


@partial(jax.custom_vjp, nondiff_argnums=(0, 7))
def root_solve(param_func: Callable, init_xs: ModelActorCriticTemp,
               params: Params, batch: Batch, discount: float, tau: float,
               target_entropy: float, solvers: Tuple[Callable, ]):

    fwd_solver = solvers[0]
    return fwd_solver(param_func, init_xs, params, batch, discount, tau,
                      target_entropy)


def root_solve_fwd(param_func: Callable, init_xs: ModelActorCriticTemp,
                   params: Params, batch: Batch, discount: float, tau: float,
                   target_entropy: float, solvers: Tuple[Callable, ]):
    sol = root_solve(param_func, init_xs, params, batch, discount, tau,
                     target_entropy, solvers)
    new_omd, merged_info = sol
    return sol, (new_omd, params, batch, discount, tau, target_entropy)


def root_solve_bwd(param_func: Callable, solvers: Tuple[Callable, ],
                   res: Tuple[Params, ], g: Tuple[ModelActorCriticTemp,
                                                  InfoDict]):
    # only the identity approximation version
    new_omd, params, batch, discount, tau, target_entropy = res

    _, vdp_fun = jax.vjp(
        lambda y: param_func(new_omd, y, batch, discount, tau, target_entropy),
        params)
    # g contains the adjoint for output of critic.update
    g_main = g[0].critic.params
    vdp = vdp_fun(g_main)[0]

    z_new_omd, z_batch, z_discount, z_tau, z_tent = jax.tree_map(
        jnp.zeros_like, (new_omd, batch, discount, tau, target_entropy))
    return z_new_omd, jax.tree_map(lambda x: -x,
                                   vdp), z_batch, z_discount, z_tau, z_tent


root_solve.defvjp(root_solve_fwd, root_solve_bwd)


def metric_loss(model_params: Params, metric_params: Params, taskmet: MetricModelActorCriticTemp, batch: Batch):
    next_observations, rewards = taskmet.model.apply({'params': model_params},
                                                batch.observations,
                                                batch.actions)
    metric = taskmet.metric.apply({'params': metric_params}, batch.observations, batch.actions)
    err = jnp.expand_dims((next_observations - batch.next_observations), 2)
    metric_loss = (err.transpose((0, 2, 1)) @ metric @ err).sum(-1).mean()
    reward_loss = ((rewards - batch.rewards)**2).mean()
    tot_loss = metric_loss + reward_loss
    return tot_loss, {"T_loss": (err[:,:,0]**2).sum(-1).mean(), "metric_loss": metric_loss, 
                  "reward_loss": reward_loss, "tot_loss": tot_loss}

def constraint_func(omd: ModelActorCriticTemp, model_params: Params,
                    batch: Batch, discount: float, tau: float,
                    target_entropy: float):
    """Get grad_Q (model-Bellman-error) = 0 constraint.
    """
    return critic.update(omd,
                         model_params,
                         batch,
                         discount,
                         soft_critic=True,
                         use_model=True,
                         return_grad=True)

def fwd_solver(constraint_func: Callable, omd: ModelActorCriticTemp,
               model_params: Params, batch: Batch, discount: float, tau: float,
               target_entropy: float):
    """Get Q_* satisfying the constraint (approximately). Makes K grad updates.
    """

    for _ in range(FLAGS.config.inner_steps):
        omd, critic_info = critic.update(omd,
                                         model_params,
                                         batch,
                                         discount,
                                         soft_critic=True,
                                         use_model=True)
        omd = critic.target_update(omd, tau)

        # note that actor and temp do not use next_observations and rewards
        omd, actor_info = actor.update(omd, batch)
        omd, alpha_info = temperature.update(omd, actor_info['entropy'],
                                             target_entropy)

    merged_info = {**critic_info, **actor_info, **alpha_info}
    return omd, merged_info

def optimality_func(taskmet: MetricModelActorCriticTemp, metric_params: Params,
                    batch: Batch):
    """Get grad_model (Model-mertic error constraint) = 0 constraint.
    """
    print(taskmet.model)
    grads, _ = jax.grad(metric_loss, has_aux=True)(taskmet.model.params, metric_params, taskmet, batch)
    return grads

def T_fwd_solver(taskmet: MetricModelActorCriticTemp,
                 metric_params: Params, batch: Batch):
    """Get T_* satisfying the constraint (approximately). Makes K grad updates.
    """
    rng, key = jax.random.split(taskmet.rng)
    metric_loss_fn = partial(metric_loss, metric_params = metric_params, taskmet=taskmet, batch=batch)
    
    for _ in range(FLAGS.config.dynamic_train_steps):
        new_model, info = taskmet.model.apply_gradient(metric_loss_fn)

    taskmet = taskmet.replace(model=new_model, rng=rng)
    return taskmet, info

def update_omd(omd: ModelActorCriticTemp, batch: Batch, discount: float,
               tau: float, target_entropy: float,
               inner_steps: int) -> Tuple[ModelActorCriticTemp, InfoDict]:
    rng, key = jax.random.split(omd.rng)

    def model_loss_fn(model_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # Update critic_params on a batch w.r.t. loss yielded by model_params
        new_omd, merged_info = root_solve(constraint_func, omd, model_params,
                                          batch, discount, tau, target_entropy,
                                          (fwd_solver, ))

        _, model_info = critic.update(new_omd,
                                      None,
                                      batch,
                                      discount,
                                      soft_critic=True,
                                      use_model=False)

        return model_info['critic_loss'], ({
            **merged_info, 'model_loss':
            model_info['critic_loss']
        }, new_omd)

    new_model, (info, new_omd) = omd.model.apply_gradient(model_loss_fn)
    new_omd = new_omd.replace(model=new_model, rng=rng)

    return new_omd, info

def update_metric(taskmet: MetricModelActorCriticTemp, batch: Batch, discount: float,
               tau: float, target_entropy: float,
               inner_steps: int, cg_iters:int = 10) -> Tuple[ModelActorCriticTemp, InfoDict]:
    rng, key = jax.random.split(taskmet.rng)

    def metric_loss_fn(metric_params: Params) -> Tuple[jnp.ndarray, InfoDict]:        
        solve = partial(jaxopt.linear_solve.solve_normal_cg, maxiter=cg_iters)
        
        @jaxopt.implicit_diff.custom_root(optimality_func, solve=solve)
        def fit_dx(taskmet, metric_params, batch):
            new_taskmet, merged_info = T_fwd_solver(taskmet, metric_params, batch)
            return new_taskmet, merged_info

        new_taskmet, dynamics_info = fit_dx(taskmet, metric_params, batch)

        # Update critic_params on a batch w.r.t. loss yielded by model_params
        new_taskmet, merged_info = root_solve(constraint_func, new_taskmet, new_taskmet.model.params,
                                          batch, discount, tau, target_entropy,
                                          (fwd_solver, ))
        
        _, model_info = critic.update(new_taskmet,
                                    None,
                                    batch,
                                    discount,
                                    soft_critic=True,
                                    use_model=False)
        
        return model_info['critic_loss'], ({
            **merged_info, **dynamics_info, 'model_loss':
            model_info['critic_loss']
        }, new_taskmet)
        

    new_metric, (info, new_taskmet) = taskmet.metric.apply_gradient(metric_loss_fn)
    new_taskmet = new_taskmet.replace(metric=new_metric, rng=rng)

    return new_taskmet, info

def update_mle(mle: ModelActorCriticTemp, batch: Batch, discount: float,
               tau: float,
               inner_steps: int) -> Tuple[ModelActorCriticTemp, InfoDict]:
    rng, key = jax.random.split(mle.rng)

    def model_loss_fn(model_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        next_observations, rewards = mle.model.apply({'params': model_params},
                                                     batch.observations,
                                                     batch.actions)

        model_loss = ((next_observations -
                       batch.next_observations)**2).sum(-1).mean()
        model_loss += ((rewards - batch.rewards)**2).mean()

        return model_loss, {'model_loss': model_loss}

    new_model, info = mle.model.apply_gradient(model_loss_fn)

    mle = mle.replace(model=new_model, rng=rng)

    return mle, info
