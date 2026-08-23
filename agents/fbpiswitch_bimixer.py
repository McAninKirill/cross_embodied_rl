"""FB pi-Switch agent with a bidirectional multi-subgoal Mixer."""

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from agents.fbpiswitch import FBpiSwitchAgent, get_config as get_base_config
from utils.flax_utils import ModuleDict, TrainState
from utils.networks import BidirectionalSubgoalActor, GCActor, GCValue


class FBpiSwitchBiMixerAgent(FBpiSwitchAgent):
    """FB pi-Switch variant that predicts an ordered intention sequence."""

    def high_actor_loss(self, batch, grad_params, rng):
        """Compute one AWR objective for an entire ordered subgoal chain."""
        observations = batch['observations']
        subgoals = batch['high_actor_targets']
        goals = batch['high_actor_goals']
        target_masks = batch['high_actor_target_masks']
        transition_masks = batch['high_actor_transition_masks']
        batch_size, num_subgoals = subgoals.shape[:2]

        rng, subgoal_rng, goal_rng = jax.random.split(rng, 3)
        target_latents, raw_target_latents = jax.lax.stop_gradient(
            self.sample_latents(subgoals, subgoal_rng, 0.0)
        )
        goal_latents, raw_goal_latents = jax.lax.stop_gradient(
            self.sample_latents(goals, goal_rng, self.config['critic_latent_mix_prob'])
        )

        predecessors = jnp.concatenate([observations[:, None], subgoals[:, :-1]], axis=1)
        repeated_goal_latents = jnp.broadcast_to(
            raw_goal_latents[:, None, :],
            raw_target_latents.shape,
        )

        def extract_chain_values(states, value_latents, intention_latents):
            values = self.successor_measure_extract(
                states.reshape((-1, *states.shape[2:])),
                value_latents.reshape((-1, value_latents.shape[-1])),
                intention_latents.reshape((-1, intention_latents.shape[-1])),
            )
            return values.reshape((*values.shape[:-1], batch_size, num_subgoals))

        Msww = extract_chain_values(predecessors, raw_target_latents, raw_target_latents)
        Mwww = extract_chain_values(subgoals, raw_target_latents, raw_target_latents)
        Vswr = extract_chain_values(predecessors, repeated_goal_latents, raw_target_latents)
        Vwrr = extract_chain_values(subgoals, repeated_goal_latents, repeated_goal_latents)
        Vrstar = extract_chain_values(predecessors, repeated_goal_latents, repeated_goal_latents)

        step_advantages = Vswr + Msww / (Mwww + 1e-8) * Vwrr - Vrstar
        advantage_masks = transition_masks.reshape(
            (1,) * (step_advantages.ndim - transition_masks.ndim) + transition_masks.shape
        )
        transition_counts = jnp.maximum(transition_masks.sum(axis=-1), 1.0)
        chain_advantages = (step_advantages * advantage_masks).sum(axis=-1) / transition_counts
        exp_a = jnp.exp(jnp.clip(chain_advantages * self.config['high_alpha'], max=5.0))
        exp_a = jax.lax.stop_gradient(exp_a)

        dist = self.network.select('high_actor')(
            observations,
            goal_latents,
            params=grad_params,
        )
        token_log_probs = dist.log_prob(target_latents)
        target_counts = jnp.maximum(target_masks.sum(axis=-1), 1.0)
        chain_log_probs = (token_log_probs * target_masks).sum(axis=-1) / target_counts

        awr_loss = -(exp_a * chain_log_probs).mean()
        bc_loss = -chain_log_probs.mean()
        actor_loss = awr_loss + self.config['chain_bc_coef'] * bc_loss

        squared_errors = jnp.mean((dist.mode() - target_latents) ** 2, axis=-1)
        masked_mse = (squared_errors * target_masks).sum() / jnp.maximum(target_masks.sum(), 1.0)

        return actor_loss, {
            'actor_loss': actor_loss,
            'awr_loss': awr_loss,
            'bc_loss': bc_loss,
            'chain_adv': chain_advantages.mean(),
            'chain_adv_std': chain_advantages.std(),
            'chain_log_prob': chain_log_probs.mean(),
            'mse': masked_mse,
            'valid_subgoals': target_masks.sum(axis=-1).mean(),
            'unique_transitions': transition_masks.sum(axis=-1).mean(),
            'active_exps': (jnp.abs(exp_a) > 1e-3).mean(),
            'saturated': jnp.mean(chain_advantages * self.config['high_alpha'] >= 5.0),
        }

    @jax.jit
    def sample_latents(self, targets, rng, latent_mix_prob):
        """Encode tensors with arbitrary leading batch and subgoal dimensions."""
        batch_shape = targets.shape[:-1]
        rng, latent_rng, mix_rng = jax.random.split(rng, 3)

        latents = jax.random.normal(latent_rng, shape=(*batch_shape, self.config['latent_dim']))
        norm_latents = self.normalize_z(latents)
        backward_reprs = self.network.select('backward_repr')(targets)
        norm_backward_reprs = self.normalize_z(backward_reprs)

        flags_latents = jax.random.uniform(mix_rng, (*batch_shape, 1)) < latent_mix_prob
        latents = jnp.where(flags_latents, latents, backward_reprs)
        norm_latents = jnp.where(flags_latents, norm_latents, norm_backward_reprs)
        return norm_latents, latents

    @jax.jit
    def sample_plan(self, observations, latents, seed=None, temperature=1.0):
        """Sample all high-level intentions in nearest-to-farthest order."""
        high_dist = self.network.select('high_actor')(
            observations,
            latents,
            goal_encoded=True,
            temperature=temperature,
        )
        return self.normalize_z(high_dist.sample(seed=seed))

    @jax.jit
    def sample_actions(self, observations, latents=None, seed=None, temperature=1.0):
        """Execute the nearest intention from the jointly predicted plan."""
        high_seed, low_seed = jax.random.split(seed)
        plan = self.sample_plan(observations, latents, seed=high_seed, temperature=temperature)
        nearest_intention = plan[..., 0, :]

        low_dist = self.network.select('actor')(
            observations,
            nearest_intention,
            goal_encoded=True,
            temperature=temperature,
        )
        actions = low_dist.sample(seed=low_seed)
        return jnp.clip(actions, -1, 1)

    @classmethod
    def create(cls, seed, ex_batch, config):
        """Create frozen FB modules and a trainable bidirectional Mixer."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = ex_batch['observations']
        ex_actions = ex_batch['actions']
        action_dim = ex_actions.shape[-1]
        ex_latents = jnp.ones((*ex_actions.shape[:-1], config['latent_dim']))

        forward_repr_def = GCValue(
            hidden_dims=config['forward_repr_hidden_dims'],
            value_dim=config['latent_dim'],
            activations=getattr(nn, config['activation']),
            layer_norm=config['forward_repr_layer_norm'],
            num_ensembles=2,
        )
        backward_repr_def = GCValue(
            hidden_dims=config['backward_repr_hidden_dims'],
            value_dim=config['latent_dim'],
            activations=getattr(nn, config['activation']),
            layer_norm=config['backward_repr_layer_norm'],
            num_ensembles=1,
        )
        actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            state_dependent_std=False,
            const_std=config['const_std'],
        )
        high_actor_def = BidirectionalSubgoalActor(
            latent_dim=config['latent_dim'],
            num_subgoals=config['num_subgoals'],
            model_dim=config['mixer_model_dim'],
            num_blocks=config['mixer_num_blocks'],
            token_hidden_dim=config['mixer_token_hidden_dim'],
            channel_hidden_dim=config['mixer_channel_hidden_dim'],
            const_std=config['const_std'],
        )

        network_info = dict(
            forward_repr=(forward_repr_def, (ex_observations, ex_latents, None, None, True)),
            backward_repr=(backward_repr_def, (ex_observations,)),
            actor=(actor_def, (ex_observations, ex_latents, True)),
            high_actor=(high_actor_def, (ex_observations, ex_latents, True)),
        )

        networks = {key: value[0] for key, value in network_info.items()}
        network_args = {key: value[1] for key, value in network_info.items()}
        network_def = ModuleDict(networks)
        network_params = network_def.init(init_rng, **network_args)['params']

        def mask_fn(params):
            flat_params = flax.traverse_util.flatten_dict(params)
            flat_mask = {
                key: 'train' if key[0] == 'modules_high_actor' else 'frozen'
                for key in flat_params
            }
            return flax.traverse_util.unflatten_dict(flat_mask)

        network_tx = optax.multi_transform(
            {
                'train': optax.adam(config['lr']),
                'frozen': optax.set_to_zero(),
            },
            mask_fn(network_params),
        )
        network = TrainState.create(network_def, network_params, tx=network_tx)
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = get_base_config()
    config.agent_name = 'fbpiswitch_bimixer'
    config.dataset_class = 'HGCMultiGoalDataset'
    config.num_subgoals = 4
    config.mixer_model_dim = 256
    config.mixer_num_blocks = 2
    config.mixer_token_hidden_dim = 64
    config.mixer_channel_hidden_dim = 512
    config.chain_bc_coef = 0.1
    return config
