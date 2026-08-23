from typing import Any, Optional, Sequence
import distrax
import flax.linen as nn
import jax.numpy as jnp


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0},
        split_rngs={'params': True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class Identity(nn.Module):
    """Identity layer."""

    def __call__(self, x):
        return x


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    initial_activation: Any = nn.tanh
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        # This architecture is adapted from the HILP implementation: https://github.com/seohongpark/HILP/blob/be2431bbb75e3b13cbdb1dec11776c42ef0f1593/hilp_zsrl/url_benchmark/agent/fb_modules.py#L148-L193.
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i == 0:
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.initial_activation(x)
            elif i + 1 < len(self.hidden_dims) or self.activate_final:
                 x = self.activations(x)
        return x

class LengthNormalize(nn.Module):
    """Length normalization layer.

    It normalizes the input along the last dimension to have a length of sqrt(dim).
    """

    @nn.compact
    def __call__(self, x):
        return x / jnp.linalg.norm(x, axis=-1, keepdims=True) * jnp.sqrt(x.shape[-1])
    

class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class GCActor(nn.Module):
    """Goal-conditioned actor.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    activations: Any = nn.gelu
    layer_norm: bool = False
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    gc_encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(
            self.hidden_dims,
            activations=self.activations,
            activate_final=True,
            layer_norm=self.layer_norm,
        )

        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        goals=None,
        goal_encoded=False,
        temperature=1.0,
    ):
        """Return the action distribution.

        Args:
            observations: Observations.
            goals: Goals (optional).
            goal_encoded: Whether the goals are already encoded (optional).
            temperature: Scaling factor for the standard deviation (optional).
        """
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution


class BidirectionalMixerBlock(nn.Module):
    """MLP-Mixer block with unrestricted communication between token positions."""

    num_tokens: int
    token_hidden_dim: int
    channel_hidden_dim: int

    @nn.compact
    def __call__(self, x):
        # Token mixing uses the token axis as the feature dimension.
        residual = x
        y = nn.LayerNorm()(x)
        y = jnp.swapaxes(y, -1, -2)
        y = nn.Dense(self.token_hidden_dim, kernel_init=default_init())(y)
        y = nn.gelu(y)
        y = nn.Dense(self.num_tokens, kernel_init=default_init())(y)
        x = residual + jnp.swapaxes(y, -1, -2)

        # This full matrix replaces CoGHP's lower-triangular causal mixer.
        residual = x
        y = nn.LayerNorm()(x)
        mixing_matrix = self.param(
            'mixing_matrix',
            nn.initializers.normal(stddev=0.02),
            (self.num_tokens, self.num_tokens),
        )
        y = jnp.einsum('ij,...jc->...ic', mixing_matrix, y)
        x = residual + y

        residual = x
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.channel_hidden_dim, kernel_init=default_init())(y)
        y = nn.gelu(y)
        y = nn.Dense(x.shape[-1], kernel_init=default_init())(y)
        return residual + y


class BidirectionalSubgoalActor(nn.Module):
    """Predict a fixed-length sequence of continuous FB intentions in parallel."""

    latent_dim: int
    num_subgoals: int
    model_dim: int = 256
    num_blocks: int = 2
    token_hidden_dim: int = 64
    channel_hidden_dim: int = 512
    const_std: bool = True
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    final_fc_init_scale: float = 1e-2

    def setup(self):
        self.state_encoder = MLP(
            (self.model_dim, self.model_dim),
            activate_final=True,
            layer_norm=True,
        )
        self.goal_encoder = MLP(
            (self.model_dim, self.model_dim),
            activate_final=True,
            layer_norm=True,
        )
        self.blocks = [
            BidirectionalMixerBlock(
                num_tokens=self.num_subgoals + 2,
                token_hidden_dim=self.token_hidden_dim,
                channel_hidden_dim=self.channel_hidden_dim,
            )
            for _ in range(self.num_blocks)
        ]
        self.output_norm = nn.LayerNorm()
        self.mean_net = nn.Dense(self.latent_dim, kernel_init=default_init(self.final_fc_init_scale))
        self.slot_tokens = self.param(
            'slot_tokens',
            nn.initializers.normal(stddev=0.02),
            (self.num_subgoals, self.model_dim),
        )
        self.position_embeddings = self.param(
            'position_embeddings',
            nn.initializers.normal(stddev=0.02),
            (self.num_subgoals + 2, self.model_dim),
        )

        if not self.const_std:
            self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.num_subgoals, self.latent_dim))

    def __call__(self, observations, goals, goal_encoded=True, temperature=1.0):
        """Return one latent distribution per ordered subgoal slot."""
        del goal_encoded  # Goals are always FB latents for this actor.

        state_token = self.state_encoder(observations)
        goal_token = self.goal_encoder(goals)

        slot_tokens = jnp.broadcast_to(
            self.slot_tokens,
            (*state_token.shape[:-1], self.num_subgoals, self.model_dim),
        )
        tokens = jnp.concatenate(
            [state_token[..., None, :], goal_token[..., None, :], slot_tokens],
            axis=-2,
        )

        tokens = tokens + self.position_embeddings
        for block in self.blocks:
            tokens = block(tokens)

        means = self.mean_net(self.output_norm(tokens[..., 2:, :]))
        means = means / (jnp.linalg.norm(means, axis=-1, keepdims=True) + 1e-8) * jnp.sqrt(self.latent_dim)

        if self.const_std:
            log_stds = jnp.zeros_like(means)
        else:
            log_stds = jnp.broadcast_to(self.log_stds, means.shape)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        return distrax.MultivariateNormalDiag(
            loc=means,
            scale_diag=jnp.exp(log_stds) * temperature,
        )


class GCValue(nn.Module):
    """Goal-conditioned value/critic function.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        value_dim: Value dimension.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        gc_encoder: GCEncoder module to encode the inputs (optional).
    """

    hidden_dims: Sequence[int]
    value_dim: int = 1
    activations: Any = nn.gelu
    layer_norm: bool = True
    num_ensembles: int = 1
    gc_encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP

        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class,  self.num_ensembles)

        self.value_net = mlp_class(
            (*self.hidden_dims, self.value_dim),
            activations=self.activations,
            activate_final=False,
            layer_norm=self.layer_norm
        )

    def __call__(self, observations, goals=None, actions=None, goal_actions=None, goal_encoded=False):
        """Return the value/critic function.

        Args:
            observations: Observations.
            goals: Goals (optional).
            actions: Actions (optional).
            goal_encoded: Whether the goals are already encoded (optional).
        """
        if self.gc_encoder is not None:
            inputs = [self.gc_encoder(observations, goals, goal_encoded=goal_encoded)]
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
        if actions is not None:
            inputs.append(actions)
        if goal_actions is not None:
            inputs.append(goal_actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        if self.value_dim == 1:
            v = self.value_net(inputs).squeeze(-1)
        else:
            v = self.value_net(inputs)

        return v

class GCBilinearValue(nn.Module):
    """Goal-conditioned bilinear value/critic function.

    This module computes the value function as V(s, g) = phi(s)^T psi(g) / sqrt(d) or the critic function as
    Q(s, a, g) = phi(s, a)^T psi(g) / sqrt(d), where phi and psi output d-dimensional vectors.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        layer_norm: Whether to apply layer normalization.
        ensemble: Whether to ensemble the value function.
        value_exp: Whether to exponentiate the value. Useful for contrastive learning.
        state_encoder: Optional state encoder.
        goal_encoder: Optional goal encoder.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    activations: Any = nn.gelu
    layer_norm: bool = True
    num_ensembles: int = 1
    state_encoder: nn.Module = None
    goal_encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class,  self.num_ensembles)
        
        self.phi_net = mlp_class(
            (*self.hidden_dims, self.latent_dim),
            activations=self.activations,
            activate_final=False,
            layer_norm=self.layer_norm
        )
        
    def __call__(self, observations, goals, intents=None):
        """Return the value/critic function.

        Args:
            observations: Observations.
            goals: Goals.
            intents: Intentions (optional).
        """
        if intents is None:
            intents = goals
    
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)
            intents = self.goal_encoder(intents)

        phi_inputs = jnp.concatenate([observations, intents], axis=-1)
        phi = self.phi_net(phi_inputs)
        v = (phi * goals / jnp.sqrt(self.latent_dim)).sum(axis=-1)

        return v

        
        
    
    
class ICVFValue(nn.Module):
    """ICVF value/critic function.

    This module computes the value function using the following parameterizations: V(s, g, z) = phi(s)^T T(z) psi(g)

    Attributes:
        hidden_dims: Hidden layer dimensions.
        value_dim: Value dimension.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        gcic_encoder: GCIntentionEncoder module to encode the inputs (optional).
    """

    hidden_dims: Sequence[int]
    value_dim: int = 1
    activations: Any = nn.gelu
    layer_norm: bool = True
    num_ensembles: int = 1
    icvf_encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP

        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class,  self.num_ensembles)

        self.phi_net = mlp_class(
            (*self.hidden_dims, self.value_dim),
            activations=self.activations,
            activate_final=False,
            layer_norm=self.layer_norm
        )
        self.transition_net = mlp_class(
            (*self.hidden_dims, self.value_dim * self.value_dim),
            activations=self.activations,
            activate_final=False,
            layer_norm=self.layer_norm
        )
        self.psi_net = mlp_class(
            (*self.hidden_dims, self.value_dim),
            activations=self.activations,
            activate_final=False,
            layer_norm=self.layer_norm
        )

    def __call__(self, observations, goals=None, intentions=None, actions=None, 
                 goal_actions=None, intention_actions=None, 
                 goal_encoded=False, intention_encoded=False,
                 phis=None, psis=None, transitions=None,
                 info=False):
        """Return the value/critic function.

        Args:
            observations: Observations.
            goals: Goals (optional).
            intentions: Intentions (optional).
            actions: Actions (optional).
            goal_actions: Goal actions (optional).
            intention_actions: Intention actions (optional).
            goal_encoded: Whether the goals are already encoded (optional).
            intention_encoded: Whether the intentions are already encoded (optional).
            phis: Precomputed phis representations (optional).
            psis: Precomputed psis representations (optional).
            transitions: precomputed transitions (optional)
            info: Whether to return phis, psis, and transitions.
            
        """
        psi_inputs = []
        transition_inputs = []
        if self.icvf_encoder is not None:
            phi_inputs, psi_inputs, transition_inputs = self.icvf_encoder(
                observations, goals, intentions, 
                goal_encoded=goal_encoded,
                intention_encoded=intention_encoded
            )
        else:
            phi_inputs = [observations]
            if goals is not None:
                psi_inputs.append(goals)
            if intentions is not None:
                transition_inputs.append(intentions)
        if actions is not None:
            phi_inputs.append(actions)
        if goal_actions is not None:
            psi_inputs.append(goal_actions)
        if intention_actions is not None:
            transition_inputs.append(intention_actions)
        if phis is None:
            phi_inputs = jnp.concatenate(phi_inputs, axis=-1)
            phis = self.phi_net(phi_inputs)
        if psis is None:
            if len(psi_inputs) > 0:
                psi_inputs = jnp.concatenate(psi_inputs, axis=-1)
                psis = self.psi_net(psi_inputs)
            else:
                psis = None
        if transitions is None:
            if len(transition_inputs) > 0:
                transition_inputs = jnp.concatenate(transition_inputs, axis=-1)
                transitions = self.transition_net(transition_inputs)
                transitions = transitions.reshape(
                    *transitions.shape[:-1], 
                    self.value_dim, 
                    self.value_dim
                )
            else:
                transitions = None
        
        if phis is not None and psis is not None and transitions is not None:
            if self.num_ensembles > 1:
                inners = jnp.einsum('eij,eijk->eik', phis, transitions)
            else:
                inners = jnp.einsum('ij,ijk->ik', phis, transitions)
            vs = jnp.sum(inners * psis, axis=-1)
            
            if self.value_dim == 1:
                vs = vs.squeeze(-1)
        else:
            vs = None
        
        if info:
            return vs, phis, psis, transitions
        else:
            return vs
