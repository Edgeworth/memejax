from typing import cast

import flax.linen as nn
import jax.numpy as jnp
from chex import assert_shape
from jax import Array

from memejax.jax.model.util.initializers import initialize_identity


# Based on this paper: https://doi.org/10.1007/s11265-020-01624-0
class DynamicNormalization(nn.Module):
    @nn.compact
    def __call__(self, X: Array) -> Array:
        # Input: X = (D, L) matrix of D time series of length L.
        # Output: Dynamically normalised time series, (D, L)

        D = X.shape[0]
        L = X.shape[1]

        # Compute means for each timestamp.
        s_alpha = jnp.mean(X, axis=1)
        assert_shape(s_alpha, (D,))

        # Want to shift and scale each time series with learned parameters.
        W_alpha = self.param("W_alpha", initialize_identity, (D, D))
        b_alpha = self.param("b_alpha", nn.initializers.zeros, (D,))
        # Alpha is the shift amount.
        alpha = jnp.dot(W_alpha, s_alpha) + b_alpha
        assert_shape(alpha, (D,))

        # Compute standard deviations for each timestamp, after shifting by alpha.
        s_beta = jnp.sqrt(jnp.mean(jnp.square(X - alpha.reshape(-1, 1)), axis=1) + 1e-8)
        assert_shape(s_beta, (D,))

        # Beta is the scale amount.
        W_beta = self.param("W_beta", initialize_identity, (D, D))
        b_beta = self.param("b_beta", nn.initializers.zeros, (D,))
        beta = jnp.dot(W_beta, s_beta) + b_beta
        assert_shape(beta, (D,))

        # Note that setting W_alpha, W_beta to identity and b_alpha, b_beta to
        # zero is equivalent to sample based normalisation. Calculate it here
        # since we mix it with the learned normalisation to increase robustness.
        # sample normed = X - mean / stddev
        # Add small value to variance before sqrt to keep it differentiable at 0.
        X_sample = (X - s_alpha.reshape(-1, 1)) / jnp.sqrt(jnp.var(X, axis=1).reshape(-1, 1) + 1e-8)

        # Parameter lambda starting at 0 saying how much of the learnt
        # normalisation to include.
        lambda_ = self.param("lambda", nn.initializers.constant(0.5), (1,))

        X_learned = (X - alpha.reshape(-1, 1)) / beta.reshape(-1, 1)
        X_norm = lambda_ * X_learned + (1 - lambda_) * X_sample
        assert_shape(X_norm, (D, L))

        # Now do a non-linear gating function on X_norm to see what features are
        # important or not.
        s_gamma = jnp.mean(X_norm, axis=1)
        assert_shape(s_gamma, (D,))

        W_gamma = self.param("W_gamma", nn.initializers.lecun_normal(), (D, D))
        b_gamma = self.param("b_gamma", nn.initializers.zeros, (D,))
        gamma = nn.sigmoid(jnp.dot(W_gamma, s_gamma) + b_gamma)
        assert_shape(gamma, (D,))

        X_done = gamma.reshape(-1, 1) * X_norm
        assert_shape(X_done, (D, L))

        return cast(Array, X_done)
