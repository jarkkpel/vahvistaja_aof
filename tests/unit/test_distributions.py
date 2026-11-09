import pytest
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from util import ClippedDistribution

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class TestClippedDistribution:
    """Unit tests for the ClippedDistribution class."""

    @pytest.mark.ml
    def test_clipped_beta(self):
        """Test that clipped beta distribution works appropriately."""

        low = 0.001
        high = 0.999

        # Test with extreme parameters that the bounds are respected.
        alpha = tf.constant([0.001])
        beta = tf.constant([0.001])
        beta_distr = tfd.Beta(concentration1=alpha, concentration0=beta)
        clipped_beta_distr = ClippedDistribution(
            base_distribution=beta_distr,
            low=low,
            high=high
        )
        samples = clipped_beta_distr.sample(1000)
        m = tf.reduce_min(samples).numpy()
        M = tf.reduce_max(samples).numpy()
        assert low <= m <= M <= high, f"Samples out of bounds: min={m}, max={M}."

        # Test that out-of-bounds log_probabilities return -inf.
        assert clipped_beta_distr.log_prob(tf.constant([0.0])).numpy() == -np.inf
        assert clipped_beta_distr.log_prob(tf.constant([1.0])).numpy() == -np.inf
        assert clipped_beta_distr.log_prob(tf.constant([low])).numpy() != -np.inf
        assert clipped_beta_distr.log_prob(tf.constant([high])).numpy() != -np.inf
