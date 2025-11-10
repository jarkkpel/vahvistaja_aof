import pytest
import tensorflow as tf
import os
import sys

from util import set_seed

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def seed():
    """Provide a consistent seed for testing."""

    return 42

@pytest.fixture(autouse=True)
def auto_set_seed_for_ml_tests(request, seed):
    """Automatically set seed for all tests marked with 'ml'."""

    if request.node.get_closest_marker("ml"):
        set_seed(seed)

@pytest.fixture(scope="session", autouse=True)
def setup_tensorflow():
    """Configure TensorFlow for testing."""

    # Disable GPU for consistent testing.
    tf.config.set_visible_devices([], "GPU")
    # Set deterministic operations.
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
