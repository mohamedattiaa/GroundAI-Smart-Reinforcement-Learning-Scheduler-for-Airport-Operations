"""
TensorFlow compatibility fix for stable-baselines3 integration.
Run this import before importing stable_baselines3 or tensorboard.
"""

import os
import sys
import warnings

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging

try:
    import tensorflow as tf
    # Ensure TensorFlow 2.x behavior
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    # Check TensorFlow version
    tf_version = tuple(map(int, tf.__version__.split('.')[:2]))
    if tf_version < (2, 10):
        warnings.warn(
            f"TensorFlow {tf.__version__} detected. "
            "For best compatibility with stable-baselines3, use TensorFlow >= 2.10"
        )
except ImportError:
    print("TensorFlow not installed. Some features may be unavailable.")

# Torch/PyTorch compatibility
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
except ImportError:
    print("PyTorch not installed.")

print("✓ TensorFlow compatibility layer loaded successfully")