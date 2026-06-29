import pytest
import os
from pathlib import Path
from tensorflow.keras.models import Model
import sys

# Ensure the parent directory is in sys.path so we can import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import build_multi_input_model

def test_model_architecture():
    """Smoke test to check if the multi-input model builds correctly."""
    model = build_multi_input_model()
    
    assert isinstance(model, Model), "Model should be an instance of tf.keras.models.Model"
    
    # Check if model has the two inputs (image and sex)
    assert len(model.inputs) == 2, "Model should have exactly 2 inputs"
    assert model.inputs[0].shape[1:] == (300, 300, 3), "Image input shape mismatch"
    assert model.inputs[1].shape[1:] == (1,), "Sex input shape mismatch"
    
    # Check if model output is correct (a single linear output for regression)
    assert len(model.outputs) == 1, "Model should have exactly 1 output"
    assert model.outputs[0].shape[1:] == (1,), "Model output shape mismatch"
