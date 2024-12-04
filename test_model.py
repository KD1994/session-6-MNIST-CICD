import pytest
import torch.nn as nn
from model import MnistNet

@pytest.fixture
def model():
    return MnistNet()

def test_total_parameter_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 20000, "Model has more than 20000 parameters"

def test_use_of_batch_normalization(model):
    batch_norm_layers = [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
    assert len(batch_norm_layers) > 0, "Model has no Batch Normalization layers"

def test_use_of_dropout(model):
    dropout_layers = [module for module in model.modules() if isinstance(module, nn.Dropout)]
    assert len(dropout_layers) > 0, "Model has no Dropout layers"

def test_use_of_fully_connected_or_gap(model):
    # Check for fully connected layer
    fully_connected_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
    # Check for Global Average Pooling
    gap_layers = [module for module in model.modules() if isinstance(module, nn.AvgPool2d) or isinstance(module, nn.AdaptiveAvgPool2d)]
    
    assert len(fully_connected_layers) > 0 or len(gap_layers) > 0, "Model should use a Fully Connected Layer or Global Average Pooling"
