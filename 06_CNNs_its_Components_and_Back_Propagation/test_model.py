import torch
import sys
import pytest
from model import Net

def count_parameters(model):
    """Count the total number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_parameter_count():
    """Test if model has less than 20k parameters"""
    model = Net()
    param_count = count_parameters(model)
    assert param_count < 20000, f"Model has {param_count} parameters, which is more than 20k"

def test_batch_norm_usage():
    """Test if model uses batch normalization"""
    model = Net()
    has_batchnorm = any(isinstance(module, torch.nn.BatchNorm2d) for module in model.modules())
    assert has_batchnorm, "Model doesn't use Batch Normalization"

def test_dropout_usage():
    """Test if model uses dropout"""
    model = Net()
    has_dropout = any(isinstance(module, torch.nn.Dropout) for module in model.modules())
    assert has_dropout, "Model doesn't use Dropout"

def test_gap_or_fc():
    """Test if model uses either Global Average Pooling or Fully Connected layers"""
    model = Net()
    has_gap = any(isinstance(module, (torch.nn.AvgPool2d, torch.nn.AdaptiveAvgPool2d)) 
                 for module in model.modules())
    has_fc = any(isinstance(module, torch.nn.Linear) for module in model.modules())
    assert has_gap or has_fc, "Model doesn't use either Global Average Pooling or Fully Connected layers"

if __name__ == "__main__":
    pytest.main([__file__]) 