import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

def initialize_weights(model, init_weights_func=None, **kwargs):
    """
    Initialize the weights of a model using a specified initialization function.

    Args:
        model (nn.Module): The model to initialize.
        init_weights_func (str, optional): The name of the initialization function to use. Defaults to None.

    Raises:
        ValueError: If the specified initialization function is invalid.
    """
    # If init weights_func is provided, call it to initialize the weights of the model. Otherwise, use the default initialization.
    if init_weights_func is not None:
        init_func = getattr(sys.modules[__name__], init_weights_func, None)
        if init_func is None or not callable(init_func):
            raise ValueError(f"Invalid init_weights_func: {init_weights_func}. Must be a callable function in the initialization module.")
        init_func(model, **kwargs)
    else:
        pass

def orthogonal_with_gain(model, *, activation):
    # Orthogonal init avoids vanishing forward/backward signal that default init causes at large depth
        gain = nn.init.calculate_gain('relu') if isinstance(activation, nn.ReLU) else 1.0
        for layer in model.hidden:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.orthogonal_(model.classifier.weight)

def ResNet_kaiming_normal(model, **kwargs):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def true_kaiming_normal(model, **kwargs):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def true_kaiming_uniform(model, **kwargs):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def Specific_Distribution(model, **kwargs):
    # Example: Initialize weights with a specific distribution (e.g., normal distribution)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

