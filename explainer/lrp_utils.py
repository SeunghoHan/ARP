"""Script with helper function."""
from explainer.lrp_layers import *

class Clone(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.clone().detach()
    
class Add(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return torch.add(x1, x2)


def layers_lookup() -> dict:
    """Lookup table to map network layer to associated LRP operation.

    Returns:
        Dictionary holding class mappings.
    """
    lookup_table = {
        torch.nn.modules.linear.Linear: RelevancePropagationLinear,
        torch.nn.modules.conv.Conv2d: RelevancePropagationConv2d,
        torch.nn.modules.activation.ReLU: RelevancePropagationReLU,
        torch.nn.modules.dropout.Dropout: RelevancePropagationDropout,
        torch.nn.modules.flatten.Flatten: RelevancePropagationFlatten,
        torch.nn.modules.pooling.AvgPool2d: RelevancePropagationAvgPool2d,
        torch.nn.modules.pooling.MaxPool2d: RelevancePropagationMaxPool2d,
        torch.nn.modules.pooling.AdaptiveAvgPool2d: RelevancePropagationAdaptiveAvgPool2d,
        torch.nn.modules.batchnorm.BatchNorm2d: RelevancePropagationBatchNorm2d,        
        Clone().__class__.__name__: RelevancePropagationClone,
        Add().__class__.__name__: RelevancePropagationAdd
    }
    
    return lookup_table


