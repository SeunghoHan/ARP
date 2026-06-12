"""Layers for layer-wise relevance propagation.

Layers for layer-wise relevance propagation can be modified.

"""
import torch
from torch import nn
from explainer.lrp_filter import relevance_filter

class RelevancePropagationAdaptiveAvgPool2d(nn.Module):
    """Layer-wise relevance propagation for 2D adaptive average pooling.

    Attributes:
        layer: 2D adaptive average pooling layer.
        eps: A value added to the denominator for numerical stability.

    """

    def __init__(
        self,
        layer: torch.nn.AdaptiveAvgPool2d,
        mode: None,
        eps: float = 1.0e-13,
        top_k: float = 0.0,
    ) -> None:
        super().__init__()
        self.layer = layer
        self.eps = eps
        self.top_k = top_k

    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        if self.top_k:
            r = relevance_filter(r, top_k_percent=self.top_k)
        
        z = self.layer.forward(a) + self.eps
        s = (r / z).data
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        return r


class RelevancePropagationAvgPool2d(nn.Module):
    """Layer-wise relevance propagation for 2D average pooling.

    Attributes:
        layer: 2D average pooling layer.
        eps: A value added to the denominator for numerical stability.

    """

    def __init__(
        self, 
        layer: torch.nn.AvgPool2d, 
        mode: None,
        eps: float = 1.0e-13, 
        top_k: float = 0.0
    ) -> None:
        super().__init__()
        self.layer = layer
        self.eps = eps
        self.top_k = top_k

    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        if self.top_k:
            r = relevance_filter(r, top_k_percent=self.top_k)
            
        z = self.layer.forward(a) + self.eps
        s = (r / z).data
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        return r


class RelevancePropagationMaxPool2d(nn.Module):
    """Layer-wise relevance propagation for 2D max pooling.

    Attributes:
        layer: 2D max pooling layer.
        eps: a value added to the denominator for numerical stability.

    """

    def __init__(
        self,
        layer: torch.nn.MaxPool2d,
        mode: None,
        eps: float = 1.0e-13,
        top_k: float = 0.0,
    ) -> None:
        super().__init__()

        self.layer = layer
        self.eps = eps
        self.top_k = top_k

    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        if self.top_k:
            r = relevance_filter(r, top_k_percent=self.top_k)
        
        z = self.layer.forward(a) + self.eps
        s = (r / z).data
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        return r


class RelevancePropagationConv2d(nn.Module):
    """Layer-wise relevance propagation for 2D convolution.

    Optionally modifies layer weights according to propagation rule. Here z^+-rule

    Attributes:
        layer: 2D convolutional layer.
        eps: a value added to the denominator for numerical stability.

    """

    def __init__(
        self,
        layer: torch.nn.Conv2d,
        mode: str = "z_plus",
        eps: float = 1.0e-13,
        top_k: float = 0.0,
    ) -> None:
        super().__init__()

        self.layer = layer

        if mode == "z_plus":
            if self.layer.weight != None: self.layer.weight = torch.nn.Parameter(self.layer.weight.clamp(min=0.0))
            if self.layer.bias != None: self.layer.bias = torch.nn.Parameter(torch.zeros_like(self.layer.bias))

        self.eps = eps
        self.top_k = top_k

    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        if self.top_k:
            r = relevance_filter(r, top_k_percent=self.top_k)
            
        z = self.layer.forward(a) + self.eps
        s = (r / z).data
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        return r


class RelevancePropagationLinear(nn.Module):
    """Layer-wise relevance propagation for linear transformation.

    Optionally modifies layer weights according to propagation rule. Here z^+-rule

    Attributes:
        layer: linear transformation layer.
        eps: a value added to the denominator for numerical stability.

    """

    def __init__(
        self,
        layer: torch.nn.Linear,
        mode: str = "z_plus",
        eps: float = 1.0e-13,
        top_k: float = 0.0,
    ) -> None:
        super().__init__()

        self.layer = layer

        if mode == "z_plus":
            self.layer.weight = torch.nn.Parameter(self.layer.weight.clamp(min=0.0))
            self.layer.bias = torch.nn.Parameter(torch.zeros_like(self.layer.bias))

        self.eps = eps
        self.top_k = top_k

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        if self.top_k:
            r = relevance_filter(r, top_k_percent=self.top_k)
        
        z = self.layer.forward(a) + self.eps
        s = r / z
        c = torch.mm(s, self.layer.weight)
        r = (a * c).data
        return r


class RelevancePropagationFlatten(nn.Module):
    """Layer-wise relevance propagation for flatten operation.

    Attributes:
        layer: flatten layer.

    """

    def __init__(self, 
                 layer: torch.nn.Flatten, 
                 mode: None,
                 top_k: float = 0.0) -> None:
        super().__init__()
        self.layer = layer

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        r = r.view(size=a.shape)
        return r


class RelevancePropagationReLU(nn.Module):
    """Layer-wise relevance propagation for ReLU activation.

    Passes the relevance scores without modification. Might be of use later.

    """

    def __init__(self, 
                 layer: torch.nn.ReLU, 
                 mode: None,
                 top_k: float = 0.0) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        return r
    
class RelevancePropagationBatchNorm2d(nn.Module):
    def __init__(self,
                 layer: torch.nn.BatchNorm2d, 
                 mode: None,
                 top_k: float = 0.0) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        return r

class RelevancePropagationClone(nn.Module):
    def __init__(self,
                 layer: torch.clone = None, 
                 mode: str = None,
                 top_k: float = 0.0) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, r1: torch.tensor, r2: torch.tensor) -> torch.tensor:
        r = (r1+r2)/2
        # r = r1+r2
        return r
    
class RelevancePropagationAdd(nn.Module):
    def __init__(self,
                 layer: torch.add = None, 
                 mode: str = None,
                 eps: float = 1.0e-13,
                 top_k: float = 0.0) -> None:
        super().__init__()
        
        self.eps = eps

    @torch.no_grad()
    def forward(self, a1: torch.tensor, a2: torch.tensor, r: torch.tensor) -> torch.tensor:
        z = (a1 + a2) + self.eps
        s = r / z
        # r = a * s
        return a1*s, a2*s
    
# class RelevancePropagationAdd(nn.Module):
#     def __init__(self,
#                  layer: torch.add = None, 
#                  mode: str = None,
#                  top_k: float = 0.0) -> None:
#         super().__init__()

#     @torch.no_grad()
#     def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
#         """
#            Args:
#                a: a / added_a (one of the input of Add layer / output of Add layer)
#         """
#         r = a * r
#         return r 
    

class RelevancePropagationDropout(nn.Module):
    """Layer-wise relevance propagation for dropout layer.

    Passes the relevance scores without modification. Might be of use later.

    """

    def __init__(self, 
                 layer: torch.nn.Dropout, 
                 mode: None,
                 top_k: float = 0.0) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        return r


class RelevancePropagationIdentity(nn.Module):
    """Identity layer for relevance propagation.

    Passes relevance scores without modifying them.

    """

    def __init__(self,
                 layer: nn.Module, 
                 mode: None,
                 top_k: float = 0.0) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        return r