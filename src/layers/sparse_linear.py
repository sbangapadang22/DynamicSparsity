import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SparseLinear(nn.Module):
    
    # initializing number of input features, output features, and fractions of weights to train
    def __init__(self, in_features, out_features, sparsity_level=0.5):
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity_level = sparsity_level
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    # initializes weights and biases
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    # applies sparsities to weights before the linear transformation
    def forward(self, input):
        if self.training:
            sparse_weight = self.apply_sparsity(self.weight)
            return F.linear(input, sparse_weight, self.bias)
        else:
            return F.linear(input, self.weight, self.bias)

    # prunes weights with the smallest magnitudes
    def apply_sparsity(self, weights):
        k = int(self.sparsity_level * weights.numel())
        abs_weights = torch.abs(weights)
        threshold = torch.kthvalue(abs_weights.view(-1), k).values.item()
        mask = abs_weights >= threshold
        sparse_weights = torch.where(mask, weights, torch.zeros_like(weights))
        return sparse_weights
