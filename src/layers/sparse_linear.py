import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features, sparsity_level=0.5):
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity_level = sparsity_level
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, input):
        if self.training:
            # Apply sparsity during training
            sparse_weight = self.apply_sparsity(self.weight)
            return F.linear(input, sparse_weight, self.bias)
        else:
            return F.linear(input, self.weight, self.bias)

    def apply_sparsity(self, weights):
        num_elements = weights.numel()
        k = int(self.sparsity_level * num_elements)
        k = min(k, num_elements)
        k = max(k, 1)  # Ensure k is at least 1

        abs_weights = torch.abs(weights)
        top_k_values, _ = torch.topk(abs_weights.view(-1), k)
        threshold = top_k_values[-1]
        mask = abs_weights >= threshold
        sparse_weights = torch.where(mask, weights, torch.zeros_like(weights))
        return sparse_weights
