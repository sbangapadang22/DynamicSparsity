import torch
import torch.nn as nn
import math

class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features, sparsity_level=0.5):
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity_level = sparsity_level
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.prune_mask = nn.Parameter(torch.ones(out_features, in_features), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        pruned_weight = self.weight * self.prune_mask
        return nn.functional.linear(input, pruned_weight, self.bias)

    def apply_sparsity(self, weights):
        num_elements = weights.numel()
        k = int(self.sparsity_level * num_elements)
        k = max(1, k)  # Ensure k is at least 1

        abs_weights = weights.abs()
        threshold = abs_weights.view(-1).kthvalue(k).values.item()
        mask = abs_weights >= threshold
        sparse_weights = weights * mask.float()
        return sparse_weights

    def update_prune_mask(self):
        self.prune_mask.data = self.apply_sparsity(self.weight).ne(0).float()
