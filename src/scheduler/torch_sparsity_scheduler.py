from src.layers.sparse_linear import SparseLinear

class SparsityScheduler:
    def __init__(self, model, initial_sparsity=0.5, final_sparsity=0.1, total_epochs=100, criteria='linear', weight_threshold=0.1):
        self.model = model
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.total_epochs = total_epochs
        self.criteria = criteria
        self.weight_threshold = weight_threshold

    def adjust_sparsity(self, epoch):
        if self.criteria == 'linear':
            # Linearly decrease sparsity from initial_sparsity to final_sparsity
            sparsity = self.initial_sparsity - (self.initial_sparsity - self.final_sparsity) * (epoch / self.total_epochs)
        elif self.criteria == 'weight_magnitude':
            # Adjust sparsity based on weight magnitude
            sparsity = self.adjust_sparsity_based_on_weights()
        else:
            raise ValueError(f"Unknown sparsity adjustment criteria: {self.criteria}")

        # Update sparsity level in all sparse layers
        for layer in self.model.modules():
            if isinstance(layer, SparseLinear):
                layer.sparsity_level = sparsity

    def adjust_sparsity_based_on_weights(self):
        total_weights = 0
        prune_weights = 0

        for layer in self.model.modules():
            if isinstance(layer, SparseLinear):
                weights = layer.weight.data
                total_weights += weights.numel()
                prune_weights += (weights < self.weight_threshold).sum().item()

        sparsity = prune_weights / total_weights
        return sparsity
