from src.layers.sparse_linear import SparseLinear

class SparsityScheduler:
    def __init__(self, model, initial_sparsity=0.5, final_sparsity=0.1, total_epochs=100):
        self.model = model
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.total_epochs = total_epochs

    def adjust_sparsity(self, epoch):
        # Linearly decrease sparsity from initial_sparsity to final_sparsity
        sparsity = self.initial_sparsity - (self.initial_sparsity - self.final_sparsity) * (epoch / self.total_epochs)
        # Update sparsity level in all sparse layers
        for layer in self.model.modules():
            if isinstance(layer, SparseLinear):
                layer.sparsity_level = sparsity
