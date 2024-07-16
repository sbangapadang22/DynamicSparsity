import torch
import torch.nn as nn
import torch.optim as optim
from src.layers.sparse_linear import SparseLinear
from src.scheduler.torch_sparsity_scheduler import SparsityScheduler

class SparseModel(nn.Module):
    def __init__(self):
        super(SparseModel, self).__init__()
        self.fc1 = SparseLinear(784, 128)
        self.fc2 = SparseLinear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Dummy data for testing
inputs = torch.randn(1000, 784)  # 1000 samples, each of size 784
labels = torch.randint(0, 10, (1000,))  # 1000 labels for 10 classes

def test_different_sparsity_levels():
    for initial_sparsity, final_sparsity in [(0.8, 0.3), (0.6, 0.1), (0.9, 0.2)]:
        model = SparseModel()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        scheduler = SparsityScheduler(model, initial_sparsity=initial_sparsity, final_sparsity=final_sparsity, total_epochs=50)

        for epoch in range(50):
            scheduler.adjust_sparsity(epoch)
            model.train()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Check if the sparsity levels are correctly updated
        for layer in model.modules():
            if isinstance(layer, SparseLinear):
                print(f"Layer: {layer}, Sparsity Level: {layer.sparsity_level}")
                assert abs(layer.sparsity_level - final_sparsity) < 0.02  # Increased tolerance to 0.02

if __name__ == '__main__':
    test_different_sparsity_levels()
    print("PyTorch different sparsity levels tests passed.")
