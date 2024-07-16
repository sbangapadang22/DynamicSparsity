import torch
import torch.nn as nn
import torch.optim as optim
from src.layers.sparse_linear import SparseLinear

class GradientSparsityScheduler:
    def __init__(self, model, initial_sparsity=0.5, final_sparsity=0.1, total_epochs=100):
        self.model = model
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.total_epochs = total_epochs

    def adjust_sparsity(self, epoch):
        sparsity = self.initial_sparsity - (self.initial_sparsity - self.final_sparsity) * (epoch / self.total_epochs)
        for layer in self.model.modules():
            if isinstance(layer, SparseLinear):
                layer.sparsity_level = sparsity

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

def test_gradient_sparsity_scheduler():
    model = SparseModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    scheduler = GradientSparsityScheduler(model, initial_sparsity=0.5, final_sparsity=0.1, total_epochs=50)

    for epoch in range(50):
        scheduler.adjust_sparsity(epoch)
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Check if the sparsity levels are correctly updated
    final_sparsity = scheduler.final_sparsity
    for layer in model.modules():
        if isinstance(layer, SparseLinear):
            print(f"Layer: {layer}, Sparsity Level: {layer.sparsity_level}")
            assert abs(layer.sparsity_level - final_sparsity) < 1e-2  # Tolerance

if __name__ == '__main__':
    test_gradient_sparsity_scheduler()
    print("PyTorch GradientSparsityScheduler tests passed.")
