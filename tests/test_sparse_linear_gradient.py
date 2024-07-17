import torch
from src.layers.sparse_linear import SparseLinear

def test_sparse_linear_gradient():
    # Create a sample input
    input_data = torch.randn(10, 20)  # Batch of 10, input size 20
    labels = torch.randint(0, 10, (10,))  # Random labels

    # Create the SparseLinear layer and a simple model
    sparse_layer = SparseLinear(20, 30, sparsity_level=0.5)  # Input size 20, output size 30
    model = torch.nn.Sequential(
        sparse_layer,
        torch.nn.Linear(30, 10)
    )

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Train the model for a few epochs
    for epoch in range(5):
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

    # Check if gradients are correctly applied
    for param in model.parameters():
        if param.grad is not None:
            assert torch.any(param.grad != 0), "Gradient is zero, backpropagation might not be working properly."

if __name__ == '__main__':
    test_sparse_linear_gradient()
    print("PyTorch SparseLinear gradient tests passed.")
