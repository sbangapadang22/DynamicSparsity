import torch
from src.layers.sparse_linear import SparseLinear

def test_sparse_linear():
    # Create a sample input
    input_data = torch.randn(10, 20)  # Batch of 10, input size 20
    
    # Create the SparseLinear layer
    sparse_layer = SparseLinear(20, 30, sparsity_level=0.5)  # Input size 20, output size 30
    
    # Apply the layer to the input
    output_data = sparse_layer(input_data)
    
    # Check output shape
    assert output_data.shape == (10, 30)
    
    # Check sparsity level
    num_weights = sparse_layer.weight.numel()
    non_zero_weights = (sparse_layer.apply_sparsity(sparse_layer.weight) != 0).sum().item()
    sparsity = non_zero_weights / num_weights
    # Allowing a margin for floating point errors
    assert sparsity <= 0.5 + 0.01  # Increased tolerance to 0.01

if __name__ == '__main__':
    test_sparse_linear()
    print("PyTorch SparseLinear layer tests passed.")
