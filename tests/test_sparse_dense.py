import tensorflow as tf
from src.layers.sparse_dense import SparseDense

def test_sparse_dense():
    # Create a sample input
    input_data = tf.random.normal((10, 20))  # Batch of 10, input size 20
    
    # Create the SparseDense layer
    sparse_layer = SparseDense(30, sparsity_level=0.5)  # Output size 30
    
    # Build the layer (initialize weights)
    sparse_layer.build(input_data.shape)
    
    # Apply the layer to the input
    output_data = sparse_layer(input_data, training=True)
    
    # Check output shape
    assert output_data.shape == (10, 30)
    
    # Check sparsity level
    num_weights = tf.size(sparse_layer.w).numpy()
    non_zero_weights = tf.math.count_nonzero(sparse_layer.apply_sparsity(sparse_layer.w)).numpy()
    sparsity = non_zero_weights / num_weights
    assert sparsity <= 0.5

if __name__ == '__main__':
    test_sparse_dense()
    print("TensorFlow SparseDense layer tests passed.")
