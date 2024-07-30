import tensorflow as tf
from src.layers.sparse_dense import SparseDense

def test_sparse_dense_forward_backward():
    # Create a sample input
    input_data = tf.random.normal((10, 20))  # Batch of 10, input size 20
    labels = tf.random.uniform((10, 1), maxval=10, dtype=tf.int32)  # Random labels

    # Create the SparseDense layer and a simple model
    sparse_layer = SparseDense(30, sparsity_level=0.5)  # Output size 30
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(20,)),
        sparse_layer,
        tf.keras.layers.Dense(10)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    # Check forward pass
    output = model(input_data)
    assert output.shape == (10, 10), "Forward pass output shape is incorrect."

    # Train the model for a few epochs34e
    model.fit(input_data, labels, epochs=5)

    # Check if gradients are correctly applied
    with tf.GradientTape() as tape:
        predictions = model(input_data, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)

    for grad in gradients:
        assert tf.reduce_any(tf.not_equal(grad, 0)), "Gradient is zero, backpropagation might not be working properly."

if __name__ == '__main__':
    test_sparse_dense_forward_backward()
    print("TensorFlow SparseDense forward and backward tests passed.")
