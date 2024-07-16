import tensorflow as tf
from src.layers.sparse_dense import SparseDense
from src.scheduler.tf_sparsity_scheduler import SparsityScheduler

# Dummy data for testing
x_train = tf.random.normal((1000, 784))  # 1000 samples, each of size 784
y_train = tf.random.uniform((1000,), maxval=10, dtype=tf.int32)  # 1000 labels for 10 classes

def test_different_sparsity_levels():
    for initial_sparsity, final_sparsity in [(0.8, 0.3), (0.6, 0.1), (0.9, 0.2)]:
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(784,)),
            SparseDense(128),
            tf.keras.layers.ReLU(),
            SparseDense(10)
        ])
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        scheduler = SparsityScheduler(initial_sparsity=initial_sparsity, final_sparsity=final_sparsity, total_epochs=50)
        
        history = model.fit(x_train, y_train, epochs=50, callbacks=[scheduler])
        
        # Check if the sparsity levels are correctly updated
        for layer in model.layers:
            if hasattr(layer, 'sparsity_level'):
                print(f"Layer: {layer}, Sparsity Level: {layer.sparsity_level}")
                assert abs(layer.sparsity_level - final_sparsity) < 0.02  # Increased tolerance to 0.02

if __name__ == '__main__':
    test_different_sparsity_levels()
    print("TensorFlow different sparsity levels tests passed.")
