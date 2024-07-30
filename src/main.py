import tensorflow as tf
from layers.sparse_dense import SparseDense
from scheduler.tf_sparsity_scheduler import SparsityScheduler
from training.custom_training import custom_training_loop

def create_model():
    return tf.keras.Sequential([
        tf.keras.Input(shape=(784,)),
        SparseDense(128),
        tf.keras.layers.ReLU(),
        SparseDense(10)
    ])

def main():
    # Create the model
    model = create_model()

    # Initialize the SparsityScheduler
    scheduler = SparsityScheduler(initial_sparsity=0.5, final_sparsity=0.1, total_epochs=10, criteria='linear')

    # Manually set the model for the scheduler
    scheduler.set_model(model)

    # Prepare the data
    train_data = tf.random.normal((1000, 784))
    train_labels = tf.random.uniform((1000,), maxval=10, dtype=tf.int32)
    train_data = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).shuffle(buffer_size=1000).batch(32)

    # Train the model with the custom training loop
    custom_training_loop(model, scheduler, epochs=10, train_data=train_data)

if __name__ == '__main__':
    main()
