import tensorflow as tf
from layers.sparse_dense import SparseDense
from scheduler.tf_sparsity_scheduler import SparsityScheduler
from training.custom_training import custom_training_loop

def create_model():
    return tf.keras.Sequential([
        tf.keras.Input(shape=(784,)),
        SparseDense(256, kernel_initializer='he_normal'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.5),
        SparseDense(128, kernel_initializer='he_normal'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.5),
        SparseDense(64, kernel_initializer='he_normal'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.5),
        SparseDense(10, kernel_initializer='he_normal')
    ])

def main():
    # Create the model
    model = create_model()

    # Initialize the SparsityScheduler
    scheduler = SparsityScheduler(initial_sparsity=0.5, final_sparsity=0.1, total_epochs=10, criteria='linear')

    # Manually set the model for the scheduler
    scheduler.set_model(model)

    # Prepare the data
    train_data = tf.random.normal((800, 784))
    train_labels = tf.random.uniform((800,), maxval=10, dtype=tf.int32)
    val_data = tf.random.normal((200, 784))
    val_labels = tf.random.uniform((200,), maxval=10, dtype=tf.int32)
    train_data = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).map(lambda x, y: (x / 255.0, y)).shuffle(buffer_size=1000).batch(32)
    val_data = tf.data.Dataset.from_tensor_slices((val_data, val_labels)).map(lambda x, y: (x / 255.0, y)).batch(32)

    # Train the model with the custom training loop
    custom_training_loop(model, scheduler, epochs=10, train_data=train_data, val_data=val_data)

if __name__ == '__main__':
    main()
