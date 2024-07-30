import tensorflow as tf

def custom_training_loop(model, scheduler, epochs, train_data, train_labels):
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    
    # Manually set the model in the scheduler
    scheduler.set_model(model)

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        
        # Adjust sparsity
        scheduler.on_epoch_begin(epoch)
        
        for step, (batch_data, batch_labels) in enumerate(zip(train_data, train_labels)):
            with tf.GradientTape() as tape:
                logits = model(batch_data, training=True)
                loss_value = loss_fn(batch_labels, logits)
            
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
        print(f'Loss: {loss_value.numpy()}')
