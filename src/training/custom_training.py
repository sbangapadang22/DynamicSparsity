import tensorflow as tf

def custom_training_loop(model, scheduler, epochs, train_data):
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # Manually set the model in the scheduler
    scheduler.set_model(model)

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')

        # Adjust sparsity
        scheduler.on_epoch_begin(epoch)

        for step, (batch_data, batch_labels) in enumerate(train_data):
            with tf.GradientTape() as tape:
                logits = model(batch_data, training=True)
                loss_value = loss_fn(batch_labels, logits)

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            if step % 10 == 0:
                print(f'Step {step}, Loss: {loss_value.numpy()}')

        print(f'Epoch {epoch+1} Loss: {loss_value.numpy()}')

        # Debug: Print sparsity levels
        for layer in model.layers:
            if hasattr(layer, 'sparsity_level'):
                print(f'Layer {layer.name} sparsity level: {layer.sparsity_level}')
