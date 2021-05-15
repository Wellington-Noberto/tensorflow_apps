import os
import time

import tensorflow as tf
# Keras

import coremltools
from src.data_io import split_images


def keras_callbacks(model_path):
    """ Defines the keras callbacks during the model training

        ModelCheckpoint: Saves the best model
        EarlyStopping: Stops the training if the models stop improving
        TensorBoard: Store the training information for TensorBoard visualization
    Args:
        model_path (str): CNN model to be trained. tf.h5

    Returns:
        my_callbacks (list of callbacks): List of callbacks
    """
    run_logdir = os.path.join('logs', time.strftime("run_%Y_%m_%d-%H_%M_%S"))
    best_model = model_path[:-3] + '{epoch:02d}-{val_loss:.2f}.h5'
    # ModelCheckpoint
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(best_model, monitor='val_loss', save_best_only=True)
    # Early Stopping
    early_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, mode='auto')
    # tensorboard --logdir=./logs --port=6006
    tensorboard_callback = tf.keras.callbacks.TensorBoard(run_logdir, update_freq=15)

    my_callbacks = [checkpoint_callback, early_callback, tensorboard_callback]

    return my_callbacks


def train_model(model_path, train_data, epochs, validation_batches, num_training_steps, num_val_steps):
    """ Trains the CNN model

    Args:
        model_path (str): CNN model to be trained. tf.h5
        train_data (tf.Dataset): Batches of training images
        epochs (int): Number of epochs during training
        validation_batches (tf.Dataset): Batches of validation images
        num_training_steps (int): Number of training steps per epoch
        num_val_steps (int): Number of steps during validation
    """
    model = tf.keras.models.load_model(model_path)
    model_name = os.path.basename(model_path)[:-3]
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    my_callbacks = keras_callbacks(model_path)

    model.fit(train_data,
              steps_per_epoch=num_training_steps,
              epochs=epochs,
              validation_data=validation_batches,
              validation_steps=num_val_steps,
              callbacks=my_callbacks,
              verbose=1
              )


def convert_cnn(model_path, model_format):
    """ Converts a .h5 model to .tflite or .mlmodel file

    Args:
        model_path (tf.h5): CNN model that should be converted
        model_format (string): Desired format for the model, should be: 'tflite' or 'mlmodel'
    Raises:
        ValueError: If model_format is not one of 'tflite' or 'mlmodel'
    """
    format_options = ('\ntflite', '\nmlmodel')
    if model_format not in ('tflite', 'mlmodel'):
        raise ValueError('preprocess_id has to be one of {0}'
                         ''.format(format_options))

    if model_format == 'tflite':
        # Load .h5 model
        model = tf.keras.models.load_model(model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        # Save .tflite model
        open(model_path.replace(".h5", ".tflite"), "wb").write(tflite_model)
    elif model_format == 'mlmodel':
        # Load .h5 model
        model = tf.keras.models.load_model(model_path)
        model = coremltools.convert(model, input_names=['image'], image_input_names='image')
        # Save .mlmodel model
        model.save(model_path.replace(".h5", ".mlmodel"))
