import os
import time

import tensorflow as tf
# Keras
from keras import Input, Model
from keras.layers import add, Conv2D, BatchNormalization, Activation, SeparableConv2D, MaxPooling2D, Conv2DTranspose, \
    UpSampling2D

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


def train_model(model_path, train_data, validation_batches, epochs, num_training_steps, num_val_steps, segmentation):
    """ Trains the CNN model

    Args:
        model_path (str): CNN model to be trained. tf.h5
        train_data (tf.Dataset): Batches of training images
        epochs (int): Number of epochs during training
        validation_batches (tf.Dataset): Batches of validation images
        num_training_steps (int): Number of training steps per epoch
        num_val_steps (int): Number of steps during validation
        segmentation (bool):
    """
    model = tf.keras.models.load_model(model_path)
    if segmentation:
        # ToDo: apply new callbacks
        loss = 'sparse_categorical_crossentropy'
    else:
        loss = 'categorical_crossentropy'

    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    my_callbacks = keras_callbacks(model_path)

    model.fit(train_data,
              steps_per_epoch=num_training_steps,
              epochs=epochs,
              validation_data=validation_batches,
              validation_steps=num_val_steps,
              callbacks=my_callbacks,
              verbose=1
              )


def get_model(img_size, num_classes):
    inputs = Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = Activation("relu")(x)
        x = SeparableConv2D(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = SeparableConv2D(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = Activation("relu")(x)
        x = Conv2DTranspose(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = Conv2DTranspose(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = UpSampling2D(2)(x)

        # Project residual
        residual = UpSampling2D(2)(previous_block_activation)
        residual = Conv2D(filters, 1, padding="same")(residual)
        x = add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = Model(inputs, outputs)
    model.save('unet.h5')


def up_sample(filters, filter_size, norm_type='batchnorm', apply_dropout=False):
    """ Upsamples an input.
    Conv2DTranspose => Batchnorm => Dropout => Relu
    Args:
        filters (int): number of filters
        filter_size (int): filter size
        norm_type (str): Normalization type; either 'batchnorm' or 'instancenorm'.
        apply_dropout (bool): If True, adds the dropout layer
    Returns:
        up_sample_layer: Upsample Sequential Model
    """

    kernel_initializer = tf.random_normal_initializer(0., 0.02)

    up_sample_layer = tf.keras.Sequential()
    up_sample_layer.add(tf.keras.layers.Conv2DTranspose(filters,
                                                        filter_size,
                                                        strides=2,
                                                        padding='same',
                                                        kernel_initializer=kernel_initializer,
                                                        use_bias=False))

    if norm_type.lower() == 'batchnorm':
        up_sample_layer.add(tf.keras.layers.BatchNormalization())

    # elif norm_type.lower() == 'instancenorm':
    #    result.add(InstanceNormalization())

    if apply_dropout:
        up_sample_layer.add(tf.keras.layers.Dropout(0.5))

    up_sample_layer.add(tf.keras.layers.ReLU())

    return up_sample_layer


def unet_mobilenet(img_height, img_width, input_channels, output_channels):
    """
    Args:
        img_height:
        img_width:
        input_channels:
        output_channels:

    Returns:

    """

    FILTER_SIZE = 3
    layer_names = (
        'block_1_expand_relu',  # 64x64
        'block_3_expand_relu',  # 32x32
        'block_6_expand_relu',  # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',  # 4x4
    )

    input_shape = (img_height, img_width, input_channels)
    #
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False)
    # Encoder layers
    layers = [base_model.get_layer(name).output for name in layer_names]
    # Encoder model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
    down_stack.trainable = False

    # Inputs
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs

    # Downsampling
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Decoder layers
    up_stack = [up_sample(512, 3),  # 4x4 -> 8x8
                up_sample(256, 3),  # 8x8 -> 16x16
                up_sample(128, 3),  # 16x16 -> 32x32
                up_sample(64, 3),  # 32x32 -> 64x64
                ]

    # Upsampling and setting skip-connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    output_layer = tf.keras.layers.Conv2DTranspose(output_channels, FILTER_SIZE, strides=2, padding='same',
                                                   activation='softmax')  # 64x64 -> 128x128
    x = output_layer(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.save('unet_mobile_v2.h5')


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
