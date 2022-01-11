import os
from typing import Tuple
import random

import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Lambda
from tensorflow.keras import layers
import Salamandres

RANDOM_SEED: int = 2021


def make_model(img_size: Tuple[int, int], num_classes: int = 2) -> keras.Model:
    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()

    inputs = keras.Input(shape=img_size + (3,))

    # Rescaling to have an input in range [0, 255] to be in the [0,1] range
    # x = Rescaling(1./255, 0.0)(inputs)
    # x = Lambda(lambda x: x/255.0)(inputs)

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs, name="u-net")

    # model.summary()
    return model


def make_validation_set(input_img_paths: list, target_img_paths: list,
                        batch_size: int, img_size: Tuple[int, int]):
    # Split our img paths into a training and a validation set
    total_nb_of_samples = len(input_img_paths)
    # We take one fifth of the total to make our validation sample
    val_samples = total_nb_of_samples // 5
    print("Total Samples : {} | Validation Set Samples : {}".format(total_nb_of_samples, val_samples))
    random.Random(RANDOM_SEED).shuffle(input_img_paths)
    random.Random(RANDOM_SEED).shuffle(target_img_paths)
    train_input_img_paths = input_img_paths[:-val_samples]
    train_target_img_paths = target_img_paths[:-val_samples]
    val_input_img_paths = input_img_paths[-val_samples:]
    val_target_img_paths = target_img_paths[-val_samples:]

    # Instantiate data Sequences for each split
    train_gen = Salamandres.Salamandre(
        batch_size, img_size, train_input_img_paths, train_target_img_paths
    )
    val_gen = Salamandres.Salamandre(batch_size, img_size, val_input_img_paths, val_target_img_paths)

    return train_gen, val_gen, val_input_img_paths, val_target_img_paths
