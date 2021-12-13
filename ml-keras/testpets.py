import os

import numpy as np
from IPython.display import Image, display
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import layers
import PIL
from PIL import ImageOps
import random
import oxford_pets

def unet_model():


    def get_model(img_size, num_classes):
        inputs = keras.Input(shape=img_size + (3,))

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
        model = keras.Model(inputs, outputs)
        return model

    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()

    # Build model
    model = get_model(img_size, num_classes)
    model.summary()
    return model

def make_validation_set():
    # Split our img paths into a training and a validation set
    val_samples = 1000
    random.Random(1337).shuffle(input_img_paths)
    random.Random(1337).shuffle(target_img_paths)
    train_input_img_paths = input_img_paths[:-val_samples]
    train_target_img_paths = target_img_paths[:-val_samples]
    val_input_img_paths = input_img_paths[-val_samples:]
    val_target_img_paths = target_img_paths[-val_samples:]

    # Instantiate data Sequences for each split
    train_gen = oxford_pets.OxfordPets(
        batch_size, img_size, train_input_img_paths, train_target_img_paths
    )
    val_gen = oxford_pets.OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)

    return train_gen, val_gen, val_input_img_paths, val_target_img_paths


if __name__ == '__main__':

    input_dir = "images/"
    target_dir = "annotations/trimaps/"
    img_size = (160, 160)
    num_classes = 3
    batch_size = 32

    input_img_paths = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.endswith(".jpg")
        ]
    )
    target_img_paths = sorted(
        [
            os.path.join(target_dir, fname)
            for fname in os.listdir(target_dir)
            if fname.endswith(".png") and not fname.startswith(".")
        ]
    )

    print("Number of samples:", len(input_img_paths))

    for input_path, target_path in zip(input_img_paths[:10], target_img_paths[:10]):
        print(input_path, "|", target_path)

    # Display input image #7
    display(Image(filename=input_img_paths[9]))

    # Display auto-contrast version of corresponding target (per-pixel categories)
    img = PIL.ImageOps.autocontrast(load_img(target_img_paths[9]))
    display(img)

    model = unet_model()
    train_gen, val_gen, val_input_img_paths, val_target_img_paths = make_validation_set()

    # Configure the model for training.
    # We use the "sparse" version of categorical_crossentropy
    # because our target data is integers.
    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

    callbacks = [
        keras.callbacks.ModelCheckpoint("oxford_segmentation.h5", save_best_only=True)
    ]

    # Train the model, doing validation at the end of each epoch.
    epochs = 15
    model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)

    # Generate predictions for all images in the validation set

    val_gen = oxford_pets.OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)
    val_preds = model.predict(val_gen)


    def display_mask(i):
        """Quick utility to display a model's prediction."""
        mask = np.argmax(val_preds[i], axis=-1)
        mask = np.expand_dims(mask, axis=-1)
        img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
        img.show()


    # Display results for validation image #10
    i = 10

    # Display input image
    display(Image(filename=val_input_img_paths[i]))
    # Display ground-truth target mask
    img = PIL.ImageOps.autocontrast(load_img(val_target_img_paths[i]))
    # display(img)
    img.show()
    # Display mask predicted by our model
    display_mask(i)  # Note that the model only sees inputs at 150x150.

