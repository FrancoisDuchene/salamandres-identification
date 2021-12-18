import os
from os import path

import keras
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import array_to_img, load_img
from keras_preprocessing.image import  ImageDataGenerator

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL
from PIL import ImageOps

import maskGen
import unet_model as um
import Salamandres as sal

if __name__ == '__main__':
    print("Running unet configuration")
    # maskGen.generate_masks_v3()

    img_size = (256, 256)
    num_classes = 2
    batch_size = 20

    # source_dir = os.path.join(os.getcwd(), "images")
    source_dir = os.path.join(os.getcwd(), "images_ready")
    input_img_paths = []
    target_img_paths = []
    for folder in os.listdir(source_dir):
        if os.path.isdir(os.path.join(source_dir, folder)):
            sample_path = os.path.join(source_dir, folder)
            img_path: str = os.path.join(sample_path,
                                         "images",
                                         os.listdir(os.path.join(sample_path, "images"))[0]
                                         )
            mask_path: str = os.path.join(sample_path,
                                          "masks",
                                          os.listdir(os.path.join(sample_path, "masks"))[0]
                                          )
            # Both img and mask are at the same index in both lists
            input_img_paths.append(img_path)
            target_img_paths.append(mask_path)
    sorted(input_img_paths)
    sorted(target_img_paths)

    print("Number of samples:", len(input_img_paths))

    # for input_path, target_path in zip(input_img_paths[:10], target_img_paths[:10]):
    #     print(input_path, "|", target_path)

    # Show image number 7 as for example
    img = mpimg.imread(input_img_paths[4])
    imgplot = plt.imshow(img)
    plt.show()

    mask = mpimg.imread(target_img_paths[4])
    maskplot = plt.imshow(mask)

    plt.show()

    print("Creating the model")
    model: keras.Model = um.make_model(img_size=img_size, num_classes=num_classes)
    # model.summary()
    # exit(0)
    print("Creating the validation set")
    train_gen, val_gen, val_input_img_paths, val_target_img_paths = \
        um.make_validation_set(input_img_paths, target_img_paths, batch_size, img_size)

    # Configure the model for training.
    # We use the "sparse" version of categorical_crossentropy
    # because our target data is integers.
    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy",
                  run_eagerly=False,
                  metrics=[
                      "MeanSquaredError"
                  ])

    # model.summary()
    callbacks = [
        ModelCheckpoint("salamandres_segmentation.h5", save_best_only=True)
    ]

    # Train the model, doing validation at the end of each epoch.
    EPOCHS = 15
    model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen, callbacks=callbacks)

    # Generate predictions for all images in the validation set

    val_gen = sal.Salamandre(batch_size=batch_size, img_size=img_size,
                             input_img_paths=val_input_img_paths,
                             target_img_paths=val_target_img_paths)
    val_preds = model.predict(val_gen)

    def display_mask(i):
        """Quick utility to display a model's prediction."""
        mask_d = np.argmax(val_preds[i], axis=-1)
        mask_d = np.expand_dims(mask_d, axis=-1)
        img_d = PIL.ImageOps.autocontrast(array_to_img(mask_d))
        img_d.show()
        return img_d

    # Display results for validation image #7
    i = 0

    img = PIL.ImageOps.autocontrast(load_img(val_target_img_paths[i]))
    img.show()

    display_mask(i)
