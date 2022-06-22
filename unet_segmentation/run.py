import os
from os import path

from tensorflow.keras import optimizers
from tqdm import tqdm

from tensorflow.keras import metrics as kmetrics
from tensorflow.keras import losses
import keras
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import array_to_img, load_img
import pandas as pd
from keras_preprocessing.image import  ImageDataGenerator

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import PIL
from PIL import ImageOps
import cv2

import maskGen
import unet_model as um
import DataGenerator as sal
import extra_losses as exl
import extra_metrics as exm

if __name__ == '__main__':
    print("Running unet configuration")
    # maskGen.generate_masks_v3()

    params = {
        "img_size": (256, 256),
        "num_classes": 2,
        "batch_size": 15,
        "EPOCHS": 5,
        "step_per_epoch": 200,
        "learning_rate": 0.001
    }

    # source_dir = os.path.join(os.getcwd(), "images")
    source_dir = os.path.join(os.getcwd(), "images_ready")
    if not os.path.exists(source_dir):
        os.mkdir(source_dir)
    output_predictions_images_dir = os.path.join(os.getcwd(), "output_pred_images")
    if not os.path.exists(output_predictions_images_dir):
        os.mkdir(output_predictions_images_dir)
    input_img_paths = []
    target_img_paths = []
    print("Loading Dataset")
    for folder in tqdm(os.listdir(source_dir)):
        if os.path.isdir(os.path.join(source_dir, folder)):
            sample_path = os.path.join(source_dir, folder)
            contents_dir_images = sorted(os.listdir(os.path.join(sample_path, "images")))
            contents_dir_masks = sorted(os.listdir(os.path.join(sample_path, "masks")))

            for i in range(0, len(contents_dir_images)):
                img_path: str = os.path.join(sample_path,
                                             "images",
                                             os.listdir(os.path.join(sample_path, "images"))[i]
                                             )
                mask_path: str = os.path.join(sample_path,
                                              "masks",
                                              os.listdir(os.path.join(sample_path, "masks"))[i]
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
    # img = mpimg.imread(input_img_paths[4])
    # imgplot = plt.imshow(img)
    # plt.show()
    #
    # mask = mpimg.imread(target_img_paths[4])
    # maskplot = plt.imshow(mask)
    #
    # plt.show()

    print("Creating the model")
    model: keras.Model = um.make_model(img_size=params["img_size"], num_classes=params["num_classes"])

    print("Creating the validation set")
    train_gen, val_gen, val_input_img_paths, val_target_img_paths, train_input_img_paths, train_target_img_paths, \
    test_gen, test_input_img_paths, test_target_img_paths = \
        um.make_validation_set(input_img_paths, target_img_paths, params["batch_size"], params["img_size"])

    # Configure the model for training.
    # We use the "sparse" version of categorical_crossentropy
    # because our target data is integers.
    #loss_function = exl.jaccard_distance
    #loss_function_name = "jaccard"
    loss_function = losses.SparseCategoricalCrossentropy()
    loss_function_name = "sparse_cat_crossentr"
    optimizer = optimizers.Adam(learning_rate=params["learning_rate"])
    model.compile(optimizer=optimizer,
                  loss=loss_function,#"sparse_categorical_crossentropy",
                  run_eagerly=False,
                  metrics=[
                      #kmetrics.MeanIoU(num_classes=2),
                      "accuracy",
                      exm.jaccard_coef
                      # exm.iou,
                      # exm.iou_thresholded
                  ])

    # model.summary()
    callbacks = [
        ModelCheckpoint(
            "salamandres_segmentation.h5",
            save_best_only=True,
            verbose=1,
            monitor="val_loss"
        )
    ]

    # Train the model, doing validation at the end of each epoch.

    history = model.fit(train_gen, epochs=params["EPOCHS"], batch_size=params["batch_size"],
                        steps_per_epoch=params["step_per_epoch"],
                        validation_data=val_gen, callbacks=callbacks)

    # Generate predictions for all images in the test set

    test_preds = model.predict(test_gen)

    def display_mask(i, old_model=False):
        """Quick utility to display a model's prediction."""
        if old_model:
            img_d = array_to_img(test_preds[i])
        else:
            mask_d = np.argmax(test_preds[i], axis=-1)
            mask_d = np.expand_dims(mask_d, axis=-1)
            img_d = PIL.ImageOps.autocontrast(array_to_img(mask_d))
        img_d.show()
        return img_d

    # Display results for validation image #0
    i = 0

    img = PIL.ImageOps.autocontrast(load_img(test_target_img_paths[i]))
    img.show()

    display_mask(i)

    def display_result(index: int, old_model=False):
        img_source = load_img(test_input_img_paths[index])
        img_source.show()
        img_target = load_img(test_target_img_paths[index])
        img_target.show()
        img_mask = display_mask(index, old_model)
        # we need to resize the mask to its original size
        img_mask = img_mask.resize((img_source.width, img_source.height))
        # we eliminate the blurry borders of the mask to make it either white or black
        threshold = 100     # 100 to take a bit more pixels than would with an perfect half-half threshold (127)
        img_mask = img_mask.point(lambda p: 255 if p > threshold else 0)
        img_mask.show()
        img_mask_path = os.path.join(
            output_predictions_images_dir,
            test_input_img_paths[index].split("\\")[-1][:-4] + "_pred.png")
        img_mask.save(img_mask_path)
        img_mask.close()

        img_mask_cv = cv2.imread(img_mask_path, cv2.IMREAD_GRAYSCALE)
        img_source_cv = cv2.imread(test_input_img_paths[index])
        img_mask_cv = cv2.bitwise_and(img_source_cv, img_source_cv, mask=img_mask_cv)
        cv2.imwrite(img_path, img_mask_cv)

    model.save(os.path.join(os.getcwd(), "model_{}_loss_{}_lr_{}_{}_samples_{}_epochs_{}_spe_adam"
                            .format(model.name, loss_function_name, params["learning_rate"], len(input_img_paths),
                                    params["EPOCHS"], params["step_per_epoch"])))
    history_xlsx_file_path = "results_{}_loss_{}_lr_{}_totsamples{}_epo{}_{}_spe_batchsize{}_{}x{}.xlsx"\
        .format(model.name, loss_function_name, params["learning_rate"], len(input_img_paths), params["EPOCHS"],
                params["step_per_epoch"], params["batch_size"], params["img_size"][0], params["img_size"][1])
    hist_df = pd.DataFrame(history.history)
    GFG = pd.ExcelWriter(history_xlsx_file_path)
    hist_df.to_excel(GFG, index=True)
    GFG.save()
    GFG.close()
