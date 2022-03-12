from tensorflow import keras
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import array_to_img
import os
import numpy as np

from DataGenerator import DataGenerator
import extra_metrics, extra_losses
from imageOperations import unet_mask_to_production_mask

batch_size = 1
img_size = (512, 512)

input_img_paths = (
    os.path.join(os.getcwd(), "prediction_image_set", "1ADF1051-3926-4B00-B818-A818B32A08AD (1).JPG"),
    os.path.join(os.getcwd(), "prediction_image_set", "1FA53FEE-6EC6-4654-8402-861C574337A8.JPG"),
    os.path.join(os.getcwd(), "prediction_image_set", "5E22B6E0-DEA9-48F0-B63A-DB86214808E7.JPG"),
)
target_img_paths = (
    os.path.join(os.getcwd(), "prediction_image_set", "00000024.png"),
    os.path.join(os.getcwd(), "prediction_image_set", "00000044.png"),
    os.path.join(os.getcwd(), "prediction_image_set", "mandeldemo_00007244.png"),
)

lol_paths = (
    os.path.join(os.getcwd(), "prediction_image_set", "00000024.jpg"),
    os.path.join(os.getcwd(), "prediction_image_set", "00000044.jpg"),
    os.path.join(os.getcwd(), "prediction_image_set", "mandeldemo_00007244.jpg"),
)


def get_test_set_for_predictions(_batch_size=batch_size, _img_size=img_size, _input_img_paths=input_img_paths,
                                 _target_img_paths=target_img_paths) -> Sequence:
    return DataGenerator(batch_size, img_size, list(_input_img_paths), list(_target_img_paths), only_predict=True)


if __name__ == "__main__":
    vidhya = "model_vidhya_u-net_3068_samples_15_epochs_adam"
    chollet = "model_3068_samples_50_epochs_adam"
    jaccard_30_epo = "model_vidhya_u-net_loss_jaccard_3068_samples_30_epochs_adam"
    jaccard_best = "model_vidhya_u-net_loss_jaccard_lr_0.001_3068_samples_15_epochs_400_stepsperepoch_adam"
    modelpath = os.path.join(os.getcwd(), jaccard_best)
    save_predictions_folder = os.path.join(os.getcwd(), "output_pred_images")

    model = keras.models.load_model(modelpath, custom_objects={
        'iou': extra_metrics.iou,
        "iou_thresholded": extra_metrics.iou_thresholded,
        "jaccard_distance": extra_losses.jaccard_distance
    })

    val_gen = get_test_set_for_predictions()
    val_preds: np.ndarray = model.predict(val_gen)

    # input_img_paths = lol_paths

    i = 0
    for val_p in val_preds:
        img_p = array_to_img(val_p)
        raw_mask_path = os.path.join(save_predictions_folder, "jac_400spe_" + input_img_paths[i].split("\\")[-1][:-4] + ".png")
        img_p.save(raw_mask_path)
        unet_mask_to_production_mask(input_img_paths[i],
                                     raw_mask_path,
                                     os.path.join(save_predictions_folder,
                                                  "masked_images",
                                                  "vidhya_jac_400spe_" + input_img_paths[i].split("\\")[-1][:-4] + ".png"
                                                  )
                                     )
        i += 1