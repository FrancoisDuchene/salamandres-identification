from typing import Tuple

from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img


class DataGenerator(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    batch_size: int
    img_size: Tuple[int, int]
    input_img_paths: list
    target_img_paths: list
    only_predict: bool

    def __init__(self, batch_size: int, img_size: Tuple[int, int],
                 input_img_paths: list, target_img_paths: list, only_predict=False
                ):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.only_predict = only_predict

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx: int):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size

        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        x: np.ndarray = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img

        y: np.ndarray = np.zeros(0)
        if self.only_predict is False:
            batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]
            y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
            for j, path in enumerate(batch_target_img_paths):
                img = load_img(path, target_size=self.img_size, color_mode="grayscale")
                y[j] = np.expand_dims(img, 2)

        # Preprocessing step, for target where we have either 0 or 255 as value, we set 255 as 1
        x = x / 255
        if self.only_predict is False:
            y = y / 255
        return x, y
