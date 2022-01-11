from typing import Tuple

from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img


class Salamandre(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    batch_size: int
    img_size: Tuple[int, int]
    input_img_paths: list
    target_img_paths: list

    def __init__(self, batch_size: int, img_size: Tuple[int, int],
                 input_img_paths: list, target_img_paths: list):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx: int):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]
        x: np.ndarray = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y: np.ndarray = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        # print("pomme")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            # print(y[j].shape)
        # print("x.shape: {} | min {} | max {}".format(x.shape, x.min(), x.max()))
        # print("y.shape: {} | min {} | max {}".format(y.shape, y.min(), y.max()))
        # Preprocessing step, for target where we have either 0 or 255 as value, we set 255 as 1
        x = x / 255
        y = y / 255
        return x, y
