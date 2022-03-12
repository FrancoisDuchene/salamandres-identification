import color_segmentation as cs
import os
import os.path
from tqdm import tqdm

def test_color_seg():
    input_folder = "input_images"
    files = [
        "00007138.jpg",
        "00007169.jpg",
        "s3_salam_1_iphone_2.jpeg"
    ]
    color_spaces = ["HSV", "lab", "rgb", "ycc"]
    channels = ["all", "01", "02", "12"]
    num_clusters = [2, 3, 5, 10]
    for image in files:
        print(image)
        print(os.path.join(input_folder, image))
        for space in color_spaces:
            for channel in channels:
                for k in num_clusters:
                    cs.main(image=os.path.join(input_folder, image),
                            width=480, color_space=space, channels=channel,
                            num_clusters=k, output_file=True, output_format="png",
                            output_color="gray",
                            verbose=True, show_results="false")


def color_segmentation(inputs: list):
    color_space = "lab"
    channels = "12"
    num_clusters = 2
    width = 1024    # un bon compromis vu que les images N&B ne prennent pas beaucoup de place
    for image in tqdm(inputs):
        cs.main(image=image, width=width, color_space=color_space, channels=channels,
                num_clusters=num_clusters, output_file=True, output_format="png",
                output_color="gray", verbose=False, show_results="false")


if __name__ == '__main__':
    # test_color_seg()
    source_dir = os.path.join(os.getcwd(), "unet_segmentation", "segmented_images_trainset")
    inputs_paths = []
    print("Loading Dataset")
    for file in tqdm(os.listdir(source_dir)):
        filename_complete = os.path.join(source_dir, file)
        if not os.path.isdir(os.path.join(source_dir, file)):
            inputs_paths.append(filename_complete)

    color_segmentation(inputs_paths)
