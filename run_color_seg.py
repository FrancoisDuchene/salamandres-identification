import color_segmentation as cs
import os.path

if __name__ == '__main__':
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
