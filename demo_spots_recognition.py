import os.path
import cv2
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import matplotlib.pyplot as plt

folder_demo = os.path.join("MandelMatcher", "ManderMatcher v02.02 DEMO WITH DATA", "ManderMatcher v02.02 DEMO WITH DATA",
                           "Images")

files = [
    os.path.join("..", folder_demo, "00007138.jpg"),
    os.path.join("..", folder_demo, "00007140.jpg")
]

window_name = 'Image'


def convert_img(src):
    # convert the image from the RGB color space to the L*a*b*
    # color space -- since we will be clustering using k-means
    # which is based on the euclidean distance, we'll use the
    # L*a*b* color space where the euclidean distance implies
    # perceptual meaning
    image = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    (h, w) = image.shape[:2]
    # reshape the image into a feature vector so that k-means
    # can be applied
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    return h, w, image


def meth_mini_batch(src):
    h, w, image = convert_img(src)

    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 3

    # apply k-means using the specified number of clusters and
    # then create the quantized image based on the predictions
    clt = MiniBatchKMeans(n_clusters=k)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]

    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))
    image = image.reshape((h, w, 3))
    # convert from L*a*b* to RGB
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    # display the images and wait for a keypress
    cv2.imshow("src", image)
    cv2.imshow(window_name, quant)
    cv2.waitKey(0)


def meth_classic_kmeans(src):
    h, w, image = convert_img(src)

    print(image)
    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image.reshape((-1, 3))
    # print(pixel_values)
    # convert to float
    pixel_values = np.float32(pixel_values)
    # print(pixel_values)

    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    # number of clusters
    k = 3

    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # convert back to 8 bit values
    centers = np.uint8(centers)

    # flatten the labels array
    labels = labels.flatten()

    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]

    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image.shape)
    # show the image
    plt.imshow(segmented_image)
    plt.show()
    # segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_LAB2RGB)
    # cv2.imshow("res", segmented_image)
    # cv2.waitKey(0)


if __name__ == '__main__':
    src1 = cv2.imread(files[0])
    src2 = cv2.imread(files[1])
    meth_classic_kmeans(src1)



