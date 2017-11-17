import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from PIL import Image, ImageOps

import keras
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util


def show_weights(layer):
    nrow = 2
    ncol = 5
    w = layer.get_weights()[0]
    plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(nrow, ncol, wspace=0.1, hspace=0.1)
    for i in range(nrow):
        for j in range(ncol):
            im = w[:, i * 5 + j]
            ax = plt.subplot(gs[i, j])
            ax.imshow(im.reshape(28, 28), cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
    plt.show()


def get_mnist_demo():
    # the data, shuffled and split between train and test sets
    (x_train, y_train), _ = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_train = x_train.astype('float32')
    x_train /= 255

    x_train = x_train[:1000]
    y_train = y_train[:1000]

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)

    model = Sequential()
    model.add(Dense(10, activation='softmax', input_shape=(784,)))
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    return model, x_train, y_train


def load_labels(path_to_labels, num_classes):
    label_map = label_map_util.load_labelmap(path_to_labels)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=num_classes, use_display_name=True
    )
    return label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def open_image(image_path):
    image = Image.open(image_path)
    w, h = image.size
    ratio = w / 1350
    image = image.resize((1350, int(h // ratio)))
    image = ImageOps.expand(image, border=20, fill='white')
    return load_image_into_numpy_array(image)


def visualize_image(image_np, boxes, classes, scores, category_index, min_score_thresh=.5):
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np, np.squeeze(boxes), np.squeeze(classes).astype(np.int32),
        np.squeeze(scores), category_index, use_normalized_coordinates=True,
        line_thickness=8, min_score_thresh=min_score_thresh
    )

    f = plt.figure(figsize=(12, 8))
    plt.imshow(image_np)
    ax = f.get_axes()[0]
    ax.set_xticks([])
    ax.set_yticks([])
