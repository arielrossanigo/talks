import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageOps

from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util


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
