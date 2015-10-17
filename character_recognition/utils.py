import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

import numpy as np
import math
import cv2


def show_images(images, new_shape=None, columns=None):
    f = plt.figure()
    if columns is not None:
        columns = columns
        rows = math.ceil(len(images) / columns)
    else:
        columns = rows = math.ceil(math.sqrt(len(images)))

    for i, image in enumerate(images):
        if new_shape:
            image = image.reshape(new_shape)
        ax = f.add_subplot(rows, columns, 1 + i)
        ax.imshow(image, cmap=cm.Greys_r)
        plt.xticks([])
        plt.yticks([])
        plt.subplots_adjust(hspace=.02, wspace=.02)

    plt.show()


def slice_to_points(_slice):
    r, c = _slice[:2]
    r0 = r.start
    r1 = r.stop
    c0 = c.start
    c1 = c.stop
    return (r0, c0, r1, c1)


def points_to_slice(r0, c0, r1, c1):
    return (slice(r0, r1, None), slice(c0, c1, None))


def make_animation(img, patch_list, interval=200):

    def add_patch(num, plot, patch_list):
        _slice, color, char = patch_list[num]
        or0, oc0, or1, oc1 = slice_to_points(_slice)
        p = patches.Rectangle((oc0, or0), oc1 - oc0, or1 - or0, fc='none',
                              ec=color)
        plot.add_patch(p)
        plot.text(oc0, or0, char, fontsize=30, fontweight='bold', color=color,
                  bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap=cm.Greys_r)

    ani = animation.FuncAnimation(
        fig, add_patch, len(patch_list),  fargs=(ax, patch_list),
        interval=interval, blit=False
    )
    return ani
