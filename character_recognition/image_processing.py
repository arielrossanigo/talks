# -*- coding: utf-8 *-*
import cv2
import numpy as np

from utils import make_animation, points_to_slice, slice_to_points


def get_image(path):
    # open the image
    img = cv2.imread(path, False)
    original = img.copy()
    blur = cv2.GaussianBlur(img, (15, 15), 0)
    thresh = cv2.adaptiveThreshold(blur, 255,
                                   adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                   thresholdType=cv2.THRESH_BINARY,
                                   blockSize=3, C=1)
    return original, thresh


def get_lines(img):
    '''return a list of slices for every line of text found in the image'''
    h, w = img.shape
    _slice = points_to_slice(0, 0, h, w)
    kernel = np.ones((1, w))
    return get_objects(img, _slice, kernel, False)


def get_words(img, line):
    '''return a list of slices for every word found in the line'''
    h = line[0].stop - line[0].start
    k = np.ones((7, round(h * 0.3)))
    return get_objects(img, line, k)


def get_chars(img, word):
    '''return a list of slices for every word found in the line'''
    return get_objects(img, word, np.ones((7, 1)))


def get_objects(img, _slice, kernel, original_heigth=True):
    '''return a list of slices '''
    or0, oc0, or1, oc1 = slice_to_points(_slice)
    subgrupo = img[_slice]
    subgrupo = cv2.morphologyEx(subgrupo, cv2.MORPH_OPEN, kernel)
    new_image, contours, hierarchy = cv2.findContours(255-subgrupo,
                                                      cv2.RETR_LIST,
                                                      cv2.CHAIN_APPROX_SIMPLE)
    slices = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 15:
            if original_heigth:
                abs_slice = points_to_slice(or0, oc0 + x, or1, oc0 + x + w)
            else:
                abs_slice = points_to_slice(or0 + y, oc0 + x, or0 + y + h, oc0 + x + w)
            slices.append(abs_slice)
    slices.sort(key=lambda x: x[1].start + x[0].start)
    slices = remove_nested_slices(slices)
    return slices


def remove_nested_slices(slices):
    res = []
    for s in slices:
        if not any(slice_neested(s, o) for o in slices if s is not o):
            res.append(s)
    return res


def slice_neested(s1, s2):
    return all(x.start >= y.start and x.stop <= y.stop for x, y in zip(s1, s2))


def get_digits(img):
    for line_slice in get_lines(img):
        for char_slice in get_chars(img, line_slice):
            digit = process_char(img, char_slice)
            yield digit, char_slice


def process_char(img, char_slice):
    '''return an slice of img reshaped to (28, 28) '''
    c = img[char_slice]
    h, w = c.shape
    m = max(h, w)
    quad = np.ones((m, m), dtype=np.uint8) * 255
    quad[centered_slice(h, m), centered_slice(w, m)] = c
    c = cv2.resize(quad, (28, 28))
    c = 255 - c.flatten()
    return c


def centered_slice(original, final):
    center = final // 2
    start = center - original // 2
    stop = center + (original - original // 2)
    return slice(start, stop, None)


def show_sample(path, interval=200):
    original, img = get_image(path)
    patch_list = []
    for line in get_lines(img):
        patch_list.append((line, 'blue', ''))
        for char in get_chars(img, line):
            patch_list.append((char, 'green', ''))
    make_animation(original, patch_list, interval)


if __name__ == '__main__':
    show_sample('numbers.jpg')
