from utils import make_animation
from image_processing import get_image
import matplotlib.pylab as plt

def get_digits_fake(img):
    h, w = img.shape
    for c in range(h//28):
        for r in range(w//28):
            s = (slice(28*c, 28*(c+1), None), slice(28*r, 28*(r+1), None))
            digit = img[s]
            yield digit, s

def process_image():
    numbers = list(map(int, open('numbers.txt').read().split(' ')))
    o, _ = get_image('numbers.png')
    slices = [(s, 'green', n) for ((_, s), n) in zip(get_digits_fake(o), numbers)]
    ani = make_animation(o, slices, 300)
    # plt.show()
    return ani
