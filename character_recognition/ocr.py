from digits_classifiers import get_predictor
from image_processing import get_digits, get_image
from utils import make_animation
import matplotlib.pylab as plt

predictor = get_predictor('Logistic regression')

original, img = get_image('numbers.png')

predictions = []
for digit, slice_ in get_digits(img):
    p = predictor.predict(digit)[0]
    predictions.append((slice_, 'green', p))

ani = make_animation(original, predictions, interval=300)
plt.show()
