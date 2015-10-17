from digits_classifiers import get_dataset, get_predictor
import cv2
import numpy as np

pred = get_predictor('Logistic regression')

imgs, labels = get_dataset("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", 10000)

rows = 5
columns = 10
filename = 'numbers.png'


images = []
i = 0
while len(images) < rows * columns:
    image = imgs[i]
    p = pred.predict(image)[0]

    if (p == labels[i]):
        images.append(image)
    i += 1

images = [x.reshape(28, 28) for x in imgs[:rows*columns]]

final = None
for i in range(rows):
    r = np.concatenate(images[columns*i:columns*(i+1)], axis=1)
    if final is not None:
        final = np.concatenate([final, r], axis=0)
    else:
        final = r

cv2.imwrite(filename, 255 - final)
