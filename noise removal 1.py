import cv2
import matplotlib.pyplot as plt
from morphology import *

# Removing unwanted noise (arbitrarily small stars) from an image.

img = cv2.imread('noise removal 1.jpg', 0)
img1 = binarize(img)
img2 = morph(img1, 'op', 1, strel=strel('rect', (7, 7)))
plt.figure('Ex. c - noise removal')
plt.subplot(131).set_title('original image')
plt.axis('off')
plt.imshow(img, cmap='gray')
plt.subplot(132).set_title('binarized image')
plt.axis('off')
plt.imshow(img1, cmap='gray')
plt.subplot(133).set_title('opened image')
plt.axis('off')
plt.imshow(img2, cmap='gray')

plt.show()

