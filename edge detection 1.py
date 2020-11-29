import cv2
import matplotlib.pyplot as plt
from morphology import *

# Simple edge detection for a binary image.

b = cv2.imread('edge detection 1.png', 0)
img = morph(b, 'er', 1)
img1 = b - img

plt.figure('Ex. a - edge detection 1')
plt.subplot(131).set_title('original binary image')
plt.axis('off')
plt.imshow(b, cmap='gray')
plt.subplot(132).set_title('eroded image')
plt.axis('off')
plt.imshow(img, cmap='gray')
plt.subplot(133).set_title('outline =\n[original image] - [eroded image]')
plt.axis('off')
plt.imshow(img1, cmap='gray')

plt.show()
