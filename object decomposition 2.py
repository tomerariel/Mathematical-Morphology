import cv2
import matplotlib.pyplot as plt
from morphology import *

# Performing object decomposition on a grayscale image.
# The image is binarized, reversed, and then morphologically filtered.

disk5 = strel('disk', rad=5)
disk6 = strel('disk', rad=6)

img = cv2.imread('object decomposition 2.jpg', 0)
img1 = binarize(img)
img2 = reverse(img1)
img3 = morph(img2, 'op', 3, strel=disk5)
img4 = morph(img3, 'er', 1, strel=disk5)
img5 = morph(img4, 'di', 5, strel=disk6)

plt.figure('Ex. e - object decomposition 2')
plt.subplot(231).set_title('original image')
plt.axis('off')
plt.imshow(img, cmap='gray')
plt.subplot(232).set_title('step 1:\nbinarization')
plt.axis('off')
plt.imshow(img1, cmap='gray')
plt.subplot(233).set_title('step 2:\nreversion')
plt.axis('off')
plt.imshow(img2, cmap='gray')
plt.subplot(234).set_title('step 3:\nopening')
plt.axis('off')
plt.imshow(img3, cmap='gray')
plt.subplot(235).set_title('step 4:\nerosion')
plt.axis('off')
plt.imshow(img4, cmap='gray')
plt.subplot(236).set_title('step 5:\ndilation')
plt.axis('off')
plt.imshow(img5, cmap='gray')

plt.show()



