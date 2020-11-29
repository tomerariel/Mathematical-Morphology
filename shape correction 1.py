import cv2
import matplotlib.pyplot as plt
from morphology import *

# The most important application of mathematical morphology - mending a broken heart.

img = cv2.imread('shape correction 1.jpg', 0)
img1 = binarize(img)
img2 = reverse(img1)
img3 = morph(img2, 'cl', 1, strel=strel('rect', (30, 30)))

plt.figure('Ex. f - shape correction')
plt.subplot(131).set_title('a broken heart')
plt.axis('off')
plt.imshow(img1, cmap='gray')
plt.subplot(132).set_title('a binarized & reversed broken heart')
plt.axis('off')
plt.imshow(img2, cmap='gray')
plt.subplot(133).set_title('LOVE IS AN OPEN(ED) DOOR! \n (although closing is used in this specific example)')
plt.axis('off')
plt.imshow(img3, cmap='gray')

plt.show()
