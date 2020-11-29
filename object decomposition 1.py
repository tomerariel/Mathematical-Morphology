import cv2
import matplotlib.pyplot as plt
from morphology import *

# Simple binary object decomposition.

strel = np.ones((17, 17), np.uint8)
strel[0, :] = 0
strel[strel.shape[0] - 1, :] = 0
strel[:, strel.shape[1] - 1] = 0
strel[:, 0] = 0

img = cv2.imread('object decomposition 1.png', 0)
img1 = morph(img, 'op', 1, strel=strel)
img2 = img - img1
img3 = morph(img2, 'er', 1)

plt.figure('Ex. d - object decomposition 1')
plt.subplot(141).set_title('original image')
plt.axis('off')
plt.imshow(img, cmap='gray')
plt.subplot(142).set_title('opened image')
plt.axis('off')
plt.imshow(img1, cmap='gray')
plt.subplot(143).set_title('residue')
plt.axis('off')
plt.imshow(img2, cmap='gray')
plt.subplot(144).set_title('eroded residue')
plt.axis('off')
plt.imshow(img3, cmap='gray')

plt.show()
