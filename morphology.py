import numpy as np


def morph(img=None, oper=None, iterations=1, strel=np.ones((3, 3)), bin=0):
    # This function accepts accepts a binary shape (img), morphological operation, structuring element
    # and number of iterations. It performs the specified operation of img by the structuring element
    # for the required number of iterations and returns the result.

    if bin:
        img = binarize(img)
    if oper is None:
        print('Choose specific morphological operator: erosion, dilation, closing, opening')
        return
    img = find_range(img)
    if oper == 'er':
        func = erosion
    elif oper == 'di':
        func = dilation
    elif oper == 'cl':
        func = closing
    elif oper == 'op':
        func = opening

    morphed_img = func(img, strel)
    if iterations > 1:
        for i in range(iterations - 1):
            morphed_img = func(morphed_img, strel)
            i += 1
    return morphed_img


def erosion(img, strel=np.ones((3, 3))):
    # The erosion function accepts a binary shape (img) and a structuring element. If no structuring element
    # is passed to the function, a 3-by-3 array of ones is used as a default.
    # The function returns the erosion of img by the structuring element.

    # img = binarize(img)
    img = find_range(img)
    img /= 255
    dim = img.shape
    strel = np.asarray(strel, dtype=np.uint8)
    orig = ((strel.shape[0]-1) // 2, (strel.shape[1] -1) // 2)
    x, y = strel.shape[0], strel.shape[1]
    a = np.zeros(dim)

    for i in range(dim[0]):
        for j in range(dim[1]):

            # Step I - Find the intersection of structuring element and current x-by-y subset of img.
            # If there is no intersection, continue:
            inters = img[fit(i - orig[0]):i - orig[0] + x, fit(j - orig[1]):j - orig[1] + y]
            if inters.shape[0] == 0 or inters.shape[1] == 0:
                continue

            # Step II - Find points (x0, y0), (x1, y1) such that the shape of
            # structuring element [x0:x1, y0:y1] will be equal to the shape of intersection:

            x0 = max(orig[0] - i, 0)
            x1 = (dim[0] + orig[0]) - (i + 1) if i + (x - orig[0]) > dim[0] else x - 1
            y0 = max(orig[1] - j, 0)
            y1 = (dim[1] + orig[1]) - (j + 1) if j + (y - orig[1]) > dim[1] else y - 1
            # Step III - Compute match:

            match = np.logical_and(inters, strel[x0:x1 + 1, y0:y1 + 1])
            if np.array_equal(match, strel[x0:x1 + 1, y0:y1 + 1]):
                a[i, j] = 1

    return np.asarray(a * 255, dtype=np.uint8)


def dilation(img, strel=np.ones((3, 3))):
    # The erosion function accepts a binary shape (img) and a structuring element. If no structuring element
    # is passed to the function, a 3-by-3 array of ones is used as a default.
    # The function returns the dilation of the binary shape by the structuring element.

    # img = binarize(img)
    img = find_range(img)
    img = img // 255
    dim = img.shape
    strel=np.asarray(strel, dtype=np.uint8)
    orig = ((strel.shape[0] - 1) // 2, (strel.shape[1] - 1) // 2)
    x, y = strel.shape[0], strel.shape[1]
    a = np.zeros(dim)

    for i in range(dim[0]):
        for j in range(dim[1]):

            # Step I - Find the intersection of structuring element and current x-by-y subset of img.
            # If there is no intersection, continue:
            inters = img[fit(i - orig[0]):i - orig[0] + x, fit(j - orig[1]):j - orig[1] + y]
            if inters.shape[0] == 0 or inters.shape[1] == 0:
                continue

            # Step II - find points (x0, y0), (x1, y1) such that the shape of
            # structuring element [x0:x1, y0:y1] will be equal to the shape of intersection:

            x0 = max(orig[0] - i, 0)
            x1 = (dim[0] + orig[0]) - (i + 1) if i + (x - orig[0]) > dim[0] else x - 1
            y0 = max(orig[1] - j, 0)
            y1 = (dim[1] + orig[1]) - (j + 1) if j + (y - orig[1]) > dim[1] else y - 1
            # Step III - Compute match:

            match = np.logical_and(inters, strel[x0:x1 + 1, y0:y1 + 1])
            if match.any():
                a[i, j] = 1

    return np.asarray(a * 255, dtype=np.uint8)


def opening(img, strel=np.ones((3, 3))):
    # The erosion function accepts a binary shape (img) and a structuring element. If no structuring element
    # is passed to the function, a 3-by-3 array of ones is used as a default.
    # The function returns the opening of img by the structuring element.

    return dilation(erosion(img, strel), strel)


def closing(img, strel=np.ones((3, 3))):
    # The erosion function accepts a binary shape (img) and a structuring element. If no structuring element
    # is passed to the function, a 3-by-3 array of ones is used as a default.
    # The function returns the closing of img by the structuring element.

    return erosion(dilation(img, strel), strel)


def reverse(img):
    # Auxiliary function to convert black-over-white images to white-over-black images.

    rev = segment(img)
    rev[img == 0] = 255
    rev[img == 255] = 0
    return np.asarray(rev, dtype=np.uint8)


def gray_conversion(img):
    if len(img.shape) == 2:
        return img
    gray_img = np.asarray(img, dtype=np.float64)
    gray_img = 0.07 * img[:, :, 2] + 0.72 * img[:, :, 1] + 0.21 * img[:, :, 0]
    gray_img = np.reshape(gray_img, (gray_img.shape[0], gray_img.shape[1]))
    return gray_img.astype(np.float64)


def segment(img, th=None):
    # This function performs a basic segmentation of a grayscale image according to given threshold.
    # Mean value of the image is used as default.

    if th is None:
        th = int(np.round(np.mean(img, axis=0).mean()))

    seg_img = img.copy()
    seg_img[seg_img < th] = 0
    seg_img[seg_img >= th] = 255
    return np.asarray(seg_img, dtype=np.uint8)


def hist_thresh(img):
    # This function performs segmentation of a grayscale image by computing the optimal threshold.

    mean = int(np.round(np.mean(img, axis=0).mean()))
    while True:
        fg_mean = np.mean(img[img > int(np.round(mean))])
        bg_mean = np.mean(img[img <= int(np.round(mean))])
        new_mean = (bg_mean + fg_mean) / 2
        if new_mean == mean:
            return segment(img, new_mean)
        mean = new_mean


def binarize(img, mode='hist', th=None):
    # Auxiliary function to convert color or grayscale images to black and white (binary).

    bin_img = img.copy()
    if len(img.shape) == 3:
        bin_img = gray_conversion(np.copy(img))
    if mode == 'seg':
        return segment(bin_img, th)
    if mode == 'hist':
        return hist_thresh(bin_img)


def fit(index):
    # Auxiliary function to compute intersection of a structuring element and a binary image.

    return max(index, 0)


def find_range(img):
    # Auxiliary function to turn binary images into [0, 255] format over [0, 1].

    if img[img == 1].any():
        return np.asarray(img * 255, dtype=np.float64)
    else:
        return np.asarray(img, dtype=np.float64)


def strel(shape='rect', dim=(3, 3), rad=3):
    if shape == 'rect':
        return np.ones(dim, np.uint8)
    if shape == 'cross':
        strel = np.zeros(dim, np.uint)
        strel[dim[0] // 2, :] = 1
        strel[:, dim[1] // 2] = 1
        return strel
    if shape == 'diag':
        return np.eye(dim[0], dim[1], dtype=np.uint8)
    if shape == 'anti':
        strel = np.eye(dim[0], dim[1], dtype=np.uint8)
        return np.fliplr(strel)
    if shape == 'ellipse':
        rad = abs(np.round(rad))
        strel = np.zeros((2*rad - 1, 2*rad - 1), np.uint8)
        origin = (2*rad - 1) // 2
        for x in range(2*rad - 1):
            for y in range(2*rad - 1):
                if (x - origin)**2 + (y - origin)**2 < rad**2 - 1:
                    strel[x, y] = 1
        return strel
    if shape == 'disk':
        rad = abs(np.round(rad))
        strel = np.zeros((2*rad - 1, 2*rad - 1), np.uint8)
        origin = (2*rad - 1) // 2
        for x in range(2*rad - 1):
            for y in range(2*rad - 1):
                if abs(x - origin) + abs(y - origin) < rad:
                    strel[x, y] = 1
        return strel


def hist_eq(img):
    hist = [np.sum(img == i) for i in range(256)]
    hist = np.array(hist, dtype=np.float64)
    PDF = hist/img.size
    CDF = np.cumsum(PDF)
    CDF = np.asarray(CDF, dtype=np.float64)
    mod_img = np.zeros(img.shape, dtype=np.uint8)
    for i in range(256):
        mod_img[img == i] = np.round(CDF[i] * 255)
    return mod_img


def erosion_ver2(img, strel=np.ones((3, 3))):
    # The erosion function accepts a binary shape (img) and a structuring element. If no structuring element
    # is passed to the function, a 3-by-3 array of ones is used as a default.
    # The function returns the erosion of img by the structuring element.

    img = binarize(img, 'seg')
    img = find_range(img)
    img /= 255
    dim = img.shape
    strel=np.asarray(strel, dtype=np.uint8)
    orig = ((strel.shape[0])// 2, (strel.shape[1])// 2)
    x, y = strel.shape[0], strel.shape[1]
    a = np.zeros(dim)

    for i in range(orig[0], dim[0] - orig[0]):
        for j in range(orig[1], dim[1]-orig[1]):
            intersection = img[i - orig[0]:i - orig[0] + x, j - orig[1]:j - orig[0] + y]
            intersection_sum = intersection + strel
            overlap = intersection[intersection_sum == 2]
            u = np.count_nonzero(overlap)
            v = np.count_nonzero(strel)
            if u == v:
                a[i, j] = 1

    return np.asarray(a*255 , dtype=np.uint8)


def dilation_ver2(img, strel=np.ones((3, 3))):
    # The erosion function accepts a binary shape (img) and a structuring element. If no structuring element
    # is passed to the function, a 3-by-3 array of ones is used as a default.
    # The function returns the erosion of img by the structuring element.

    # img = binarize(img, 'seg')
    img = find_range(img)
    img /= 255
    dim = img.shape
    strel = np.asarray(strel, dtype=np.uint8)
    orig = ((strel.shape[0])// 2, (strel.shape[1])// 2)
    x, y = strel.shape[0], strel.shape[1]
    a = np.zeros(dim)

    for i in range(orig[0], dim[0] - orig[0]):
        for j in range(orig[1], dim[1] - orig[1]):
            intersection = img[i - orig[0]:i - orig[0] + x, j - orig[1]:j - orig[0] + y]
            intersection_sum = intersection + strel
            overlap = intersection[intersection_sum == 2]
            u = np.count_nonzero(overlap)
            if u >= 1:
                a[i, j] = 1

    return np.asarray(a * 255, dtype=np.uint8)

