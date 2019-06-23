import cv2
import numpy as np
# import matplotlib.pyplot as plt

counter = 0
gray_img = cv2.imread('tom', 0)
gray_img = cv2.resize(gray_img, (600,600))
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#cv2.imshow("image", gray_img)
#gray_img = cv2.resize(gray_img, (300, 300))

def normalize(img):
    i, j = img.shape
    maximum_value = img[0][0]
    minimum_value = img[0][0]
    for row in range(i):
        for col in range(j):
            if maximum_value < img[row][col]:
                maximum_value = img[row][col]
            if minimum_value > img[row][col]:
                minimum_value = img[row][col]

    for row in range(i):
        for col in range(j):
            img[row][col] = (img[row][col] - minimum_value) / (maximum_value - minimum_value)

    return img


def add_pad(img):
    i, j = img.shape
    pad_image = np.zeros((i + 2, j + 2))
    for row in range(i):
        for col in range(j):
            pad_image[row + 1][col + 1] = img[row][col]

    return pad_image


def convolution_vertical(img):
    global counter
    counter = counter + 1
    #img = normalize(img)
    img = add_pad(img)
    i , j = img.shape
    vertical_edge = np.zeros((i - 2, j - 2))

    for row in range(i - 2):
        for col in range(j - 2):
            vertical_edge[row][col] = img[row][col] * 1 + img[row][col + 1] * 0 + img[row][col + 2] * (-1)\
                                      + img[row + 1][col] * 1 + img[row + 1][col + 1] * 0 + img[row + 1][col + 2] * (-1)\
                                      + img[row + 2][col] * 1 + img[row + 2][col + 1] * 0 + img[row + 2][col + 2] * (-1)
    print(counter)
    return vertical_edge


def join(img1, img2, layer):
    img = (img1 + img2) / layer
    return img

# print(pad_image(gray_img))

def relu(img):
    i, j = img.shape
    for row in range(i):
        for col in range(j):
            img[row][col] = max(0, img[row][col])

    return img


def fractal_net(x, layer):
    if layer == 2:
        return join(convolution_vertical(x), convolution_vertical(convolution_vertical(x)), layer)

    img1 = fractal_net(x, layer - 1)
    img2 = fractal_net(img1, layer - 1)
    block_img = join(convolution_vertical(x), img2 * (layer - 1), layer)
    cv2.imshow("image" + str(layer), block_img)
    return block_img


img_block1 = fractal_net(gray_img, 5)

'''
img_block2 = fractal_net(img_block1, 5)
cv2.imshow("img_block2", img_block2)

img_block3 = fractal_net(img_block2, 5)
cv2.imshow("img_block3", img_block3)

img_block4 = fractal_net(img_block3, 5)
cv2.imshow("img_block4", img_block4)

img_block5 = fractal_net(img_block4, 5)
cv2.imshow("img_block5", img_block5)
'''


# cv2.imshow('horizontal_edge',horizontal_edge)

cv2.waitKey(0)
cv2.destroyAllWindows()

