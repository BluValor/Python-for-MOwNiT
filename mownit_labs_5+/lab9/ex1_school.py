import numpy as np
from PIL import Image as im
from PIL import ImageOps as io
import matplotlib.pyplot as plt


def mark_occurrences(C, img):

    img_arr = np.array(img)
    h, w = np.shape(C)
    correlation_threshold = 0.95 * np.amax(C)
    counter = 0
    print(np.shape(img_arr))

    for y in range(h):
        for x in range(w):
            if C[y, x] > correlation_threshold:
                # img_arr[y, x] = np.array([242, 252, 106])
                img_arr[y, x] = np.array([255, 100, 100])

    return im.fromarray(img_arr)


def equalize_image(greyscale_img):

    img_arr = np.array(greyscale_img)
    h, w = np.shape(greyscale_img)

    for y in range(h):
        for x in range(w):

            # img_arr[y, x] = min(img_arr[y, x] + 25, 255)
            # img_arr[y, x] += 25
            tmp = img_arr[y, x]
            tmp += (255 - tmp) / 8
            tmp += (0 - tmp) / 4
            img_arr[y, x] = tmp
            # img_arr[y, x] += (255 - img_arr[y, x]) / 5
            # img_arr[y, x] += (0 - img_arr[y, x]) / 2
            # img_arr[y, x] = max(img_arr[y, x] - 25, 0)

    return im.fromarray(img_arr)


name = 'school.jpg'

image = im.open(name).convert('L')
image = equalize_image(image)
# image.show()

pattern = im.open('fish1.png').convert('L')
pattern = equalize_image(pattern)
# pattern.show()

n, m = np.shape(image)
C_complex = np.fft.ifft2(np.fft.fft2(image) * np.fft.fft2(np.rot90(pattern, 2), s=(n, m)))
print(n, m)
C_real = np.real(C_complex)
C_absolute = np.absolute(C_complex)
C_phase = np.angle(C_complex)

im.fromarray(np.uint8(C_absolute)).show()
im.fromarray(np.uint8(C_phase)).show()

plt.figure()
plt.axis('off')
plt.style.use('classic')
# plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
img = im.open(name)
img = mark_occurrences(C_real, img)
plt.imshow(img)
plt.margins(0)
plt.show()
im._show(img)
