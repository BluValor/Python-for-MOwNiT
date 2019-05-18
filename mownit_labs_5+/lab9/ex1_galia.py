import numpy as np
from PIL import Image as im
from PIL import ImageOps as io
import matplotlib.pyplot as plt


def put_marker(y, x):
    plt.scatter(y - 8, x - 8, s=20, c='red', marker='o')


def mark_occurrences(C):

    h, w = np.shape(C)
    correlation_threshold = 0.95 * np.amax(C)
    counter = 0

    for y in range(h):
        for x in range(w):
            if C[y, x] > correlation_threshold:
                put_marker(x, y)
                counter += 1

    return counter


name = 'galia.png'

image = im.open(name).convert('L')
image_inv = io.invert(image)
# image_inv.show()

pattern = im.open('galia_e.png').convert('L')
pattern_inv = io.invert(pattern)
# pattern_inv.show()

n, m = np.shape(image_inv)
C_complex = np.fft.ifft2(np.fft.fft2(image_inv) * np.fft.fft2(np.rot90(pattern_inv, 2), s=(n, m)))
C_real = np.real(C_complex)
C_absolute = np.absolute(C_complex)
C_phase = np.angle(C_complex)

# im.fromarray(np.uint8(C_absolute)).show()
# im.fromarray(np.uint8(C_phase)).show()

plt.figure()
plt.axis('off')
plt.imshow(im.open(name))
counter = mark_occurrences(C_real)
plt.show()

print('Number of occurrences:', counter)
