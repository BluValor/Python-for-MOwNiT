from PIL import Image
import numpy as np


def make_gif(A):

    U, S_list, VH = np.linalg.svd(A)
    S = np.diag(S_list)

    images = []

    for i in range(1, len(S_list) + 1):

        S[-i][-i] = 0

        A_compressed = U @ S @ VH

        compressed_image = Image.fromarray(A_compressed)
        compressed_image = compressed_image.convert("RGB")
        images.append(compressed_image)

    images[0].save("cat_compresion.gif", save_all=True, append_images=images[1:], duration=32, loop=0)


I = Image.open('cat.jpg').convert('L')
A = np.array(I)

# make_gif(A)

U, S_list, VH = np.linalg.svd(A)
S = np.diag(S_list)

for i in range(-1, -430, -1):
    S[i][i] = 0

A_compressed = U @ S @ VH

print(A_compressed)

im = Image.fromarray(np.uint8(A_compressed)).show()

# jakość obrazu początkowo ulega tylko niewielkiemu pogorszeniu. Dopiero ok 400 piksela zmiany zaczynają być dobrze
# widoczne, a od 450 / 470 staję się nieczytelny

