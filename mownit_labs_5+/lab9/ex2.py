from PIL import Image as im
from PIL import ImageOps as io
from PIL import ImageChops as ic
from PIL import ImageDraw as id
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.signal
import scipy.ndimage


symbols_h = 3
symbols_w = 26

symbols = [
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', 'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'Z', 'X', 'C', 'V',
     'B', 'N', 'M'],
    ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'z', 'x', 'c', 'v',
     'b', 'n', 'm'],
    ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '!',
     '?', ',', 'dot']]


def svd_add_noise(path, name, percentage):
    I = im.open(path + '/' + name).convert('L')
    A = np.array(I)
    A = A + 0.3 * np.random.standard_normal(np.shape(A))

    U, S_list, VH = np.linalg.svd(A)

    h = np.shape(VH)[0]
    S = np.diag(S_list, len(S_list) - h)
    S = np.array([np.array(line) for line in S[h - len(S_list):]])

    for i in range(-1, -(int(percentage * len(S_list))), -1):
        S[i][i - h + len(S_list)] = 0

    A_compressed = U @ S @ VH
    img = im.fromarray(np.uint8(A_compressed))

    img.save(path + '/' + name[:-4] + '_w_noise.png')

    return img


def rotate_and_save(degrees, path, name):

    im.open(path + '/' + name)\
        .rotate(degrees, im.BILINEAR, expand=True, fillcolor='white')\
        .save(path + '/_' + ('left' if degrees > 0 else 'right') + '_rot_' + str(np.abs(degrees)) + '_' + name)


# extracts symbols from specially made table in _page.png file - added for convenience
# delta format: (delta top, delta bottom)
def crop_symbols(path, delta=0):

    img = im.open(path + "/_page.png").convert('L')

    left = 154
    up = 154

    border = (left, up, left + 1280, up + 165 + delta) # left, up, right, bottom

    img = img.crop(border)
    img.save(path + "/_symbols.png")


# extracts text from pre-specified area in _page.png file - added for convenience
# delta format: (delta top, delta bottom)
def crop_text(path, delta=0):

    img = im.open(path + "/_page.png").convert('L')

    left = 140
    up = 730

    border = (left, up, left + 1310, up + 870 + delta)  # left, up, right, bottom`
    trim(img.crop(border)).save(path + "/_text.png")


def trim(img, outline=5):

    h, w = np.shape(img)

    bg = im.new(img.mode, img.size, img.getpixel((0, 0)))
    diff = ic.difference(img, bg)
    diff = ic.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()

    if bbox:
        outline = (max(0, bbox[0] - outline), max(0, bbox[1] - outline), min(w, bbox[2] + outline),
                   min(h, bbox[3] + outline))
        return img.crop(outline)


def align(img, name):

    min_size = np.prod(list(img.size))
    optimal = 0
    print(img.size)

    degrees = (range(0, -90, -1) if name.startswith('_left') else range(0, 90, 1))

    for i in degrees:
        tmp = np.prod(list(trim(img.rotate(i, im.BILINEAR, expand=True, fillcolor='white')).size))
        if tmp < min_size:
            min_size = tmp
            optimal = i

    return trim(img.rotate(optimal, im.BILINEAR, expand=True, fillcolor='white'))


def remove_noise(img, count=1):

    img2 = scipy.array(img)

    for _ in range(count):
        img2 = scipy.signal.medfilt2d(img2)

    # img2 = scipy.ndimage.gaussian_filter(img2, sigma=1)
    # img2 = scipy.ndimage.maximum_filter(img2, size=2)
    img2 = scipy.ndimage.percentile_filter(img2, percentile=50, size=3)

    return im.fromarray(np.uint8(img2))


def extract_numbers(path, name, outline=5):

    img = im.open(path + '/' + name)
    w, h = img.size
    width, height = w / symbols_w, h / symbols_h

    for n in range(symbols_h):
        for m in range(symbols_w):

            border = (int(m * width), int(n * height), int((m + 1) * width), int((n + 1) * height))
            if symbols[n][m] != ' ':
                trim(img.crop(border), outline=outline).save(path + '/' + symbols[n][m] + '.png')


def put_marker(y, x):
    plt.scatter(y - 8, x - 8, s=6, c='red', marker='o')


def mark_occurrences(img, occurrences):

    img_arr = np.array(img)
    for position in occurrences:
        put_marker(position[1], position[0])


def find_lines(av_char_size, h, mult=1.515):
    # outline = 5: eh = 2(?) * av
    eh = mult * av_char_size[0]
    estim = h / eh
    return [int((i + 0.75) * eh) for i in range(int(estim))]


def assemble_text(arr, order, av_char_size, lines, mult=1.0):

    arr2 = [[(lst[0] if len(lst) > 0 else ' ') for lst in line] for line in arr]
    order2 = order + [' ']

    float_h, float_w = av_char_size
    # outline = 3: h / 2, w / 2.9
    eh, ew = int(float_h / 1.65), int(float_w / 2.393)
    h, w = np.shape(arr)
    result = [[] for _ in lines]
    max_repeated = int(2 * eh * ew * mult)

    for i, line in enumerate(lines):
        repeated = 0
        prev_c = ' '
        result[i] += prev_c
        for x in range(w):
            for off in range(-eh, eh):

                if not(line + off < 0 or line + off > h - 1):

                    c = arr2[line + off][x]
                    if prev_c == c:
                        repeated += 1
                        if repeated == max_repeated:
                            if c == 'dot':
                                result[i] += '.'
                            else:
                                result[i] += c
                            repeated = 0
                    else:
                        if order2.index(c) < order2.index(prev_c):
                            result[i].pop()
                            if c == 'dot':
                                result[i] += '.'
                            else:
                                result[i] += c
                            repeated = 0
                            prev_c = c
                        else:
                            repeated += 1
                            if repeated == max_repeated:
                                if c == 'dot':
                                    result[i] += '.'
                                else:
                                    result[i] += c
                                repeated = 0
                                prev_c = ' '

    return '\n'.join([''.join(lst) for lst in result])


def flatten_image(img, delta=20):
    return im.fromarray(np.uint8(np.array([[min(255, x + delta) for x in line] for line in np.array(img)])))


def find_pattern(image_inv, pattern, acceptance, doflatten=True):

    pattern_inv = io.invert(pattern)
    if doflatten:
        pattern_inv = flatten_image(pattern_inv)

    w, h = image_inv.size
    C = np.real(np.fft.ifft2(np.fft.fft2(image_inv) * np.fft.fft2(np.rot90(pattern_inv, 2), s=(h, w))))

    w2, h2 = pattern.size
    max_t = np.amax(np.real(np.fft.ifft2(np.fft.fft2(pattern_inv) * np.fft.fft2(np.rot90(pattern_inv, 2), s=(h2, w2)))))

    correlation_threshold = acceptance * max_t
    occurrences = []

    for y in range(h):
        for x in range(w):
            if C[y, x] > correlation_threshold:
                occurrences.append((y, x))

    return occurrences


def ocr(path, symbols_name, image_name, acceptance=0.8, doalign=False, donoise=False, doflatten=False, linesmult=1.515, assemblemult=1.0):

    extract_numbers(path, symbols_name, outline=2)

    img = trim(im.open(path + '/' + image_name)).convert('L')
    if donoise:
        img = remove_noise(img)
    if doalign:
        img = align(img, image_name)
    img.save(path + '/' + '_tmp.png')
    img.show()
    img = io.invert(img)
    if doflatten:
        img = flatten_image(img)

    w, h = img.size
    found = [[[] for _ in range(w)] for _ in range(h)]
    counts = np.zeros((symbols_h, symbols_w), dtype=int)

    for n in range(symbols_h):
        for m in range(symbols_w):

            if symbols[n][m] != ' ':
                pattern = im.open(path + '/' + symbols[n][m] + '.png').convert('L')
                occurrences = find_pattern(img, pattern, acceptance, doflatten=doflatten)
                print(symbols[n][m], '->', len(occurrences))
                for y, x in occurrences:
                    found[y][x].append(symbols[n][m])

                # img_to_show = img.copy()
                # plt.figure()
                # plt.axis('off')
                # plt.imshow(img_to_show)
                # plt.suptitle(symbols[n][m])
                # mark_occurrences(img_to_show, occurrences)
                # plt.show()

                counts[n, m] = len(occurrences)

    symbol_counts = list(zip(np.array(symbols).flatten(), np.array(counts).flatten()))
    sorted_symbol_counts = sorted(symbol_counts, key=lambda tup: tup[1])
    order = [i for i, _ in sorted_symbol_counts if i != ' ']

    for y in range(h):
        for x in range(w):
            found[y][x] = sorted(found[y][x], key=lambda c: order.index(c))

    s_sizes = [im.open(path + '/' + symbol + '.png').size for symbol in order[:-5]]
    av_char_size = sum([i for _, i in s_sizes]) / len(s_sizes), sum([i for i, _ in s_sizes]) / len(s_sizes)
    lines = find_lines(av_char_size, h, mult=linesmult)

    draw = id.Draw(img)
    for line in lines:
        draw.line((0, line, w, line), fill=100)
    img.show()

    print(assemble_text(found, order, av_char_size, lines, mult=assemblemult))


# print("CenturySchoolbook: aligned, no noise")
# path = "./CenturySchoolbook"
# image_name = "_text.png"
# symbols_image = "_symbols.png"
# crop_symbols(path)
# crop_text(path)
# trim(im.open(path + '/' + image_name)).convert('L').show()
# ocr(path, symbols_image, image_name, acceptance=0.99)
# print("\n\n")


# print("CenturySchoolbook: aligned, no noise, flattened")
# path = "./CenturySchoolbook"
# image_name = "_text.png"
# symbols_image = "_symbols.png"
# crop_symbols(path)
# crop_text(path)
# trim(im.open(path + '/' + image_name)).convert('L').show()
# ocr(path, symbols_image, image_name, acceptance=0.99, doflatten=True)
# print("\n\n")


# print("CenturySchoolbook: not aligned, with noise")
# path = "./CenturySchoolbook"
# image_name = "_text.png"
# symbols_image = "_symbols.png"
# crop_symbols(path)
# crop_text(path)
# rotate_and_save(-23, path, image_name)
# image_name = "_right_rot_23__text.png"
# img = svd_add_noise(path, image_name, 0.25)
# image_name = "_right_rot_23__text_w_noise.png"
# trim(im.open(path + '/' + image_name)).convert('L').show()
# ocr(path, symbols_image, image_name, acceptance=0.72, doalign=True, donoise=True)
# print("\n\n")


# print("CenturySchoolbook (widened): aligned, no noise")
# path = "./CenturySchoolbook_widened"
# image_name = "_text.png"
# symbols_image = "_symbols.png"
# crop_symbols(path)
# crop_text(path, delta=140)
# trim(im.open(path + '/' + image_name)).convert('L').show()
# ocr(path, symbols_image, image_name, acceptance=0.99, assemblemult=1.40)
# print("\n\n")


# print("CenturySchoolbook: not aligned, with noise")
# path = "./CenturySchoolbook_widened"
# image_name = "_text.png"
# symbols_image = "_symbols.png"
# crop_symbols(path)
# crop_text(path, delta=140)
# rotate_and_save(16, path, image_name)
# image_name = "_left_rot_16__text.png"
# img = svd_add_noise(path, image_name, 0.20)
# image_name = "_left_rot_16__text_w_noise.png"
# trim(im.open(path + '/' + image_name)).convert('L').show()
# ocr(path, symbols_image, image_name, acceptance=0.68, assemblemult=1.40, doalign=True, donoise=True)
# print("\n\n")


# print("FreeSans: aligned, no noise")
# path = "./FreeSans"
# image_name = "_text.png"
# symbols_image = "_symbols.png"
# crop_symbols(path, delta=-9)
# crop_text(path)
# trim(im.open(path + '/' + image_name)).convert('L').show()
# ocr(path, symbols_image, image_name, acceptance=0.98, assemblemult=1.10, linesmult=1.46)
# print("\n\n")


# print("FreeSans: not aligned, with noise")
# path = "./FreeSans"
# image_name = "_text.png"
# symbols_image = "_symbols.png"
# crop_symbols(path, delta=-9)
# crop_text(path)
# rotate_and_save(-15, path, image_name)
# image_name = "_right_rot_15__text.png"
# img = svd_add_noise(path, image_name, 0.19)
# image_name = "_right_rot_15__text_w_noise.png"
# trim(im.open(path + '/' + image_name)).convert('L').show()
# ocr(path, symbols_image, image_name, acceptance=0.74, assemblemult=1.10, linesmult=1.46, doalign=True, donoise=True)
# print("\n\n")


# print("MathJax Typewriter: aligned, no noise")
# path = "./MathJax_Typewriter"
# image_name = "_text.png"
# symbols_image = "_symbols.png"
# crop_symbols(path, delta=-35)
# crop_text(path)
# trim(im.open(path + '/' + image_name)).convert('L').show()
# ocr(path, symbols_image, image_name, acceptance=0.98, assemblemult=1.35, linesmult=1.33)
# print("\n\n")


# print("MathJax Typewriter: not aligned, with noise")
# path = "./MathJax_Typewriter"
# image_name = "_text.png"
# symbols_image = "_symbols.png"
# crop_symbols(path, delta=-35)
# crop_text(path)
# rotate_and_save(10, path, image_name)
# image_name = "_left_rot_10__text.png"
# img = svd_add_noise(path, image_name, 0.13)
# image_name = "_left_rot_10__text_w_noise.png"
# trim(im.open(path + '/' + image_name)).convert('L').show()
# ocr(path, symbols_image, image_name, acceptance=0.76, assemblemult=1.35, linesmult=1.32, doalign=True, donoise=True)
# print("\n\n")
