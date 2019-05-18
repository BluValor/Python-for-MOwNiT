import PIL.Image as im
import lab9.ex2 as ex2


print("CenturySchoolbook: aligned, no noise")
path = "./CenturySchoolbook"
image_name = "_text.png"
symbols_image = "_symbols.png"
ex2.crop_symbols(path)
ex2.crop_text(path)
ex2.trim(im.open(path + '/' + image_name)).convert('L').show()
ex2.ocr(path, symbols_image, image_name, acceptance=0.99)
print("\n\n")


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