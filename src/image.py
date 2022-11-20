import cv2


def resize_256(image):
    width, height = image.shape[1], image.shape[0]

    if (width > height):
        start_row = 0
        end_row = height

        start_col = (width - height) / 2
        end_col = width - start_col
    else:
        max = width
        start_row = (height - width) / 2
        end_row = height - start_row

        start_col = 0
        end_col = width

    crop_img = image[int(start_row):int(end_row), int(start_col):int(end_col)]

    # dimensi yang diinginkan 256 x 256
    dim = (256, 256)

    resized_image = cv2.resize(crop_img, dim, interpolation=cv2.INTER_LINEAR)

    return resized_image
