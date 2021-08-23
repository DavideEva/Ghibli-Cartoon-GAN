import argparse
import os
import cv2
import numpy as np

# The raw image as found in dataset files. The important part is the width/height proportion. In this case, 16:9.
raw_shape = (1080, 1920, 3)

# The same as https://github.com/FilipAndersson245/cartoon-gan/blob/5a09f4e2cfad42accfc1792dedfba95f9ab6fb83/utils/datasets.py#L32
# Should be less than input shape
preprocess_shape = (384, 384, 3)

# Dimension after the preprocess stage
# Should be the dimension expected by the network and the loss functions
final_shape = (256, 256, 3)


def resize_image_height(img, height):
    original_h, original_w, _ = img.shape
    scale_factor = height / original_h
    scaled_w = round(original_w * scale_factor)
    scaled_h = round(original_h * scale_factor)
    scaled_size = (scaled_w, scaled_h)
    resized = cv2.resize(img, scaled_size)
    return resized


def left_crop(img):
    return img[:, 0:img.shape[0], :]


def right_crop(img):
    return img[:, -img.shape[0]:, :]


def smooth_edges(img):
    # Parameters taken from https://github.com/FilipAndersson245/cartoon-gan/blob/master/utils/datasets.py
    kernel_size = 5
    pad_size = kernel_size // 2 + 1
    gray_img = cv2.cvtColor(np.uint8(img), cv2.COLOR_RGB2GRAY)
    pad_img = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')
    edges = cv2.Canny(gray_img, 150, 500)
    dilation = cv2.dilate(edges, np.ones((kernel_size, kernel_size), np.uint8))
    gauss = cv2.getGaussianKernel(kernel_size, 0)
    gauss = gauss * gauss.transpose(1, 0)
    idx = np.where(dilation != 0)
    loops = len(idx[0])
    gauss_img = np.copy(img)
    for i in range(loops):
        # debug edges detection:
        # gauss_img[idx[0][i], idx[1][i], 0], gauss_img[idx[0][i], idx[1][i], 1], gauss_img[idx[0][i], idx[1][i], 2] = (1.0, 1.0, 0.0)
        gauss_img[idx[0][i], idx[1][i], 0] = np.sum(
            np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 0], gauss))
        gauss_img[idx[0][i], idx[1][i], 1] = np.sum(
            np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 1], gauss))
        gauss_img[idx[0][i], idx[1][i], 2] = np.sum(
            np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 2], gauss))
    return gauss_img


def save_rgb_image(file_name, image):
    print(file_name)
    cv2.imwrite(
        file_name,
        cv2.cvtColor(
            cv2.resize(
                image,
                (final_shape[1], final_shape[0]),
                interpolation=cv2.INTER_AREA
            ),
            cv2.COLOR_RGB2BGR
        )
    )


def _parse_folder(root, folder, output_dir, dest_type, files, preprocess=lambda x: x):
    rel_path = os.path.relpath(root, folder)
    dest = os.path.join(output_dir, dest_type, rel_path)
    dest_left = os.path.join(dest, "left")
    dest_right = os.path.join(dest, "right")
    os.makedirs(dest_left, exist_ok=True)
    os.makedirs(dest_right, exist_ok=True)
    for file in files:
        if os.path.exists(os.path.join(dest_left, file)):
            continue  # Skip if already exist
        img = cv2.imread(os.path.join(root, file))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        preprocessed_img = preprocess(resize_image_height(img, preprocess_shape[0]))
        left_part = left_crop(preprocessed_img)
        right_part = right_crop(preprocessed_img)
        save_rgb_image(os.path.join(dest_left, file), left_part)
        save_rgb_image(os.path.join(dest_right, file), right_part)


def parse_folder(folder, output_dir, images_are_real=False):
    if folder is None or output_dir is None:
        print(
            "Input folder not specified" if folder is None else "" + "Output folder not specified" if output_dir is None else "")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(folder):
        if len(files) > 0:
            _parse_folder(root, folder, output_dir, 'real', files)
            if not images_are_real:
                _parse_folder(root, folder, output_dir, 'smooth', files, lambda x: smooth_edges(x))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, help='Input folder')
    parser.add_argument('-o', type=str, help='Output folder', default='img')
    parser.add_argument('-r', type=bool, help='Preprocess real images', default=False)
    parse_folder(parser.parse_args().i, parser.parse_args().o, parser.parse_args().r)


if __name__ == '__main__':
    main()
