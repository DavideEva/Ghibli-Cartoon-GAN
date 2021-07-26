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


def parse_folder(folder, output_dir):
    if folder is None or output_dir is None:
        print(
            "Input folder not specified" if folder is None else "" + "Output folder not specified" if output_dir is None else "")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(folder):
        rel_path = os.path.relpath(root, folder)
        dest_real = os.path.join(output_dir, "real", rel_path)
        dest_smooth = os.path.join(output_dir, "smooth", rel_path)
        dest_real_left = os.path.join(dest_real, "left")
        dest_real_right = os.path.join(dest_real, "right")
        dest_smooth_left = os.path.join(dest_smooth, "left")
        dest_smooth_right = os.path.join(dest_smooth, "right")
        if len(files) > 0:
            os.makedirs(dest_real_left, exist_ok=True)
            os.makedirs(dest_real_right, exist_ok=True)
            os.makedirs(dest_smooth_left, exist_ok=True)
            os.makedirs(dest_smooth_right, exist_ok=True)
            for file in files:
                img = cv2.cvtColor(cv2.imread(os.path.join(root, file)), cv2.COLOR_BGR2RGB)
                target_size = (raw_shape[1] * preprocess_shape[0] // raw_shape[0], preprocess_shape[0])
                preprocess_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                left_part = left_crop(preprocess_img)
                right_part = right_crop(preprocess_img)
                left_part_smooth = smooth_edges(left_part)
                right_part_smooth = smooth_edges(right_part)
                save_rgb_image(os.path.join(dest_real_left, file), left_part)
                save_rgb_image(os.path.join(dest_real_right, file), right_part)
                save_rgb_image(os.path.join(dest_smooth_left, file), left_part_smooth)
                save_rgb_image(os.path.join(dest_smooth_right, file), right_part_smooth)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, help='Input folder')
    parser.add_argument('-o', type=str, help='Output folder', default='img')
    parse_folder(parser.parse_args().i, parser.parse_args().o)


if __name__ == '__main__':
    main()
