import argparse
import multiprocessing
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


def smooth_all(img):
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.float32) / 25
    return cv2.filter2D(img, -1, kernel)


def smooth_edges(img, return_mask=False):
    # Parameters taken from https://github.com/FilipAndersson245/cartoon-gan/blob/master/utils/datasets.py
    kernel_size = 5
    pad_size = kernel_size // 2 + 1
    gray_img = cv2.cvtColor(np.uint8(img), cv2.COLOR_RGB2GRAY)
    pad_img = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')
    edges = cv2.Canny(gray_img, 150, 500)
    dilation = cv2.dilate(edges, np.ones((kernel_size, kernel_size), np.uint8))
    if return_mask:
        return cv2.cvtColor(dilation, cv2.COLOR_GRAY2RGB)
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
    if image.shape[0] != image.shape[1]:
        return
    # slows down debug:
    #print(file_name)
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


def compare_images(image1, image2):
    avg_hash = cv2.img_hash.AverageHash_create()
    cutoff = 8
    return avg_hash.compare(avg_hash.compute(image1), avg_hash.compute(image2)) < cutoff


def _parse_folder(root, folder, output_dir, dest_type, files, preprocess=lambda x: x, to_skip=None):
    ignore_diff = True
    if to_skip is None:
        to_skip = []
        ignore_diff = False
    rel_path = os.path.relpath(root, folder)
    dest = os.path.join(output_dir, dest_type, rel_path)
    skipped = []
    if any(map(lambda f: f.endswith('.jpg') or f.endswith('.png'), files)):
        prev = None
        os.makedirs(dest, exist_ok=True)
        for idx, file in enumerate(files):
            file_dest = file.removesuffix('.jpg').removesuffix('.png')
            file_dest_left = os.path.join(dest, file_dest + '.left.jpg')
            file_dest_right = os.path.join(dest, file_dest + '.right.jpg')
            if idx in to_skip:
                continue
            if os.path.exists(file_dest_left):
                continue  # Skip if already exist
            img = cv2.imread(os.path.join(root, file))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            preprocessed_img = preprocess(resize_image_height(img, preprocess_shape[0]))
            if not ignore_diff and prev is not None and compare_images(prev, preprocessed_img):
                skipped.append(idx)
                continue
            prev = preprocessed_img
            left_part = left_crop(preprocessed_img)
            right_part = right_crop(preprocessed_img)
            save_rgb_image(file_dest_left, left_part)
            save_rgb_image(file_dest_right, right_part)
    return skipped


def parse_folder_job(root, folder, output_dir, files, images_are_real):
    skipped = _parse_folder(root, folder, output_dir, 'real', files)
    if not images_are_real:
        _parse_folder(root, folder, output_dir, 'smooth', files, lambda x: smooth_edges(x), to_skip=skipped)
        _parse_folder(root, folder, output_dir, 'mask', files, lambda x: smooth_edges(x, return_mask=True), to_skip=skipped)


def _call_parse_folder_job(x):
    parse_folder_job(*x)


def parse_folder(folder, output_dir, images_are_real=False):
    if folder is None or output_dir is None:
        print(
            "Input folder not specified" if folder is None else "" + "Output folder not specified" if output_dir is None else "")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    to_preprocess = []
    for root, dirs, files in os.walk(folder):
        if len(files) > 0:
            to_preprocess.append((root, folder, output_dir, files, images_are_real))
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        p.map(_call_parse_folder_job, to_preprocess)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, help='Input folder')
    parser.add_argument('-o', type=str, help='Output folder', default='img')
    parser.add_argument('-r', type=bool, help='Preprocess real images', default=False)
    parse_folder(parser.parse_args().i, parser.parse_args().o, parser.parse_args().r)


if __name__ == '__main__':
    main()
