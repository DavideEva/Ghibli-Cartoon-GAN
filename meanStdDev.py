import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm


def main(input_dir):
  result = []
  for root, dirs, files in os.walk(input_dir):
    if len(files) > 0:
      for file in tqdm(files):
        img = cv2.imread(os.path.join(root, file))
        result.append(cv2.meanStdDev(img))
  return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, nargs='+', help='Input Folders')
    print(parser.parse_args().i)

    parser.add_argument('-o', type=str, help='Output folder', default='..\\meanStdDev')

    meanStdDev = None
    for i in parser.parse_args().i:
      if meanStdDev is None:
        meanStdDev = main(i)
      else:
        meanStdDev = np.concatenate([meanStdDev, main(i)])

    output_dir = os.path.join(parser.parse_args().o)
    if not os.path.exists(os.path.join(os.getcwd(), output_dir)):
        os.makedirs(os.path.join(os.getcwd(), output_dir), exist_ok=True)
    output_file_name = 'stats.txt'
    with open(os.path.join(os.getcwd(), output_dir, output_file_name), mode='w') as f:
      f.write(str(np.mean(meanStdDev, axis=0)))