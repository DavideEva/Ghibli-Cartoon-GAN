import glob
import os
import pstats
import shutil

import cv2
import os.path
import argparse


from scenedetect import VideoManager, ContentDetector, StatsManager
from scenedetect import SceneManager
from scenedetect.scene_manager import save_images
from tqdm import tqdm


def find_scenes(video_path, output_dir='output'):
  video_manager = VideoManager([video_path])

  scene_manager = SceneManager()

  scene_manager.add_detector(ContentDetector(min_scene_len=30, threshold=20))

  video_manager.set_downscale_factor()

  video_manager.start()

  scene_manager.detect_scenes(frame_source=video_manager)

  scene_list = scene_manager.get_scene_list()

  print('\nSaving scenes frames...')
  save_images(
    scene_list=scene_list,
    video_manager=video_manager,
    num_images=10,
    output_dir=output_dir,
    show_progress=True,
    image_name_template='Scene-$SCENE_NUMBER/Frame-$IMAGE_NUMBER')


def analyze_video(path_to_video, output_dir):
  print(f'Analize video {path_to_video}...')
  print('Searching for scenes...')
  find_scenes(path_to_video, output_dir=output_dir)
  print(f'\nDone {path_to_video}!')


def parse_folder(folder, output_dir):
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  video_file_names = glob.glob(folder)
  for v in video_file_names:
    folder_name, _ = os.path.splitext(os.path.basename(v))
    folder_path = os.path.join(output_dir, folder_name)
    if not os.path.exists(folder_path):
      os.makedirs(folder_path)
    else:
      for f in os.listdir(folder_path):
        shutil.rmtree(os.path.join(folder_path, f))
    analyze_video(v, folder_path)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', type=str, help='Input folder')
  parser.add_argument('-o', type=str, help='Output folder', default='img')
  print(parser.parse_args().i, parser.parse_args().o)
  parse_folder(parser.parse_args().i, parser.parse_args().o)


if __name__ == '__main__':
  main()
