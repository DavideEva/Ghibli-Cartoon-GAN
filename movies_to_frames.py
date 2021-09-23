import argparse
import os
import os.path
import shutil

from scenedetect import SceneManager
from scenedetect import VideoManager, ContentDetector
from scenedetect.scene_manager import save_images


def find_scenes(video_path, output_dir='output', frames_per_scene=3):
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
    num_images=frames_per_scene,
    output_dir=output_dir,
    show_progress=True,
    image_name_template='Scene-$SCENE_NUMBER.Frame-$IMAGE_NUMBER')


def analyze_video(path_to_video, output_dir, frames_per_scene):
  print(f'Analize video {path_to_video}...')
  print('Searching for scenes...')
  find_scenes(path_to_video, output_dir=output_dir, frames_per_scene=frames_per_scene)
  print(f'\nDone {path_to_video}!')


def parse_folder(folder, output_dir, frames_per_scene):
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  for root, dirs, files in os.walk(folder):
    for v in files:
      if v.endswith('.mp4') or v.endswith('.mkv'):
        rel_path = os.path.relpath(root, folder)
        dest = os.path.join(output_dir, rel_path, os.path.basename(v))
        src = os.path.join(root, v)
        if not os.path.exists(dest):
          os.makedirs(dest)
        else:
          for f in os.listdir(dest):
            shutil.rmtree(os.path.join(dest, f))
        analyze_video(src, dest, frames_per_scene)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', type=str, help='Input folder')
  parser.add_argument('-o', type=str, help='Output folder', default='img')
  parser.add_argument('-n', type=int, help='Frames per scene', default=3)
  print(parser.parse_args().i, parser.parse_args().o, parser.parse_args().n)
  parse_folder(parser.parse_args().i, parser.parse_args().o, parser.parse_args().n)


if __name__ == '__main__':
  main()
