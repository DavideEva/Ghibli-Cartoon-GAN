import glob
import os
import cv2
import os.path
import argparse

from scenedetect import VideoManager, ContentDetector
from scenedetect import SceneManager
from scenedetect.scene_manager import save_images
from tqdm import tqdm

RESOURCE_PATH = 'res'
CASCADE_FILE_PATH = os.path.join(RESOURCE_PATH, 'cascade/lbpcascade_animeface.xml')


def find_scenes(video_path, output_dir='output'):
  video_manager = VideoManager([video_path])

  scene_manager = SceneManager()

  scene_manager.add_detector(ContentDetector(min_scene_len=30))

  scene_list = []
  video_manager.set_downscale_factor()

  video_manager.start()

  scene_manager.detect_scenes(frame_source=video_manager)

  # Obtain list of detected scenes.
  scene_list = scene_manager.get_scene_list()
  # Each scene is a tuple of (start, end) FrameTimecodes.

  save_images(
    scene_list=scene_list,
    video_manager=video_manager,
    num_images=3,
    output_dir=output_dir,
    show_progress=True)
  print()


def face_area_percent(image, cascade_file=CASCADE_FILE_PATH):
  if not os.path.isfile(cascade_file):
    raise RuntimeError("%s: not found" % cascade_file)

  cascade = cv2.CascadeClassifier(cascade_file)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  gray = cv2.equalizeHist(gray)

  faces = cascade.detectMultiScale(gray,
                                   # detector options
                                   scaleFactor=1.05,
                                   minNeighbors=2,
                                   minSize=(24, 24))
  total_area = image.shape[0] * image.shape[1]
  faces_area = 0
  for (x, y, w, h) in faces:
    faces_area += w*h

  return faces_area/total_area


def compare_images(image1, image2):
  hash = cv2.img_hash.AverageHash_create()
  cutoff = 8
  return hash.compare(hash.compute(image1), hash.compute(image2)) < cutoff


def check_face(image):
  return face_area_percent(image) > 0


def analyze_video(path_to_video, output_dir):
  temp = 'Temp'
  if not os.path.exists(temp):
    os.makedirs(temp)
  for f in os.listdir(temp):
    os.remove(os.path.join(temp, f))
  find_scenes(path_to_video, output_dir=temp)
  prev = None

  for root, _, files in tqdm(os.walk(temp), position=0):
    for file in tqdm(sorted(files), position=1, leave=False):
      file_name = os.path.join(root, file)
      newImage = cv2.imread(file_name)
      if prev is not None:
        if compare_images(prev, newImage):
          continue
      if check_face(newImage):
        continue
      cv2.imwrite(os.path.join(output_dir, file), newImage)
      prev = newImage

  for root, dirs, files in os.walk(temp, topdown=False):
    for name in files:
      os.remove(os.path.join(root, name))
    for name in dirs:
      os.rmdir(os.path.join(root, name))
  os.rmdir(temp)


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
        os.remove(os.path.join(folder_path, f))
    analyze_video(v, folder_path)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', type=str, help='Input folder')
  parser.add_argument('-o', type=str, help='Output folder', default='img')
  parse_folder(parser.parse_args().i, parser.parse_args().o)


if __name__ == '__main__':
  main()