import base64
import os.path
import re
from itertools import groupby
from operator import itemgetter
import shutil

import PySimpleGUI as sg
# First the window layout in 2 columns
import cv2

IMG_SIZE = 518

regex = r'^Scene-(\d+)\.Frame-(\d+).(?:jpg|png)$'

file_list_column = [
  [
    sg.Text("Input Frame Folder"),
    sg.In('img', size=(25, 1), enable_events=True, key="-FOLDER-IN-"),
    sg.FolderBrowse(initial_folder='./'),
  ],
  [
    sg.Text("Output Frame Folder"),
    sg.In('res/img', size=(25, 1), enable_events=True, key="-FOLDER-OUT-"),
    sg.FolderBrowse(initial_folder='./'),
  ],
  [
    sg.Text("Next scene number"),
    sg.In('1', size=(25, 1), enable_events=True, key="-SCENE-"),
  ],
  [
    sg.Text("Current Film"),
    sg.Text("???", key="-CURR-FILM-", size=(10, 1)),
  ],
  [
    sg.Text("Current frame"),
    sg.Text("0", key="-CURR-FRAME-", size=(10, 1)),
  ],
  [
    sg.Button("Start", key="-START-")
  ],
  [
    sg.Text("'Left arrow'/A/N for NO\n'Right arrow'/D/Y for YES"),
  ]
]

# For now will only show the name of the file that was chosen
image_viewer_column = [
  [sg.Text("First frame")],
  [sg.Text(size=(40, 1), key="-TOUT-1-")],
  [sg.Image(key="-IMAGE-1-", size=(IMG_SIZE, IMG_SIZE))],
]

image_viewer_column2 = [
  [sg.Text("More frames")],
  [sg.Text(size=(40, 1), key="-TOUT-2-")],
  [sg.Image(key="-IMAGE-2-", size=(IMG_SIZE, IMG_SIZE))],
]

# ----- Full layout -----
layout = [
  [
    sg.Column(file_list_column),
    sg.VSeperator(),
    sg.Column(image_viewer_column),
    sg.VSeperator(),
    sg.Column(image_viewer_column2),
  ]
]

window = sg.Window("Image Viewer", layout, return_keyboard_events=True, use_default_focus=False)


def compare_images(image1, image2):
  hash = cv2.img_hash.AverageHash_create()
  cutoff = 8
  return hash.compare(hash.compute(image1), hash.compute(image2)) < cutoff


def navigate_scenes(path_to_frames):
  for root, dirs, files in os.walk(path_to_frames):
    if len(files) > 0:
      data = map(lambda file_name: (*(list(map(int, re.findall(regex, file_name)[0]))), file_name, root), filter(lambda x: len(re.findall(regex, x)) > 0, files))
      for scene_n, frames in groupby(data, itemgetter(0)):
        frames = list(frames)
        f_files = list(map(itemgetter(2), frames))
        c_root = frames[0][3] # all the root are the same
        yield os.path.normpath(c_root), scene_n, f_files


def move_images(root, file_dir, files, output_dir, files_name_update=None):
  prev = None
  valid_index = []
  for idx, image in enumerate(map(cv2.imread, map(lambda x: os.path.join(root, x), files))):
    if prev is None:
      prev = image
      valid_index.append(idx)
      continue
    if not compare_images(prev, image):
      valid_index.append(idx)
      prev = image
  output_root = os.path.join(output_dir, file_dir)
  if not os.path.exists(output_root):
    os.makedirs(output_root)
  new_files_name = files
  if files_name_update is not None:
    new_files_name = list(map(files_name_update, files))
  for idx in valid_index:
    shutil.copy(os.path.join(root, files[idx]), os.path.join(output_root, new_files_name[idx]))


def extract_scene_number(root):
  scene_folder = root.split(os.sep)[-1]
  scene_folder = ''.join(c for c in scene_folder if c.isdigit())
  return int(scene_folder)


def image_to_base64(image_path):
  image = cv2.imread(image_path)
  image = cv2.resize(image, dsize=(IMG_SIZE, IMG_SIZE))
  _, buffer = cv2.imencode(".png", image)
  return base64.b64encode(buffer)


if __name__ == '__main__':
  input_folder = os.path.join(os.path.curdir, 'img')
  output_folder = os.path.join(os.path.curdir, 'output')
  root, files, images = None, None, []
  g = None
  i = 0
  next_scene = 1
  while True:
    event, values = window.read(timeout=75)
    # Folder name was filled in, make a list of files in the folder
    if event == "-FOLDER-IN-":
      input_folder = os.path.normpath(values["-FOLDER-IN-"])
    elif event == "-FOLDER-OUT-":
      output_folder = os.path.normpath(values["-FOLDER-OUT-"])
    elif event == "-SCENE-":
      next_scene = int('0'+''.join(c for c in values["-SCENE-"] if c.isdigit()))
    elif event == "-START-":
      print('Start with input:',input_folder, 'and output:', output_folder)
      g = navigate_scenes(input_folder)
      window.write_event_value('-NEXT-', values)
      window["-START-"].update(disabled=True)
      window.force_focus()
    elif event == "-NEXT-":
      root_t, scene_n, files = next(g, (None, None, None))
      if root_t is None:
        window.close()
        next_scene = 1
        break
      if root is not None and root_t.split(os.sep)[-1] != root.split(os.sep)[-1]:
        next_scene = scene_n + 1
      root = root_t
      window["-CURR-FILM-"].update(value=root.split(os.sep)[-1])
      if scene_n >= next_scene:
        next_scene = scene_n + 1
      else:
        window.write_event_value('-NEXT-', values)
        continue
      window["-CURR-FRAME-"].update(value=f'{scene_n}')
      files = sorted(files)
      window["-TOUT-1-"].update(files[0])
      window["-IMAGE-1-"].update(data=image_to_base64(os.path.join(root, files[0])))
      images = list(map(image_to_base64, map(lambda x: os.path.join(root, x), files)))
    elif event is not None and (event.startswith('Right') or event == 'd' or event == 'y'):
      if root is not None and files is not None:
        print('Images accepted!')
        move_images(root, str(window["-CURR-FILM-"].get()), files, output_folder)
        window.write_event_value('-NEXT-', values)
    elif event is not None and (event.startswith('Left') or event == 'a' or event == 'n'):
      if root is not None:
        print('Images rejected!')
        window.write_event_value('-NEXT-', values)
    elif files is not None and event == "__TIMEOUT__":
      i = (i+1)%len(files)
      window["-TOUT-2-"].update(files[i])
      window["-IMAGE-2-"].update(data=images[i])
    if event == "Exit" or event == sg.WIN_CLOSED:
      window.close()
      break
