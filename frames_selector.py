import base64
import glob
import shutil

import PySimpleGUI as sg
import os.path
import io

# First the window layout in 2 columns
import cv2
import matplotlib.pyplot as plt
import numpy as np

file_list_column = [
  [
    sg.Text("Input Frame Folder"),
    sg.In('./img', size=(25, 1), enable_events=True, key="-FOLDER-IN-"),
    sg.FolderBrowse(initial_folder='img'),
  ],
  [
    sg.Text("Output Frame Folder"),
    sg.In('./output', size=(25, 1), enable_events=True, key="-FOLDER-OUT-"),
    sg.FolderBrowse(initial_folder='output'),
  ],
  [
    sg.Button("Start", key="-START-")
  ],
  [
    sg.Text("'Left arrow'/A/Y for NO\n'Right arrow'/D/N for YES"),
  ]
]

# For now will only show the name of the file that was chosen
image_viewer_column = [
  [sg.Text("First frame")],
  [sg.Text(size=(40, 1), key="-TOUT-1-")],
  [sg.Image(key="-IMAGE-1-", size=(256, 256))],
]

image_viewer_column2 = [
  [sg.Text("Last frame")],
  [sg.Text(size=(40, 1), key="-TOUT-2-")],
  [sg.Image(key="-IMAGE-2-", size=(256, 256))],
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
      yield root, files


def move_images(root, files, output_dir):
  prev = None
  valid_index = []
  for idx, image in enumerate(map(cv2.imread, map(lambda x: os.path.join(root, x), files))):
    if prev is None:
      prev = image
      valid_index.append(idx)
      continue
    if not compare_images(prev, image):
      valid_index.append(idx)
  print(root.split(os.sep)[0], output_dir)
  output_root = root.replace(root.split(os.sep)[0], output_dir, 1)
  if not os.path.exists(output_root):
    os.makedirs(output_root)
  for idx in valid_index:
    shutil.move(os.path.join(root, files[idx]), os.path.join(output_root, files[idx]))


if __name__ == '__main__':
  input_folder = os.path.join(os.path.curdir, 'img')
  output_folder = os.path.join(os.path.curdir, 'output')
  root, files = None, None
  g = None
  while True:
    event, values = window.read()
    # Folder name was filled in, make a list of files in the folder
    if event == "-FOLDER-IN-":
      input_folder = values["-FOLDER-IN-"]
    elif event == "-FOLDER-OUT-":
      output_folder = values["-FOLDER-OUT-"]
    elif event == "-START-":
      print('Start with input:',input_folder, 'and output:', output_folder)
      g = navigate_scenes(input_folder)
      window.write_event_value('-NEXT-', values)
      window["-START-"].update(disabled=True)
    elif event == "-NEXT-":
      root, files = next(g, (None, None))
      if root is None:
        window.close()
        break
      files = sorted(files)
      image = cv2.imread(os.path.join(root, files[0]))
      image = cv2.resize(image, dsize=(256, 256))
      _, buffer = cv2.imencode(".png", image)
      window["-TOUT-1-"].update(files[0])
      window["-IMAGE-1-"].update(data=base64.b64encode(buffer))

      image = cv2.imread(os.path.join(root, files[-1]))
      image = cv2.resize(image, dsize=(256, 256))
      _, buffer = cv2.imencode(".png", image)
      window["-TOUT-2-"].update(files[-1])
      window["-IMAGE-2-"].update(data=base64.b64encode(buffer))
    elif event is not None and (event.startswith('Right') or event == 'd' or event == 'y'):
      if root is not None and files is not None:
        print('Images accepted!')
        move_images(root, files, output_folder)
        window.write_event_value('-NEXT-', values)
    elif event is not None and (event.startswith('Left') or event == 'a' or event == 'n'):
      if root is not None:
        print('Images rejected!')
        window.write_event_value('-NEXT-', values)
    if event == "Exit" or event == sg.WIN_CLOSED:
      window.close()
      break
