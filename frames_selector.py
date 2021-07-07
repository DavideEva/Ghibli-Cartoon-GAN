import base64
import os.path
import shutil

import PySimpleGUI as sg
# First the window layout in 2 columns
import cv2

IMG_SIZE = 518

file_list_column = [
  [
    sg.Text("Input Frame Folder"),
    sg.In('img', size=(25, 1), enable_events=True, key="-FOLDER-IN-"),
    sg.FolderBrowse(initial_folder='./'),
  ],
  [
    sg.Text("Output Frame Folder"),
    sg.In('output', size=(25, 1), enable_events=True, key="-FOLDER-OUT-"),
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
    sg.Text("'Left arrow'/A/Y for NO\n'Right arrow'/D/N for YES"),
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
  output_root = root.replace(root.split(os.sep)[0], output_dir, 1)
  if not os.path.exists(output_root):
    os.makedirs(output_root)
  for idx in valid_index:
    shutil.copy(os.path.join(root, files[idx]), os.path.join(output_root, files[idx]))


def extract_scene_number(root):
  scene_folder = root.split(os.sep)[-1]
  scene_folder = ''.join(c for c in scene_folder if c.isdigit())
  return int(scene_folder)


if __name__ == '__main__':
  input_folder = os.path.join(os.path.curdir, 'img')
  output_folder = os.path.join(os.path.curdir, 'output')
  root, files = None, None
  g = None
  i = 0
  next_scene = 1
  while True:
    event, values = window.read(timeout=75)
    # Folder name was filled in, make a list of files in the folder
    if event == "-FOLDER-IN-":
      input_folder = values["-FOLDER-IN-"]
    elif event == "-FOLDER-OUT-":
      output_folder = values["-FOLDER-OUT-"]
    elif event == "-SCENE-":
      next_scene = int('0'+''.join(c for c in values["-SCENE-"] if c.isdigit()))
    elif event == "-START-":
      print('Start with input:',input_folder, 'and output:', output_folder)
      g = navigate_scenes(input_folder)
      window.write_event_value('-NEXT-', values)
      window["-START-"].update(disabled=True)
    elif event == "-NEXT-":
      root_t, files = next(g, (None, None))
      if root_t is None:
        window.close()
        break
      if root is not None and root_t.split(os.sep)[-2] != root.split(os.sep)[-2]:
        next_scene = 1
      root = root_t
      window["-CURR-FILM-"].update(value=root.split(os.sep)[-2])
      scene_n = extract_scene_number(root)
      if scene_n >= next_scene:
        next_scene = scene_n + 1
      else:
        window.write_event_value('-NEXT-', values)
        continue
      window["-CURR-FRAME-"].update(value=f'{next_scene}')
      files = sorted(files)
      image = cv2.imread(os.path.join(root, files[0]))
      image = cv2.resize(image, dsize=(IMG_SIZE, IMG_SIZE))
      _, buffer = cv2.imencode(".png", image)
      window["-TOUT-1-"].update(files[0])
      window["-IMAGE-1-"].update(data=base64.b64encode(buffer))

    elif event is not None and (event.startswith('Right') or event == 'd' or event == 'y'):
      if root is not None and files is not None:
        print('Images accepted!')
        move_images(root, files, output_folder)
        window.write_event_value('-NEXT-', values)
    elif event is not None and (event.startswith('Left') or event == 'a' or event == 'n'):
      if root is not None:
        print('Images rejected!')
        window.write_event_value('-NEXT-', values)
    elif files is not None and event == "__TIMEOUT__":
      i = (i+1)%len(files)
      image = cv2.imread(os.path.join(root, files[i]))
      image = cv2.resize(image, dsize=(IMG_SIZE, IMG_SIZE))
      _, buffer = cv2.imencode(".png", image)
      window["-TOUT-2-"].update(files[i])
      window["-IMAGE-2-"].update(data=base64.b64encode(buffer))
    if event == "Exit" or event == sg.WIN_CLOSED:
      window.close()
      break
