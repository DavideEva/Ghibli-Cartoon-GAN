# -*- coding: utf-8 -*-
import re
from io import TextIOWrapper

vgg_16_layer = 14
epochs = 50
omega = 0.1

import os
from dotenv import load_dotenv
load_dotenv()
training_folder_suffix = os.getenv('FOLDER_SUFFIX', '')
training_folder_suffix = training_folder_suffix if training_folder_suffix == '' else f'_{training_folder_suffix}'


from glob import glob
from zipfile import ZipFile

import gdown

flickr30k_folder = 'Flickr30k-images-preprocessed'

if not os.path.exists(flickr30k_folder):
  gdown.download(id="10c0Xruu2wAE-FpQEIlXm17HlwQAJKKVM")
  with ZipFile('flickr30k-images-preprocessed.zip', 'r') as zipObj:
    zipObj.extractall(path=flickr30k_folder)
  os.remove('flickr30k-images-preprocessed.zip')

# Unfiltered dataset: 1yYk3ZyxXdP2KHjE31fSTFGCNYb2QbSHe
# Filtered dataset: 1bR_BE-ZZSXW1URBJqPIMFo9VLODYb4_k

films_data = {
    "Ghibli": "1bR_BE-ZZSXW1URBJqPIMFo9VLODYb4_k"
}
ghibli_index = 0
real_folder = 'real'
smooth_folder = 'smooth'
mask_folder = 'mask'

def load_data(id, name):
  if os.path.exists(name):
    return name
  os.mkdir(name)
  gdown.download(id=id, output=os.path.join(name, 'movies.zip'))

  zip_files = glob(f'{name}/*.zip')
  for zip_file in zip_files:
    with ZipFile(zip_file, 'r') as zipObj:
      zipObj.extractall(path=name)
    os.remove(zip_file)
  return name

folders = [load_data(id_drive, studio_name) for studio_name, id_drive in films_data.items()]

"""# Import"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import math
import shutil
import csv
import urllib
from os import path
from matplotlib import pyplot as plt
# %matplotlib inline
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import Layer, LeakyReLU, Input, Conv2D, Conv2DTranspose, BatchNormalization, AveragePooling2D, Add
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.inception_v3 import InceptionV3
from numpy import cov, trace, iscomplexobj
from scipy.linalg import sqrtm

# ! pip install tensorflow-addons
import tensorflow_addons as tfa
from tensorflow_addons.optimizers import CyclicalLearningRate

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

"""# Plot functions"""

def plot_grid(images, columns, show_axis=False, labels=None):
  if len(images) == 0 or columns <= 0:
    return
  scale = 1.5
  height = (1 + math.ceil(len(images) / columns) * 2) * scale
  width = (columns * 4) * scale
  dpi = min(max(images[0].shape[0], images[0].shape[1]) // 2, 480)
  fig = plt.figure(figsize=(width, height), dpi=dpi)
  fig.subplots_adjust(hspace=0.4)
  for index, img in enumerate(images, start=1):
    if 'float' in img.dtype.name:
      img = (img * 255).astype('uint8')
    sp = fig.add_subplot(math.ceil(len(images) / columns), columns, index)
    if not show_axis:
      plt.axis('off')
    if len(np.shape(img)) == 2 or (len(np.shape(img)) > 2 and np.shape(img)[2] == 1):
      img = img.squeeze()
      plt.imshow(img, cmap='gray', vmin=0.0, vmax=255.0)
    else:
      plt.imshow(img, vmin=0.0, vmax=255.0)
    if labels is not None:
      l = len(labels)
      sp.set_title(labels[(index-1) % l], fontsize=10)
    else:
      sp.set_title(index, fontsize=10)
  plt.show()

def float_to_int_images(outputs):
  return [np.clip(output * 255 + 0.5, 0, 255).astype(np.uint8) for output in outputs]

def image_name_to_number(name):
  return int("".join(list(filter(str.isdigit, name))))

def save_plot_grid(images, columns, name, directory='', ext='.png', show_axis=False, labels=None, title=''):
  if len(images) == 0 or columns <= 0:
    return
  scale = 1.5
  height = (1 + math.ceil(len(images) / columns) * 2) * scale
  width = (columns * 4) * scale
  dpi = min(max(images[0].shape[0], images[0].shape[1]) // 2, 480)
  fig = plt.figure(figsize=(width, height), dpi=dpi)
  fig.subplots_adjust(hspace=0.4)
  for index, img in enumerate(images, start=1):
    if 'float' in img.dtype.name:
      img = (img * 255).astype('uint8')
    sp = fig.add_subplot(math.ceil(len(images) / columns), columns, index)
    if not show_axis:
      plt.axis('off')
    if len(np.shape(img)) == 2 or (len(np.shape(img)) > 2 and np.shape(img)[2] == 1):
      img = img.squeeze()
      plt.imshow(img, cmap='gray', vmin=0.0, vmax=255.0)
    else:
      plt.imshow(img, vmin=0.0, vmax=255.0)
    if labels is not None:
      l = len(labels)
      sp.set_title(labels[(index-1) // columns], fontsize=14)
    else:
      sp.set_title(index, fontsize=10)
  if title != '':
    fig.suptitle(title, fontsize=16)
  fig.savefig(os.path.join(directory, f'{name}{ext}'))
  plt.close(fig)

"""# Global parameters"""

# Dimension after the preprocess stage
# Should be the dimension expected by the network and the loss functions
input_shape = (224, 224, 3)

# Batch size used for training and fetching images
batch_size = 16

# Images are split between train+validation and test set at this proportion
validation_split = 0.2

"""# Dataset loading and preprocessing"""

def lambda_generator(batches, f=lambda x: x):
  for batch in batches:
    if type(batch) is tuple:
      batch, labels = batch
      yield [f(i) for i in batch], labels
    else:
      yield [f(i) for i in batch]

def random_merge_generator(it_1, it_2, p=0.5):
  while True:
    rand = np.random.random()
    it, other = (it_1, it_2) if rand < p else (it_2, it_1)
    try:
      yield next(it)
    except StopIteration:
      while True:
        yield next(other)

norm_mean = np.asfarray([0.485, 0.456, 0.406])
norm_std = np.asfarray([0.229, 0.224, 0.225])

def normalize(img):
  return (img - norm_mean) / norm_std
def unnormalize(img):
  return tf.clip_by_value(img * norm_std + norm_mean, 0.0, 1.0)
def rescale_and_normalize(img):
  return normalize(img / 255.0)

def generated_to_images(outputs):
  return [unnormalize(output).numpy() for output in outputs]

data_generator_settings = {
    'data_format' : 'channels_last',
    'validation_split' : validation_split,
    'preprocessing_function' : rescale_and_normalize,
    #'rescale' : 1.0 / 255,
    'horizontal_flip' : True
}

data_flow_settings = {
    'color_mode' : 'rgb',
    'batch_size' : batch_size,
    'shuffle' : True,
    'seed' : 42, # Mandatory to allign shuffles between cartoon_real and cartoon_smooth
    'class_mode' : None,
    'interpolation' : 'bilinear',
    'target_size' : (input_shape[0], input_shape[1])
}

def cartoon_real_generator(subset='training'):
  cartoon_real_gen = ImageDataGenerator(
    **data_generator_settings
  )
  return cartoon_real_gen.flow_from_directory(
        **data_flow_settings,
        # Ghibli cartoon
        directory = path.join(folders[ghibli_index], real_folder),
        subset = subset
      )

def cartoon_real_validation_generator():
  return cartoon_real_generator('validation')

def cartoon_smooth_generator(subset='training'):
  cartoon_smooth_gen = ImageDataGenerator(
    **data_generator_settings
  )
  return cartoon_smooth_gen.flow_from_directory(
        **data_flow_settings,
        # Ghibli cartoon
        directory = path.join(folders[ghibli_index], smooth_folder),
        subset = subset
      )

def cartoon_smooth_validation_generator():
  return cartoon_smooth_generator('validation')

def real_generator(subset='training'):
  real_gen = ImageDataGenerator(
      **data_generator_settings
  )
  return real_gen.flow_from_directory(
      **data_flow_settings,
      # Flickr30k images
      directory=flickr30k_folder,
      subset=subset
  )

def real_validation_generator():
  return real_generator('validation')

def smooth_label_generator(subset='training'):
  def mask_to_label(mask):
    return (mask <= 0).astype(np.float32)
  label_data_generator_settings = {**data_generator_settings}
  label_data_generator_settings['preprocessing_function'] = mask_to_label
  smooth_mask_gen = ImageDataGenerator(
    **label_data_generator_settings
  )
  label_data_flow_settings = {**data_flow_settings}
  label_data_flow_settings['target_size'] = (56, 56) # Discriminator label size
  label_data_flow_settings['color_mode'] = 'grayscale'
  return smooth_mask_gen.flow_from_directory(
      **label_data_flow_settings,
      # Mask images
      directory=path.join(folders[ghibli_index], mask_folder),
      subset=subset
  )

def smooth_label_validation_generator(subset='training'):
  return smooth_label_generator('validation')

"""# Cartoon-GAN

## Custom Convolutional Layers
"""

class ReflectionPadding2D(Layer):
  def __init__(self, padding=(1, 1), **kwargs):
    self.padding = tuple(padding)
    # self.input_spec = [InputSpec(ndim=4)]
    super(ReflectionPadding2D, self).__init__(**kwargs)

  def compute_output_shape(self, s):
    if s[1] == None:
      return (None, None, None, s[3])
    return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

  def call(self, x, mask=None):
    w_pad, h_pad = self.padding
    return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')

  def get_config(self):
    config = super(ReflectionPadding2D, self).get_config()
    return config

class Conv2DReflection3x3(Layer):
  def __init__(self, features, stride=1):
    super().__init__()
    self.reflectionPadding2D = ReflectionPadding2D()
    self.conv2d = Conv2D(features, (3,3), strides=(stride, stride), padding='valid', use_bias=False)

  def call(self, inputs, training=True):
    x = self.reflectionPadding2D(inputs, training=training)
    return self.conv2d(x, training=training)

"""## Discriminator
Based on the Cartoon-GAN discriminator, available at [this link](https://github.com/FilipAndersson245/cartoon-gan/blob/master/models/discriminator.py).
"""

# define the discriminator model
def define_discriminator(image_shape):
  alpha = 0.2
  epsilon = 1e-5
  momentum = 0.1

  # source image input
  in_image = Input(shape=image_shape)

  # k3n32s1
  d = Conv2DReflection3x3(32, stride=1)(in_image)
  d = LeakyReLU(alpha=alpha)(d)

  # k3n64s2
  d = Conv2DReflection3x3(64, stride=2)(d)
  d = LeakyReLU(alpha=alpha)(d)
  # k3n128s1
  d = Conv2DReflection3x3(128, stride=1)(d)
  d = BatchNormalization(epsilon=epsilon, momentum=momentum)(d)
  d = LeakyReLU(alpha=alpha)(d)

  # k3n128s2
  d = Conv2DReflection3x3(128, stride=2)(d)
  d = LeakyReLU(alpha=alpha)(d)
  # k3n256s1
  d = Conv2DReflection3x3(256, stride=1)(d)
  d = BatchNormalization(epsilon=epsilon, momentum=momentum)(d)
  d = LeakyReLU(alpha=alpha)(d)

  # feature construction block
  # k3n256s1
  #d = Conv2DReflection3x3(256, stride=1)(d)
  #d = BatchNormalization(epsilon=epsilon, momentum=momentum)(d)
  #d = LeakyReLU(alpha=alpha)(d)

  # patch output
  d = Conv2DReflection3x3(1, stride=1)(d)
  #d = tf.keras.activations.sigmoid(d)

  # define model
  model = Model(in_image, d, name='Discriminator')
  return model

D = define_discriminator(input_shape)

"""## Generator
Based on the Cartoon-GAN generator, available at [this link](https://github.com/FilipAndersson245/cartoon-gan/blob/master/models/generator.py).
"""

# define the generator model
def define_generator(image_shape):
  alpha = 0.2
  epsilon = 1e-5
  momentum = 0.1

  # source image input
  in_image = Input(shape=image_shape)

  # flat block
  # k7n64s1
  g = Conv2D(64, (7,7), strides=1, padding='same', use_bias=False)(in_image)
  g = BatchNormalization(epsilon=epsilon, momentum=momentum)(g)
  g = LeakyReLU(alpha=alpha)(g)

  def down_block(x, n_features):
    # k3n?s2
    x = Conv2DReflection3x3(n_features, stride=2)(x)
    # k3n?s1
    x = Conv2DReflection3x3(n_features, stride=1)(x)
    x = BatchNormalization(epsilon=epsilon, momentum=momentum)(x)
    x = LeakyReLU(alpha=alpha)(x)
    return x

  # 1st down block
  g = down_block(g, 128)

  # 2nd down block
  g = down_block(g, 256)

  def resiual_block(x):
    skip = x
    # k3n256s1
    x = Conv2DReflection3x3(256, stride=1)(x)
    x = BatchNormalization(epsilon=epsilon, momentum=momentum)(x)
    x = LeakyReLU(alpha=alpha)(x)
    # k3n256s1
    x = Conv2DReflection3x3(256, stride=1)(x)
    x = BatchNormalization(epsilon=epsilon, momentum=momentum)(x)
    x = Add()([x, skip])
    x = LeakyReLU(alpha=alpha)(x)
    return x

  for _ in range(8):
    g = resiual_block(g)

  def up_block(x, n_features):
    # k3n?s1/2
    x = Conv2DTranspose(n_features, (3,3), strides=2)(x)
    x = AveragePooling2D(pool_size=(2,2), strides=1)(x)
    # k3n?s1
    x = Conv2DReflection3x3(n_features, stride=1)(x)
    x = BatchNormalization(epsilon=epsilon, momentum=momentum)(x)
    x = LeakyReLU(alpha=alpha)(x)
    return x

  # 1st up block
  g = up_block(g, 128)

  # 2nd up-block
  g = up_block(g, 64)

  # k7n3s1
  output = Conv2D(3, (7,7), strides=1, padding='same')(g)

  # define model
  model = Model(in_image, output, name='Generator')
  return model


G = define_generator(input_shape)

"""## Loss functions

### Binary Cross Entropy
"""

def BCEWithLogitsLoss():
  bce = tf.keras.losses.BinaryCrossentropy(
    from_logits=True,
    reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
  return lambda x, y: bce(y, x)

def BCELoss():
  bce = tf.keras.losses.BinaryCrossentropy(
    from_logits=False,
    reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
  return lambda x, y: bce(y, x)

"""### Adversarial Loss
Also called Discriminator loss.
"""

# Penalizes smoothed edges
class AdversarialLoss:
  def __init__(self, cartoon_labels, fake_cartoon_labels):
    self.base_loss = BCEWithLogitsLoss()
    self.cartoon_labels = cartoon_labels
    self.fake_cartoon_labels = fake_cartoon_labels

  def __call__(self, cartoons_outputs, generated_fakes_outputs, cartoon_edge_fakes_outputs, cartoon_edge_fake_labels):
    D_cartoon_loss = self.base_loss(cartoons_outputs, self.cartoon_labels)
    D_generated_fake_loss = self.base_loss(generated_fakes_outputs, self.fake_cartoon_labels)
    D_edge_fake_loss = self.base_loss(cartoon_edge_fakes_outputs, cartoon_edge_fake_labels)

    return D_cartoon_loss + D_generated_fake_loss + D_edge_fake_loss

# Ignores smoothed edges
class SimplifiedAdversarialLoss:
  def __init__(self, cartoon_labels, fake_cartoon_labels):
    self.base_loss = BCEWithLogitsLoss()
    self.cartoon_labels = cartoon_labels
    self.fake_cartoon_labels = fake_cartoon_labels

  def __call__(self, cartoons_outputs, generated_fakes_outputs):
    D_cartoon_loss = self.base_loss(cartoons_outputs, self.cartoon_labels)
    D_generated_fake_loss = self.base_loss(generated_fakes_outputs, self.fake_cartoon_labels)

    return D_cartoon_loss + D_generated_fake_loss

# Download ImageNet
path_imagenet = os.path.join('ImageNet', 'Keras', 'weights')
file_imagenet = os.path.join(path_imagenet, 'vgg16_imagenet.h5')
if not os.path.exists(file_imagenet):
  os.makedirs(path_imagenet)
  url_imagenet = 'https://raw.githubusercontent.com/ezavarygin/vgg16_pytorch2keras/master/vgg16_pytorch2keras.h5'
  from urllib.request import URLopener

  testfile = URLopener()
  testfile.retrieve(url_imagenet, file_imagenet)

from tensorflow.keras.applications import vgg16

vgg16_model = vgg16.VGG16(include_top=False,
                    weights=file_imagenet,
                    input_shape=input_shape,
                    )

vgg16_model.trainable = False
for l in vgg16_model.layers:
  l.trainable = False
vgg16_cut = Sequential(vgg16_model.layers[:vgg_16_layer], name="ContentLoss_VGG16") # TODO assegnare il valore giusto a vgg
vgg16_cut.layers[-1].activation = None # -1 is the last layer
vgg16_cut.trainable = False
vgg16_cut.summary()

class ContentLoss:
  def __init__(self):
    def perc(img):
      return vgg16_cut(img, training=False)
    self.perception = perc

  def __call__(self, outputs, targets):
    diff = self.perception(outputs) - self.perception(targets)
    k = tf.math.reduce_mean(tf.math.abs(diff))
    return k

"""### Generator Loss
Enforces both discriminator fooling and content fidelty from the original.
"""

class GeneratorLoss:
  def __init__(self, cartoon_labels, omega=10):
    self.omega = tf.constant(omega, dtype=tf.float32)
    self.content_loss = ContentLoss()
    self.base_loss = BCEWithLogitsLoss()
    self.cartoon_labels = cartoon_labels

  def __call__(self, outputs, inputs, outputs_labels):
    return self.base_loss(outputs_labels, self.cartoon_labels) + self.omega * self.content_loss(outputs, inputs)

"""## Learning Rate"""

cyclical_learning_rate = CyclicalLearningRate(
    initial_learning_rate=1e-3,
    maximal_learning_rate=1e-2,
    step_size=batch_size * 2,
    scale_fn=lambda x: 1 / (2. ** (x - 1))
  )

learning_rate = cyclical_learning_rate #1e-3
beta1, beta2 = (.5, .99)
weight_decay = 1e-4

discriminator_optimizer = tfa.optimizers.AdamW(
    learning_rate=learning_rate,
    beta_1=beta1, beta_2=beta2,
    weight_decay=weight_decay
)
generator_optimizer = tfa.optimizers.AdamW(
    learning_rate=learning_rate,
    beta_1=beta1, beta_2=beta2,
    weight_decay=weight_decay
)

"""## Metrics

### FID
"""

model_inception_v3 = InceptionV3(include_top=False, pooling='avg', input_shape=input_shape)

# calculate frechet inception distance
def calculate_fid(_model, images1, images2):
  # calculate activations
  act1 = _model.predict(images1)
  act2 = _model.predict(images2)

  # calculate mean and covariance statistics
  mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
  mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)

  # calculate sum squared difference between means
  ssdiff = np.sum((mu1 - mu2) ** 2.0)

  # calculate sqrt of product between cov
  covmean = sqrtm(sigma1.dot(sigma2))

  # check and correct imaginary numbers from sqrt
  if iscomplexobj(covmean):
    covmean = covmean.real

  # calculate score
  fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
  return fid

def evaluate_fid(base_images, test_images):
  return calculate_fid(model_inception_v3, base_images, test_images)

# pre_evaluate fid for cartoon
# calculate activations over cartoon val
cartoon_gen = cartoon_real_validation_generator()
act_cartoon_gen = model_inception_v3.predict(cartoon_gen)
mu_cartoon_gen, sigma_cartoon_gen = act_cartoon_gen.mean(axis=0), cov(act_cartoon_gen, rowvar=False)

def evaluate_fid_with_cartoon_val(images, mu_cartoon_gen=mu_cartoon_gen, sigma_cartoon_gen=sigma_cartoon_gen):
  actual_n_images = len(act_cartoon_gen)

  # calculate activations cap to actual_n_images
  act2 = model_inception_v3.predict(images)[:actual_n_images]
  if len(act2) < actual_n_images:
    act_cartoon_gen_crop = act_cartoon_gen[:len(act2)]
    mu_cartoon_gen, sigma_cartoon_gen = act_cartoon_gen_crop.mean(axis=0), cov(act_cartoon_gen_crop, rowvar=False)

  # calculate mean and covariance statistics
  mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)

  # calculate sum squared difference between means
  ssdiff = np.sum((mu_cartoon_gen - mu2) ** 2.0)

  # calculate sqrt of product between cov
  covmean = sqrtm(sigma_cartoon_gen.dot(sigma2))

  # check and correct imaginary numbers from sqrt
  if iscomplexobj(covmean):
    covmean = covmean.real

  # calculate score
  fid = ssdiff + trace(sigma_cartoon_gen + sigma2 - 2.0 * covmean)
  return fid

"""## Checkpoints & History
In order to save time, we save progress checkpoints.
"""

local_checkpoint_location = f'./training_checkpoints{training_folder_suffix}/'
local_pretrain_checkpoint_location = './pretraining_checkpoints'
history_file_name = 'history.csv'
history_header = ['epoch',
                  'train_discriminator_loss_mean',
                  'train_discriminator_loss_std',
                  'train_generator_loss_mean',
                  'train_generator_loss_std',
                  'val_discriminator_loss_mean',
                  'val_discriminator_loss_std',
                  'val_generator_loss_mean',
                  'val_generator_loss_std',
                  'fid_score'
                  ]
history_file_path = path.join(local_checkpoint_location, history_file_name)

def reset_history():
  with open(history_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(history_header)

def history_append(epoch,
                   train_discriminator_loss_mean,
                   train_discriminator_loss_std,
                   train_generator_loss_mean,
                   train_generator_loss_std,
                   val_discriminator_loss_mean,
                   val_discriminator_loss_std,
                   val_generator_loss_mean,
                   val_generator_loss_std,
                   fid_score):
  with open(history_file_path, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow([epoch,
                   train_discriminator_loss_mean,
                   train_discriminator_loss_std,
                   train_generator_loss_mean,
                   train_generator_loss_std,
                   val_discriminator_loss_mean,
                   val_discriminator_loss_std,
                   val_generator_loss_mean,
                   val_generator_loss_std,
                   fid_score])

"""#### Google Drive checkpoint backup"""

use_google_drive = False #@param {type:'boolean'}
google_drive_checkpoint_path = 'Anime-Frames/Checkpoints' #@param {type: 'string'}
reset_checkpoints = os.getenv('RESET_CHECKPOINTS', 'false')
if reset_checkpoints == 'true':
  reset_checkpoints = True
elif reset_checkpoints == 'false':
  reset_checkpoints = False
else:
  raise ValueError('RESET_CHECKPOINTS can be false or true')
google_drive_root = '/content/drive/'
google_drive_checkpoint_location = path.join(google_drive_root, 'MyDrive', google_drive_checkpoint_path)
image_directory = os.path.join(local_checkpoint_location, 'images')

# if use_google_drive:
#   from google.colab import drive
#   drive.mount(google_drive_root)
#   os.makedirs(google_drive_checkpoint_location, exist_ok=True)
#   shutil.rmtree(local_checkpoint_location, ignore_errors=True)
#   shutil.copytree(google_drive_checkpoint_location, local_checkpoint_location)
#   print(f'Local files at {local_checkpoint_location} will be backed inside {google_drive_checkpoint_location}')
# else:
#   try:
#     drive.flush_and_unmount()
#     !rm -rf /content/drive
#   except:
#     pass


if reset_checkpoints:
  shutil.rmtree(local_checkpoint_location, ignore_errors=True)
if not path.isfile(history_file_path):
  os.makedirs(local_checkpoint_location, exist_ok=True)
  reset_history()
if reset_checkpoints:
  shutil.rmtree(image_directory, ignore_errors=True)
os.makedirs(image_directory, exist_ok=True)

"""#### Checkpoint manager"""

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=G,
                                 discriminator=D,
                                 epoch=tf.Variable(0))
checkpoint_manager = tf.train.CheckpointManager(checkpoint,
                                                local_checkpoint_location,
                                                max_to_keep=10,
                                                keep_checkpoint_every_n_hours=2)

pretrain_checkpoint_manager = tf.train.CheckpointManager(checkpoint,
                                                local_pretrain_checkpoint_location,
                                                max_to_keep=1)


"""#### Image generation during train"""

path_to_folder = './Flickr30k-images-preprocessed/flickr30k-images-out/real/left'
imgs = ['1048710776.jpg',  # sea and sky
        '1130369873.jpg',  # dog
        '100197432.jpg',   # urban city
        '123101580.jpg'    # sport and people
       ]

selected_imgs = list(map(lambda x: os.path.join(path_to_folder, x), imgs))

test_images = np.array([rescale_and_normalize(np.array(Image.open(i).resize((input_shape[:2])))) for i in selected_imgs])

# plot_grid(np.array([generated_to_images(x) for x in test_images]), columns=4)

class ImageSaver:
  def __init__(self, epoch_variable, real_images=None, n_images=4):
    self.n_images = n_images
    self.directory = image_directory
    self.epoch_variable = epoch_variable
    if real_images is None:
      # extract random real images
      real_gen = real_validation_generator()
      self.real_batch = next(real_gen)[:self.n_images]
    else:
      # use given images
      self.real_batch = real_images
    cartoon_gen = cartoon_real_validation_generator()
    self.cartoon_batch = next(cartoon_gen)[:self.n_images]
    smooth_gen = cartoon_smooth_validation_generator()
    self.smooth_batch = next(smooth_gen)[:self.n_images]

  def save_images(self, D, G):
    real_images = self.real_batch
    cartoon_images = self.cartoon_batch
    smooth_images = self.smooth_batch

    real_images_G = G(real_images, training=False)
    real_to_cartoon_images = generated_to_images(real_images_G)

    generator_list = np.concatenate([generated_to_images(real_images), real_to_cartoon_images])

    real_to_cartoon_images_discriminator = self._labels_to_images(D(real_images_G, training=False))
    cartoon_real_discriminator = self._labels_to_images(D(cartoon_images, training=False))
    smooth_discriminator = self._labels_to_images(D(smooth_images, training=False))
    discriminator_list = np.concatenate([cartoon_real_discriminator, smooth_discriminator, real_to_cartoon_images_discriminator])

    save_plot_grid(generator_list,
                   self.n_images,
                   name=f'generator{self.epoch_variable.numpy()}',
                   directory=self.directory,
                   title=f'Epoch {self.epoch_variable.numpy()}',
                   labels=['Real images', 'Generated images'])
    save_plot_grid(discriminator_list,
                   self.n_images,
                   name=f'discriminator{self.epoch_variable.numpy()}',
                   directory=self.directory,
                   title=f'Epoch {self.epoch_variable.numpy()}',
                   labels=['cartoon images', 'smoothed images', 'real images'])

  def _labels_to_images(self, labels):
    return tf.sigmoid(labels).numpy().squeeze()

  def _G_to_images(self, images, G):
    return generated_to_images(G(images))

# ImageSaver test
# ImageSaver(tf.constant(-1), real_images=test_images).save_images(D, G)
# os.remove(os.path.join(image_directory, 'discriminator-1.png'))
# os.remove(os.path.join(image_directory, 'generator-1.png'))

"""## Pretraining"""

def pretrain(epochs=10, epochs_d=None):
  if epochs_d is None: epochs_d = epochs
  rg = real_generator()
  x = map(lambda x: (x, x), rg)
  rvg = real_validation_generator()
  x_val = map(lambda x: (x, x), rvg)
  content_loss = ContentLoss()

  G.compile(optimizer=tfa.optimizers.AdamW(
              learning_rate=1e-3,
              beta_1=.5, beta_2=.99,
              weight_decay=1e-3
    ),
    loss=lambda img_true, img_pred: content_loss(img_pred, img_true),
    metrics=[]
   )
  G.fit(x=x,
        epochs=epochs,
        steps_per_epoch=len(rg),
        validation_data=x_val,
        validation_steps=len(rvg),
        shuffle=True,
        verbose=0
       )

  base_loss = BCEWithLogitsLoss()
  fake_labels = tf.zeros((*D.compute_output_shape((batch_size, *input_shape)),))
  cartoon_labels = tf.ones((*D.compute_output_shape((batch_size, *input_shape)),))
  simple_loss = SimplifiedAdversarialLoss(cartoon_labels, fake_labels)
  optimizer=tfa.optimizers.AdamW(
              learning_rate=1e-3,
              beta_1=.5, beta_2=.99,
              weight_decay=1e-3
        )

  # discriminator training
  @tf.function
  def pretrain_step(x1, y1, x2, y2):
    with tf.GradientTape() as tape:
      pred1 = D(x1, training=True)
      loss1 = base_loss(pred1, y1)
      pred2 = D(x2, training=True)
      loss2 = base_loss(pred2, y2)
      loss = loss1 + loss2

    gradients = tape.gradient(loss, D.trainable_weights)
    optimizer.apply_gradients(
        zip(gradients, D.trainable_weights)
    )

    return loss

  epochs_count = epochs_d
  crg = cartoon_real_generator()
  rg = real_generator()
  bar_format = '{bar}{desc}: {percentage:3.0f}% {r_bar}'

  train_loss_mean = []
  for epoch in range(epochs_count):
    batches_per_epoch = min(len(crg), len(rg)) - 1
    # train phase
    epoch_progress = range(batches_per_epoch)
    train_loss_sum = 0
    train_loss_sum_squared = 0
    train_step_count = 0
    # train
    for (step,
        crb,
        rb) in zip(
           epoch_progress,
           crg,
           rg):
      loss = pretrain_step(crb, cartoon_labels, rb, fake_labels)
      loss = loss.numpy()

      train_loss_sum += loss
      train_step_count += 1

    train_loss_mean += [train_loss_sum / train_step_count]
    #print('Train mean loss:', train_loss_mean)

    # reset the generator
    crg.reset()
    rg.reset()

if checkpoint_manager.latest_checkpoint and len(checkpoint_manager.checkpoints) > 1:
  checkpoint.restore(checkpoint_manager.latest_checkpoint)
  print(f'Restored train from {checkpoint_manager.latest_checkpoint}.')
  print("Skipping pretraining: training was already started.")
elif pretrain_checkpoint_manager.latest_checkpoint:
  checkpoint.restore(pretrain_checkpoint_manager.latest_checkpoint)
  print(f'Restored pretrain from {pretrain_checkpoint_manager.latest_checkpoint}.')
  checkpoint_manager.save()
else:
  pretrain()
  pretrain_checkpoint_manager.save()
  checkpoint_manager.save()

"""## Training

### Implementation
"""

class CartoonGAN:

  def __init__(self,
               checkpoint,
               checkpoint_manager,
               cartoon_real_generator,
               cartoon_smooth_generator,
               real_image_generator,
               smooth_label_generator,
               cartoon_real_generator_val,
               cartoon_smooth_generator_val,
               real_image_generator_val,
               smooth_label_generator_val,
               omega=10.,
               starting_epoch=None):
    self.name = 'Cartoon-GAN'

    self.checkpoint = checkpoint
    self.checkpoint_manager = checkpoint_manager
    if starting_epoch is None:
      if checkpoint_manager.latest_checkpoint:
        print("-----------Restoring from {}-----------".format(
            checkpoint_manager.latest_checkpoint))
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
      else:
        print("-----------Initializing from scratch-----------")
    else:
      checkpoint_fname = local_checkpoint_location + 'ckpt-' + str(starting_epoch)# + '.data-0000-of-0001'
      print("-----------Restoring from {}-----------".format(checkpoint_fname))
      checkpoint.restore(checkpoint_fname)

    self.discriminator = checkpoint.discriminator
    self.generator = checkpoint.generator
    self.d_optimizer = checkpoint.discriminator_optimizer
    self.g_optimizer = checkpoint.generator_optimizer

    self.cartoon_real_generator = cartoon_real_generator # cartoon real gen train
    self.cartoon_smooth_generator = cartoon_smooth_generator # cartoon smooth gen train
    self.cartoon_smooth_generator_val = cartoon_smooth_generator_val # cartoon smooth gen val
    self.cartoon_real_generator_val = cartoon_real_generator_val # cartoon real gen val
    self.real_image_generator = real_image_generator # real image gen train
    self.real_image_generator_val = real_image_generator_val # real image gen val
    self.smooth_label_generator = smooth_label_generator # smooth labels train
    self.smooth_label_generator_val = smooth_label_generator_val # smooth labels val

    self.batches_per_epoch = min(len(self.cartoon_real_generator), len(self.real_image_generator)) - 1 # Last batch has unknown size
    self.val_batches_per_epoch = min(len(self.cartoon_real_generator_val), len(self.real_image_generator_val)) - 1 # Last batch has unknown size

    self.cartoon_labels = tf.ones((*D.compute_output_shape((batch_size, *input_shape)),))
    self.fake_cartoon_labels = tf.zeros((*D.compute_output_shape((batch_size, *input_shape)),))

    self.d_loss_fn = AdversarialLoss(self.cartoon_labels, self.fake_cartoon_labels)
    self.g_loss_fn = GeneratorLoss(self.cartoon_labels, omega=omega)
    self.bar_format = '{bar}{desc}: {percentage:3.0f}% {r_bar}'

    self.image_saver = ImageSaver(epoch_variable=self.checkpoint.epoch, real_images=test_images)

    self.eval_fid = evaluate_fid_with_cartoon_val

  @tf.function
  def _step(self,
              x_cartoon_batch_train,
              x_real_train,
              x_cartoon_smooth_batch_train,
              y_cartoon_smooth_batch_train):

    generated_real_train = self.generator(x_real_train, training=True)

    with tf.GradientTape() as disc_tape:
      # Discriminator loss
      predictions_cartoon_train = self.discriminator(x_cartoon_batch_train, training=True)
      predictions_generated_train = self.discriminator(generated_real_train, training=True)
      predictions_cartoon_smooth_train = self.discriminator(x_cartoon_smooth_batch_train, training=True)
      d_loss = self.d_loss_fn(predictions_cartoon_train, predictions_generated_train, predictions_cartoon_smooth_train, y_cartoon_smooth_batch_train)

    gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
    self.d_optimizer.apply_gradients(
        zip(gradients_of_discriminator, self.discriminator.trainable_variables)
    )

    with tf.GradientTape() as gen_tape:
      # Generator loss
      generated_real_train = self.generator(x_real_train, training=True)
      generated_real_labels = self.discriminator(generated_real_train, training=True)
      g_loss = self.g_loss_fn(generated_real_train, x_real_train, generated_real_labels)

    gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
    self.g_optimizer.apply_gradients(
        zip(gradients_of_generator, self.generator.trainable_variables)
    )

    return d_loss, g_loss

  @tf.function
  def _val_step(self,
              x_cartoon_batch_val,
              x_real_val,
              x_cartoon_smooth_batch_val,
              y_cartoon_smooth_batch_val):

    generated_real_val = self.generator(x_real_val, training=False)

    # Discriminator loss
    predictions_cartoon_val = self.discriminator(x_cartoon_batch_val, training=False)
    predictions_generated_val = self.discriminator(generated_real_val, training=False)
    predictions_cartoon_smooth_val = self.discriminator(x_cartoon_smooth_batch_val, training=False)
    d_loss = self.d_loss_fn(predictions_cartoon_val, predictions_generated_val, predictions_cartoon_smooth_val, y_cartoon_smooth_batch_val)

    # Generator loss (could have been computed from discriminator after one step of training)
    generated_real_labels = self.discriminator(generated_real_val, training=False)
    g_loss = self.g_loss_fn(generated_real_val, x_real_val, generated_real_labels)

    return d_loss, g_loss

  def train(self, epochs):
    starting_epoch = int(self.checkpoint.epoch)
    for epoch in range(starting_epoch, epochs):

      # train phase
      epoch_progress = range(self.batches_per_epoch)
      train_d_loss_sum = 0
      train_d_loss_sum_squared = 0
      train_g_loss_sum = 0
      train_g_loss_sum_squared = 0
      train_step_count = 0

      for (step,
          x_cartoon_batch_train,
          x_cartoon_smooth_batch_train,
          x_real_train,
          y_cartoon_smooth_batch_train) in zip(
             epoch_progress,
             self.cartoon_real_generator,
             self.cartoon_smooth_generator,
             self.real_image_generator,
             self.smooth_label_generator):

        d_loss_tensor, g_loss_tensor = self._step(x_cartoon_batch_train,
                                      x_real_train,
                                      x_cartoon_smooth_batch_train,
                                      y_cartoon_smooth_batch_train)
        d_loss = d_loss_tensor.numpy()
        g_loss = g_loss_tensor.numpy()

        train_d_loss_sum += d_loss
        train_d_loss_sum_squared += d_loss * d_loss
        train_g_loss_sum += g_loss
        train_g_loss_sum_squared += g_loss * g_loss
        train_step_count += 1

      self.cartoon_real_generator.reset()
      self.cartoon_smooth_generator.reset()
      self.real_image_generator.reset()
      self.smooth_label_generator.reset()

      # Compute mean loss and variance for historical data
      train_d_loss_mean = train_d_loss_sum / train_step_count
      train_d_loss_std = np.sqrt(train_d_loss_sum_squared / train_step_count - train_d_loss_mean * train_d_loss_mean)
      train_g_loss_mean = train_g_loss_sum / train_step_count
      train_g_loss_std = np.sqrt(train_g_loss_sum_squared / train_step_count - train_g_loss_mean * train_g_loss_mean)

      # validation phase
      epoch_val_progress = range(self.val_batches_per_epoch)

      val_d_loss_sum = 0
      val_d_loss_sum_squared = 0
      val_g_loss_sum = 0
      val_g_loss_sum_squared = 0
      val_step_count = 0

      for crgv, csgv, rigv, slgv, _ in zip(self.cartoon_real_generator_val,
                                     self.cartoon_smooth_generator_val,
                                     self.real_image_generator_val,
                                     self.smooth_label_generator_val,
                                     range(self.val_batches_per_epoch)):

        d_val_loss, g_val_loss = self._val_step(crgv, rigv, csgv, slgv)
        d_val_loss = d_val_loss.numpy()
        g_val_loss = g_val_loss.numpy()

        val_d_loss_sum += d_val_loss
        val_d_loss_sum_squared += d_val_loss * d_val_loss
        val_g_loss_sum += g_val_loss
        val_g_loss_sum_squared += g_val_loss * g_val_loss
        val_step_count += 1


      # Compute mean loss and variance for historical data
      val_d_loss_mean = val_d_loss_sum / val_step_count
      val_d_loss_std = np.sqrt(val_d_loss_sum_squared / val_step_count - val_d_loss_mean * val_d_loss_mean)
      val_g_loss_mean = val_g_loss_sum / val_step_count
      val_g_loss_std = np.sqrt(val_g_loss_sum_squared / val_step_count - val_g_loss_mean * val_g_loss_mean)

      self.cartoon_real_generator_val.reset()
      self.cartoon_smooth_generator_val.reset()
      self.real_image_generator_val.reset()
      self.smooth_label_generator_val.reset()

      fid_score = self.eval_fid(self.real_image_generator_val)
      # print('FID for epoch[', epoch, '] =', fid_score)
      self.real_image_generator_val.reset()

      # checkpoint phase
      self.checkpoint.epoch.assign_add(1)
      save_path = self.checkpoint_manager.save()

      # history save
      history_append(epoch, train_d_loss_mean, train_d_loss_std, train_g_loss_mean, train_g_loss_std, val_d_loss_mean, val_d_loss_std, val_g_loss_mean, val_g_loss_std, fid_score)

      # training image save
      self.image_saver.save_images(self.discriminator, self.generator)

"""### Save model"""

def download_load_image(url):
  if os.path.isfile(url):
    import matplotlib.image as mpimg
    return mpimg.imread(url)
  temp_file = "temp.jpg"
  urllib.request.urlretrieve(url, temp_file)
  return np.array(Image.open(temp_file))

def image_split_coordinate(shape, shape_rect, stride, add_last_rect=True):
  """
  :param shape:
  :param shape_rect:
  :param stride:
  :param add_last_rect:
  :return: a list of rectangle of type (x, y, w, h) represent rectangles in an image of shape=shape
  """
  image_w, image_h = shape
  rect_w, rect_h = shape_rect
  stride_x, stride_y = stride
  extra_w, extra_h = [], []
  if add_last_rect:
    extra_h = [image_h - rect_h, ]
    extra_w = [image_w - rect_w, ]

  output_rect = []
  for y in list(range(0, image_h - rect_h + 1, stride_y)) + extra_h:
    for x in list(range(0, image_w - rect_w + 1, stride_x)) + extra_w:
      output_rect.append((x, y))

  tuples = np.unique(output_rect, axis=0)

  return [(x, y, rect_w, rect_h) for (x, y) in tuples]

def merge_images(final_size, images, positions):
  nw, nh = final_size[:2]
  imgs = []
  temp_img = Image.new('RGBA', size=(nw, nh), color=(0, 0, 0, 0))
  for img, (x, y, w, h) in zip(images, positions):
    temp_img.paste(Image.fromarray(np.uint8(img)), (x, y))
  temp_img.show()
  return np.array(temp_img)

def marge_images_with_weight(final_size, images, positions):
  def point_in_rect(point, rect):
    x1, y1, w, h = rect
    x2, y2 = x1+w, y1+h
    x, y = point
    if (x1 <= x and x < x2):
        if (y1 <= y and y < y2):
            return True
    return False

  nw, nh = final_size[:2]
  weigths = np.zeros((nh, nw))
  for y in range(nh):
    for x in range(nw):
      n = len(list(filter(lambda r: point_in_rect((x, y), r), positions)))
      weigths[y, x] = n
  weigths1 = np.expand_dims(weigths, -1)
  weigths = np.repeat(weigths1, 3, axis=-1)
  output = np.zeros((nh, nw, 3))
  for image, (x, y, w, h) in zip(images, positions):
    output[y:y+h, x:x+h] += image
  return np.uint8(output / weigths)

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def convert_image_to_anime(image, generator=G, rescale_for_computing=2., stride_percent=0.2, plot=False):
  '''
    The image have to be bigger than 224 x 224
  '''
  if isinstance(image, str):
    image = download_load_image(image)


  real_image_dim = image.shape[1::-1]

  # rescale the image
  img = Image.fromarray(np.uint8(image))
  new_size = [int(real_image_dim[0] // rescale_for_computing), int(real_image_dim[1] // rescale_for_computing)]
  if new_size[0] < 224 or new_size[1] < 224:
    raise ValueError("Image should be bigger than 224x224")
  elif new_size == real_image_dim:
    new_size = real_image_dim
    scaled_image = image
  else:
    scaled_image = np.array(img.resize(new_size, Image.ANTIALIAS))

  image = scaled_image
  image_dim = new_size

  rectangles = image_split_coordinate(image_dim, (224, 224), (int(224*stride_percent), int(224*stride_percent)))

  # scompose the image
  sub_images = []
  for (x, y, w, h) in rectangles:
    sub_image = image[y:y+h, x:x+w].copy()
    sub_images.append(sub_image)

  g_out = []
  for mini_batch in np.array_split(sub_images, max(1, len(sub_images)//10)):
    # predict
    g_out += [*generator(rescale_and_normalize(np.array(mini_batch)), training=False)]


  # unnormalize and rescale images
  pred_images = [x*255 for x in generated_to_images(g_out)]

  img_merged_g = marge_images_with_weight((*image_dim, 3), pred_images, rectangles)
  if new_size != real_image_dim:
    img_merged_g = Image.fromarray(img_merged_g).resize(real_image_dim)
  else:
    img_merged_g = Image.fromarray(img_merged_g)

  merged_img = get_concat_h(img, img_merged_g)
  if plot: merged_img.show()
  return np.array(merged_img)

images_path = [
#   'https://www.donnad.it/sites/default/files/styles/r_visual_d/public/202107/sfondi-desktop-paesaggi-primaverili-gratis-7.jpg?itok=bWAh9RZj',
#   'https://fotocomefare.com/wp-content/uploads/2020/03/2414671551_efeec4177e.jpg',
#   'https://www.unicoebello.it/wp-content/uploads/2020/03/Londra-1.jpg',
#   'https://siviaggia.it/wp-content/uploads/sites/2/2021/04/malcesine-passeggiate-italia.jpg',
#   'https://img.100r.systems/img/f66f3819f0698a577171667f6fef913b.jpg',
  'https://camo.githubusercontent.com/7cb58019ea8637e8473b2b1dba8335b5f449ba660ef11dc9e9fe4a7cbec772c3/68747470733a2f2f692e696d6775722e636f6d2f5944766c6d67422e706e67',
  'https://camo.githubusercontent.com/581738a9aa8d98ab23f7bb5d35c5e0839093b81350b95529a3acfbb1de0dff80/68747470733a2f2f692e696d6775722e636f6d2f6544797333765a2e706e67',
  'https://camo.githubusercontent.com/bba844b46204e6c073443265bc1f8747a7c2572253d85da673504a4484274493/68747470733a2f2f692e696d6775722e636f6d2f35303170777a392e706e67',
  'https://camo.githubusercontent.com/b7c44e901a66ee93993010c87fc24728d6269171cdd551a4a17eedbfa958ce9c/68747470733a2f2f692e696d6775722e636f6d2f523168374d75432e706e67',
  'https://camo.githubusercontent.com/9e66e94b5db7c20e16245bffddc3d6095055abcc83ab3acb4797a8f9f51a1850/68747470733a2f2f692e696d6775722e636f6d2f6d616d725a58412e706e67',
  'https://camo.githubusercontent.com/9d5119090f91eabfcea13c56a31c16f7a525f96edcdef1ef12104b77380b3f8d/68747470733a2f2f692e696d6775722e636f6d2f6f467a585148532e706e67',
  'https://camo.githubusercontent.com/ef80a8fb5f4a742886864c61d45a9b2b2f4a8d18d10ed5200c3734093dd398f0/68747470733a2f2f692e696d6775722e636f6d2f3774475a6551792e706e67',
  'https://camo.githubusercontent.com/9ff4644cade20ba143e7ba68f3af8fb9aa6dde459d814173b25be43898b43d71/68747470733a2f2f692e696d6775722e636f6d2f5a67424878576d2e706e67',
  'https://camo.githubusercontent.com/487ee69b8384aea4e5b68157f9a90b60746c3e000f7a59b48b4c0c4bd6e6da27/68747470733a2f2f692e696d6775722e636f6d2f376a33797376302e706e67',
  'https://camo.githubusercontent.com/e2d0ae0fdb33d32a8625b2536e1b204e336eaaf18e1048c200b291b567b4203f/68747470733a2f2f692e696d6775722e636f6d2f41336e497551642e706e67',
  'https://camo.githubusercontent.com/1735135e57e3681749aa26b578d846ffe99ab975461f6367a89b2c3e77a3b018/68747470733a2f2f692e696d6775722e636f6d2f506a71575a4a6f2e706e67',
  'https://camo.githubusercontent.com/ae784a131e893d1850177ad88087c948e63e279fd2a809cabed8d698036a3398/68747470733a2f2f692e696d6775722e636f6d2f334435594650592e706e67',
]

def get_checkpoint_path(epoch):
  checkpoint_fnames = [os.path.join(local_checkpoint_location, 'ckpt-' + str(epoch) + '.data-00000-of-00001'),
                       os.path.join(local_checkpoint_location, 'ckpt-' + str(epoch) + '.index')]
  return checkpoint_fnames

def load_checkpoint(epoch):
  checkpoint_fname = local_checkpoint_location + 'ckpt-' + str(epoch)# + '.data-0000-of-0001'
  print("-----------Restoring from {}-----------".format(checkpoint_fname))
  checkpoint.restore(checkpoint_fname)
  return checkpoint

def load_evaluate_and_save_epoch(epoch, omega, vgg16_layer, folder, images):
  '''
    1. load the model at epcoh = 'epoch'
    2. evaluate images with the Generator loaded
    3. save model '{folder}/model_e{epoch}_o{omega}_v{vgg16_layer}/ckpt-{epoch}'
    4. save images in '{folder}/model_e{epoch}_o{omega}_v{vgg16_layer}/images/x.jpj' where x is the position of the image in the array
  '''
  # convert images to anime
  checkpoint = load_checkpoint(epoch)
  generator = checkpoint.generator
  cartoon_images = [convert_image_to_anime(image=i, generator=generator, rescale_for_computing=1., stride_percent=0.1)
            for i in images_path]

  # creates folders
  if not os.path.exists(folder):
    os.mkdir(folder, )
  model_folder = os.path.join(folder, f'model_e{epoch}_o{omega}_v{vgg16_layer}')
  if not os.path.exists(model_folder):
    os.mkdir(model_folder)
  image_folder = os.path.join(model_folder, 'images')
  if not os.path.exists(image_folder):
    os.mkdir(image_folder)

  # save the model
  for checkpoint_f in get_checkpoint_path(epoch):
    shutil.copy(checkpoint_f, os.path.join(model_folder, os.path.basename(checkpoint_f)))

  # save the images
  for i, image in enumerate(cartoon_images):
    img = Image.fromarray(image)
    img.save(os.path.join(image_folder, f'{i}.jpg'))

  # copy history.csv
  shutil.copy(os.path.join(local_checkpoint_location, history_file_name), os.path.join(model_folder, history_file_name))

def save_best_checkpoints(omega, vgg_lvl=18, path='bests_checkpoints', min_epoch=0):
  # find available checkpoints
  available_checkpoints = [int(re.search(r'(ckpt-)([0-9]+)(.index)', x)[2]) for x in glob(f'{local_checkpoint_location}/ckpt-*.index')]
  available_checkpoints = list(filter(lambda x: x >= min_epoch, available_checkpoints))
  # load dataset from csv+
  df = pd.read_csv(history_file_path, sep=',', header=0)
  df = df.set_index('epoch')
  df = df[~df.index.duplicated(keep='last')]
  df_available_checkpoints = df[[x in available_checkpoints for x in df.index]]

  # useful columns
  val_gen_loss_mean = 'val_generator_loss_mean'
  fid_score = 'fid_score'

  # search the best epochs
  df_val_gen_loss_mean = df_available_checkpoints[val_gen_loss_mean]
  best_ep_val_gen_loss = df_available_checkpoints.index[np.argmin(df_val_gen_loss_mean)]
  df_fid_score = df_available_checkpoints[fid_score]
  best_ep_fid_score = df_available_checkpoints.index[np.argmin(df_fid_score)]
  last_ep = np.max(df_available_checkpoints.index)

  best_eps = np.unique([best_ep_val_gen_loss, best_ep_fid_score, last_ep])

  for ep in best_eps:
    print(f'Saving epoch {ep}')
    load_evaluate_and_save_epoch(ep, omega, vgg_lvl, path, images_path)

def main(epochs=None, omega=None):
  epochs = epochs if epochs is not None else 50
  omega = omega if omega is not None else 1
  model = CartoonGAN(checkpoint,
                     checkpoint_manager,
                     cartoon_real_generator(),
                     cartoon_smooth_generator(),
                     real_generator(),
                     smooth_label_generator(),
                     cartoon_real_validation_generator(),
                     cartoon_smooth_validation_generator(),
                     real_validation_generator(),
                     smooth_label_validation_generator(),
                     omega=omega,
                     starting_epoch=None)

  """### Training"""

  model.train(epochs)

  save_best_checkpoints(omega=omega, vgg_lvl=vgg_16_layer)

if __name__ == '__main__':
  import sys
  args = sys.argv
  len_argv = len(args)
  if len_argv > 1:
    epochs = int(args[1])
  if len_argv > 2:
    omega = float(args[2])
  main(epochs=epochs, omega=omega)
