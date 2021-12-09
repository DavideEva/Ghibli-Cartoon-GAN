import argparse
import cv2
from tqdm import tqdm

import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from imutils import is_cv3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Add, AveragePooling2D, Layer

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

def define_generator(image_shape=(224, 224, 3)):
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

# RGB channels mean values based on Imagenet
norm_mean = np.asfarray([0.485, 0.456, 0.406])
# RGB channels standanrd deviation values based on Imagenet
norm_std = np.asfarray([0.229, 0.224, 0.225])

def normalize(img):
  return (img - norm_mean) / norm_std
def unnormalize(img):
  return tf.clip_by_value(img * norm_std + norm_mean, 0.0, 1.0)
def rescale_and_normalize(img):
  return normalize(img / 255.0)
def generated_to_images(outputs):
  return [unnormalize(output).numpy() for output in outputs]

def image_split_coordinate(shape, shape_rect, stride, add_last_rect=True):
  image_w, image_h = shape
  rect_w, rect_h = shape_rect
  stride_x, stride_y = stride
  extra_x = []
  extra_y = []
  if add_last_rect:
    if image_h % stride_y != 0:
      extra_x.append(image_h - rect_h)
    if image_w % stride_x != 0:
      extra_y.append(image_w - rect_w)

  [x, y] = np.ogrid[0:image_h - rect_h + 1:stride_y, 0:image_w - rect_w + 1:stride_x]
  x = np.append(x, extra_x)
  y = np.append(y, extra_y)
  return np.array(np.meshgrid(y, x, [rect_w], [rect_h]), dtype=np.uint32).T.reshape(-1, 4)


def merge_images_with_weight(final_size, images, positions):
  output = np.zeros(final_size)
  output_sum = np.zeros(final_size)
  for image, (x, y, w, h) in zip(images, positions):
    output[y:y+w, x:x+h] += image
    output_sum[y:y+w, x:x+h] += np.ones((w, h, final_size[2]))
  return np.uint8(output / output_sum)

def concat_images(im1, im2, RGB=True):
  im1w, im1h = np.shape(im1)[1::-1]
  im2w, im2h = np.shape(im1)[1::-1]
  assert im1h == im2h

  dst = Image.new('RGB' if RGB else 'L', (im1w + im2w, im1h))
  dst.paste(Image.fromarray(im1), (0, 0))
  dst.paste(Image.fromarray(im2), (im1w, 0))
  return np.array(dst)

def convert_image_to_anime(image, generator, stride_fraction=0.22):
  real_image_dim = image.shape[1::-1]
  image_dim = [int(real_image_dim[0]), int(real_image_dim[1])]
  img = Image.fromarray(np.uint8(image))

  # generate the coordinates of the sub_square that will be passed 
  # to the generator
  sub_squares = image_split_coordinate(shape=image_dim, 
                                          shape_rect=(224, 224), 
                                          stride=(int(224*stride_fraction), int(224*stride_fraction)))


  g_out = []
  # cut the sub_square from the original image
  for mini_batch in np.array_split([image[y:y+h, x:x+w] for [x, y, w, h] in sub_squares], max(1, len(sub_squares)//8)):
    # predict
    g_out += [*generator(rescale_and_normalize(np.array(mini_batch)), training=False)]


  # unnormalize and rescale images
  pred_images = [x*255 for x in generated_to_images(g_out)]

  img_merged_g = merge_images_with_weight(final_size=(*image_dim, 3), 
                                          images=pred_images, 
                                          positions=sub_squares)
  return img_merged_g

def count_frames(path, override=False):
	# grab a pointer to the video file and initialize the total
	# number of frames read
	video = cv2.VideoCapture(path)
	total = 0
	# if the override flag is passed in, revert to the manual
	# method of counting frames
	if override:
		total = count_frames_manual(video)
	# otherwise, let's try the fast way first
	else:
		# lets try to determine the number of frames in a video
		# via video properties; this method can be very buggy
		# and might throw an error based on your OpenCV version
		# or may fail entirely based on your which video codecs
		# you have installed
		try:
			# check if we are using OpenCV 3
			if is_cv3():
				total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
			# otherwise, we are using OpenCV 2.4
			else:
				total = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
		# uh-oh, we got an error -- revert to counting manually
		except:
			total = count_frames_manual(video)
	# release the video file pointer
	video.release()
	# return the total number of frames in the video
	return total
    
def count_frames_manual(video):
	# initialize the total number of frames read
	total = 0
	# loop over the frames of the video
	while True:
		# grab the current frame
		(grabbed, frame) = video.read()
	 
		# check to see if we have reached the end of the
		# video
		if not grabbed:
			break
		# increment the total number of frames read
		total += 1
	# return the total number of frames in the video file
	return total

def convert(input, output, weights_path):
  G = define_generator()
  G.load_weights(weights_path)

  total_frames = count_frames(input)

  vidcap = cv2.VideoCapture(input)
  if not vidcap.isOpened():
    print("Error opening video stream or file")
    exit(1)
  vidcap_out = cv2.VideoWriter(output, 
          cv2.VideoWriter_fourcc(*'mp4v'),
          vidcap.get(cv2.CAP_PROP_FPS),
          (int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)),
          int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
  hasFrames, image = vidcap.read()
  for _ in tqdm(range(total_frames)):
    if not hasFrames:
      break
    vidcap_out.write(convert_image_to_anime(image, G))
    hasFrames, image = vidcap.read()
  vidcap.release()
  vidcap_out.release()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Convert movies to anime.')
  parser.add_argument('-i', '--input', help='Input file.', required=True)
  parser.add_argument('-o', '--output', help='Output file.', required=True)
  parser.add_argument('-w', '--weights', help='Weights file.', required=True)
  args = parser.parse_args()
  convert(args.input, args.output, args.weights)