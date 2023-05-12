import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import os
import time

def extract_hog_features(image, cell_size, num_orientations):
  # Convert the image to grayscale
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Calculate the HOG features
  hog = cv2.HOGDescriptor(_winSize=(gray_image.shape[1] // cell_size[1] * cell_size[1],
                                      gray_image.shape[0] // cell_size[0] * cell_size[0]),
                          _blockSize=(cell_size[1] * 2, cell_size[0] * 2),
                          _blockStride=(cell_size[1], cell_size[0]),
                          _cellSize=(cell_size[1], cell_size[0]),
                          _nbins=num_orientations)
  features = hog.compute(gray_image)

  # Return the features
  return features

def extract_lbp_features(image, cell_size, radius):
  # Convert the image to grayscale
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Calculate the LBP features
  lbp = local_binary_pattern(gray_image, cell_size[1] * 8, radius, method='uniform')
  features = lbp.ravel()

  # Return the features
  return features

filename = 'data/men/3/3_men (3).JPG'
if not os.path.isfile(filename):
  print(f"File '{filename}' not found!")
else:
  image = cv2.imread(filename)
  # Measure the time for HOG feature extraction
  start_time = time.time()
  features_vector = np.asarray(extract_hog_features(image, np.array([8, 8]), 9))
  end_time = time.time()
  print("HOG Features: ", features_vector.shape)
  print(f"Time taken for HOG feature extraction: {end_time - start_time} seconds")

  # Measure the time for LBP feature extraction
  start_time = time.time()
  # In this example, I used a radius of 1 for the LBP algorithm,
  # which means that the algorithm considers 8 neighboring pixels around each pixel
  # to calculate the LBP code. You can change the radius parameter according to your needs.
  features_vector = np.asarray(extract_lbp_features(image, np.array([8, 8]), 1))
  end_time = time.time()
  print("LBP Features: ", features_vector.shape)
  print(f"Time taken for LBP feature extraction: {end_time - start_time} seconds")













# The HOG (Histogram of Oriented Gradients) algorithm is a feature descriptor that is
# commonly used in computer vision and image processing for object detection and 
# recognition. HOG features describe the distribution of edge gradients in an image, which 
# can be useful for identifying object shapes and boundaries. The output of the HOG algorithm 
# is a feature vector, which is a one-dimensional array that contains information about the 
# orientation and magnitude of edge gradients in the image. In this example, the size of the 
# feature vector is 6686100, which means that the image was divided into multiple cells, and
# for each cell, a histogram of gradients was computed. These histograms are concatenated
# to form the final feature vector.

# On the other hand, LBP (Local Binary Pattern) is a texture descriptor that is commonly used
# in image analysis and texture classification. LBP features describe the local texture patterns
# of an image by comparing each pixel to its surrounding neighbors. The output of the LBP
# algorithm is a feature vector that contains information about the frequency of each local
# pattern in the image. In this example, the size of the feature vector is 11943936, which means
# that the LBP algorithm considers 8 neighboring pixels around each pixel to calculate the 
# LBP code. This results in a large feature vector that contains information about the texture patterns in the image.

# In terms of impact, the choice of feature extraction algorithm depends on the specific task
# and requirements of the application. HOG features are useful for object detection and
# recognition tasks, where the shapes and boundaries of objects need to be identified. LBP
# features, on the other hand, are useful for texture classification tasks, where the local texture
# patterns of an image need to be analyzed. The time taken for feature extraction also varies
# between algorithms, with HOG being generally faster than LBP. In this example, the HOG
# algorithm took 0.24 seconds to extract the features, while the LBP algorithm took 11.65 seconds.

# Both HOG and LBP are commonly used in hand gesture recognition systems, but they have their own strengths and weaknesses.

# HOG is well-suited for detecting object shapes and contours, which makes it a popular
# choice for object detection and recognition tasks. In hand gesture recognition, HOG can
# help identify the general shape and orientation of the hand and fingers, which can be useful
# for recognizing gestures such as thumbs up or peace sign. However, HOG may struggle with
# detecting finer details or texture information, which could make it less effective for
# recognizing more complex hand gestures.

# LBP, on the other hand, is more focused on detecting texture and pattern information, which
# makes it well-suited for recognizing fine-grained details in an image. In hand gesture
# recognition, this could be useful for recognizing features such as wrinkles or creases in the
# skin, which could help distinguish between similar-looking gestures. However, LBP may not
# be as effective at capturing shape information, which could make it less reliable for
# recognizing gestures that rely more on hand shape than texture.

# Overall, both HOG and LBP can be effective for hand gesture recognition, but the choice of
# algorithm will depend on the specific needs of your application and the types of gestures
# you are trying to recognize.