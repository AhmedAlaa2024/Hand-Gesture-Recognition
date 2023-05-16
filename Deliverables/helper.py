from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from skimage import exposure

def hog_preprocessing(image):
  image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  filtered_image  = cv2.GaussianBlur(image, (3, 3), 1.0)
  resized_image = cv2.resize(filtered_image,(500,500))
  #======================================Color Segmentation========================#
  hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2HSV)
  light_skin = (0, 50, 50 )
  dark_skin = (50, 250, 255)
  mask = cv2.inRange(hsv_image, light_skin, dark_skin)
  #Define the kernel for closing
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
  # Perform closing on the thresholded image)
  mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,iterations=10)
  result = cv2.bitwise_and(resized_image, resized_image, mask=mask)
  resized_image = cv2.resize(result,(128,128))
  grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
  #=====================================Contrast enhancement========================#
  # Apply histogram equalization to enhance contrast
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10,10))
  normalized_image = clahe.apply(grayscale_image)
  gamma_corrected = exposure.adjust_gamma(normalized_image, gamma=2.5)
  return gamma_corrected


def extract_hog_features(image, cell_size, num_orientations):

  # Calculate the HOG features
  hog = cv2.HOGDescriptor(_winSize=(image.shape[1] // cell_size[1] * cell_size[1],
                                    image.shape[0] // cell_size[0] * cell_size[0]),
                          _blockSize=(cell_size[1] * 2, cell_size[0] * 2),
                          _blockStride=(cell_size[1], cell_size[0]),
                          _cellSize=(cell_size[1], cell_size[0]),
                          _nbins=num_orientations)
  features = hog.compute(image)

  # Return the features
  return features

def HOG_box(image):
   preprocessed_image = hog_preprocessing(image)
   hog_feature_vector = extract_hog_features(preprocessed_image, np.array([10, 10]), 9) 
   return hog_feature_vector

def load_images_from_folder(folder_path):
  # Set the path to the folder containing the images
  # Create an empty list to store the images
  feature_vectors = []
  labels = []
  # Loop through all the images in the folder
  for imagefile in os.listdir(folder_path):
    # Read the image
    image = cv2.imread(os.path.join(folder_path, imagefile))

    # Convert the color channels from BGR to RGB
    hog_feature_vector =   extract_hog_features(image, np.array([10, 10]), 9)
    # Append the image to the list of images
    try:
      labels.append(int(imagefile[0]))
      feature_vectors.append(hog_feature_vector)
    except:
      del image
      continue
    # Append the label to the list of labels
    del image

  return labels, feature_vectors


def load_dataset_linux():
  X = []
  Y = []

  Y, X = load_images_from_folder('/home/ahmedalaa/Study/College/NN/Hand-Gesture-Recognition/resizedData')

  return np.array(Y), np.array(X)


# For the windows farmers
def load_dataset_win():
  X = []
  Y = []
  Y_train, X_train = load_images_from_folder('E:/2nd term 3rd year/Neural Network/Project/Hand-Gesture-Recognition/resizedData_training')
  Y_valid, X_valid = load_images_from_folder('E:/2nd term 3rd year/Neural Network/Project/Hand-Gesture-Recognition/resizedData_validation')

  return np.array(Y_train),np.array(X_train),np.array(Y_valid),np.array(X_valid)

# For the windows farmers
def load_all_dataset__win():
  X = []
  Y = []
  Y_train, X_train = load_images_from_folder('E:/2nd term 3rd year/Neural Network/Project/resizedData')

  return np.array(Y_train),np.array(X_train)