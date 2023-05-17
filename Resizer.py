# I want to load all the images in the data directory
# and resize them to 500x500
# and make some sort of processing
# then resize them to 128x128

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
import time

def load_images_from_folder(folder_path):
    # Set the path to the folder containing the images
    # folder_path = 'E:/2nd term 3rd year/Neural Network/Project/Distrubute_dataset' + gender + '/' + str(number)
    # Create an empty list to store the images
    feature_vectors = []
    labels = []
    # Loop through all the images in the folder
    for imagefile in os.listdir(folder_path):
        # Read the image 
        image = cv2.imread(os.path.join(folder_path, imagefile))
        
        # Convert the color channels from BGR to RGB
        feature_vector = HOG_box(image)
        # Append the image to the list of images
        try:
            labels.append(int(imagefile[2]))
            feature_vectors.append(feature_vector)
        except:
            del image
            continue
        # Append the label to the list of labels
        del image

    return labels, feature_vectors

def hog_preprocessing(image):
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image,(500,500))
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


def load_images(folder_path):
    resizedImage = []
    labels = []

    for imagefile in os.listdir(folder_path):
        print( imagefile)
        # Read the image 
        image = cv2.imread(os.path.join(folder_path, imagefile))
        resizedImage.append(hog_preprocessing(image))
        labels.append(int(imagefile[2]))
        del image
    return labels, resizedImage

def save_images(folder_path, resizedImage, labels):
    for i in range(len(resizedImage)):
        cv2.imwrite(folder_path + '/' + str(labels[i]) + '_' + str(i) + '.jpg', resizedImage[i])

if __name__ == "__main__":
  start_time = time.time()
  labels, resizedImages = load_images('E:/2nd term 3rd year/Neural Network/Project/Hand-Gesture-Recognition/training')
  print("Loading Time: %s seconds" % (time.time() - start_time))

  start_time = time.time()
  save_images('E:/2nd term 3rd year/Neural Network/Project/Hand-Gesture-Recognition/resizedData_training', resizedImages, labels)
  print("Saving Time: %s seconds" % (time.time() - start_time))

  start_time = time.time()
  labels, resizedImages = load_images('E:/2nd term 3rd year/Neural Network/Project/Hand-Gesture-Recognition/validation')
  print("Loading Time: %s seconds" % (time.time() - start_time))

  start_time = time.time()
  save_images('E:/2nd term 3rd year/Neural Network/Project/Hand-Gesture-Recognition/resizedData_validation', resizedImages, labels)
  print("Saving Time: %s seconds" % (time.time() - start_time))