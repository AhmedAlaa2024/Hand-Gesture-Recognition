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
def hog_preprocessing(image):
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image,(500,500))
    #======================================Color Segmentation========================#
    filtered_image  = cv2.GaussianBlur(resized_image, (7, 7), 1)
    hsv_image = cv2.cvtColor(filtered_image, cv2.COLOR_RGB2HSV)
    light_skin = (0, 40, 50 )
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
 labels, resizedImages = load_images('E:/2nd term 3rd year/Neural Network/Project/data')
 print("Loading Time: %s seconds" % (time.time() - start_time))

 start_time = time.time()
 save_images('E:/2nd term 3rd year/Neural Network/Project/resizedData', resizedImages, labels)
 print("Saving Time: %s seconds" % (time.time() - start_time))