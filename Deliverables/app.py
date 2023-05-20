import cv2
import joblib
import os
from helper import *
import numpy as np
import time
# Stage 6: Model Inference
# Load the model
with open('models/soft_training_voting.hdf5', 'rb') as file:
  soft_voting = joblib.load(file)
    # Initialize an empty list to store the numbers
# Load the images
directory = "data"
image_paths = []
count=1
for filename in os.listdir(directory):
  image_paths.append("./data/"+str(count)+".png")
  count=count+1
if not image_paths:
  print('No images found in the data directory')
  exit()
# Make predictions and record times
soft_predictions = []
times = []
resizedImage = []
for image_path in image_paths:
  image = cv2.imread(image_path)
  start_time = time.time()
  image = hog_preprocessing(image)
  resizedImage.append(image)
  hog_feature_vector = extract_hog_features(image, np.array([10, 10]), 9)
  hog_feature_vector = np.asarray(hog_feature_vector).reshape(1, -1)
  soft_prediction = soft_voting.predict(hog_feature_vector)
  end_time = time.time()
  soft_predictions.append(soft_prediction[0])
  times.append(end_time - start_time)
# Create the output directory if it doesn't exist
if not os.path.exists('output'):
  os.makedirs('output')

# Write results to files
with open('output/results.txt', 'w') as f:
  for prediction in soft_predictions:
    f.write(str(prediction) + '\n')

with open('output/time.txt', 'w') as f:
  for time_taken in times:
    f.write('{:.3f}\n'.format(time_taken))
print('Done!')
print('Results written to output/results.txt')
print('Time taken written to output/time.txt with average time = {:.3f} seconds'.format(np.mean(times)))