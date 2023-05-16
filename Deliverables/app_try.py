import cv2
import joblib
import os
from helper import *
import numpy as np
import time
# Stage 6: Model Inference
# Load the model
def save_images(folder_path, resizedImage, labels):
    for i in range(len(resizedImage)):
        cv2.imwrite(folder_path + '/' + str(labels) + '_' + str(i) + '.jpg', resizedImage[i])

with open('models/soft_training_voting.hdf5', 'rb') as file:
  soft_voting = joblib.load(file)
    # Initialize an empty list to store the numbers
lables=[]
# Load the images
directory = "data"
image_paths = []
for filename in os.listdir(directory):
  if filename.endswith('.jpg') or filename.endswith('.png'):
    # lables.append(int(filename[2]))
    image_paths.append(os.path.join(directory, filename))
# print(lables)
if not image_paths:
  print('No images found in the test directory')
  exit()

# Make predictions and record times
soft_predictions = []
times = []
resizedImage = []
hog_feature_vectors=[]
for image_path in image_paths:
  image = cv2.imread(image_path)
  image = hog_preprocessing(image)
  resizedImage.append(image)
  hog_feature_vector = extract_hog_features(image, np.array([10, 10]), 9)
  #hog_feature_vector = np.asarray(hog_feature_vector).reshape(1, -1)
  hog_feature_vectors.append(hog_feature_vector)
start_time = time.time()
soft_predictions = soft_voting.predict(hog_feature_vectors)
end_time = time.time()
# soft_predictions.append(soft_prediction)
times.append(end_time - start_time)
save_images('E:/2nd term 3rd year/Neural Network/Project/Hand-Gesture-Recognition/Deliverables/output/preprocess_result', resizedImage, 1)

#===============================================

# score_soft = accuracy_score(lables, soft_predictions)
# print(" ======================================================== ")
# print("| [Stage 4]: Model Evaluation                            |")
# print("| Accuracy soft:", score_soft*100, '%                      |')
# print(" ======================================================== ")
#===========================================
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