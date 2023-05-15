import cv2
import joblib
import os
from helper import hog_preprocessing, extract_hog_features
import numpy as np
import time

# Stage 6: Model Inference
# Load the model
with open('model/hard_voting.hdf5', 'rb') as file:
  hard_voting = joblib.load(file)

with open('model/soft_voting.hdf5', 'rb') as file:
  soft_voting = joblib.load(file)

# Load the image
directory = 'test'
image_path = None
for filename in os.listdir(directory):
  if filename.endswith('.jpg') or filename.endswith('.png'):
    image_path = os.path.join(directory, filename)
    break

if image_path is not None:
  image = cv2.imread(image_path)
  image = hog_preprocessing(image)
  hog_feature_vector = extract_hog_features(image, np.array([10, 10]), 9)
  hog_feature_vector = np.asarray(hog_feature_vector).reshape(1, -1)
else:
  print('No image found in the test directory')

# Make predictions
start_time = time.time()
hard_prediction = hard_voting.predict(hog_feature_vector)
soft_prediction = soft_voting.predict(hog_feature_vector)
end_time = time.time()

# Print the predictions
print(" ======================================================== ")
print("| [Stage 6]: Model Inference                             |")
print("| Hard Voting Prediction:", hard_prediction[0], "                             |")
print("| Soft Voting Prediction:", soft_prediction[0], "                             |")
print("| Inference Time = {:.4f}                                |".format(end_time - start_time))
print(" ======================================================== ")

cv2.imshow("Prediction", cv2.imread('labels/{0}.png'.format(hard_prediction[0])))
cv2.waitKey(0)
cv2.destroyAllWindows()