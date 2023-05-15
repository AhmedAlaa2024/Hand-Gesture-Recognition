from helper import *
import joblib

preprocessed_images = []
feature_vectors = []

# Stage 1: Data Loading, Preprocessing and Feature Extraction
Y, X = load_dataset_linux()
print(" =============================================================== ")
print("| [Stage 1]: Data Loading, Preprocessing and Feature Extraction |")
print(f"| Number of images loaded: {len(X)}                                 |")
print(f"| Number of labels loaded: {len(Y)}                                 |")
print(" =============================================================== ")

# Stage 2: Data Splitting into Training and Testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
X_train, X_vaild, Y_train, Y_valid = train_test_split(X_train,Y_train, test_size=0.15)
print(" ======================================================== ")
print("| [Stage 2]: Data Splitting into Training and Testing    |")
print(f"| Number of training images: {len(X_train)}                        |")
print(f"| Number of training labels: {len(Y_train)}                        |")
print(f"| Number of testing images: {len(X_test)}                          |")
print(f"| Number of testing labels: {len(Y_test)}                          |")
print(f"| Number of validation images: {len(X_vaild)}                       |")
print(f"| Number of validation labels: {len(Y_valid)}                       |")
print(" ======================================================== ")

# Stage 3: Model Training and Validation
estimator = []
estimator.append(('FR', RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=2, random_state=1)))
estimator.append(('SVC', svm.SVC(kernel='rbf', C=10, gamma='scale', probability=True)))
estimator.append(('KNN', KNeighborsClassifier(n_neighbors=5)))

vot_hard = VotingClassifier(estimators = estimator, voting ='hard')
vot_hard.fit(X_train, Y_train)
y_pred_hard = vot_hard.predict(X_test)

vot_soft = VotingClassifier(estimators = estimator, voting ='soft')
vot_soft.fit(X_train, Y_train)
y_pred_soft = vot_soft.predict(X_test)

print(" ======================================================== ")
print("| [Stage 3]: Model Training and Validation               |")
print(f"| Number of training images: {len(X_train)}                        |")
print(f"| Number of training labels: {len(Y_train)}                        |")
print(f"| Number of testing images: {len(X_test)}                          |")
print(f"| Number of testing labels: {len(Y_test)}                          |")
print(f"| Number of validation images: {len(X_vaild)}                       |")
print(f"| Number of validation labels: {len(Y_valid)}                       |")
print(" ======================================================== ")

# Stage 4: Model Evaluation
score_hard = accuracy_score(Y_test, y_pred_hard)
print(" ======================================================== ")
print("| [Stage 4]: Model Evaluation                            |")
print("| Accuracy hard:", score_hard*100, '%                     |')
print(" ======================================================== ")

score_soft = accuracy_score(Y_test, y_pred_soft)
print(" ======================================================== ")
print("| [Stage 4]: Model Evaluation                            |")
print("| Accuracy soft:", score_soft*100, '%                      |')
print(" ======================================================== ")

# Check if the file exists
if os.path.exists('model/accuracy.txt'):
  print("[News] Found accuracy.txt!")
  # Read the current accuracy values from the text file
  with open('model/accuracy.txt', 'r') as f:
    lines = f.readlines()
    old_soft_accuracy = float(lines[1].split('=')[1].strip()[:-1])
    old_hard_accuracy = float(lines[0].split('=')[1].strip()[:-1])
else:
  print("[Warnings] accuracy.txt not found!")
  old_soft_accuracy = 0
  old_hard_accuracy = 0

# Calculate the new accuracy values
new_hard_accuracy = score_hard * 100
new_soft_accuracy = score_soft * 100

# Compare the new accuracy values with the current accuracy values
if new_hard_accuracy > old_hard_accuracy:
  print("[News] New hard voting accuracy is higher than the old one!")
  print("[News] Hard Accuracy improved by {:.4f}%".format(new_hard_accuracy - old_hard_accuracy))
  old_hard_accuracy = new_hard_accuracy

if new_soft_accuracy > old_soft_accuracy:
  print("[News] New soft voting accuracy is higher than the old one!")
  print("[News] Soft Accuracy improved by {:.4f}%".format(new_soft_accuracy - old_soft_accuracy))
  old_soft_accuracy = new_soft_accuracy

# Replace the accuracy values in the text file
with open('model/accuracy.txt', 'w') as f:
  f.write('SOFT_VOTING_ACCURACY={:.4f}%\n'.format(old_soft_accuracy))
  f.write('HARD_VOTING_ACCURACY={:.4f}%\n'.format(old_hard_accuracy))


# Stage 5: Model Saving
if old_hard_accuracy == new_hard_accuracy:
  with open('model/hard_voting.hdf5', 'wb') as file:
    joblib.dump(vot_hard, file)
    print(" ======================================================== ")
    print("| [Stage 5]: Hard model saved successfully!             |")
    print(" ======================================================== ")

if old_soft_accuracy == new_soft_accuracy:
  with open('model/soft_voting.hdf5', 'wb') as file:
    joblib.dump(vot_soft, file)
    print(" ======================================================== ")
    print("| [Stage 5]: Soft model saved successfully!             |")
    print(" ======================================================== ")