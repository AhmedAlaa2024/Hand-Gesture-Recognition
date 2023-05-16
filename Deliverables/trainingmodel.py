from helper import *
import joblib

preprocessed_images = []
feature_vectors = []

# Stage 1: Data Loading, Preprocessing and Feature Extraction
Y_train, X_train = load_all_dataset__win()
print(" ======================================================== ")
print("| [Stage 2]: Data Splitting into Training and Testing    |")
print(f"| Number of training images: {len(X_train)}              |")
print(f"| Number of training labels: {len(Y_train)}              |")
print(" ======================================================== ")

# Stage 3: Model Training and Validation
estimator = []
estimator.append(('FR', RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=2, random_state=1)))
estimator.append(('SVC', svm.SVC(kernel='rbf', C=10, gamma='scale', probability=True)))
estimator.append(('KNN', KNeighborsClassifier(n_neighbors=5)))

vot_hard = VotingClassifier(estimators = estimator, voting ='hard')

vot_hard.fit(X_train, Y_train)

vot_soft = VotingClassifier(estimators = estimator, voting ='soft')
vot_soft.fit(X_train, Y_train)

with open('models/hard_training_voting.hdf5', 'wb') as file:
    joblib.dump(vot_hard, file)
    print(" ======================================================== ")
    print("| [Stage 5]: Hard model saved successfully!             |")
    print(" ======================================================== ")
with open('models/soft_training_voting.hdf5', 'wb') as file:
    joblib.dump(vot_soft, file)
    print(" ======================================================== ")
    print("| [Stage 5]: Soft model saved successfully!             |")
    print(" ======================================================== ")