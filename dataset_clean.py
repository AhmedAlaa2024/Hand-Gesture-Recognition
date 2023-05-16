import os
import random
import shutil
import time

# the path of the original directory
path = "E:/2nd term 3rd year/Neural Network/Project/dataset"

# the path of the new directory
new_path = "E:/2nd term 3rd year/Neural Network/Project/data"

# delete the new directory if it exists
if os.path.exists(new_path):
    shutil.rmtree(new_path)

# create the new directory
os.mkdir(new_path)

# Counter to count all the images
men_counter = [0] * 6
women_counter = [0] * 6

snapshot_time = time.time()

# traverse recursivly on the original directory
for root, dirs, files in os.walk(path):
    for file in files:
        # check if the file is JPG
        if file.endswith(".JPG"):
            # check if the file is in a subdirectory
            if root != path:
                # get the name of the subdirectory
                sub_dir = root.split(os.path.sep)[1]

                # check if the subdirectory is "M" or "W"
                if sub_dir == "men":
                    # get the number of the image
                    num = file.split(".")[0][0]
                    men_counter[int(num)] += 1

                    # get the ID of the image
                    id = file.split(".")[0][7:-1]

                    # copy the image to the new directory and rename it
                    shutil.copy(os.path.join(root, file), os.path.join(new_path, "M_" + num + "_" + id + ".jpg"))

                    print("Creating " + "M_" + num + "_" + id + ".jpg")
                elif sub_dir == "Women":
                    # get the number of the image
                    num = file.split(".")[0][0]
                    women_counter[int(num)] += 1

                    # get the ID of the image
                    id = file.split(".")[0][9:-1]

                    # copy the image to the new directory and rename it
                    shutil.copy(os.path.join(root, file), os.path.join(new_path, "W_" + num + "_" + id + ".jpg"))

                    print("Creating " + "W_" + num + "_" + id + ".jpg")
            else:
                # get the number of the image
                num = file.split(".")[0]

                # get the ID of the image
                id = file.split(".")[1]

                # copy the image to the new directory and rename it
                shutil.copy(os.path.join(root, file), os.path.join(new_path, "M_" + num + "_" + id))

                # copy the image to the new directory and rename it
                shutil.copy(os.path.join(root, file), os.path.join(new_path, "W_" + num + "_" + id))

print('\n')
print(" ===========================================================================")
print("|              Gathering Dataset is completed successfully                  |")
print(" ===========================================================================")

# print the number of images in each class
total_men_counter = 0
total_women_counter = 0

for i in range(6):
    total_men_counter += men_counter[i]
    total_women_counter += women_counter[i]

for i in range(6):
    print("Men[{i}]: " + str(men_counter[i] / total_men_counter * 100) + "%")
print("\nTotal Men Images: " + str(total_men_counter))
print('\n')

for i in range(6):
    print("Women[{i}]: " + str(women_counter[i] / total_women_counter * 100) + "%")
print("\nTotal Women Images: " + str(total_women_counter))

print("\nTotal time for gathering dataset: " + str(time.time() - snapshot_time) + " seconds")
print('\n')
# print(" ===========================================================================")
# print("|               the data                                                    |")
# print(" ===========================================================================")

# shutil.copy2(os.path.join(path, file), os.path.join(new_path, file))


print(" ===========================================================================")
print("|              Split the data in training, validation, test                  |")
print(" ===========================================================================")

# Define the paths to the training, validation, and test directories
train_path = "training"
val_path = "validation"
test_path = "test"

# Define the percentage of data to use for each set
train_percent = 0.7
val_percent = 0.15
test_percent = 0.15

training_counter = 0
validation_counter = 0
test_counter = 0

# Create the training, validation, and test directories if they don't exist
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

snapshot_time = time.time()
# Loop over each file in the dataset directory
for file in os.listdir(new_path):
    # Generate a random number between 0 and 1
    rand = random.random()

    # Copy the file to the appropriate directory based on the random number
    if rand < train_percent:
        training_counter += 1
        shutil.copy2(os.path.join(new_path, file), os.path.join(train_path, file))
    elif rand < train_percent + val_percent:
        validation_counter += 1
        shutil.copy2(os.path.join(new_path, file), os.path.join(val_path, file))
    else:
        test_counter += 1
        shutil.copy2(os.path.join(new_path, file), os.path.join(test_path, file))

shutil.rmtree(new_path)
os.mkdir(new_path)

print("Training dataset count: " + str(training_counter))
print("Validation dataset count: " + str(validation_counter))
print("Test dataset count: " + str(test_counter))
print("Total time for partitioning the dataset: " + str(time.time() - snapshot_time) + " seconds")

print('\n')
print(" ===========================================================================")
print("|                                Done                                       |")
print(" ===========================================================================")