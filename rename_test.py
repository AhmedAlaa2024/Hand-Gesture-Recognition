import os
import shutil

# Define the source and destination folders
src_folder = 'E:/2nd term 3rd year/Neural Network/Project/Hand-Gesture-Recognition/remame'
dst_folder = 'E:/2nd term 3rd year/Neural Network/Project/Hand-Gesture-Recognition/Deliverables/test'

# Get a list of all the image files in the source folder
img_files = [f for f in os.listdir(src_folder) if f.endswith('.jpg') or f.endswith('.png')]

# Sort the image files alphabetically
img_files.sort()

# Rename and write the image files to the destination folder
for i, img_file in enumerate(img_files):
    # Construct the new file name using the index and file extension
    new_file_name = f'image_{i+1}{os.path.splitext(img_file)[1]}'
    
    # Construct the full paths for the source and destination files
    src_file_path = os.path.join(src_folder, img_file)
    dst_file_path = os.path.join(dst_folder, new_file_name)
    
    # Copy the source file to the destination folder with the new name
    shutil.copy(src_file_path, dst_file_path)