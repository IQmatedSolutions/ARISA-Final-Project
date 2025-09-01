# Split between train and val folders
from pathlib import Path
import random
import os
import sys
import shutil
import yaml

def train_val_split(data_path,train_percent):
    # Check for valid entries
    if not os.path.isdir(data_path):
        print("Directory specified by --data_path not found. Verify the path is correct  and try again.")
        sys.exit(0)
    if train_percent < .01 or train_percent > 0.99:
        print("Invalid entry for train_percent. Please enter a number between .01 and .99.")
        sys.exit(0)

    # Define path to input dataset 
    input_image_path = os.path.join(data_path,"images")
    input_label_path = os.path.join(data_path,"labels")

    # Define paths to image and annotation folders
    cwd = os.getcwd()
    train_img_path = os.path.join(cwd,"dataset/train/images")
    train_txt_path = os.path.join(cwd,"dataset/train/labels")
    val_img_path = os.path.join(cwd,"dataset/validation/images")
    val_txt_path = os.path.join(cwd,"dataset/validation/labels")

    # Create folders if they don"t already exist
    for dir_path in [train_img_path, train_txt_path, val_img_path, val_txt_path]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created folder at {dir_path}.")

    # Get list of all images and annotation files
    img_file_list = [path for path in Path(input_image_path).rglob("*")]
    txt_file_list = [path for path in Path(input_label_path).rglob("*")]

    print(f"Number of image files: {len(img_file_list)}")
    print(f"Number of annotation files: {len(txt_file_list)}")

    # Determine number of files to move to each folder
    file_num = len(img_file_list)
    train_num = int(file_num*train_percent)
    val_num = file_num - train_num
    print("Images moving to train: %d" % train_num)
    print("Images moving to validation: %d" % val_num)

    # Select files randomly and copy them to train or val folders
    for i, set_num in enumerate([train_num, val_num]):
        for ii in range(set_num):
            img_path = random.choice(img_file_list)
            img_fn = img_path.name
            base_fn = img_path.stem
            txt_fn = base_fn + ".txt"
            txt_path = os.path.join(input_label_path,txt_fn)

            if i == 0: # Copy first set of files to train folders
                new_img_path, new_txt_path = train_img_path, train_txt_path
            elif i == 1: # Copy second set of files to the validation folders
                new_img_path, new_txt_path = val_img_path, val_txt_path

            shutil.copy(img_path, os.path.join(new_img_path,img_fn))
            if os.path.exists(txt_path): # If txt path does not exist, this is a background image, so skip txt file
                shutil.copy(txt_path,os.path.join(new_txt_path,txt_fn))

            img_file_list.remove(img_path)

def create_data_yaml(path_to_classes_txt, path_to_data_yaml):
  # Python function to automatically create data.yaml config file
  # 1. Reads "classes.txt" file to get list of class names
  # 2. Creates data dictionary with correct paths to folders, number of classes, and names of classes
  # 3. Writes data in YAML format to data.yaml
  # Read class.txt to get class names
  if not os.path.exists(path_to_classes_txt):
    print(f"classes.txt file not found! Please create a classes.txt labelmap and move it to {path_to_classes_txt}. Current path is {os.getcwd()}. List of files {os.listdir()}")
    return
  with open(path_to_classes_txt, "r") as f:
    classes = []
    for line in f.readlines():
      if len(line.strip()) == 0: continue
      classes.append(line.strip())
  number_of_classes = len(classes)

  # Create data dictionary
  data = {
      "path": "/dataset",
      "train": "train/images",
      "val": "validation/images",
      "nc": number_of_classes,
      "names": classes,
      "augmentations": {
                    "hsv_h": 0.015,  # hue
                    "hsv_s": 0.7,   # saturation
                    "hsv_v": 0.4,   # value
                    "degrees": 0.0, # rotation
                    "translate": 0.1, # translation
                    "scale": 0.5,   # scaling
                    "shear": 0.0,   # shearing
                    "perspective": 0.0, # perspective distortion
                    "flipud": 0.0,  # flip up and down
                    "fliplr": 0.5,  # flip left and right
                    "mosaic": 1.0,  # mosaic
                    "mixup": 0.0,  # mixup
                    "copy_paste": 0.0 # copy-paste for segmentation}
                    }
        }

  # Write data to YAML file
  with open(path_to_data_yaml, "w") as f:
    yaml.dump(data, f, sort_keys=False)
  print(f"Created config file at {path_to_data_yaml}")

  return

# Define path to classes.txt and run function
path_to_classes_txt = "./dataset/classes.txt"
path_to_data_yaml = "./dataset/data.yaml"

train_val_split("dataset",0.9)
create_data_yaml(path_to_classes_txt, path_to_data_yaml)
