import os
import random
import shutil
from pathlib import Path

def split_yolo_data(data_path="data/yolo_dataset", split_ratio=0.2):
    """ orders the data in a Yolo structure with train/val splits.
    Args:     data_path (str): The base path where the 'images' and 'labels' folders are located.
        split_ratio (float): The proportion of data to be used for validation.
    """
    BASE_DIR = Path(__file__).resolve().parent.parent
    path = BASE_DIR / data_path
    
    img_src = path / "images"
    lbl_src = path / "labels"
    
    if not img_src.exists():
        print(f"Error : Folder {img_src} can't be found.")
        return

    # Yolo structure is data/yolo_dataset/{train,val}/images and data/yolo_dataset/{train,val}/labels
    for split in ['train', 'val']:
        (path / split / "images").mkdir(parents=True, exist_ok=True)
        (path / split / "labels").mkdir(parents=True, exist_ok=True)
    images = [f for f in os.listdir(img_src) if f.endswith('.jpg')]
    random.seed(42)
    random.shuffle(images)
    val_count = int(len(images) * split_ratio)
    val_images = images[:val_count]
    train_images = images[val_count:]

    def move_files(file_list, target_split):
        """
        Move the specified files from the source directories to the target split directories.
        """
        count = 0
        for img_name in file_list:
            # Source
            s_img = img_src / img_name # Construct the source path for the image file
            s_lbl = lbl_src / img_name.replace('.jpg', '.txt') # Construct the source path for the corresponding label file by replacing the .jpg extension with .txt
            
            # Destination
            d_img = path / target_split / "images" / img_name
            d_lbl = path / target_split / "labels" / img_name.replace('.jpg', '.txt')
            
            # copy
            if s_img.exists(): # Check if the source image file exists before attempting to move it
                shutil.move(str(s_img), str(d_img)) # Move the image file from the source to the destination directory
            if s_lbl.exists():
                shutil.move(str(s_lbl), str(d_lbl))
            count += 1
        return count

    print(f"data repartition  ({100-int(split_ratio*100)}/{int(split_ratio*100)}) :")
    c_train = move_files(train_images, "train") # Move training images and labels to the 'train' split directories and count how many were moved
    print(f" - {c_train} images to TRAIN")
    
    c_val = move_files(val_images, "val") # Move validation images and labels to the 'val' split directories and count how many were moved
    print(f" - {c_val} images to VAL")
    
    
    try:
        if not os.listdir(img_src): os.rmdir(img_src)
        if not os.listdir(lbl_src): os.rmdir(lbl_src)
        print("\nOrdered data structure created successfully.")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    split_yolo_data()