import pandas as pd
import cv2
import os
import numpy as np
import tarfile
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from pathlib import Path
'''
This script performs a fast streaming conversion of the FRSign dataset from its original tar.gz 
format to a YOLO-compatible structure.
'''
def process_image_task(binary_data, meta_entry, out_images, out_labels):
    """Convert every image to YOLO format and save it. This function is designed to be run in parallel.
    Args:     binary_data (bytes): The raw binary data of the image extracted from the tar archive.
        meta_entry (dict): A dictionary containing the base name and annotations for the image.
        out_images (Path): The output directory for the processed images.
        out_labels (Path): The output directory for the corresponding YOLO label files.
        """
    try:
        """
        This function takes the raw binary data of an image, decodes it, resizes it to 640x640 using letterboxing,
        and converts the annotations to YOLO format. It then saves the processed image and its corresponding label file in the specified output directories. 
        The function is designed to be run in parallel using a ThreadPoolExecutor.
        """
        base_name = meta_entry['base_name']
        annos = meta_entry['annotations']
        
        img_array = np.asarray(bytearray(binary_data), dtype=np.uint8) # Convert bytes to numpy array
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR) # Decode the image from the numpy array, resulting in a BGR image
        if img is None: return

        # Letterbox resizing to 640x640
        h, w = img.shape[:2] # Get original dimensions of the image
        scale = min(640/w, 640/h) # Calculate the scaling factor to fit the image within 640x640 while maintaining aspect ratio
        nw, nh = int(w * scale), int(h * scale) # Calculate new dimensions after scaling
        img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA) # Resize the image to the new dimensions using area interpolation for better quality
        
        canvas = np.zeros((640, 640, 3), dtype=np.uint8) # Create a black canvas of size 640x640 to place the resized image
        pad_x, pad_y = (640 - nw) // 2, (640 - nh) // 2 # Calculate padding to center the resized image on the canvas
        canvas[pad_y:pad_y+nh, pad_x:pad_x+nw] = img_resized # Place the resized image onto the canvas, effectively letterboxing it to fit within 640x640
        
        # Annotation conversion to YOLO format and clamping to ensure all coordinates are between 0 and 1
        yolo_lines = []
        for a in annos: 
            cx_raw = ((a['x'] * scale) + pad_x + (a['w'] * scale) / 2) / 640 # Calculate the raw center x-coordinate in YOLO format, accounting for scaling and padding
            cy_raw = ((a['y'] * scale) + pad_y + (a['h'] * scale) / 2) / 640
            nw_y_raw = (a['w'] * scale) / 640 # Calculate the raw width in YOLO format, accounting for scaling
            nh_y_raw = (a['h'] * scale) / 640
            # Security! We need coors between 0 and 1! Avoid vanishing gradients during training or errors during inference.
            # It happens because during resizing some boxes can end a bit outside because of python rounding i think
            cx = max(0.0, min(1.0, cx_raw))
            cy = max(0.0, min(1.0, cy_raw))
            nw_y = max(0.0, min(1.0, nw_y_raw))
            nh_y = max(0.0, min(1.0, nh_y_raw))
            yolo_lines.append(f"{a['class']} {cx:.6f} {cy:.6f} {nw_y:.6f} {nh_y:.6f}") # Format the annotation line in YOLO format with class index and normalized coordinates, ensuring all values are between 0 and 1
        
        cv2.imwrite(str(out_images / f"{base_name}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85]) # Save the processed image to the output directory with a specified JPEG quality of 85
        with open(out_labels / f"{base_name}.txt", "w") as f: # Save the corresponding YOLO label file containing the annotations in the output directory
            f.write("\n".join(yolo_lines))
    except Exception:
        pass

def fast_streaming_conversion():
    """
    This function performs a fast streaming conversion of the FRSign dataset from its original tar.gz format to a YOLO-compatible structure.
    It extracts images and annotations directly from the tar archive, processes them in memory, and saves the results in a new directory structure suitable for YOLO training.
    The function also generates a data.yaml file with the necessary dataset information for YOLO.
    
    Steps:
    
1. Define paths for the input tar.gz archive, the HDF5 file containing annotations,
    and the output directories for images and labels.
    
2. Load the annotations from the HDF5 file into pandas DataFrames and perform necessary preprocessing 
to create a lookup structure for images and their corresponding annotations.

3. Use a ThreadPoolExecutor to process images in parallel. For each image in the tar archive that has corresponding annotations,
    extract the image data, convert it to the required format, and save both the processed image and its YOLO label file.
    """
    # Path definitions (1)
    BASE_DIR = Path(__file__).resolve().parent.parent
    archive_path = BASE_DIR / "data" / "FRSign.tar.gz"
    h5_path = BASE_DIR / "data" / "data" / "datasets" / "frsign" / "FRSign_modified" / "FRSign" / "frsign_v1.0.h5"
    
    # Output folders
    dataset_root = BASE_DIR / "data" / "yolo_dataset"
    out_images = dataset_root / "images"
    out_labels = dataset_root / "labels"
    
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    # Load and filter annotations (2)
    print("Loading annotations from HDF5... and building lookup structure for images and their annotations.")
    with pd.HDFStore(str(h5_path), mode='r') as store:
        df_img = store.select('images').reset_index()
        df_meta = store.select('dataframe').reset_index()
    
    s_col = 'sequence' if 'sequence' in df_img.columns else df_img.columns[0]
    i_col = 'image' if 'image' in df_img.columns else df_img.columns[1]
    
    # Mapping sequence/image to type using the metadata dataframe, creating a new 'type' column in the df_img DataFrame
    type_dict = df_meta.set_index(df_meta.columns[0])['type'].to_dict()
    df_img['type'] = df_img[s_col].map(type_dict)
    
    class_names = sorted(df_img['type'].unique().tolist())
    class_map = {name: i for i, name in enumerate(class_names)}

    # Build a lookup dictionary where the key is the image filename and the value is a dictionary containing the base name for output files and a list of annotations for that image.
    meta_lookup = {}
    ignored_count = 0
    
    for _, row in df_img.iterrows():
        # --- FILTRE SNCF : On ne garde que les panneaux sur la voie ---
        if 'on_track' in row and row['on_track'] == 0:
            ignored_count += 1
            continue

        fname = os.path.basename(row['fullpath'])
        if fname not in meta_lookup:
            meta_lookup[fname] = {
                'base_name': f"seq{int(row[s_col])}_img{int(row[i_col])}",
                'annotations': []
            }
        meta_lookup[fname]['annotations'].append({
            'class': class_map[row['type']],
            'x': row['x'], 'y': row['y'], 'w': row['w'], 'h': row['h']
        })

    print(f"Ended: {ignored_count} Ignored smaples (not on track).")  
    print(f"Conversion of {len(meta_lookup)} unique images...")

    #  Process images in parallel (3)
    with ThreadPoolExecutor(max_workers=4) as executor: # Adjust the number of workers based on your CPU capabilities for optimal performance
        with tarfile.open(archive_path, "r:gz") as tar: # Open the tar.gz archive for reading
            pbar = tqdm(total=len(meta_lookup), desc="Progression") # Initialize a progress bar to track the conversion process
            
            for member in tar: # Iterate through each member (file) in the tar archive
                if not member.isfile(): continue # Skip directories and non-file members
                
                fname = os.path.basename(member.name) # Extract the filename from the member's name (which may include directory paths)
                if fname in meta_lookup: # Check if the filename has corresponding annotations in the meta_lookup dictionary
                    meta_entry = meta_lookup[fname]
                    
                    if (out_images / f"{meta_entry['base_name']}.jpg").exists(): # If the processed image already exists in the output directory, skip processing to avoid redundant work
                        pbar.update(1) # Update the progress bar 
                        continue
                    
                    f = tar.extractfile(member) # Extract the file object for the current member from the tar archive
                    if f: # If the file object is successfully extracted, read its binary data and submit a task to the ThreadPoolExecutor 
                        # to process the image and its annotations in parallel
                        binary_data = f.read() # Read the raw binary data of the image from the file object
                        executor.submit(process_image_task, binary_data, meta_entry, out_images, out_labels) # Submit the image processing task to the ThreadPoolExecutor, passing the binary data, metadata entry, and output directories as arguments
                        pbar.update(1) # update progress bar
            pbar.close()

    #  YAML file creation
    with open(dataset_root / "data.yaml", "w") as f:
        f.write(f"path: {dataset_root.absolute()}\n")
        f.write(f"train: train/images\n")
        f.write(f"val: val/images\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")

    print(f"\nExtracted in {dataset_root}")

if __name__ == "__main__":
    fast_streaming_conversion()