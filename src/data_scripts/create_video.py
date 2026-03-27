import shutil
from pathlib import Path
"""We want to createa folder for only one sequence"""
def filter_and_copy_images(src_dir, dest_dir, pattern):
    src_path = Path(src_dir)
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    files_to_copy = list(src_path.glob(f"{pattern}*"))
    print(f"File  numbers : {len(files_to_copy)}")
    for file in files_to_copy:
        if file.is_file():
            shutil.copy2(file, dest_path / file.name)



SOURCE = "./data/yolo_dataset/val/images"
DESTINATION = "./data/video"
PATTERN = "seq83"

if __name__ == "__main__":
    filter_and_copy_images(SOURCE, DESTINATION, PATTERN)