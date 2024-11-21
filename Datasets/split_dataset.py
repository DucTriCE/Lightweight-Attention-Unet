import os
import random
import shutil

source_dir = 'uitcar/images/train'
val_dir = 'uitcar/images/val'


# all_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

# num_files_to_move = int(0.1 * len(all_files))
# files_to_move = random.sample(all_files, num_files_to_move)

# for file_name in files_to_move:
#     shutil.move(os.path.join(source_dir, file_name), os.path.join(val_dir, file_name))

# print(f"Moved {num_files_to_move} files to {val_dir} folder.")

images = os.listdir(val_dir)

for img in images:
    shutil.move(os.path.join(source_dir.replace('images', 'labels'), img.replace('jpg','png')), os.path.join(val_dir.replace('images', 'labels'), img.replace('jpg','png')))
