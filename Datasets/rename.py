import os

folder_path = 'bridge/labels'
files = sorted(os.listdir(folder_path))

for i, filename in enumerate(files, start=1):
    new_name = f"image{i+620}.png"
    
    old_file = os.path.join(folder_path, filename)
    new_file = os.path.join(folder_path, new_name)
    
    os.rename(old_file, new_file)

print("Images have been renamed successfully!")
