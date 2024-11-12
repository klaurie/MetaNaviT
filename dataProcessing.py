import os
from datetime import datetime

def get_file_path():
    # get file path from user

    while True:
        file_path = input("Enter the file path to set root dir: ")

        if os.path.exists(file_path):
            return file_path
        else:
            print("Invalid file path.")

file_path = get_file_path()
print("this is the file_path: ", file_path)

# collection of all things in file 

parsedfile = []

for root, dirs, files in os.walk(file_path,):
    for filename in files:

        file_path = os.path.join(root, filename)

        metaData = {
            "file_path": file_path,
            "file_name": filename,
            "file_type": os.path.splitext(filename)[1],  # Get file extension
            "file_size": os.path.getsize(file_path),      # File size in bytes
            "last_modified": datetime.fromtimestamp(os.path.getmtime(file_path))  # Last modified date
        }

        parsedfile.append(metaData)

for file in parsedfile:
    print(file)
