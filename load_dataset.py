import zipfile
import re
import os


files = os.listdir()
zip_files = [file for file in files if re.match(r'dataset.*\.zip', file)]


if len(zip_files) == 0:
    print("No zip file found")
    exit(1)
else:
    print("Zip files found: ", zip_files)


for zip_file in zip_files:
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall()
print("Dataset extracted successfully")

