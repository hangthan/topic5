import zipfile
import os

zip_path = r"D:\2026\DL\topic5\kaggle_output\topic05_outputs_full_training.zip"
extract_path = r"D:\2026\DL\topic5\kaggle_output\extracted"

if not os.path.exists(extract_path):
    os.makedirs(extract_path)

print(f"Opening zip: {zip_path}")
with zipfile.ZipFile(zip_path, 'r') as z:
    for file_info in z.infolist():
        if file_info.filename.endswith(('.csv', '.json', '.png')):
            # Don't extract large model checkpoints
            print(f"Extracting: {file_info.filename}")
            z.extract(file_info, extract_path)
print("Done extracting summary files.")
