import os
from preprocess_files import Preprocess

BASE_DIR = os.getcwd()
IMAGES_DIR = os.path.join(BASE_DIR, 'data')

files = Preprocess()
files = files.search_files(path=IMAGES_DIR, pattern='*.*')

print(files)


files