import os
from preprocess_files import Preprocess

BASE_DIR = os.getcwd()
IMAGES_DIR = os.path.join(BASE_DIR, 'data')

files = Preprocess(IMAGES_DIR)
#files.rename_files(IMAGES_DIR, pattern='*.docx')
files.get_empty_extention()

files.search_files()

files.rename_files(IMAGES_DIR, new_name='fake-', pattern='2x*.jpg')