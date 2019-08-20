import os
from util.preprocess_files import *

SOURCE_DIR = os.path.join('.', 'qr-dataset')
split_rate= 0.15

model_dir = ModelData()
source_data = SourceData(SOURCE_DIR)

model_dir.delete_structure()