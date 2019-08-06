import os
from util.preprocess_files import Preprocess

BASE_DIR = os.getcwd()
SOURCE_DIR = 'qr-dataset'
DATASET_DIR = 'dataset'
split_rate= 0.15

prepare_files = Preprocess(path=BASE_DIR, 
                           source_data=SOURCE_DIR, 
                           dataset_folder=DATASET_DIR)

prepare_files.train_test_split(split_rate=split_rate)