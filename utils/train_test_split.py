# -*- coding: utf-8 -*-
import os
from shutil import copyfile
import random

BASE_DIR ='C:\\Users\\tuchapski\\Documents\\Projetos\\rotulos-qr\\'
src_dir = BASE_DIR + 'qr-dataset\\'
dst_dir = BASE_DIR + 'dataset\\'

def dataset_split(src_dir, dst_dir, split_rate=0.1):
    dst_dir = dst_dir
    training_dir = dst_dir + '\\Training\\'
    validation_dir = dst_dir + '\\Validation\\'
    split_rate= split_rate
    
    if os.path.exists(dst_dir): 
        print(dst_dir + ' Found.')
        if not os.path.exists(training_dir):
            os.mkdir(training_dir)
        if not os.path.exists(validation_dir):
            os.mkdir(validation_dir)
    else:
        print('Directory not found, creating..')
        os.mkdir(dst_dir)
        os.mkdir(training_dir)
        os.mkdir(validation_dir)
 
    #Copy subdirectories Structure from Source Directory
    dir_list = [folder for folder in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir,folder))]
    
    for subfolder in dir_list:
        try:
            os.mkdir(training_dir + subfolder)
        except:
            print(training_dir + subfolder + ' Already exists, Skipping')
        try:
            os.mkdir(validation_dir + subfolder)
        except:
            print(validation_dir + subfolder + ' Already exists, Skipping')
    
    #Validate, Shuffle and copy directories to Training and Validation folders at destination
    for subfolder in dir_list:
        folder_path = src_dir + subfolder           
        files_list = []
        
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            
            if os.path.isfile(file_path) and os.path.getsize(file_path) == 0:
                print('Ignoring: ' + file_path)
            else:
                files_list.append(file)
                
        random.shuffle(files_list)       
        n_training_files = int(len(files_list) * split_rate)
        
        for file in files_list[n_training_files:]:
            try:
                copyfile(src_dir + subfolder + '\\' + file, training_dir + subfolder + '\\' + file)
            except:
                print('Already exists, ignoring!')
        for file in files_list[:n_training_files + 1]:
            try:
                copyfile(src_dir + subfolder + '\\' + file, validation_dir + subfolder + '\\' + file)
            except:
                print('Already exists, Ignoring!')
                
    print('Done!')
                
dataset_split(src_dir, dst_dir)
        