import os, fnmatch
import random
from shutil import copyfile

class Preprocess:
    
    def __init__(self, path='.', 
                 source_data='data',
                 dataset_folder='dataset',
                 train_dir_name='Training',
                 validation_dir_name='Validation'):
        self.path = path
        self.source_data = os.path.join(self.path, source_data)
        self.dataset_folder = os.path.join(self.path, dataset_folder)
        self.train_dir_name = train_dir_name
        self.validation_dir_name = validation_dir_name
    
    def get_extention(self, file):
        self.file = file
        os.path.join(self.path, self.file)
        return os.path.splitext(file)[1]
    
    def get_empty_extention(self):
        self.file_list = self.search_files(pattern='*')
        self.empty_files = []        
        for self.file in self.file_list:
            if self.get_extention(self.file) == '':
                self.empty_files.append(self.file)
        return self.empty_files
   
    def remove_files(self, files=[]):
        self.files = files
        for self.file in self.files:
            os.remove(os.path.join(self.path, self.file))
            
    def search_files(self, pattern='*.*'):
        self.pattern = pattern                       
        self.search_result = []
        for self.item in os.listdir(self.path):
            if fnmatch.fnmatch(self.item, self.pattern) and os.path.isfile(os.path.join(self.path, self.item)):
                self.search_result.append(self.item)
        return self.search_result                        
                
    def rename_files(self, pattern='*',         
                     new_name='file_',
                     shuffle=False):
        self.pattern = pattern
        self.new_name = new_name
        self.shuffle = shuffle
        self.i = 1 
        self.file_list = self.search_files(self.pattern)
        if self.shuffle == True:
            self.file_list = random.sample(self.file_list, len(self.file_list))
        self.ext = []    
        for self.file in self.file_list:
            self.ext = self.get_extention(self.file)
            self.new = self.new_name + "{0:02d}".format(self.i) + self.ext
            self.n = 1
            while True:
                if self.new in self.search_files(self.new):
                    self.new = self.new_name + "{0:02d}".format(self.n) + self.ext
                    self.n +=1
                else:
                    break
            self.src = os.path.join(self.path, self.file)
            self.dst = os.path.join(self.path, self.new)
            os.rename(self.src, self.dst) 
            self.i += 1
            
    def create_training_folder_structure(self):
        self.train_dir_name = os.path.join(self.dataset_folder, self.train_dir_name)
        self.validation_dir_name = os.path.join(self.dataset_folder, self.validation_dir_name)      
        if not os.path.exists(self.dataset_folder): 
            print('Created: ' + self.dataset_folder)
            os.mkdir(self.dataset_folder)
        if not os.path.exists(self.train_dir_name):
            print('Created: ' + self.train_dir_name)
            os.mkdir(self.train_dir_name)
        if not os.path.exists(self.validation_dir_name):
            print('Created: ' + self.validation_dir_name)
            os.mkdir(self.validation_dir_name)
            
    def get_src_dir_structure(self, src_dir):
        self.src_dir = src_dir
        self.dir_list = [self.folder for self.folder in os.listdir(self.src_dir) if os.path.isdir(os.path.join(self.src_dir, self.folder))]
        return self.dir_list
    
    def copy_scr_dir_structure(self):
        for self.subfolder in self.get_src_dir_structure(self.source_data):
            try:
                os.mkdir(os.path.join(self.dataset_folder, self.train_dir_name, self.subfolder))
            except:
                print(os.path.join(self.train_dir_name, self.subfolder) + 'Skipping')
            try:
                os.mkdir(os.path.join(self.dataset_folder, self.validation_dir_name, self.subfolder))
            except:
                print(os.path.join(self.validation_dir_name, self.subfolder) + 'Skipping')
                    
    def train_test_split(self, split_rate=0.2):
        self.split_rate = split_rate
        self.create_training_folder_structure()
        self.copy_scr_dir_structure()
        
        #Validate, Shuffle and copy directories to Training and Validation folders at destination
        for self.subfolder in self.get_src_dir_structure(self.source_data):    
            self.folder_list = os.path.join(self.source_data, self.subfolder)
            self.files_list = []
            for self.file in os.listdir(self.folder_list):
                self.file_path = os.path.join(self.folder_list, self.file)                
                if (os.path.isfile(self.file_path) == False) and (os.path.getsize(self.file_path) == 0):
                    print('Ignoring: ' + self.file_path)
                else:
                    self.files_list.append(self.file)                    
            self.n_training_files = int(len(self.files_list) * self.split_rate)     
            
            for self.file in self.files_list[self.n_training_files:]:
                try:
                    copyfile(self.file_path, os.path.join(self.dataset_folder, self.train_dir_name, self.subfolder, self.file))
                except:
                    print('Already exists, ignoring!')
            for self.file in self.files_list[:self.n_training_files + 1]:
                try:
                    copyfile(self.file_path, os.path.join(self.dataset_folder, self.validation_dir_name, self.subfolder, self.file))
                except:
                    print('Already exists, Ignoring!')
                        
        print('Done!')
        
    
    
    
    
    
    
    
    
    
    
    

