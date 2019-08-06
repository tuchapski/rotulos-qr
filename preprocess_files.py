import os, fnmatch

class Preprocess:
    
    def __init__(self, path='.'):
        self.path = path
    
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
                
    def rename_files(self, path,
                     new_name='file_',
                     pattern = '*'):
        self.path = path
        self.new_name = new_name
        self.pattern = pattern
        self.i = 1 
        self.file_list = self.search_files(self.pattern)
        self.ext = []

        
        for self.file in self.file_list:
            self.ext = self.get_extention(self.file)
            if self.i < 10:
                self.new = self.new_name + '0' + str(self.i) + self.ext
            else:
                self.new = self.new_name + str(self.i) + self.ext
            
            self.src = os.path.join(self.path, self.file)
            self.dst = os.path.join(self.path, self.new)

            os.rename(self.src, self.dst) 
            self.i += 1