import os, fnmatch

class Preprocess:
    
    def __init__(self, base_dir=None):
        self.base_dir = base_dir
        if self.base_dir == None:
            self.base_dir = os.getcwd()
            
    def list_from_dir(self, folder=''):
        self.folder = os.path.join(self.base_dir, folder)
        self.found_files = []          
        for self.filename in os.listdir(self.folder):
#           if os.path.isfile(os.path.join(self.folder, self.filename)):
            self.found_files.append(self.filename)
        return self.folder, self.found_files   
    
    def list_from_dir_recursive(self, folder=''):
        self.folder = folder
        self.found_files = []        
        for self.d in os.walk(os.path.join(self.base_dir, folder)):        
            self.found_files.append(self.d)
        return self.found_files
    
    def search_files(self, pattern='*.*', recursive='n'):
        self.pattern = pattern
        self.recursive = recursive
        
    """def rename_files(self, prefix='file_', extention):
        pass
        self.prefix = prefix
        self.extention = extention
        self.i = 1
        if len(self.scanned_files) == 0:
            self.scan_files(extention=self.extention)
        for self.filename in self.scanned_files:
            if i < 10:
                dst = self.prefix + '0' + str(self.i) + self.extention
            else:
                dst = prefix + str(i) + self.extention
                    
                self.src = path + filename 
                dst = path + dst 
                    
                # rename() function will 
                # rename all the files 
                os.rename(src, dst) 
                i += 1"""
            
yum = Preprocess()

yum.list_files_from_dir()