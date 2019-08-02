import os, fnmatch

class Preprocess:
    
    def __init__(self):
        pass
    
    def search_files(self, path='.',   
                     pattern='*.*'):
        self.path = path
        self.pattern = pattern                       
        self.search_result = []
        for self.item in os.listdir(self.path):
            if fnmatch.fnmatch(self.item, self.pattern) and os.path.isfile(os.path.join(self.path, self.item)):
                self.search_result.append(self.item)
        return self.search_result                        
                
    def rename_files(self, path=None,
                     new_name='',
                     pattern = '*.*'):
        self.path = path
        SELF.new_name = new_name
        self.pattern = pattern
        self.serial = serial
        self.i = 1        
        self.file_list = self.search_files(self.path, self.pattern)
        
        for self.f in self.file_list:
            if self.i < 10:
                self.serie = '0' + str(self.i)
                    for self.filename in self.scanned_files:
                        if i < 10:
                            self.dst = self.prefix + self + str(self.i) + self.extention
            else:
                dst = prefix + str(i) + self.extention
                    
                self.src = path + filename 
                dst = path + dst 
                    
                # rename() function will 
                # rename all the files 
                os.rename(src, dst) 
                i += 1       