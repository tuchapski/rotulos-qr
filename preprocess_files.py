import os, glob

class Preprocess:
    
    def __init__(self):
        if dir_path == None:
            self.dir_path = os.getcwd()
                
    def scan_files(self, pattern='*.*'):
        os.chdir(dir_path)
        self.pattern = pattern
        return glob.glob(os.path.join(self.dir_path, self.extention))
    
    def 
        
    def summary(self):
        print('Summary')
        print('files_path: {0}'.format(self.dir_path))
        print('n_files: {0}'.format(len(self.scanned_files))) 
        
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
                


dir_path = os.getcwd()

found = glob.glob(('*.*'))

print(found)