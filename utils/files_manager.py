import os

class ImageTools:
    
    def __init__(self, dir_path, extentions=['jpg']):
        self.dir_path = dir_path    
        self.extentions = extentions
    
    def run_all_extentions(self):
        self.file_list = []
        self.extentions_located = []
        for self.file in os.listdir(self.dir_path):
            for self.ext in self.extentions:
                if self.file.endswith('.' + self.ext):
                    self.file_list.append(self.file)
                    if self.ext not in self.extentions_located:
                        self.extentions_located.append(self.ext)              
        return self.file_list
    
        
    def summary(self):
        print('Summary \n')
        print('dir_path: {0}'.format(self.dir_path))
        print('n_files: {0}'.format(len(self.run_all_extentions()))) 
        print('extentions located: {0}'.format(self.extentions_located))
        
    def rename_files(self):
        self.i = 1
                
        for self.ext in self.extentions:
            for filename in os.listdir(path): 
                if i < 10:
                    dst = prefix + '0' + str(i) + ".jpg"
                else:
                    dst = prefix + str(i) + ".jpg"
                        
                    src = path + filename 
                    dst = path + dst 
                        
                    # rename() function will 
                    # rename all the files 
                    os.rename(src, dst) 
                    i += 1
        

file_maker = ImageTools('C:\\Users\\tuchapski\\Documents\\sage', extentions=['txt', 'docx'])

file_maker.summary()

file_maker.file_list


