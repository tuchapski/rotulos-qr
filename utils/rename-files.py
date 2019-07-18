# -*- coding: utf-8 -*-
import os

BASE_DIR ='C:\\Users\\tuchapski\\Documents\\Projetos\\rotulos-qr\\'
src = BASE_DIR + 'prediction-files\\'


files = os.listdir(src)        

def rename_files(path, prefix):
    i = 1
    path = path
      
    for filename in os.listdir(path): 
        dst = prefix + str(i) + ".jpg"
        src = path + filename 
        dst = path + dst 
          
        # rename() function will 
        # rename all the files 
        os.rename(src, dst) 
        i += 1
        

rename_files(src, 'img-')