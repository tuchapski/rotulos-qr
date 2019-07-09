# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.preprocessing import image
from shutil import copyfile
import random
import matplotlib.image as mpimg


#::::::::::::::::::::::::::::
#::::::PREPARE THE DATASET
#::::::::::::::::::::::::::::

src_dir = 'C:\\Users\\tuchapski\\OneDrive\\TENSOR\\Dogs_Race\\Images\\'
dst_dir = 'C:\\Users\\tuchapski\\OneDrive\\TENSOR\\Dogs_Race\\dataset\\'

def dataset_split(src_dir, dst_dir, split_rate=0.2):
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
        
        for file in files_list[:n_training_files + 1]:
            try:
                copyfile(src_dir + subfolder + '\\' + file, training_dir + subfolder + '\\' + file)
            except:
                print('Already exists, ignoring!')
        for file in files_list[n_training_files::]:
            try:
                copyfile(src_dir + subfolder + '\\' + file, validation_dir + subfolder + '\\' + file)
            except:
                print('Already exists, Ignoring!')
        
                
#dataset_split(src_dir, dst_dir, split_rate=0.8)    


#:::::::::::::::::::::::::::::::::::
#::::::::LOAD AND SETUP PRE-TRAINED MODEL
#:::::::::::::::::::::::::::::::::::

local_weights_file = 'C:\\Users\\tuchapski\\OneDrive\\TENSOR\\Rotulos\\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
img_width = 200
img_height = 200

pre_trained_model = InceptionV3(input_shape = (img_width, img_height, 3),
                                include_top = False,
                                weights = None)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
  layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7') #Corta o modelo InceptionV3 na camada mixed7
#print('last layer output shape', last_layer.output_shape)
last_output = last_layer.output

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x) #Concatena o modelo carregado com as camadas definidas em x

model.compile(optimizer = RMSprop(lr = 0.0001),
              loss = 'binary_crossentropy',
              metrics = ['acc'])

train_dir = 'C:\\Users\\tuchapski\\OneDrive\\TENSOR\\Rotulos\\dataset\\Training'
validation_dir = 'C:\\Users\\tuchapski\\OneDrive\\TENSOR\\Rotulos\\dataset\\Validation'

train_datagen = ImageDataGenerator(rescale = 1.0/255.,
                                   rotation_range = 30,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

validation_datagen = ImageDataGenerator(rescale = 1.0/255.)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = 32,
                                                    class_mode = 'binary',
                                                    target_size = (img_width, img_height))

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                  batch_size = 32,
                                                  class_mode = 'binary',
                                                  target_size = (img_width, img_height))

#Run model training
history = model.fit_generator(train_generator,
                              validation_data = validation_generator,
                              steps_per_epoch = 100,
                              epochs = 10,
                              validation_steps = 10,
                              verbose = 1)

train_labels = train_generator.class_indices

#model.save_weights('Dogs_Race\\dog_races_acc59.h5')
#model.save('dog_races_acc59.h5') #To save the entire model en load later with load_model...
#model.load_weights('Dogs_Race\\dog_races_acc61.h5')

#model = tf.keras.models.load_model('dog_races_acc61.h5') #To load full model after...

#print(history.history.keys())


#:::::::::::::::::::::::::::::::::::::::::
#::::::::PLOT THE TRANING/VALIDATION GRAPHS
#:::::::::::::::::::::::::::::::::::::::::
def plot_model_accuracy(model_history, show_loss=False):
    # summarize history for accuracy
    plt.plot(model_history.history['acc'])
    plt.plot(model_history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    if show_loss == True:
        # summarize history for loss
        plt.plot(model_history.history['loss'])
        plt.plot(model_history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

#plot_model_accuracy(history, show_loss=True)


#::::::::::::::::::::::::::::::::::::::
#:::::::MAKING PREDICTIONS
#::::::::::::::::::::::::::::::::::::::

def remove_labelID(name):
    s = name[1:]
    result = ''.join([i for i in s if not i.isdigit() and i.isalnum()])
    return result

def predict_images(prediction_dir):
    
    for fn in os.listdir(prediction_dir):
        path=prediction_dir + fn
        img=image.load_img(path, target_size=(img_width, img_height))
        x=image.img_to_array(img)
        x /= 255.
        x=np.expand_dims(x, axis=0)
        images = np.vstack([x]) #O que é vstack ? Os valores estão normalizados como no treino ? 
  
        classes = model.predict(images, batch_size=10)
  
        max_element = np.amax(classes)
        search = np.where(classes == max_element)
        search = search[1]

        for item, value in train_labels.items():
            if value == search:       
                sample_path = train_dir + '\\' + item
                sample_path = sample_path + '\\' + os.listdir(sample_path)[random.randint(0,20)]
                predicted_item = item
  
        # plot images side by side!
        path = mpimg.imread(path)
        sample_path = mpimg.imread(sample_path)
        
        fig = plt.figure(figsize=(15,15))
        ax1 = fig.add_subplot(2,2,1)
        ax1.set_title(fn)
        ax1.imshow(path)
        ax2 = fig.add_subplot(2,2,2)
        ax2.set_title(remove_labelID(predicted_item))
        ax2.imshow(sample_path)
  
prediction_dir = 'C:\\Users\\tuchapski\\OneDrive\\TENSOR\\Dogs_Race\\prediction-files\\'
predict_images(prediction_dir)




