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
                
#dataset_split(src_dir, dst_dir, split_rate=0.8)    

#:::::::::::::::::::::::::::::::::::
#::::::::LOAD AND SETUP PRE-TRAINED MODEL
#:::::::::::::::::::::::::::::::::::
                
BASE_DIR = "C:\\Users\\tuchapski\\Documents\\Projetos\\rotulos-qr\\"                
local_weights_file = BASE_DIR + 'weights\\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
img_width = 200
img_height = 200
batch_size = 8
learning_rate = 0.0001
epochs = 5

pre_trained_model = InceptionV3(input_shape = (img_width, img_height, 3),
                                include_top = False,
                                weights = None)

pre_trained_model.load_weights(local_weights_file)

#for layer in pre_trained_model.layers:
#  layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7') #Corta o modelo InceptionV3 na camada mixed7
#print('last layer output shape', last_layer.output_shape)
last_output = last_layer.output

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(2, activation='softmax')(x)

model = Model(pre_trained_model.input, x) #Concatena o modelo carregado com as camadas definidas em x

model.compile(optimizer = RMSprop(lr=learning_rate),
              loss = 'categorical_crossentropy',
              metrics = ['acc'])

train_dir = BASE_DIR + 'dataset\\Training'
validation_dir = BASE_DIR + 'dataset\\Validation'

train_datagen = ImageDataGenerator(rescale = 1.0/255.,
                                   rotation_range = 10,
                                   width_shift_range = 0.1,
                                   height_shift_range = 0.1,
                                   shear_range = 0.1,
                                   zoom_range = 0.1,
                                   horizontal_flip = True)

validation_datagen = ImageDataGenerator(rescale = 1.0/255.)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = batch_size,
                                                    class_mode = 'categorical',
                                                    target_size = (img_width, img_height))

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                  batch_size = batch_size,
                                                  class_mode = 'categorical',
                                                  target_size = (img_width, img_height))

#Run model training
history = model.fit_generator(train_generator,
                              validation_data = validation_generator,
                              epochs = epochs,
                              verbose = 1)

train_labels = train_generator.class_indices

#::::::::::::::::::::::::::::::::::::
#::::::::::Save/Load Model
#::::::::::::::::::::::::::::::::::::

model.save_weights(BASE_DIR + 'weights\\weights-rotulos-softmax--10000.h5')
#model.save('label_antartica_5833.h5') #To save the entire model en load later with load_model...
model.load_weights(BASE_DIR + 'weights\\weights-rotulos-softmax--6250.h5')

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

#plot_model_accuracy(history, show_loss=False)


#::::::::::::::::::::::::::::::::::::::
#:::::::MAKING PREDICTIONS
#::::::::::::::::::::::::::::::::::::::

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
                predicted = item
        
        prob = max(classes[0])
        
        if prob < 0.7:
            print('Imagem {0} é {1} - Prob {2:.3f} % -- (BAIXA PROBABILIDADE)'.format(fn, predicted, prob ))
        else:
            print('Imagem {0} é {1} - Prob: {2:.3f} %'.format(fn, predicted, prob ))
        
        
prediction_dir = BASE_DIR + 'prediction-files\\'
predict_images(prediction_dir)
