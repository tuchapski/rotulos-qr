# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.preprocessing import image


#::::::::::
#:::Model
#::::::::::

BASE_DIR = 'C:\\Users\\atuchapski\\Documents\\Projetos\\rotulos-qr\\'
batch_size = 8
num_classes = 2
img_width = 200
img_height = 200
model_name = 'keras_rotulos-qr_acc8125_custom.h5'


model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', 
                 input_shape = (img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

opt = tf.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics = ['accuracy'])

train_dir = BASE_DIR + 'dataset\\Training'
validation_dir = BASE_DIR + 'dataset\\Validation'

train_datagen = ImageDataGenerator(rescale = 1.0/255.,
                                   rotation_range = 30,
                                   width_shift_range = 0.1,
                                   height_shift_range = 0.1,
                                   shear_range = 0.1,
                                   zoom_range = 0.1,
                                   horizontal_flip = False)

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
                              epochs = 15,                              
                              verbose = 1)

train_labels = train_generator.class_indices

model.save_weights(BASE_DIR + model_name)
#model.save('label_antartica_5833.h5') #To save the entire model en load later with load_model...
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

plot_model_accuracy(history, show_loss=True)


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
            print('Imagem {0} é {1} - Prob {2:.3f} % -- Baixa Prob.'.format(fn, predicted, prob ))
        else:
            print('Imagem {0} é {1} - Prob: {2:.3f} %'.format(fn, predicted, prob ))
        
        
prediction_dir = BASE_DIR + 'prediction-files\\' 
predict_images(prediction_dir)


