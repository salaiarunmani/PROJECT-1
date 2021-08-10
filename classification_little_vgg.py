from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization #
from keras.layers import Conv2D,MaxPooling2D #convert2d and maxpooling
import os

num_classes = 5 #5 emotion
img_rows,img_cols = 48,48 
batch_size = 8 # how many models train at a time since my RAM is low i go with 8 no

train_data_dir = r'H:\6th sem\project\images\train' #r for forward slash python not accept /
validation_data_dir = r'H:\6th sem\project\images\validation' 

train_datagen = ImageDataGenerator(
					rescale=1./255,
					rotation_range=30,
					shear_range=0.3,
					zoom_range=0.3,
					width_shift_range=0.4,
					height_shift_range=0.4,
					horizontal_flip=True,
					fill_mode='nearest')
                    #5 to 6 image will create from 1 image,hor_flip=mirror photo

validation_datagen = ImageDataGenerator(rescale=1./255) #it is used for crosscheck the image

train_generator = train_datagen.flow_from_directory(
					train_data_dir,
					color_mode='grayscale',
					target_size=(img_rows,img_cols),
					batch_size=batch_size,
					class_mode='categorical',
					shuffle=True)
                    #cate=happy,sad,...

validation_generator = validation_datagen.flow_from_directory(
							validation_data_dir,
							color_mode='grayscale',
							target_size=(img_rows,img_cols),
							batch_size=batch_size,
							class_mode='categorical',
							shuffle=True)
                            #target color mode=greyscale image


model = Sequential() #simple model

# Block-1  ( 3,3 size taken and  stread = 2,2 padding means = how many pixels taking 

model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))#1 for grey scale ,padding amount of pixel added to an image when (32 neurons)
model.add(Activation('elu'))# x x>e0 and alpha(e^x-1) x<0
model.add(BatchNormalization()) #nerual network
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))#cross the thersold value o/p then only it go for output
model.add(BatchNormalization())#it is used to speed up the training neral network(performance and stubilty)
model.add(MaxPooling2D(pool_size=(2,2)))#2x2
model.add(Dropout(0.2))

# Block-2 

model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))#64 neurons
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-3

model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-4 

model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-5

model.add(Flatten())
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block-6

model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu')) #elu for setting the threshold value
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block-7

model.add(Dense(num_classes,kernel_initializer='he_normal'))
model.add(Activation('softmax')) #used to get the output from 5 sample/cat

print(model.summary())

from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
#checkpoint is pick the model in best accuracy
checkpoint = ModelCheckpoint('Emotion_little_vgg.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)#if model accuracy is not improving , vgg.h5 is generated one

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True
                          )
                          #used to stop the traning if accuracy is not best one

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001) #reduce learning rate slow down the learning .

callbacks = [earlystop,checkpoint,reduce_lr] #callback lst

model.compile(loss='categorical_crossentropy',
              optimizer = Adam(lr=0.001),
              metrics=['accuracy']) #compiler part , adam is optimizer

nb_train_samples = 24176 #no of training samples 7000*5
nb_validation_samples = 3006 #no of validation samples
epochs=25  # 25 times it train the program
#fit for traning purpose
history=model.fit_generator( 
                train_generator,
                steps_per_epoch=nb_train_samples//batch_size,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=validation_generator,
                validation_steps=nb_validation_samples//batch_size) #fitgen is start to generate the training part
