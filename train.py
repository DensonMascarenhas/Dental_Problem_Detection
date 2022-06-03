from keras.layers import Input, Lambda, Dense, Flatten,Dropout
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt


train_path="Img/Training"
test_path="Img/Testing"
val_path="Img/Validation"

x_train=[]

for folder in os.listdir(train_path):

    sub_path=train_path+"/"+folder

    for img in os.listdir(sub_path):

        image_path=sub_path+"/"+img

        img_arr=cv2.imread(image_path)

        img_arr=cv2.resize(img_arr,(224,224))

        x_train.append(img_arr)

x_test=[]

for folder in os.listdir(test_path):

    sub_path=test_path+"/"+folder

    for img in os.listdir(sub_path):

        image_path=sub_path+"/"+img

        img_arr=cv2.imread(image_path)

        img_arr=cv2.resize(img_arr,(224,224))

        x_test.append(img_arr)

x_val=[]

for folder in os.listdir(val_path):

    sub_path=val_path+"/"+folder

    for img in os.listdir(sub_path):

        image_path=sub_path+"/"+img

        img_arr=cv2.imread(image_path)

        img_arr=cv2.resize(img_arr,(224, 224))

        x_val.append(img_arr)

# x_train,x_test, and x_val must be divided by 255.0 for normalization.
train_x=np.array(x_train)
test_x=np.array(x_test)
val_x=np.array(x_val)

train_x=train_x/255.0
test_x=test_x/255.0
val_x=val_x/255.0

# train_datagen = ImageDataGenerator(rescale = 1./255)
# test_datagen = ImageDataGenerator(rescale = 1./255)
# val_datagen = ImageDataGenerator(rescale = 1./255)

train_datagen=ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    brightness_range=(0.9, 1.1),
    fill_mode='nearest')


test_datagen=ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    brightness_range=(0.9, 1.1),
    fill_mode='nearest')


val_datagen=ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    brightness_range=(0.9, 1.1),
    fill_mode='nearest')




# label generation
training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                save_prefix='aug',
                                                save_format='jpg',
                                                 class_mode = 'sparse')
test_set = test_datagen.flow_from_directory(test_path,
                                            save_prefix='aug',
                                            save_format='jpg',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'sparse')
val_set = val_datagen.flow_from_directory(val_path,
                                            save_prefix='aug',
                                            save_format='jpg',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'sparse')

train_y=training_set.classes
test_y=test_set.classes
val_y=val_set.classes


training_set.class_indices

train_y.shape,test_y.shape,val_y.shape

# Model training

vgg = VGG19(input_shape=[224,224]+ [3], weights='imagenet', include_top=False,pooling='max')

#do not train the pre-trained layers of VGG-19
for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)


#adding output layer.Softmax classifier is used as it is multi-class classification
prediction = Dense(5, activation='softmax')(x)

model = Model(inputs=vgg.input, outputs=prediction)
# view the structure of the model
model.summary()

# compiling
model.compile(
  loss='sparse_categorical_crossentropy',
  optimizer="adam",
  metrics=['accuracy']
)

from keras.callbacks import EarlyStopping
early_stop=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=5)
#Early stopping to avoid overfitting of model

# fit the model
history = model.fit(
  train_x,
  train_y,
  validation_data=(val_x,val_y),
  epochs=50,
  callbacks=[early_stop],
  batch_size=32,shuffle=True)

# accuracies
plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(history.history['accuracy'], label='train acc')

plt.plot(history.history['val_accuracy'], label='val acc')

plt.legend()

plt.savefig('vgg-acc-rps-1.png')


plt.subplot(2, 2, 2)
# loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.savefig('vgg-loss-rps-1.png')


# evaluation
model.evaluate(test_x,test_y,batch_size=32)
model.save("main_model.h5")

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import numpy as np

#predict
y_pred=model.predict(test_x)
y_pred=np.argmax(y_pred,axis=1)

#get classification report
print(classification_report(y_pred,test_y))

# confusion matrix
print(confusion_matrix(y_pred,test_y))

plt.show()

