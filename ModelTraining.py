#!/usr/bin/env python
# coding: utf-8

# In[3]:


from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import load_model
import os
import pprint
import smtplib
import tensorflow as tf
from tensorflow.python import keras
from keras.callbacks import *
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers


# In[4]:


print(keras.__version__)


# In[4]:


train_dir = '/home/pa-logi/Desktop/Data (copy)/Training'
validation_dir = '/home/pa-logi/Desktop/Data (copy)/Validation'
img_height, img_width = 640, 390
EMAIL_ADDRESS = "<YOUREMAIL>" #send Statusinformation after every Epoch to your Email Adress
PASSWORD = "<YOURPASSWORD>"
da = np.array(['Beauty and Fashion', 'Cooking', 'Gaming',  'Comedy', 'Music' ])
for d in da:
  print(len(os.listdir("/home/pa-logi/Desktop/Data (copy)/Training/{}".format(d))))
  print(len(os.listdir("/home/pa-logi/Desktop/Data (copy)/Validation/{}".format(d))))


# In[ ]:


filepath = "/home/pa-logi/Desktop/Data (copy)/Epoch/epochs:{epoch:03d}-val_acc:{val_acc:.3f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=False, mode='max')
callbacks_list = [checkpoint]


# This Code to register a callback handler that will send a notification Email after each Epoch with the Accuracy and the Epoch name 

# In[ ]:


class MyCustomCallback(ModelCheckpoint):
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        send_email(subject, self.filepath.format(epoch=epoch +1, **logs))
        
    def _init_(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(MyCustomCallback, self)._init_(filepath, monitor, verbose,
                 save_best_only, save_weights_only,
                 mode, period)

def send_email(subject, msg):
    try:
        server = smtplib.SMTP('smtp.gmail.com:587')
        server.ehlo()
        server.starttls()
        server.login(EMAIL_ADDRESS, PASSWORD)
        message = 'Subject: {}\n\n{}'.format(subject, msg)
        server.sendmail(EMAIL_ADDRESS, EMAIL_ADDRESS, message)
        server.quit()
        print("Success: Email sent!")
    except:
        print("Email failed to send.")
subject = "Update - VGG16 5 classes"
msg1 = "Done. "


# In[5]:


#Load the VGG model
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width,3))

# Freeze all the layers
for layer in vgg_conv.layers[:]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)

# Create the model
model = models.Sequential()

# Add the vgg convolutional base model
model.add(vgg_conv)

# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(5, activation='softmax'))
# Show a summary of the model. Check the number of trainable parameters
model.summary()


# In[6]:


train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Choose the batchsize suitable to the system RAM
train_batchsize = 50
val_batchsize = 10

# Data Generator for Training data
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=train_batchsize,
        class_mode='categorical')

# Data Generator for Validation data
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(img_height, img_width),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=1e-4),
              metrics=['acc']) 

# Train the Model
# Uncomment the next line in case you want to start from a specific epoch (it is useful when the system fail during the Epochs generation)
#model = load_model('/home/pa-logi/Desktop/Epoch Run 1 ohne Data Augmentation/epochs:016-val_acc:0.912.h5')
history = model.fit_generator(
      train_generator,
      steps_per_epoch=train_generator.samples/train_generator.batch_size ,
      epochs =10,
      initial_epoch=0,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size,
      verbose=1,
      callbacks=[MyCustomCallback(filepath, monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=False, mode='max')])


# Save the Model
model.save('all_freezed.h5')

# Plot the accuracy and loss curves
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# In[7]:


# Create a generator for prediction
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(img_height, img_width),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)

# Get the filenames from the generator
fnames = validation_generator.filenames

# Get the ground truth from generator
ground_truth = validation_generator.classes

# Get the label to class mapping from the generator
label2index = validation_generator.class_indices

# Getting the mapping from class index to class label
idx2label = dict((v,k) for k,v in label2index.items())

# Get the predictions from the model using the generator
predictions = model.predict_generator(validation_generator, steps=validation_generator.samples/validation_generator.batch_size,verbose=1)
predicted_classes = np.argmax(predictions,axis=1)

errors = np.where(predicted_classes != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors),validation_generator.samples))
import seaborn as sns
from sklearn.metrics import confusion_matrix


mat = confusion_matrix(ground_truth, predicted_classes )
sns.heatmap(mat, annot=True, fmt='d', cmap="coolwarm_r",xticklabels = label2index,yticklabels = label2index,  linewidth = 0.5)
plt.title("confusion matrix")
plt.xlabel('Predicted')
plt.ylabel('Truth')
b, t = plt.ylim() 
b += 0.5 
t -= 0.5 
plt.ylim(b, t)
plt.show()


# Show the errors
for i in range(len(errors)):
    pred_class = np.argmax(predictions[errors[i]])
    pred_label = idx2label[pred_class]
    
    title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
        fnames[errors[i]].split('/')[0],
        pred_label,
        predictions[errors[i]][pred_class])
    
    original = load_img('{}/{}'.format(validation_dir,fnames[errors[i]]))
    plt.figure(figsize=[7,7])
    plt.axis('off')
    plt.title(title)
    plt.imshow(original)
    plt.show()

