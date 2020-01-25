#!/usr/bin/env python
# coding: utf-8

# Libraries

# In[12]:


from __future__ import print_function
import cv2
import time
import glob
import errno
import keras
import numpy as np
import os
import pprint
import smtplib
import tensorflow as tf
import keras.callbacks
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, save_img
from keras.models import load_model
from keras import models
from keras import layers
from keras import optimizers
from keras.applications import VGG16
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Paths and Models

# In[13]:


path = "/home/pa-logi/Desktop/grinch/Videos/" #Videopath
pathout = "/home/pa-logi/Desktop/grinch/Picture/Cooking/" #captured Framepath
errorpath = "/home/pa-logi/Desktop/grinch/errorPicture"
model = load_model('/home/pa-logi/Desktop/all_freezed.h5')
# the same you used to create the model before training
img_height, img_width = 640, 390
val_batchsize = 10


# Framing Code

# In[14]:


def framing(path, pathout):
    sep = "/"
    i = 1
    cap = 1
    n = 1
    frames = 2500
    kat = "Test"
    counter = 1
    files = glob.glob(path + "*") 
    for name in files:
        print(name)
        try:
            with open(name) as f: 
                string = str(name)
                print(string) 
                string = string.split(sep) 
                print(string[-1])
                kat = string[-3] 
                print(string[-3]) 
                print("______________________")
                cap = cv2.VideoCapture(path + string[-1]) 
                frames_per_second = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                every_sec = total_frames//frames_per_second
                frames = total_frames//every_sec
                counter = 1
                while(cap.isOpened()):
                    ret, frame = cap.read()
                    if ret == False:
                        break
                    if i%frames == 0:
                        cv2.imwrite(pathout+'Foto'+"_"+ str(n) + "_" + str(counter) + "_" + string[-1] + "_" + str(kat) + '.jpg',frame)
                        counter += 1
                    i+=1
                n+=1
                cap.release()
        except IOError as exc:
            if exc.errno != errno.EISDIR:
                raise
    


# Validation Code

# In[15]:


#Test the Frames against the Model 
def val(pathout, model):
    validation_dir = pathout
    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(img_height, img_width),
            batch_size=val_batchsize,
            class_mode='categorical',
            shuffle=False)

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
    
    return predictions, ground_truth, idx2label, predicted_classes, errors, label2index, fnames, validation_dir


# In[ ]:


Errorplot Code
1. Heatmap
2. Print missclassified Pictures 


# In[16]:


def errorplot(predictions, ground_truth, idx2label, predicted_classes, errors, label2index, fnames, validation_dir, errorpath):
    mat = confusion_matrix(ground_truth, predicted_classes )
    sns.heatmap(mat, annot=True, fmt='d',cbar=False, cmap="coolwarm_r",xticklabels = label2index,yticklabels = label2index,  linewidth = 1)
    plt.title("confusion matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    b, t = plt.ylim()
    b += 1 
    t -= 1
    plt.ylim(b, t)
    plt.savefig(errorpath + '/heatmap.pdf')
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
        save_img('{}/{}'.format(errorpath,fnames[errors[i]]), original)
        plt.figure(figsize=[7,7])
        plt.axis('off')
        plt.title(title)
        plt.imshow(original)
        plt.show()


# Plot Pictures Probability over time

# In[17]:


def summaryplot(predictions, idx2label, errorpath):
    summary = np.sum(predictions, axis = 1)
    for i in range(len(idx2label)):
        plt.plot(predictions[:,i], label=idx2label[i])
    plt.plot(summary, label="sum")
    plt.ylabel('Probability')
    plt.xlabel('#Frame')
    plt.legend()
    plt.savefig(errorpath + '/the_best_plot.pdf')
    plt.show()
    


# cleans the Path before we create new frames 

# In[18]:


def cleanoutpath(pathout):
    files = glob.glob(pathout+"*")
    for f in files:
        os.remove(f)


# In[19]:


def executeall(path, pathout, model, errorpath):
    cleanoutpath(pathout)
    framing(path, pathout)
    predictions, ground_truth, idx2label, predicted_classes, errors, label2index, fnames, validation_dir = val("/home/pa-logi/Desktop/grinch/Picture",model)
    errorplot(predictions, ground_truth, idx2label, predicted_classes, errors, label2index, fnames, validation_dir, errorpath)
    summaryplot(predictions, idx2label, errorpath)
    print(os.listdir(path))


# In[20]:


executeall(path, pathout, model, errorpath)


# In[ ]:




