import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import itertools
import shutil
import random
import glob
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import time
import sys
import multiprocessing as mp
from multiprocessing import dummy as multiprocessing
import logging
import PIL
import matplotlib.pyplot as plt
import numpy as np
import socket
import inception_utils
import inception
import h5py
from collections import Counter
from sklearn.metrics import confusion_matrix
import pickle
slim = tf.contrib.slim
CURRENT_PATH=os.getcwd()

# Checkpoint path of InceptionV3 model
CHECKPOINT = os.path.join(CURRENT_PATH,'inception_v3.ckpt')
# Checkpoints path of InceptionV3 model saved after removing least responsible neurons. 
#CHECKPOINT = os.path.join(CURRENT_PATH,'inception_checkpoints','basenji.ckpt')
def set_checkpoint(name):
     CHECKPOINT = os.path.join(CURRENT_PATH,'inception_checkpoints',name+'.ckpt')
        
            
   

# Scope of model
model_scope = 'InceptionV3'
model = inception.inpcetion_instance(checkpoint=CHECKPOINT)
model_variables = tf.global_variables(scope=model_scope)
convs = ['/'.join(k.name.split('/')[:-1]) for k in model_variables if 'weights'
     in k.name and 'Aux' not in k.name and 'Logits' not in k.name]
layer_dic = {conv: [var for var in model_variables if conv in var.name]
             for conv in convs}


# Funtion to get top n number of most or least responsible neurons with their shapley value
# Input: Number of responsible neurons, experiment name, activity name, is required most responsible or least responsible neurons
# Output: shapley values,responsible neurons
def top_n_neurons(n,experiment,activity,isMostImp=True):
    val=np.array(open(os.path.join(cb_dir1,experiment,activity,'vals.txt')).read().split(',')).astype(float)
    chosen_players=np.array(open(os.path.join(cb_dir1,experiment,activity,'chosen_players.txt')).read().split(',')).astype(int)
#     l=[(v,i) for i,v in zip(chosen_players[val>0],val[val>0])]
    l=[(v,i) for i,v in zip(chosen_players[val!=0],val[val!=0])]
    if isMostImp:
        l=sorted(l,reverse=True)#reverse=True
    else:
        l=sorted(l)
    shapley_val=[]
    player=[]
    for data in l[:n]:
        shapley_val.append(data[0])
        player.append(players[data[1]])
    print('#'*10,'shapley_val: ',shapley_val)
#     print('#'*10,'val: ',len(val))
    print('#'*10,'players(Counts:{}): {}'.format(len(player),player))
    print('*'*100)
    return shapley_val,player
    
# function to get most probability value with classifying label
# Input: List of predicted values with corresponding labels
# Output: label of class having most probability, probability
def show_prob(prediction):
    probs={}
    for i, prob in enumerate(prediction):
            probs[prob]=i
    prob=prediction;
    prob.sort()
#     print('#'*50)
#     for i in range(1,5):
#         print('{}:{}\n'.format(model.id_to_label(probs[prob[-1*i]]),prob[-1*i]))
    return model.id_to_label(probs[prob[-1]]),prob[-1]


# Function to get accuracy of model for a specific class of image.
# Input: Model, input image folder, class of image, is accuracy required from deafult checkpoint or saved modified checkpoints
# Output:  Accuracy in %
def get_accuracy(model,img_folder,class_label,from_checkpoit=False):
    images=os.path.join(CURRENT_PATH,img_folder,class_label)
    files=os.listdir(images)
    imgs = []
    for file in files:
         imgs.append(os.path.join(images,file))
    imgs=inception_utils.load_images_from_files(imgs)
    if(from_checkpoit):
        predictions=get_predictions(model,imgs)
    else:
        predictions=model.get_predictions(imgs)
#     print('model.run_imgs:{}'.format(model.run_imgs(imgs[:2],'mixed_10')))
    # print(predictions[0])
    result={}
    counts=0
    for img in predictions:
        label,accuracy=show_prob(img)
        result[label]=accuracy
        if label==class_label:
            counts+=1
    
    accuracy=(counts/len(files))*100
    print('Image class:{},Accuracy:{}'.format(class_label,accuracy))
    print('#'*100)
    return round(accuracy,2)



# Function to remove selected filter from InceptionV3 network
# Input: inceptionV3 Model instance, filters to be removed
def remove_players(model, players):
    '''Remove selected players (filters) in the Inception-v3 network.'''
    if isinstance(players, str):
        players = [players]
    for player in players:
        variables = layer_dic['_'.join(player.split('_')[:-1])]
        var_vals = model.sess.run(variables)
#         print('variables:{},var_vals:{}'.format(variables,var_vals))
        for var, var_val in zip(variables, var_vals):
            if 'variance' in var.name:
                var_val[..., int(player.split('_')[-1])] = 1.
#                 print("#"*12,var_val)
            elif 'beta' in var.name:
                pass
            else:
                var_val[..., int(player.split('_')[-1])] = 0.
            var.load(var_val, model.sess)
            
# Function to save the checkpoint of model 
# Input: name of checkpoint, InceptionV3 model instance
def save_model(name,model):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(tf.global_variables(scope='InceptionV3'))
    new_name=name+'.ckpt'
    saver.save(model.sess,  os.path.join(os.getcwd(),'inception_checkpoints',new_name)) 

        
# Get prediction from saved model checkpoint
# Input: InceptionV3 model instance, input images dataset
# Output: model Predicted values for each input image
def get_predictions(model, imgs):
    return  model.sess.run(model.ends['prediction'], {model.ends['input']: imgs})

# Model to save checkpoint after removing least responsible neurons.
# Input: Number of neurons to be removed, name of checkpoint of model to be saved
def save_model_after_removing_players(players_to_remove,name):
    shapley_val,player=top_n_neurons(players_to_remove,experiment,activity,isMostImp=False)
    remove_players(model,player)
    print("*"*10,'checkpoint has been saved','*'*10)
    save_model(class_label,model)
    set_checkpoint(name)
    print('####### ACCURACY: ',get_accuracy(model,'imagenet',class_label,True))
    
    

                         
                     
# Name of input image folder                     
img_folder='imagenet'
# Name of class of image
class_label='hartebeest'
# Name of experiment
experiment='accuracy_hartebeest_new'
# Name of activty
activity='cb_Bernstein_0.2_25'
# Path of class result saved during neuron shapley algorithm
cb_dir1=os.path.join(CURRENT_PATH,'results/NShap/inceptionv3')
# Path for file players.txt having Players name in it
player_path=os.path.join(CURRENT_PATH,cb_dir1,experiment,'players.txt')
# List having all filter in InceptionV3 model
players=np.array(open(player_path).read().split(',')).astype(str)    
t=time.clock()

# print('#############model_accuracy:{} ,model_sample_accuracy: {}'.format(model.accuracy,model.sample_accuracy))
# print('###############: Accuracy: ',get_accuracy(model,'imagenet',class_label))
# print('Time taken: ',time.clock()-t)
x=[]
y=[]
for i in range(0,21,15):
            x.append(i)
            shapley_val,player=top_n_neurons(i,experiment,activity,isMostImp=True)
            remove_players(model,player)
            y.append(get_accuracy(model,'imagenet',class_label))
            
print('Number of Removal Players: ',x)
print('Accuracy: ',y)
# save_model_after_removing_players(400,class_label)
# print("############Over all acccuracy: ",get_overall_accuracy(model))

