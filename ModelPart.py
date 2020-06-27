#!/usr/bin/env python
# coding: utf-8

# In[1]:
get_ipython().system('pip install langid')


# In[2]:
import numpy as np #used  for data manipulation
import pandas as pd
from sklearn import utils as ut
import json          #used to save the model
import keras         #importing Tenorflow's Keras API and preprocessing tools
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
import nltk          # impoting NLTK to use for stopwords removal
nltk.download('stopwords')
import tensorflow as tf
from keras import regularizers
nltk.download('punkt')
from langid import classify #used for language filtering
from textblob import TextBlob #used for spell corrections and sentiment analysis


# In[3]:
#reading csv data. We skip certain lines because the data file was 
#corrupted and we could not properly import
data = pd.read_csv('trained-balanced-sarcasm.csv',skiprows= [997045,112610,291851]) #I reduced the dataset as this file is too large to upload to github, so the estimation rate is decreasing.
print("Dataset columns:")
#print(list(dataset.columns.values))
print("Initial dataframe shape: ",data.shape)
data = data[['label','comment']]

print("Dataframe shape for eperiment: ",data.shape)




# In[4]:
#Language filtering
#Applying pandas  lambda function to create a new dataframe column with language tags
data['langid']= data['comment'].dropna().apply(lambda x: classify(x)[0])


#dropping comments
data = data[data['langid'] == 'en']



#Sentiment analysis 
data['sentiment'] = data['comment'].apply(lambda x: TextBlob(x).sentiment)
#splittin sentiment analysis tuple into separate columns
data['polarity'] = data['sentiment'].apply(lambda x: x[0] )
data['subjectivity'] = data['sentiment'].apply(lambda x: x[1] )




dataset = data[['label','comment','polarity','subjectivity']]

dataset.to_csv("clean_set.csv", sep=',', encoding='utf-8')


# In[5]:
from nltk.stem.api import StemmerI
import nltk
# Simple stemming function for sentences 
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import LancasterStemmer
def stemSentence(sentence):
    
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(lancaster.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)


# In[6]:
#This is just loading code for the clean csv file
#in case you have already processed it
data = pd.read_csv('clean_set.csv',skiprows= [12693,24722,997045,112610,291851]) 
print(data.shape)
print(list(data.columns.values))

data = data[['label','comment']]

# In[7]:
from nltk.corpus import stopwords
from sklearn.utils import shuffle
import string
from nltk.stem import LancasterStemmer

stop = stopwords.words('english')  #downloading stopwords


#staring preprocessin
import re
#removing punctuation and numbers from texts
#data['comment'] = data['comment'].str.translate(None, string.punctuation)

data['comment'] =  data['comment'].str.replace('[^\w\s]','')
data['comment'] =  data['comment'].str.replace('[0-9]+','')
#data['comment'] = data['comment'].apply(lambda x: TextBlob(x).correct())
#creating a sentiment analysis feature to use while training and validation

lancaster=LancasterStemmer()
data['comment'] = data['comment'].dropna().apply(lambda x: stemSentence(x))



#splitting comments and changing uppercase to lowercase 
data['comment'] = data['comment'].str.lower().str.split()
#removing stopwords
data['comment'] = data['comment'].dropna().apply(lambda x: [item for item in x if item not in stop])


#creating dataset and shuffling containts using sklearn
dataset = data[['label','comment']]
dataset = shuffle(dataset)

#printing dataset to visualize he final shape of the data
print(dataset.shape)

# In[8]:
dataset_x = dataset[['comment']]
#print(dataset_x)

vocab_size = 20000 # number of words in our vocabulary 

#encoding texts using one hot encoding
encoded_docs = [one_hot(str(d), vocab_size) for d in dataset_x['comment'] ]

max_length = 20

#padding texts
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

#reseting dataset as the padded versionof itself
dataset_x = padded_docs
#print(dataset_x)

# In[9]:
dataset_x_final = pd.DataFrame(dataset_x)
dataset_y_final = pd.DataFrame(dataset['label'])

print(dataset_x_final.shape)
train_x = dataset_x_final.head(800000)

test_x = dataset_x_final.tail(100000)

train_y = dataset_y_final.head(800000)

test_y = dataset_y_final.tail(100000)


# In[10]:
#the network's architecture 
#importing layers 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D,MaxPooling1D, Flatten
from keras.layers import LSTM , Embedding
from keras.regularizers import l2

#Bulding the model architecure using the Sequentia API
model = Sequential()
model.add(Embedding(50000, 128, input_length=20))
model.add(Dropout(0.5))
model.add(Conv1D(264, 5, activation='relu'))#,bias_initializer='glorot_uniform',kernel_initializer='glorot_uniform'))
model.add(MaxPooling1D(pool_size=3))
model.add((Dropout(0.25)))
model.add(Conv1D(128, 5, activation='relu'))#,bias_initializer='glorot_uniform',kernel_initializer='glorot_uniform'))
model.add(MaxPooling1D(pool_size=1))
model.add((Dropout(0.25)))
model.add(Conv1D(32, 1 ,activation='relu'))#,bias_initializer='glorot_uniform',kernel_initializer='glorot_uniform'))
model.add(MaxPooling1D(pool_size=1)) #global max pooling layer
model.add((Dropout(0.25)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1, activation='tanh'))


# In[11]:
print(model.summary())


# In[12]:
#this is a custom metric used while training
#in order to use the metric you have
#to add this callback to the fitting function

from sklearn.metrics import f1_score

class Metrics(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs={}):
        predict = np.asarray(self.model.predict(self.validation_data[0]))
        targ = self.validation_data[1]
        self.f1s=f1_score(targ, predict)
        return
      
metrics = Metrics()


# In[13]:
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
opt = keras.optimizers.Adam(lr=0.001)

earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=1, verbose=1, mode='auto')
callbacks_list = [earlystop]

model.compile(loss='mean_squared_logarithmic_error',
  optimizer=opt,
  metrics=['accuracy'])

history = model.fit(train_x, train_y,
  batch_size=128,
  epochs=10,
  verbose=1,
  #callbacks=callbacks_list,
  validation_split=0.2,
  shuffle=True)

import pickle

# Save the Modle to file in the current working directory

Pkl_Filename = "irony.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model, file)



# In[14]:
model.evaluate(x=test_x, y=test_y, batch_size=32, verbose=1, sample_weight=None, steps=None)






