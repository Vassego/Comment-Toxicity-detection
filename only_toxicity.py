import os
import pandas as pd
import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import TextVectorization  
from keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding
from keras.metrics import Precision, Recall, BinaryAccuracy


#Loading data

print(os.path.join('data', 'train.csv'))
df = pd.read_csv(os.path.join('data', 'train.csv'))


# Preprocessing

X = df['comment_text'] 
y = df[df.columns[2]].values
MAX_FEATURES = 200000 
vectorizer = TextVectorization(max_tokens=MAX_FEATURES, output_sequence_length=1800, output_mode='int')
vectorizer.adapt(X.values)

vectorized_text = vectorizer(X.values)
print(vectorized_text.shape)
print(len(X))
dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
dataset = dataset.cache()
dataset = dataset.shuffle(160000)
dataset = dataset.batch(16)
dataset = dataset.prefetch(tf.data.AUTOTUNE)  

batch_X, batch_y = dataset.as_numpy_iterator().next()  
print("Inputs: ", batch_X, "Outputs: ", batch_y)
print("input Shape: ", batch_X.shape, "Output Shape: ", batch_y.shape)

print("Total no.of batches = ", len(dataset))
train = dataset.take(int(len(dataset)*.7))
val = dataset.skip(int(len(dataset)*.7)).take(int(len(dataset)*.2))
test = dataset.skip(int(len(dataset)*.9)).take(int(len(dataset)*.1))

print("no.of training batches = ", len(train))
print("no.of val batches = ", len(val))
print("no.of test batches = ", len(test))

# Creating the deep learning model

model = Sequential()
model.add(Embedding(MAX_FEATURES+1, 32, input_length=1800))  
model.add(Bidirectional(LSTM(32, activation='tanh')))
model.add(Dense(128, activation='relu')) 
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile('adam', loss=tf.losses.BinaryCrossentropy(),metrics=['accuracy'])
model.summary()
history = model.fit(train, epochs=10, validation_data=val)
model.save(os.path.join('models', 'Toxicity_final.h5'))

print(history.history)

fig = plt.figure()
plt.plot(history.history['loss'], color='teal', label='loss')
plt.plot(history.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()
fig = plt.figure()
plt.plot(history.history['accuracy'], color='teal', label='Accuracy')
plt.plot(history.history['val_accuracy'], color='orange', label='Val_Accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

#Evaluating the model

pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator():
    X_true, y_true = batch
    yhat = model.predict(X_true)
    y_true = y_true.flatten()
    yhat = yhat.flatten()
    pre.update_state(y_true, yhat)
    re.update_state(y_true, yhat)
    acc.update_state(y_true, yhat)
    
    
print("Precision = {} \t Recall = {} \t Accuracy = {}".format(pre.result().numpy(), re.result().numpy(), acc.result().numpy()))
