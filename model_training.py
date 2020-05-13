#!/usr/bin/env python
# coding: utf-8

# In[22]:


import tensorflow as tf
print(tf.__version__)
import numpy as np
import matplotlib.pyplot as plt
import tools
from sklearn.model_selection import train_test_split


# In[98]:


train_path = "trainset.npz"
test_path = "testset.npz"

X, y = tools.load_data(train_path, one_hot=True)
X_test, y_test = tools.load_data(test_path, one_hot=True)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

X_train = tf.convert_to_tensor(np.expand_dims(X_train,-1))
X_val = tf.convert_to_tensor(np.expand_dims(X_val,-1))
X_test = tf.convert_to_tensor(np.expand_dims(X_test,-1))
assert X_train.shape[1] == 13
assert X_train.shape[2] == 190
assert X_train.shape[3] == 1
assert X_val.shape[1] == 13
assert X_val.shape[2] == 190
assert X_val.shape[3] == 1
assert X_val.shape[1] == 13
assert X_val.shape[2] == 190
assert X_val.shape[3] == 1


# In[97]:



model = tf.keras.models.Sequential()

#model.add(tf.keras.layers.InputLayer(input_shape=(1,13,190)))
NR_FILTERS_1 = 3
NR_FILTERS_2 = 15
NR_FILTERS_3 = 65

KERNEL_SIZE_1 = (46,1)
KERNEL_SIZE_2 = (10,1)
KERNEL_SIZE_3 = (1,1)

model.add(tf.keras.layers.Conv2D(NR_FILTERS_1, KERNEL_SIZE_1, strides=(1,4), padding = "same", input_shape=(13,190,1)))
model.add(tf.keras.layers.Conv2D(NR_FILTERS_2, KERNEL_SIZE_2, strides=(1,4), padding = "same"))
model.add(tf.keras.layers.Conv2D(NR_FILTERS_3, KERNEL_SIZE_3, strides=(1,4),padding = "same"))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Softmax())

loss = tf.keras.losses.CategoricalCrossentropy()

model.compile(optimizer="adam", loss=loss, metrics=[tf.keras.metrics.CategoricalAccuracy()])
model.summary()
model.fit(X_train, y_train, validation_data=(X_val,y_val), batch_size=32, epochs=10)

#y_test_pred = model.predict(X_test)
#y_test_pred = tf.math.argmax(y_test_pred, axis=1)
#y_test=tf.math.argmax(y_test, axis=1)
#tf.metrics.accuracy(y_test_pred,y_test)
y_train.shape

