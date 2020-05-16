#!/usr/bin/env python
# coding: utf-8

# In[22]:

import seaborn as sn
import pandas as pd
import tensorflow as tf
print(tf.__version__)
import numpy as np
import matplotlib.pyplot as plt
import tools
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os
import datetime

train_path = "trainset.npz"
test_path = "testset.npz"

X, y, ids = tools.load_data(train_path, one_hot=True)
X_test, y_test, ids_test = tools.load_data(test_path, one_hot=True)
ids = np.array(ids)
ids_test=np.array(ids_test)

ids_unique = list(set(ids))
print(ids_unique)
ids_train, ids_val = train_test_split(ids_unique, test_size=0.1)

X_train = []
X_val = []
y_train = []
y_val = []
ids_train_list = []
ids_val_list = []
for XX, yy, id in zip(X, y, ids):
    if str(id) in ids_train:
        X_train.append(XX)
        y_train.append(yy)
        ids_train_list.append(id)
    elif str(id) in ids_val:
        X_val.append(XX)
        y_val.append(yy)
        ids_val_list.append(id)
    else:
        #print(id in ids_test)
        #print(ids_train)
        raise Exception('id {} was not in ids_val or ids_train'.format(id))
        
X_train = np.array(X_train)
X_val = np.array(X_val)
y_train = np.array(y_train)
y_val = np.array(y_val)

#X_train, X_val, y_train, y_val, ids_val, ids_train = train_test_split(X, y, ids, test_size=0.1)

X_train = tf.convert_to_tensor(np.expand_dims(X_train,-1))
X_val = tf.convert_to_tensor(np.expand_dims(X_val,-1))
X_test = tf.convert_to_tensor(np.expand_dims(X_test,-1))
ids_train_list = np.array(ids_train_list)
ids_val_list = np.array(ids_val_list)
assert X_train.shape[1] == 13
assert X_train.shape[2] == 190
assert X_train.shape[3] == 1
assert X_val.shape[1] == 13
assert X_val.shape[2] == 190
assert X_val.shape[3] == 1
assert X_val.shape[1] == 13
assert X_val.shape[2] == 190
assert X_val.shape[3] == 1

assert set(ids_train_list).intersection(set(ids_val_list)) == set()
print("intersection {}".format(set(ids_train_list).intersection(set(ids_val_list))))
#print(list(set(ids_val_list)))


def run_experiment(model, job_dir, X_train, y_train, ids_train, X_val, y_val, ids_val, X_test, y_test,ids_test, hyperparams, class_names):
    BATCH_SIZE = hyperparams["batch_size"]
    EPOCHS = hyperparams["epochs"]
    y_true_list = []
    y_pred_list = []
    from pathlib import Path
    Path(job_dir).mkdir(parents=True, exist_ok=True)
    #os.mkdir(job_dir)
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    callbacks=[callback]
    
    history = model.fit(X_train, y_train, validation_data=(X_val,y_val), batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, callbacks=callbacks)
    print(history.history.keys())
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(job_dir+"/training_accs")
    plt.clf()
    print("X_test shape {}".format(X_test.shape))
    

    y_pred = model.predict(X_test)
    
    for test_id in list(set(ids_test)):
        #print(test_id)
        y_maj_vote = np.zeros(y_train[0].shape)
        
        #print(y_pred.shape)
        for y_pred_temp, y, id in zip(y_pred, y_test, ids_test):
            #X= np.expand_dims(X, 0)
            #print("X shape: {}".format(X.shape))
            #print("ypred temp {}".format(y_pred_temp))
            if id == test_id:
                #y_pred_temp = model.predict(X)
                y_maj_vote += tf.one_hot(tf.argmax(y_pred_temp, axis=0), y.shape[0])
                true_label = tf.argmax(y, axis=0)
        final_vote = tf.argmax(y_maj_vote, axis=0)
        
        y_pred_list.append(int(final_vote))
        y_true_list.append(int(true_label))
        #print(test_id, y_and_y_pred[test_id])
    #class_names = [str(i) for i in range(10)]
    cm = confusion_matrix(y_pred_list, y_true_list)
    df_cm = pd.DataFrame(cm, index = class_names,
                  columns = class_names)
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, cmap = "Blues")
    plt.xlabel("True labels")
    plt.ylabel("Predicted labels")
    plt.savefig(job_dir+"/conf_matrix")
    plt.clf()
    return y_pred_list, y_true_list

def get_default_model():
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
    return model
#model.fit(X_train, y_train, validation_data=(X_val,y_val), batch_size=32, epochs=10)
job_dir = "results/" + datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
example_model = get_default_model()
params = {"batch_size": 32, "epochs": 40}
class_names = [i for i in "ABCDEFGHIJ"]
result = run_experiment(example_model, job_dir, X_train, y_train, ids_train_list, X_val, y_val, ids_val_list, X_test, y_test, ids_test, params, class_names)
print(result)
#y_test_pred = model.predict(X_test)
#y_test_pred = tf.math.argmax(y_test_pred, axis=1)
#y_test=tf.math.argmax(y_test, axis=1)
#tf.metrics.accuracy(y_test_pred,y_test)
y_train.shape

