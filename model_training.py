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
GENRE_LIST = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

train_path = "trainset.npz"
test_path = "testset.npz"

def import_and_filter(train_path, test_path, genres_to_include):
    
    class_to_index_dict = {j:i for i,j in enumerate(genres_to_include)}
    index_to_class_dict = {i:j for i,j in enumerate(genres_to_include)}

    print(class_to_index_dict)
    
    X, y, ids = tools.load_data(train_path, one_hot=False, genres_to_include=genres_to_include)
    #print("y before{}".format(y))
    y= [class_to_index_dict[i] for i in y]
    #print("y after{}".format(y))
    y= tf.one_hot(y, max(y)+1)
    X_test, y_test, ids_test = tools.load_data(test_path, one_hot=False, genres_to_include=genres_to_include)
    #print("y before{}".format(y_test))
    
    y_test= [class_to_index_dict[i] for i in y_test]
    #print("y after{}".format(y_test))
    y_test= tf.one_hot(y_test, max(y_test)+1)
    
    
    ids = np.array(ids)
    ids_test=np.array(ids_test)

    ids_unique = list(set(ids))
    #print("ids unique {}".format(ids_unique))
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
    print("ytest returned: {}".format(len(y_test)))
    return X_train, y_train, ids_train_list, X_val, y_val, ids_val_list, X_test, y_test, ids_test, index_to_class_dict

def run_experiment(model, job_dir, X_train, y_train, ids_train, X_val, y_val, ids_val, X_test, y_test,ids_test, hyperparams, class_names):
    BATCH_SIZE = hyperparams["batch_size"]
    EPOCHS = hyperparams["epochs"]
    y_true_list = []
    y_pred_list = []
    
    #os.mkdir(job_dir)
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)
    callbacks=[callback]
    callbacks=[]
    #print("test_y {}".format(len()))

    history = model.fit(X_train, y_train, validation_data=(X_val,y_val), batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, callbacks=callbacks)
    print(history.history.keys())
    plt.figure()
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('model accuracy', size=18)
    plt.ylabel('accuracy', size=16)
    plt.xlabel('epoch', size=16)
    plt.legend(['train', 'val'], loc='upper left',prop={'size': 16})
    plt.savefig(job_dir+"_training_accs")
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
    plt.figure(figsize = (20,14))
    sn.set(font_scale=2)
    sn.heatmap(df_cm, annot=True, cmap = "Blues",annot_kws={"size": 42})
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlabel("True labels", size=32)
    plt.ylabel("Predicted labels", size=32)
    plt.savefig(job_dir+"_conf_matrix")
    plt.clf()
    return [1-x for x in history.history['categorical_accuracy']]

def get_default_model(nr_targets, use_dropout=False, dropout=0):
    model = tf.keras.models.Sequential()

    #model.add(tf.keras.layers.InputLayer(input_shape=(1,13,190)))
    NR_FILTERS_1 = 3
    NR_FILTERS_2 = 15
    NR_FILTERS_3 = 65

    KERNEL_SIZE_1 = (13,10)
    KERNEL_SIZE_2 = (1,10)
    KERNEL_SIZE_3 = (1,10)

    model.add(tf.keras.layers.Conv2D(NR_FILTERS_1, KERNEL_SIZE_1, strides=(1,4), padding = "valid", input_shape=(13,190,1)))
    model.add(tf.keras.layers.Activation("sigmoid"))
    model.add(tf.keras.layers.Conv2D(NR_FILTERS_2, KERNEL_SIZE_2, strides=(1,4), padding = "valid"))
    model.add(tf.keras.layers.Activation("sigmoid"))
    model.add(tf.keras.layers.Conv2D(NR_FILTERS_3, KERNEL_SIZE_3, strides=(1,4),padding = "valid"))
    model.add(tf.keras.layers.Activation("sigmoid"))
    model.add(tf.keras.layers.Flatten())
    
    if use_dropout:
        model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(nr_targets))
    model.add(tf.keras.layers.Softmax())

    loss = tf.keras.losses.CategoricalCrossentropy()

    model.compile(optimizer="SGD", loss=loss, metrics=[tf.keras.metrics.CategoricalAccuracy()])
    model.summary()
    return model

dropout_list = [0,0.2,0.5,0.8]
dropout_list = [0]
current_date_time = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
from pathlib import Path
Path("results/{}".format(current_date_time)).mkdir(parents=True, exist_ok=True)
for dropout in dropout_list:
    USE_DROPOUT = False

    
    genres_to_test = [ [1,5,9], [1,5,9,3], [1,5,9,3,7], [1,5,9,3,7,0]]
    genres_to_test = [list(range(10))]

    training_errors = {}
    for genres_to_include in genres_to_test:
    
        X_train, y_train, ids_train_list, X_val, y_val, ids_val_list, X_test, y_test, ids_test, index_to_class_dict = import_and_filter(train_path, test_path, genres_to_include)
        #print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)
        #print("y_test again {}".format(len(y_test)))
        #model.fit(X_train, y_train, validation_data=(X_val,y_val), batch_size=32, epochs=10)
        job_dir = "results/{}/{}_dropout_{}_genres".format(current_date_time,str(dropout)[-1], len(genres_to_include))
        example_model = get_default_model(nr_targets= len(genres_to_include), use_dropout=USE_DROPOUT)
        params = {"batch_size": 32, "epochs": 200}
        class_names = list(np.array(GENRE_LIST)[genres_to_include])
        print("xtrain shape",X_train.shape)
        result = run_experiment(example_model, job_dir, X_train, y_train, ids_train_list, X_val, y_val, ids_val_list, X_test, y_test, ids_test, params, class_names)
        training_errors[len(genres_to_include)] = result
    plt.figure()
    for key, vals in training_errors.items():
        plt.plot(vals)
        
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('model accuracy', size=18)
    plt.ylabel('error rate', size=16)
    plt.xlabel('epoch', size=16)

        
        
    
    plt.legend(["nr genres: {}".format(len(k)) for k in genres_to_test], prop={'size': 16})
    plt.title("training error", size=18)
    plt.savefig("results/{}/{}_dropout_training_errors".format(current_date_time,str(dropout)[-1]))
    plt.clf()
