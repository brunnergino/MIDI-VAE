from settings import *
from keras.models import Sequential
from keras import regularizers
from keras.layers import Input, RepeatVector
from keras.models import Model
from keras.layers.recurrent import LSTM, GRU
from keras.layers import TimeDistributed
from keras.layers import Dense, Activation
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop, Adam
from keras.utils import to_categorical
from keras.layers.wrappers import Bidirectional
from random import shuffle
import progressbar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import _pickle as pickle
import time
import data_class

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import pretty_midi as pm
import sys

from matplotlib2tikz import save as tikz_save

from import_midi import import_midi_from_folder


model_path = 'models/velocityclustering/'
model_filetype = '.pickle'

input_dim = 1
verbose = False
show_plot = False
save_plot = True
lstm_size = 256
batch_size = 512
learning_rate = 0.00002 #1e-06
step_size = 1
save_step = 10
shuffle_train_set = True
bidirectional = False
embedding = False
optimizer = 'Adam'
activity_regularizer = None
reset_states = True
num_layers = 2
test_step = 1

scale_velocity_between_0_and_1 = False

#wheter to set the velocity to 1 everywhere where it is > 1
#by comparing this test accuracy with the evaluation without this
# you can see how much influence the actual velocity info has
only_train_note_starts = False
#function that runs over V if only_train_note_starts
def set_to_1_if_nonzero(V):
    V_normalized = np.copy(V)
    for sample in range(V.shape[0]):
        for step in range(V.shape[1]):
            if V[sample, step,0] > 0:
                V_normalized[sample, step, 0] = 1
    return V_normalized


print('loading data...')
# Get Train and test sets


folder = source_folder

V_train, V_test, D_train, D_test, T_train, T_test, I_train, I_test, Y_train, Y_test, X_train, X_test, C_train, C_test, train_paths, test_paths = import_midi_from_folder(folder)

train_set_size = len(X_train)
test_set_size = len(X_test)


print(len(train_paths))
print(len(test_paths))
print(C_test)


class_string = ''
for class_name in classes:
    class_string += class_name

fd = {'highcrop': high_crop, 'lowcrop':low_crop, 'lr': learning_rate, 'opt': optimizer,
'bi': bidirectional, 'lstm_size': lstm_size, 'trainsize': train_set_size, 
'testsize': test_set_size, 'input_length': input_length, 'reset_states': reset_states, 
'num_layers':num_layers, 'only_train_note_starts': only_train_note_starts, 
'velocity_threshold_such_that_it_is_a_played_note': velocity_threshold_such_that_it_is_a_played_note, 
'scale': scale_velocity_between_0_and_1, 'classes': class_string}
t = str(int(round(time.time())))
model_name = t+'-num_layers_%(num_layers)s_maxlen_%(input_length)s_otns_%(only_train_note_starts)s_lstmsize_%(lstm_size)s_trainsize_%(trainsize)s_testsize_%(testsize)s_thresh_%(velocity_threshold_such_that_it_is_a_played_note)s_scale_%(scale)s_classes_%(classes)s' % fd

model_path = model_path + model_name + '/'
if not os.path.exists(model_path):
    os.makedirs(model_path)


# Define an input sequence and process it.
inputs = Input(shape=(None, input_dim))
lstm_outputs = inputs
for layer_no in range(num_layers-1):
    lstm_outputs = GRU(lstm_size, return_state=False, return_sequences=True)(lstm_outputs)
#last layer, that does not return sequences
lstm_outputs = GRU(lstm_size, return_state=False, return_sequences=False)(lstm_outputs)
dense = Dense(num_classes, activation='softmax')
outputs = dense(lstm_outputs)
model = Model(inputs, outputs)


#compile autoencoder
if optimizer == 'RMS': optimizer = RMSprop(lr=learning_rate)
if optimizer == 'Adam': optimizer = Adam(lr=learning_rate)
loss = 'categorical_crossentropy'
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

print(model.summary())

# initialize loss arrays
total_test_loss_array = [] 
total_test_accuracy_array = []
total_train_loss_array = []
total_train_loss = 0
total_train_accuracy_array = []
total_train_accuracy = 0


if scale_velocity_between_0_and_1:
    for V in (V_train + V_test):
        V[np.nonzero(V)] = (V[np.nonzero(V)] - velocity_threshold_such_that_it_is_a_played_note) / (1.0-velocity_threshold_such_that_it_is_a_played_note)

if only_train_note_starts:
    for V in (V_train + V_test):
        V[np.nonzero(V)] = 1

# Test function
def test(testID):
    print('\nTesting:')
    total_test_loss = 0
    total_test_loss_length = 0
    total_test_loss_number = 0

    confusion_matrix = np.zeros((num_classes, num_classes))

    bar = progressbar.ProgressBar(max_value=test_set_size, redirect_stdout=False)
    for i, test_song in enumerate(X_test):

        X = V_test[i]
        X = np.expand_dims(X, 2)
        num_samples = X.shape[0]
        c = C_test[i]
        Y = np.asarray([to_categorical(c, num_classes=num_classes)]*num_samples).squeeze()

        scores = model.evaluate(X,Y , batch_size=batch_size, verbose=verbose)
        if reset_states:
            model.reset_states()
        total_test_loss += scores[0]

        Y_predicted = model.predict(X, batch_size=batch_size, verbose=verbose)
        for y_val, y_predicted in zip(Y, Y_predicted):
            y_class_test = np.argmax(y_val)
            y_class_predicted = np.argmax(y_predicted)
            confusion_matrix[y_class_predicted, y_class_test] += 1
        bar.update(i+1)

    accuracy = np.sum(np.diagonal(confusion_matrix)) / np.sum(confusion_matrix)
    total_test_loss_array.append(total_test_loss/test_set_size)
    total_test_accuracy_array.append(accuracy)
    print('\nTotal test loss: ', total_test_loss/test_set_size)
    print('Total accuracy: ' + str(accuracy*100) + "%") 
    print('-'*50)
    plt.figure()
    plt.title('Style classification on velocity information')
    plt.plot(total_test_loss_array, label='Total test loss')
    plt.plot(total_train_loss_array, label='Total train loss')
    plt.plot(total_test_accuracy_array, label='Total test accuracy')
    plt.plot(total_train_accuracy_array, label='Total train accuracy')
    plt.legend(loc='lower left', prop={'size': 8})
    if show_plot: plt.show()
    if save_plot: 
        plt.savefig(model_path+t+'velocity_train.png')
        tikz_save(model_path+t+'velocity_train.tex', encoding='utf-8', show_info=False)
    pickle.dump(total_test_loss_array,open(model_path+'total_test_loss_array.pickle', 'wb'))
    pickle.dump(total_test_accuracy_array,open(model_path+'total_test_accuracy_array.pickle', 'wb'))
    pickle.dump(total_train_accuracy_array,open(model_path+'total_train_accuracy_array.pickle', 'wb'))
    pickle.dump(total_train_loss_array,open(model_path+'total_train_loss_array.pickle', 'wb'))

    if testID % save_step is 0:
        confusion_matrix = confusion_matrix/confusion_matrix.sum(axis=1, keepdims=True)
        plt.figure()
        plt.imshow(confusion_matrix, interpolation='nearest')
        plt.title('Total accuracy: ' + str(accuracy) + '%')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.xticks(np.arange(0,num_classes), classes)
        plt.yticks(np.arange(0,num_classes), classes)
        plt.colorbar()
        if show_plot: plt.show()
        if save_plot: 
            plt.savefig(model_path+'confusion_matrix' + str(testID) + '.png')
            tikz_save(model_path+'confusion_matrix' + str(testID) + '.tex', encoding='utf-8', show_info=False)


# Save Parameters to text file
with open(model_path + 'params.txt', "w", encoding='utf-8') as text_file:
    text_file.write("velocity_threshold_such_that_it_is_a_played_note: %s" % velocity_threshold_such_that_it_is_a_played_note + '\n')
    text_file.write("epochs: %s" % epochs + '\n')
    text_file.write("train_set_size: %s" % train_set_size + '\n')
    text_file.write("test_set_size: %s" % test_set_size + '\n')
    text_file.write("only_train_note_starts: %s" % only_train_note_starts + '\n')
    text_file.write("learning_rate: %s" % learning_rate + '\n')
    text_file.write("save_step: %s" % save_step + '\n')
    text_file.write("shuffle_train_set: %s" % shuffle_train_set + '\n')
    text_file.write("test_step: %s" % test_step + '\n')
    text_file.write("bidirectional: %s" % bidirectional + '\n')
    text_file.write("load_from_pickle_instead_of_midi: %s" % load_from_pickle_instead_of_midi + '\n')
    text_file.write("pickle_load_path: %s" % pickle_load_path + '\n')
    text_file.write("train_paths: %s" % train_paths + '\n')
    text_file.write("test_paths: %s" % test_paths + '\n')

# Train model
print('training model...')
for e in range(1, epochs+1):

    total_train_loss = 0
    total_train_accuracy = 0
    
    print('Epoch ', e, 'of ', epochs, 'Epochs\nTraining:')


    if shuffle_train_set:

        permutation = np.random.permutation(len(X_train))

        train_paths = [train_paths[i] for i in permutation]
        X_train = [X_train[i] for i in permutation]
        Y_train = [Y_train[i] for i in permutation]
        C_train = [C_train[i] for i in permutation]
        I_train = [I_train[i] for i in permutation]
        V_train = [V_train[i] for i in permutation]
        D_train = [D_train[i] for i in permutation]
        T_train = [T_train[i] for i in permutation]

    bar = progressbar.ProgressBar(max_value=train_set_size)
    
    # Train model with each song seperately
    for i, train_song in enumerate(X_train):

        X = V_train[i]
        X = np.expand_dims(X, 2)
        num_samples = X.shape[0]

        if num_samples > 1:
            c = C_train[i]
            Y = np.asarray([to_categorical(c, num_classes=num_classes)]*num_samples).squeeze()


            hist = model.fit(X, Y,
                        epochs=1,
                        batch_size=batch_size,
                        shuffle=False,
                        verbose=verbose)

            if reset_states:
                model.reset_states()

            total_train_loss += np.mean(hist.history['loss'])
            total_train_accuracy += np.mean(hist.history['acc'])
        bar.update(i+1)
    if (e+1)%test_step is 0:
        total_train_loss = total_train_loss/train_set_size
        total_train_loss_array.append(total_train_loss)
        total_train_accuracy = total_train_accuracy/train_set_size
        total_train_accuracy_array.append(total_train_accuracy)
        test(e)
        

    if e%save_step is 0:
        print('saving model')
        model_save_path = model_path + 'model' + 'Epoch' + str(e) + model_filetype
        model.save(model_save_path)


