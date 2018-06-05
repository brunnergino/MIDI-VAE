
# ----------------------------------------------------------------------------------------------
# Import dependencies
# ----------------------------------------------------------------------------------------------

from settings import *
from keras.utils import to_categorical
from random import shuffle
import progressbar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import _pickle as pickle
import time
import vae_definition
from vae_definition import VAE
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
import pretty_midi as pm
import sys
from import_midi import import_midi_from_folder
import data_class
from matplotlib2tikz import save as tikz_save


# ----------------------------------------------------------------------------------------------
# Set parameters for training session (not for VAE)
# ----------------------------------------------------------------------------------------------

# Path where the polyphonic models are saved:
model_path = 'models/autoencode/vae/'
model_filetype = '.pickle'

assert(output_length > 0)
assert(input_length > 0)

# ----------------------------------------------------------------------------------------------
# Build VAE model
# ----------------------------------------------------------------------------------------------

print('creating model...')

model = VAE()
model.create( input_dim=input_dim, 
    output_dim=output_dim, 
    use_embedding=use_embedding, 
    embedding_dim=embedding_dim, 
    input_length=input_length,
    output_length=output_length, 
    latent_rep_size=latent_dim, 
    vae_loss=vae_loss,
    optimizer=optimizer, 
    activation=activation, 
    lstm_activation=lstm_activation, 
    lstm_state_activation=lstm_state_activation,
    epsilon_std=epsilon_std, 
    epsilon_factor=epsilon_factor,
    include_composer_decoder=include_composer_decoder,
    num_composers=num_composers, 
    composer_weight=composer_weight, 
    lstm_size=lstm_size, 
    cell_type=cell_type,
    num_layers_encoder=num_layers_encoder, 
    num_layers_decoder=num_layers_decoder, 
    bidirectional=bidirectional, 
    decode=decode, 
    teacher_force=teacher_force, 
    learning_rate=learning_rate, 
    split_lstm_vector=split_lstm_vector, 
    history=history, 
    beta=beta, 
    prior_mean=prior_mean,
    prior_std=prior_std,
    decoder_additional_input=decoder_additional_input, 
    decoder_additional_input_dim=decoder_additional_input_dim, 
    extra_layer=extra_layer,
    meta_instrument= meta_instrument,
    meta_instrument_dim= meta_instrument_dim,
    meta_instrument_length=meta_instrument_length,
    meta_instrument_activation=meta_instrument_activation,
    meta_instrument_weight = meta_instrument_weight,
    signature_decoder = signature_decoder,
    signature_dim = signature_dim,
    signature_activation = signature_activation,
    signature_weight = signature_weight,
    composer_decoder_at_notes_output=composer_decoder_at_notes_output,
    composer_decoder_at_notes_weight=composer_decoder_at_notes_weight,
    composer_decoder_at_notes_activation=composer_decoder_at_notes_activation,
    composer_decoder_at_instrument_output=composer_decoder_at_instrument_output,
    composer_decoder_at_instrument_weight=composer_decoder_at_instrument_weight,
    composer_decoder_at_instrument_activation=composer_decoder_at_instrument_activation,
    meta_velocity=meta_velocity,
    meta_velocity_length=meta_velocity_length,
    meta_velocity_activation=meta_velocity_activation,
    meta_velocity_weight=meta_velocity_weight,
    meta_held_notes=meta_held_notes,
    meta_held_notes_length=meta_held_notes_length,
    meta_held_notes_activation=meta_held_notes_activation,
    meta_held_notes_weight=meta_held_notes_weight,
    meta_next_notes=meta_next_notes,
    meta_next_notes_output_length=meta_next_notes_output_length,
    meta_next_notes_weight=meta_next_notes_weight,
    meta_next_notes_teacher_force=meta_next_notes_teacher_force,
    activation_before_splitting=activation_before_splitting
    )

encoder = model.encoder
decoder = model.decoder
autoencoder = model.autoencoder

print(encoder.summary())
print(decoder.summary())
print(autoencoder.summary())


if load_previous_checkpoint:
    autoencoder.load_weights(previous_checkpoint_path +'autoencoder'+'Epoch'+str(previous_epoch)+'.pickle', by_name=False)
    encoder.load_weights(previous_checkpoint_path+'encoder'+'Epoch'+str(previous_epoch)+'.pickle', by_name=False)
    decoder.load_weights(previous_checkpoint_path+'decoder'+'Epoch'+str(previous_epoch)+'.pickle', by_name=False)

    print("Successfully loaded previous epochs")

    if reset_states:
        autoencoder.reset_states()
        encoder.reset_states()
        decoder.reset_states()

# ----------------------------------------------------------------------------------------------
# Import and preprocess data
# ----------------------------------------------------------------------------------------------

print('loading data...')
# Get Train and test sets


folder = source_folder

V_train, V_test, D_train, D_test, T_train, T_test, I_train, I_test, Y_train, Y_test, X_train, X_test, C_train, C_test, train_paths, test_paths = import_midi_from_folder(folder)

train_set_size = len(X_train)
test_set_size = len(X_test)


print(len(train_paths))
print(len(test_paths))
print(C_test)


# ----------------------------------------------------------------------------------------------
# Prepare model path
# ----------------------------------------------------------------------------------------------


fd = {'include_composer_feature': include_composer_feature, 'highcrop': high_crop, 'lowcrop':low_crop, 'lr': learning_rate, 'opt': optimizer,
'bi': bidirectional, 'lstm_size': lstm_size, 'latent': latent_dim, 'trainsize': train_set_size, 'testsize': test_set_size, 'input_length': input_length,
'output_length': output_length, 'reset_states': reset_states, 'compdec': include_composer_decoder, 'num_layers_encoder': num_layers_encoder, 'num_layers_decoder': num_layers_decoder, 
'beta': beta, 'lr': learning_rate, 'epsstd': epsilon_std}
model_name = t+'-_ls_inlen_%(input_length)s_outlen_%(output_length)s_beta_%(beta)s_lr_%(lr)s_lstmsize_%(lstm_size)s_latent_%(latent)s_trainsize_%(trainsize)s_testsize_%(testsize)s_epsstd_%(epsstd)s' % fd

model_path = model_path + model_name + '/'
if not os.path.exists(model_path):
    os.makedirs(model_path)

# ----------------------------------------------------------------------------------------------
# Test function
# ----------------------------------------------------------------------------------------------

enumerated_metric_names = []
metric_names_total_dict = dict()
metric_names_count_dict = dict()
for name in autoencoder.metrics_names:
    if name in metric_names_count_dict.keys():
        metric_names_total_dict[name] += 1
    else:
        metric_names_total_dict[name] = 1
        #initialize count dict
        metric_names_count_dict[name] = 0
for name in autoencoder.metrics_names:
    if metric_names_total_dict[name] > 1:
        metric_names_count_dict[name] += 1
        enumerated_metric_names.append(name + "_" + str(metric_names_count_dict[name]))
    else:
        enumerated_metric_names.append(name)

# initialize loss arrays
total_test_notes_loss_array = []
total_train_notes_loss_array = []
total_test_loss_array = [] 
total_train_loss_array = []
total_train_accuracy_array = []
total_test_accuracy_array = []
max_test_accuracy = 0

total_train_meta_instrument_accuracy_array = []
total_test_meta_instrument_accuracy_array = []
total_train_meta_instrument_loss_array = []
total_test_meta_instrument_loss_array = []

total_train_meta_velocity_accuracy_array = []
total_test_meta_velocity_accuracy_array = []
total_train_meta_velocity_loss_array = []
total_test_meta_velocity_loss_array = []

total_train_meta_held_notes_accuracy_array = []
total_test_meta_held_notes_accuracy_array = []
total_train_meta_held_notes_loss_array = []
total_test_meta_held_notes_loss_array = []

total_train_meta_next_notes_accuracy_array = []
total_test_meta_next_notes_accuracy_array = []
total_train_meta_next_notes_loss_array = []
total_test_meta_next_notes_loss_array = []

total_train_composer_accuracy_array = []
total_train_composer_loss_array = []
total_test_composer_accuracy_array = []
total_test_composer_loss_array = []

total_train_signature_accuracy_array = []
total_train_signature_loss_array = []
total_test_signature_accuracy_array = []
total_test_signature_loss_array = []

total_test_kl_loss_array = []
total_train_kl_loss_array = []

total_train_composer_instrument_accuracy_array = []
total_train_composer_instrument_loss_array = []
total_test_composer_instrument_accuracy_array = []
total_test_composer_instrument_loss_array = []

total_train_composer_notes_accuracy_array = []
total_train_composer_notes_loss_array = []
total_test_composer_notes_accuracy_array = []
total_test_composer_notes_loss_array = []


# Test function
def test():
    global max_test_accuracy
    print('\nTesting:')
    total_test_loss = 0
    total_test_accuracy = 0
    total_test_notes_loss = 0

    total_test_meta_instrument_loss = 0
    total_test_meta_instrument_accuracy = 0
    total_test_meta_velocity_loss = 0
    total_test_meta_velocity_accuracy = 0
    total_test_meta_held_notes_loss = 0
    total_test_meta_held_notes_accuracy = 0

    total_test_meta_next_notes_loss = 0
    total_test_meta_next_notes_accuracy = 0


    total_test_loss_composer = 0
    total_test_accuracy_composer = 0

    total_test_loss_signature = 0
    total_test_signature_accuracy = 0

    total_test_loss_composer_notes = 0
    total_test_composer_notes_accuracy = 0
    total_test_loss_composer_instrument = 0
    total_test_composer_instrument_accuracy = 0
    
    bar = progressbar.ProgressBar(max_value=test_set_size, redirect_stdout=False)
    for test_song_num in range(len(X_test)):

        X = X_test[test_song_num]
        Y = Y_test[test_song_num]
        C = C_test[test_song_num]
        I = I_test[test_song_num]
        V = V_test[test_song_num]
        D = D_test[test_song_num]
        S = normalized_S_test[test_song_num]

        T = T_test[test_song_num] #not yet used

        #calculate history if desired
        if history:
            #get the representation by feeding the inputs into the encoder
            encoder_input_list = vae_definition.prepare_encoder_input_list(X,I,V,D)
            representation_list = encoder.predict(encoder_input_list, batch_size=batch_size, verbose=False)
            #roll the list by one to save the representation of the last sample for each input
            H = np.zeros(representation_list.shape)
            H[1:] = representation_list[:-1]

        else:
            H = np.zeros((X.shape[0], latent_dim))


        input_list, output_list = vae_definition.prepare_autoencoder_input_and_output_list(X,Y,C,I,V,D,S,H, return_sample_weight=False)
        
        loss = autoencoder.evaluate(input_list, output_list, batch_size=batch_size, verbose=False)

        total_test_loss += loss[0]
        if meta_instrument or meta_velocity or meta_held_notes or meta_next_notes:

            count = 1
            total_test_notes_loss += loss[enumerated_metric_names.index('decoder_loss_' + str(count))]
            total_test_accuracy += loss[enumerated_metric_names.index('decoder_acc_1')]
            
            if meta_instrument:
                count +=1
                total_test_meta_instrument_loss += loss[enumerated_metric_names.index('decoder_loss_' + str(count))]
                total_test_meta_instrument_accuracy += loss[enumerated_metric_names.index('decoder_acc_' + str(count))]

            if meta_velocity:
                count += 1
                total_test_meta_velocity_loss += loss[enumerated_metric_names.index('decoder_loss_' + str(count))]
                total_test_meta_velocity_accuracy += loss[enumerated_metric_names.index('decoder_acc_' + str(count))]

            if meta_held_notes:
                count += 1
                total_test_meta_held_notes_loss += loss[enumerated_metric_names.index('decoder_loss_' + str(count))]
                total_test_meta_held_notes_accuracy += loss[enumerated_metric_names.index('decoder_acc_' + str(count))]

            if meta_next_notes:
                count += 1
                total_test_meta_next_notes_loss += loss[enumerated_metric_names.index('decoder_loss_' + str(count))]
                total_test_meta_next_notes_accuracy += loss[enumerated_metric_names.index('decoder_acc_' + str(count))]

        else:
            if len(enumerated_metric_names) > 2:
                total_test_accuracy += loss[enumerated_metric_names.index('decoder_acc')]
                total_test_notes_loss += loss[enumerated_metric_names.index('decoder_loss')]
            else:
                total_test_notes_loss += loss[0]
                total_test_accuracy += loss[1]

        if include_composer_decoder:
            total_test_loss_composer += loss[enumerated_metric_names.index('composer_decoder_loss')]
            total_test_accuracy_composer += loss[enumerated_metric_names.index('composer_decoder_acc')]

        if signature_decoder:
            total_test_loss_signature += loss[enumerated_metric_names.index('signature_decoder_loss')]
            total_test_signature_accuracy += loss[enumerated_metric_names.index('signature_decoder_acc')]

        if composer_decoder_at_notes_output:
            total_test_loss_composer_notes += loss[enumerated_metric_names.index('composer_decoder_at_notes_loss')]
            total_test_composer_notes_accuracy += loss[enumerated_metric_names.index('composer_decoder_at_notes_acc')]

        if composer_decoder_at_instrument_output:
            total_test_loss_composer_instrument += loss[enumerated_metric_names.index('composer_decoder_at_instruments_loss')]
            total_test_composer_instrument_accuracy += loss[enumerated_metric_names.index('composer_decoder_at_instruments_acc')]
        
        if reset_states:
            autoencoder.reset_states()
        
              
        bar.update(test_song_num+1)

    plt.close('all')
    f, axarr = plt.subplots(3,2, sharex=True, figsize=(15.0, 20.0))
    f.suptitle(t)
    
    
    if include_composer_decoder:
        composer_accuracy = total_test_accuracy_composer/test_set_size
        composer_loss = total_test_loss_composer/test_set_size
        total_test_composer_loss_array.append(composer_loss)
        total_test_composer_accuracy_array.append(composer_accuracy)
        print('\nTest composer accuracy: ', composer_accuracy)
        print('Test composer loss: ', composer_loss)
        axarr[1,1].plot(total_test_composer_accuracy_array,  label='Test composer accuracy')
        axarr[1,0].plot(total_train_composer_accuracy_array,  label='Train composer accuracy')
        axarr[0,1].plot(total_test_composer_loss_array,  label='Test composer loss')
        axarr[0,0].plot(total_train_composer_loss_array,  label='Train composer loss')
        pickle.dump(total_test_composer_loss_array,open(model_path+'total_test_composer_loss_array.pickle', 'wb'))
        pickle.dump(total_train_composer_loss_array,open(model_path+'total_train_composer_loss_array.pickle', 'wb'))
        pickle.dump(total_test_composer_accuracy_array,open(model_path+'total_test_composer_accuracy_array.pickle', 'wb'))
        pickle.dump(total_train_composer_accuracy_array,open(model_path+'total_train_composer_accuracy_array.pickle', 'wb'))

    if meta_instrument:
        meta_instrument_accuracy = total_test_meta_instrument_accuracy/test_set_size
        meta_instrument_loss = total_test_meta_instrument_loss/test_set_size
        total_test_meta_instrument_loss_array.append(meta_instrument_loss)
        total_test_meta_instrument_accuracy_array.append(meta_instrument_accuracy)
        print('Test meta instrument accuracy: ', meta_instrument_accuracy)
        print('Test meta instrument loss: ', meta_instrument_loss)
        axarr[1,1].plot(total_test_meta_instrument_accuracy_array, label='Test instrument accuracy')
        axarr[1,0].plot(total_train_meta_instrument_accuracy_array, label='Train instrument accuracy')
        axarr[0,1].plot(total_test_meta_instrument_loss_array, label='Test instrument loss')
        axarr[0,0].plot(total_train_meta_instrument_loss_array, label='Train instrument loss')
        pickle.dump(total_test_meta_instrument_loss_array,open(model_path+'total_test_meta_instrument_loss_array.pickle', 'wb'))
        pickle.dump(total_test_meta_instrument_accuracy_array,open(model_path+'total_test_meta_instrument_accuracy_array.pickle', 'wb'))
        pickle.dump(total_train_meta_instrument_loss_array,open(model_path+'total_train_meta_instrument_loss_array.pickle', 'wb'))
        pickle.dump(total_train_meta_instrument_accuracy_array,open(model_path+'total_train_meta_instrument_accuracy_array.pickle', 'wb'))

    if meta_held_notes:
        meta_held_notes_accuracy = total_test_meta_held_notes_accuracy/test_set_size
        meta_held_notes_loss = total_test_meta_held_notes_loss/test_set_size
        total_test_meta_held_notes_loss_array.append(meta_held_notes_loss)
        total_test_meta_held_notes_accuracy_array.append(meta_held_notes_accuracy)
        print('Test meta held_notes accuracy: ', meta_held_notes_accuracy)
        print('Test meta held_notes loss: ', meta_held_notes_loss)
        axarr[1,1].plot(total_test_meta_held_notes_accuracy_array, label='Test held_notes accuracy')
        axarr[1,0].plot(total_train_meta_held_notes_accuracy_array, label='Train held_notes accuracy')
        axarr[0,1].plot(total_test_meta_held_notes_loss_array, label='Test held_notes loss')
        axarr[0,0].plot(total_train_meta_held_notes_loss_array, label='Train held_notes loss')
        pickle.dump(total_test_meta_held_notes_loss_array,open(model_path+'total_test_meta_held_notes_loss_array.pickle', 'wb'))
        pickle.dump(total_test_meta_held_notes_accuracy_array,open(model_path+'total_test_meta_held_notes_accuracy_array.pickle', 'wb'))
        pickle.dump(total_train_meta_held_notes_loss_array,open(model_path+'total_train_meta_held_notes_loss_array.pickle', 'wb'))
        pickle.dump(total_train_meta_held_notes_accuracy_array,open(model_path+'total_train_meta_held_notes_accuracy_array.pickle', 'wb'))

    if meta_next_notes:
        meta_next_notes_accuracy = total_test_meta_next_notes_accuracy/test_set_size
        meta_next_notes_loss = total_test_meta_next_notes_loss/test_set_size
        total_test_meta_next_notes_loss_array.append(meta_next_notes_loss)
        total_test_meta_next_notes_accuracy_array.append(meta_next_notes_accuracy)
        print('Test meta next_notes accuracy: ', meta_next_notes_accuracy)
        print('Test meta next_notes loss: ', meta_next_notes_loss)
        axarr[1,1].plot(total_test_meta_next_notes_accuracy_array, label='Test next_notes accuracy')
        axarr[1,0].plot(total_train_meta_next_notes_accuracy_array, label='Train next_notes accuracy')
        axarr[0,1].plot(total_test_meta_next_notes_loss_array, label='Test next_notes loss')
        axarr[0,0].plot(total_train_meta_next_notes_loss_array, label='Train next_notes loss')
        pickle.dump(total_test_meta_next_notes_loss_array,open(model_path+'total_test_meta_next_notes_loss_array.pickle', 'wb'))
        pickle.dump(total_test_meta_next_notes_accuracy_array,open(model_path+'total_test_meta_next_notes_accuracy_array.pickle', 'wb'))
        pickle.dump(total_train_meta_next_notes_loss_array,open(model_path+'total_train_meta_next_notes_loss_array.pickle', 'wb'))
        pickle.dump(total_train_meta_next_notes_accuracy_array,open(model_path+'total_train_meta_next_notes_accuracy_array.pickle', 'wb'))

    if composer_decoder_at_notes_output:
        composer_notes_accuracy = total_test_composer_notes_accuracy/test_set_size
        composer_notes_loss = total_test_loss_composer_notes/test_set_size
        total_test_composer_notes_loss_array.append(composer_notes_loss)
        total_test_composer_notes_accuracy_array.append(composer_notes_accuracy)
        print('Test composer_notes accuracy: ', composer_notes_accuracy)
        print('Test composer_notes loss: ', composer_notes_loss)
        axarr[1,1].plot(total_test_composer_notes_accuracy_array, label='Test composer_notes accuracy')
        axarr[1,0].plot(total_train_composer_notes_accuracy_array, label='Train composer_notes accuracy')
        axarr[0,1].plot(total_test_composer_notes_loss_array, label='Test composer_notes loss')
        axarr[0,0].plot(total_train_composer_notes_loss_array, label='Train composer_notes loss')
        pickle.dump(total_test_composer_notes_loss_array,open(model_path+'total_test_composer_notes_loss_array.pickle', 'wb'))
        pickle.dump(total_test_composer_notes_accuracy_array,open(model_path+'total_test_composer_notes_accuracy_array.pickle', 'wb'))
        pickle.dump(total_train_composer_notes_loss_array,open(model_path+'total_train_composer_notes_loss_array.pickle', 'wb'))
        pickle.dump(total_train_composer_notes_accuracy_array,open(model_path+'total_train_composer_notes_accuracy_array.pickle', 'wb'))

    if composer_decoder_at_instrument_output:
        composer_instrument_accuracy = total_test_composer_instrument_accuracy/test_set_size
        composer_instrument_loss = total_test_loss_composer_instrument/test_set_size
        total_test_composer_instrument_loss_array.append(composer_instrument_loss)
        total_test_composer_instrument_accuracy_array.append(composer_instrument_accuracy)
        print('Test composer_instrument accuracy: ', composer_instrument_accuracy)
        print('Test composer_instrument loss: ', composer_instrument_loss)
        axarr[1,1].plot(total_test_composer_instrument_accuracy_array, label='Test composer_instrument accuracy')
        axarr[1,0].plot(total_train_composer_instrument_accuracy_array, label='Train composer_instrument accuracy')
        axarr[0,1].plot(total_test_composer_instrument_loss_array, label='Test composer_instrument loss')
        axarr[0,0].plot(total_train_composer_instrument_loss_array, label='Train composer_instrument loss')
        pickle.dump(total_test_composer_instrument_loss_array,open(model_path+'total_test_composer_instrument_loss_array.pickle', 'wb'))
        pickle.dump(total_test_composer_instrument_accuracy_array,open(model_path+'total_test_composer_instrument_accuracy_array.pickle', 'wb'))
        pickle.dump(total_train_composer_instrument_loss_array,open(model_path+'total_train_composer_instrument_loss_array.pickle', 'wb'))
        pickle.dump(total_train_composer_instrument_accuracy_array,open(model_path+'total_train_composer_instrument_accuracy_array.pickle', 'wb'))

    accuracy = total_test_accuracy/test_set_size
    if max_test_accuracy < accuracy:
        max_test_accuracy = accuracy
    total_test_accuracy_array.append(accuracy)
    notes_loss = total_test_notes_loss / test_set_size
    total_test_notes_loss_array.append(notes_loss)
    print('Test notes accuracy: ', accuracy)
    print('Test notes loss: ', notes_loss)
    axarr[1,1].plot(total_test_accuracy_array, label='Test notes accuracy')
    axarr[1,0].plot(total_train_accuracy_array, label='Train notes accuracy')
    axarr[0,1].plot(total_test_notes_loss_array, label='Test notes loss')
    axarr[0,0].plot(total_train_notes_loss_array, label='Train notes loss')
    pickle.dump(total_train_accuracy_array,open(model_path+'total_train_accuracy_array.pickle', 'wb'))
    pickle.dump(total_test_accuracy_array,open(model_path+'total_test_accuracy_array.pickle', 'wb'))
    pickle.dump(total_test_notes_loss_array,open(model_path+'total_test_notes_loss_array.pickle', 'wb'))
    pickle.dump(total_train_notes_loss_array,open(model_path+'total_train_notes_loss_array.pickle', 'wb'))

    if meta_velocity:
        meta_velocity_accuracy = total_test_meta_velocity_accuracy/test_set_size
        meta_velocity_loss = total_test_meta_velocity_loss/test_set_size
        total_test_meta_velocity_loss_array.append(meta_velocity_loss)
        total_test_meta_velocity_accuracy_array.append(meta_velocity_accuracy)
        #Accuracy is logged for meta_velocity (it outputs accuracy metric for all losses) but it does not make sense, so don't show it or save it
        #only plot and save it if it is combined with the held notes (which have accuracy)
        if combine_velocity_and_held_notes or velocity_threshold_such_that_it_is_a_played_note >= 0.5:
            print('Test meta velocity accuracy: ', meta_velocity_accuracy)
        print('Test meta velocity loss: ', meta_velocity_loss)
        if combine_velocity_and_held_notes:
            axarr[1,1].plot(total_test_meta_velocity_accuracy_array, label='Test velocity accuracy')
            axarr[1,0].plot(total_train_meta_velocity_accuracy_array, label='Train velocity accuracy')
        axarr[0,1].plot(total_test_meta_velocity_loss_array, label='Test velocity loss')
        axarr[0,0].plot(total_train_meta_velocity_loss_array, label='Train velocity loss')
        pickle.dump(total_test_meta_velocity_loss_array,open(model_path+'total_test_meta_velocity_loss_array.pickle', 'wb'))
        if combine_velocity_and_held_notes or velocity_threshold_such_that_it_is_a_played_note >= 0.5:
            pickle.dump(total_test_meta_velocity_accuracy_array,open(model_path+'total_test_meta_velocity_accuracy_array.pickle', 'wb'))
            pickle.dump(total_train_meta_velocity_accuracy_array,open(model_path+'total_train_meta_velocity_accuracy_array.pickle', 'wb'))
        pickle.dump(total_train_meta_velocity_loss_array,open(model_path+'total_train_meta_velocity_loss_array.pickle', 'wb'))

    

    if signature_decoder:
        signature_accuracy = total_test_signature_accuracy/test_set_size
        signature_loss = total_test_loss_signature/test_set_size
        total_test_signature_loss_array.append(signature_loss)
        total_test_signature_accuracy_array.append(signature_accuracy)
        #Don't plot signature accuracy since it makes no sense in regression problem
        #print('Test signature accuracy: ', signature_accuracy)
        print('Test signature loss: ', signature_loss)
        #axarr[1,1].plot(total_test_signature_accuracy_array, label='Test signature accuracy')
        #axarr[1,0].plot(total_train_signature_accuracy_array, label='Train signature accuracy')
        axarr[0,1].plot(total_test_signature_loss_array, label='Test signature loss')
        axarr[0,0].plot(total_train_signature_loss_array, label='Train signature loss')
        pickle.dump(total_test_signature_loss_array,open(model_path+'total_test_signature_loss_array.pickle', 'wb'))
        #pickle.dump(total_test_signature_accuracy_array,open(model_path+'total_test_signature_accuracy_array.pickle', 'wb'))
        pickle.dump(total_train_signature_loss_array,open(model_path+'total_train_signature_loss_array.pickle', 'wb'))
        #pickle.dump(total_train_signature_accuracy_array,open(model_path+'total_train_signature_accuracy_array.pickle', 'wb'))
    

    

    test_loss = total_test_loss/test_set_size
    total_test_loss_array.append(test_loss)

    

    if beta > 0:
        #TODO. adjust by weights?
        kl_loss = test_loss - notes_loss * 1.0
        if include_composer_decoder: kl_loss -= composer_loss * composer_weight
        if meta_instrument: kl_loss -= meta_instrument_loss * meta_instrument_weight
        if meta_velocity: kl_loss -= meta_velocity_loss * meta_velocity_weight
        if meta_held_notes: kl_loss -= meta_held_notes_loss * meta_held_notes_weight
        if meta_next_notes: kl_loss -= meta_next_notes_loss * meta_next_notes_weight
        if signature_decoder: kl_loss -= signature_loss * signature_weight
        if composer_decoder_at_notes_output: kl_loss -= composer_notes_loss * composer_decoder_at_notes_weight
        if composer_decoder_at_instrument_output: kl_loss -= composer_instrument_loss * composer_decoder_at_instrument_weight
        #since you get the value back weighted, scale back by dividing by beta
        kl_loss = kl_loss / beta
        total_test_kl_loss_array.append(kl_loss)
        axarr[2,1].plot(total_test_kl_loss_array, label='Test KL loss')
        axarr[2,0].plot(total_train_kl_loss_array, label='Train KL loss')
        print('Test KL loss: ', kl_loss)
        pickle.dump(total_test_kl_loss_array,open(model_path+'total_test_kl_loss_array.pickle', 'wb'))
        pickle.dump(total_train_kl_loss_array,open(model_path+'total_train_kl_loss_array.pickle', 'wb'))

    print('Total test loss: ', test_loss)


    axarr[0,1].plot(total_test_loss_array, label='Total test loss')
    axarr[0,0].plot(total_train_loss_array, label='Total train loss')
    pickle.dump(total_test_loss_array,open(model_path+'total_test_loss_array.pickle', 'wb'))
    pickle.dump(total_train_loss_array,open(model_path+'total_train_loss_array.pickle', 'wb'))

    axarr[2,1].set_title("Test KL loss",fontsize=10)
    axarr[2,0].set_title("Train KL loss", fontsize=10)
    axarr[1,1].set_title("Test accuracies - Max notes acc: %4.2f" % max_test_accuracy, fontsize=10)
    axarr[1,0].set_title("Train accuracies", fontsize=10)
    axarr[0,1].set_title("Test losses",fontsize=10)
    axarr[0,0].set_title("Train losses", fontsize=10)
    axarr[2,1].legend(loc='upper right', prop={'size': 8})
    axarr[2,0].legend(loc='upper right', prop={'size': 8})
    axarr[1,1].legend(loc='lower right', prop={'size': 8})
    axarr[1,0].legend(loc='lower right', prop={'size': 8})
    axarr[0,1].legend(loc='upper right', prop={'size': 8})
    axarr[0,0].legend(loc='upper right', prop={'size': 8})

    if show_plot: f.show()
    if save_plot: f.savefig(model_path+'plot.png')
    print('-'*50)
    



# ----------------------------------------------------------------------------------------------
# Save parameters file
# ----------------------------------------------------------------------------------------------

# Save Parameters to text file
with open(model_path + 'params.txt', "w", encoding='utf-8') as text_file:
    text_file.write("load_from_pickle_instead_of_midi: %s" % load_from_pickle_instead_of_midi + '\n')
    text_file.write("pickle_load_path: %s" % pickle_load_path + '\n')
    text_file.write("epochs: %s" % epochs + '\n')
    text_file.write("input_dim: %s" % input_dim + '\n')
    text_file.write("output_dim: %s" % output_dim + '\n')
    text_file.write("attach_instruments: %s" % attach_instruments + '\n')
    text_file.write("instrument_dim: %s" % instrument_dim + '\n')
    text_file.write("include_only_monophonic_instruments: %s" % include_only_monophonic_instruments + '\n')
    text_file.write("instrument_attach_method: %s" % instrument_attach_method + '\n')
    text_file.write("equal_mini_songs: %s" % equal_mini_songs + '\n')
    text_file.write("train_set_size: %s" % train_set_size + '\n')
    text_file.write("test_set_size: %s" % test_set_size + '\n')
    text_file.write("batch_size: %s" % batch_size + '\n')
    text_file.write("learning_rate: %s" % learning_rate + '\n')
    text_file.write("beta: %s" % beta + '\n')
    text_file.write("prior_mean: %s" % prior_mean + '\n')
    text_file.write("prior_std: %s" % prior_std + '\n')
    text_file.write("save_step: %s" % save_step + '\n')
    text_file.write("shuffle_train_set: %s" % shuffle_train_set + '\n')
    text_file.write("test_step: %s" % test_step + '\n')
    text_file.write("bidirectional: %s" % bidirectional + '\n')
    text_file.write("teacher_force: %s" % teacher_force + '\n')
    text_file.write("include_composer_decoder: %s" % include_composer_decoder + '\n')
    text_file.write("composer_weight: %s" % composer_weight + '\n')
    text_file.write("include_composer_feature: %s" % include_composer_feature + '\n')
    text_file.write("max_voices: %s" % max_voices + '\n')
    text_file.write("num_layers_encoder: %s" % num_layers_encoder + '\n')
    text_file.write("num_layers_decoder: %s" % num_layers_decoder + '\n')
    text_file.write("optimizer: %s" % optimizer + '\n')
    text_file.write("cell_type: %s" % cell_type + '\n')
    text_file.write("lstm_size: %s" % lstm_size + '\n')
    text_file.write("latent_dim: %s" % latent_dim + '\n')
    text_file.write("split_lstm_vector: %s" % split_lstm_vector + '\n')
    text_file.write("extra_layer: %s" % extra_layer + '\n')
    text_file.write("history: %s" % history + '\n')
    text_file.write("include_silent_note: %s" % include_silent_note + '\n')
    text_file.write("silent_weight: %s" % silent_weight + '\n')
    text_file.write("activation: %s" % activation + '\n')
    text_file.write("lstm_activation: %s" % lstm_activation + '\n')
    text_file.write("lstm_state_activation: %s" % lstm_state_activation + '\n')
    text_file.write("decoder_additional_input: %s" % decoder_additional_input + '\n')
    text_file.write("decoder_additional_input_dim: %s" % decoder_additional_input_dim + '\n')
    text_file.write("decoder_input_composer: %s" % decoder_input_composer + '\n')
    text_file.write("epsilon_std: %s" % epsilon_std + '\n')
    text_file.write("epsilon_factor: %s" % epsilon_factor + '\n')
    text_file.write("append_signature_vector_to_latent: %s" % append_signature_vector_to_latent + '\n')
    text_file.write("song_completion: %s" % song_completion + '\n')
    text_file.write("meta_instrument: %s" % meta_instrument + '\n')
    text_file.write("meta_instrument_dim: %s" % meta_instrument_dim + '\n')
    text_file.write("meta_instrument_length: %s" % meta_instrument_length + '\n')
    text_file.write("meta_instrument_activation: %s" % meta_instrument_activation + '\n')
    text_file.write("meta_instrument_weight: %s" % meta_instrument_weight + '\n')
    text_file.write("signature_decoder: %s" % signature_decoder + '\n')
    text_file.write("signature_dim: %s" % signature_dim + '\n')
    text_file.write("signature_activation: %s" % signature_activation + '\n')
    text_file.write("signature_weight: %s" % signature_weight + '\n')
    text_file.write("composer_decoder_at_notes_output: %s" % composer_decoder_at_notes_output + '\n')
    text_file.write("composer_decoder_at_notes_weight: %s" % composer_decoder_at_notes_weight + '\n')
    text_file.write("composer_decoder_at_notes_activation: %s" % composer_decoder_at_notes_activation + '\n')
    text_file.write("composer_decoder_at_instrument_output: %s" % composer_decoder_at_instrument_output + '\n')
    text_file.write("composer_decoder_at_instrument_weight: %s" % composer_decoder_at_instrument_weight + '\n')
    text_file.write("composer_decoder_at_instrument_activation: %s" % composer_decoder_at_instrument_activation+ '\n')
    text_file.write("meta_velocity: %s" % meta_velocity +"\n")
    text_file.write("meta_velocity_activation: %s" % meta_velocity_activation +"\n")
    text_file.write("meta_velocity_weight: %s" % meta_velocity_weight +"\n")
    text_file.write("meta_held_notes: %s" % meta_held_notes +"\n")
    text_file.write("meta_held_notes_length: %s" % meta_held_notes_length +"\n")
    text_file.write("meta_held_notes_activation: %s" % meta_held_notes_activation +"\n")
    text_file.write("meta_held_notes_weight: %s" % meta_held_notes_weight +"\n")
    text_file.write("meta_next_notes: %s" % meta_next_notes +"\n")
    text_file.write("meta_next_notes_output_length: %s" % meta_next_notes_output_length +"\n")
    text_file.write("meta_next_notes_weight: %s" % meta_next_notes_weight +"\n")
    text_file.write("meta_next_notes_teacher_force: %s" % meta_next_notes_teacher_force +"\n")
    text_file.write("activation_before_splitting: %s" % activation_before_splitting+"\n")
    text_file.write("train_paths: %s" % train_paths + '\n')
    text_file.write("test_paths: %s" % test_paths + '\n')

# ----------------------------------------------------------------------------------------------
# Final preprocessing / Calculate signature vectors for set
# ----------------------------------------------------------------------------------------------

total_notes = 0
for train_song_num in range(len(X_train)):
    x = X_train[train_song_num]
    total_notes += input_length * x.shape[0]

print("Total steps (notes + silent): ", total_notes)
print("Total samples: ", total_notes // input_length)

all_S = []
S_train = []
for train_song_num in range(len(Y_train)):
    Y = Y_train[train_song_num]
    num_samples = Y.shape[0]
    signature_vectors = np.zeros((num_samples, signature_vector_length))
    for sample in range(num_samples):

        poly_sample = data_class.monophonic_to_khot_pianoroll(Y[sample], max_voices)
        if include_silent_note:
            poly_sample = poly_sample[:,:-1]
        signature = data_class.signature_from_pianoroll(poly_sample)
        signature_vectors[sample] = signature
    S_train.append(signature_vectors)
    all_S.extend(signature_vectors)

all_S = np.asarray(all_S)

mean_signature = np.mean(all_S, axis=0)
print(mean_signature)
std_signature = np.std(all_S, axis=0)

#make sure you don't divide by zero if std is 0
for i, val in enumerate(std_signature):
    if val == 0:
        std_signature[i] = 1.0e-10
print(std_signature)


normalized_S_train = []
for signature_vectors in S_train:
    normalized_signature_vectors = (signature_vectors - mean_signature) / std_signature
    normalized_S_train.append(normalized_signature_vectors)

normalized_S_test = []
for test_song_num in range(len(Y_test)):
    Y = Y_test[test_song_num]
    num_samples = Y.shape[0]
    signature_vectors = np.zeros((num_samples, signature_vector_length))
    for sample in range(num_samples):

        poly_sample = data_class.monophonic_to_khot_pianoroll(Y[sample], max_voices)

        if include_silent_note:
            poly_sample = poly_sample[:,:-1]
        signature = data_class.signature_from_pianoroll(poly_sample)
        signature = (signature - mean_signature) / std_signature
        signature_vectors[sample] = signature
    normalized_S_test.append(signature_vectors)


# ----------------------------------------------------------------------------------------------
# Train and test
# ----------------------------------------------------------------------------------------------

# Train model
print('Training model...')
start_epoch = 0
if load_previous_checkpoint:
    start_epoch = previous_epoch
for e in range(start_epoch, epochs):

    #total_switched_notes = 0
    total_train_loss = 0.0
    total_train_accuracy = 0.0
    total_train_meta_instrument_accuracy = 0.0
    total_train_meta_instrument_loss = 0.0
    total_train_meta_velocity_accuracy = 0.0
    total_train_meta_velocity_loss = 0.0
    total_train_meta_held_notes_accuracy = 0.0
    total_train_meta_held_notes_loss = 0.0
    total_train_meta_next_notes_accuracy = 0.0
    total_train_meta_next_notes_loss = 0.0
    total_train_composer_accuracy = 0.0
    total_train_composer_loss = 0.0
    total_train_signature_accuracy = 0.0
    total_train_signature_loss = 0.0
    total_train_notes_loss = 0.0
    total_train_kl_loss = 0.0
    total_train_composer_notes_accuracy = 0.0
    total_train_composer_notes_loss = 0.0
    total_train_composer_instrument_accuracy = 0.0
    total_train_composer_instrument_loss = 0.0
    
    
    print('Epoch ', e, 'of ', epochs, 'Epochs\nTraining:')

    print("Beta: ", beta)
    print("Epsilon std: ", epsilon_std)

    if shuffle_train_set:

        permutation = np.random.permutation(len(X_train))

        train_paths = [train_paths[i] for i in permutation]
        X_train = [X_train[i] for i in permutation]
        Y_train = [Y_train[i] for i in permutation]
        C_train = [C_train[i] for i in permutation]
        I_train = [I_train[i] for i in permutation]
        V_train = [V_train[i] for i in permutation]
        D_train = [D_train[i] for i in permutation]
        S_train = [S_train[i] for i in permutation]
        normalized_S_train = [normalized_S_train[i] for i in permutation]
        T_train = [T_train[i] for i in permutation]
        

    bar = progressbar.ProgressBar(max_value=train_set_size)
    for train_song_num in range(len(X_train)):

        X = X_train[train_song_num]
        Y = Y_train[train_song_num]
        C = C_train[train_song_num] 
        I = I_train[train_song_num]
        V = V_train[train_song_num]
        D = D_train[train_song_num]
        S = normalized_S_train[train_song_num]

        T = T_train[train_song_num] #not yet used

        #calculate history if desired
        if history:
            #don't use the history on the 0'th epoch since the encoder is not trained yet
            if e == 0:
                H = np.zeros((X.shape[0], latent_dim))
            else:
                #get the representation by feeding the inputs into the encoder
                encoder_input_list = vae_definition.prepare_encoder_input_list(X,I,V,D)
                representation_list = encoder.predict(encoder_input_list, batch_size=batch_size, verbose=False)
                #roll the list by one to save the representation of the last sample for each input
                H = np.zeros(representation_list.shape)
                H[1:] = representation_list[:-1]
        else:
            H = np.zeros((X.shape[0], latent_dim))

        input_list, output_list, sample_weight = vae_definition.prepare_autoencoder_input_and_output_list(X,Y,C,I,V,D,S,H, return_sample_weight=True)

        hist = autoencoder.fit(input_list, output_list,
                epochs=1,
                batch_size=batch_size,
                shuffle=False,
                sample_weight=sample_weight,
                verbose=False)

        if reset_states:
            autoencoder.reset_states()

        bar.update(train_song_num+1)


        total_train_loss += np.mean(hist.history['loss'])

        #make sure you have installed keras=2.0.8 if you receive only one loss instead of decoder_loss_0,1,2... for each output
        #did not work for keras=2.1.4
        if meta_instrument or meta_velocity or meta_held_notes or meta_next_notes:
            count = 1
            total_train_accuracy += np.mean(hist.history['decoder_acc_' + str(count)])
            total_train_notes_loss += np.mean(hist.history['decoder_loss_' + str(count)])
            if meta_instrument:
                count += 1
                total_train_meta_instrument_accuracy += np.mean(hist.history['decoder_acc_' + str(count)])
                total_train_meta_instrument_loss += np.mean(hist.history['decoder_loss_' + str(count)])
            if meta_velocity:
                count += 1
                total_train_meta_velocity_accuracy += np.mean(hist.history['decoder_acc_' + str(count)])
                total_train_meta_velocity_loss += np.mean(hist.history['decoder_loss_' + str(count)])
            if meta_held_notes:
                count += 1
                total_train_meta_held_notes_accuracy += np.mean(hist.history['decoder_acc_' + str(count)])
                total_train_meta_held_notes_loss += np.mean(hist.history['decoder_loss_' + str(count)])
            if meta_next_notes:
                count += 1
                total_train_meta_next_notes_accuracy += np.mean(hist.history['decoder_acc_' + str(count)])
                total_train_meta_next_notes_loss += np.mean(hist.history['decoder_loss_' + str(count)])
        else:
            if len(hist.history.keys()) > 2:
                total_train_accuracy += np.mean(hist.history['decoder_acc'])
                total_train_notes_loss += np.mean(hist.history['decoder_loss'])
            else:
                total_train_accuracy += np.mean(hist.history['acc'])
                total_train_notes_loss += np.mean(hist.history['loss'])


        if include_composer_decoder:

            total_train_composer_accuracy += np.mean(hist.history['composer_decoder_acc'])
            total_train_composer_loss += np.mean(hist.history['composer_decoder_loss'])
        if signature_decoder:
            total_train_signature_accuracy += np.mean(hist.history['signature_decoder_acc'])
            total_train_signature_loss += np.mean(hist.history['signature_decoder_loss'])

        if composer_decoder_at_notes_output:
            total_train_composer_notes_accuracy += np.mean(hist.history['composer_decoder_at_notes_acc'])
            total_train_composer_notes_loss += np.mean(hist.history['composer_decoder_at_notes_loss'])

        if composer_decoder_at_instrument_output:
            total_train_composer_instrument_accuracy += np.mean(hist.history['composer_decoder_at_instruments_acc'])
            total_train_composer_instrument_loss += np.mean(hist.history['composer_decoder_at_instruments_loss'])
            

    total_train_loss = total_train_loss/train_set_size
    total_train_accuracy = total_train_accuracy/train_set_size

    total_train_notes_loss = total_train_notes_loss/train_set_size
    total_train_notes_loss_array.append(total_train_notes_loss)

    total_train_loss_array.append(total_train_loss)
    total_train_accuracy_array.append(total_train_accuracy)

    if meta_instrument:
        train_meta_instrument_accuracy = total_train_meta_instrument_accuracy/train_set_size
        train_meta_instrument_loss = total_train_meta_instrument_loss/train_set_size
        total_train_meta_instrument_accuracy_array.append(train_meta_instrument_accuracy) 
        total_train_meta_instrument_loss_array.append(train_meta_instrument_loss)
        print("Train instrument meta accuracy: ", train_meta_instrument_accuracy) 
        print("Train instrument meta loss: ", train_meta_instrument_loss)

    if meta_velocity:
        train_meta_velocity_accuracy = total_train_meta_velocity_accuracy/train_set_size
        train_meta_velocity_loss = total_train_meta_velocity_loss/train_set_size
        total_train_meta_velocity_accuracy_array.append(train_meta_velocity_accuracy) 
        total_train_meta_velocity_loss_array.append(train_meta_velocity_loss)
        if combine_velocity_and_held_notes:
            print("Train velocity meta accuracy: ", train_meta_velocity_accuracy) 
        print("Train velocity meta loss: ", train_meta_velocity_loss)

    if meta_held_notes:
        train_meta_held_notes_accuracy = total_train_meta_held_notes_accuracy/train_set_size
        train_meta_held_notes_loss = total_train_meta_held_notes_loss/train_set_size
        total_train_meta_held_notes_accuracy_array.append(train_meta_held_notes_accuracy) 
        total_train_meta_held_notes_loss_array.append(train_meta_held_notes_loss)
        print("Train held_notes meta accuracy: ", train_meta_held_notes_accuracy) 
        print("Train held_notes meta loss: ", train_meta_held_notes_loss)

    if meta_next_notes:
        train_meta_next_notes_accuracy = total_train_meta_next_notes_accuracy/train_set_size
        train_meta_next_notes_loss = total_train_meta_next_notes_loss/train_set_size
        total_train_meta_next_notes_accuracy_array.append(train_meta_next_notes_accuracy) 
        total_train_meta_next_notes_loss_array.append(train_meta_next_notes_loss)
        print("Train next_notes meta accuracy: ", train_meta_next_notes_accuracy) 
        print("Train next_notes meta loss: ", train_meta_next_notes_loss)

    if include_composer_decoder:
        train_composer_accuracy = total_train_composer_accuracy/train_set_size
        train_composer_loss = total_train_composer_loss/train_set_size
        total_train_composer_accuracy_array.append(train_composer_accuracy)
        total_train_composer_loss_array.append(train_composer_loss)
        print("Train composer accuracy: ", train_composer_accuracy) 
        print("Train composer loss: ", train_composer_loss)

    if signature_decoder:
        train_signature_accuracy = total_train_signature_accuracy/train_set_size
        train_signature_loss = total_train_signature_loss/train_set_size
        total_train_signature_accuracy_array.append(train_signature_accuracy)
        total_train_signature_loss_array.append(train_signature_loss)
        #print("Train signature accuracy: ", train_signature_accuracy) 
        print("Train signature loss: ", train_signature_loss)

    if composer_decoder_at_notes_output:
        train_composer_notes_accuracy = total_train_composer_notes_accuracy/train_set_size
        train_composer_notes_loss = total_train_composer_notes_loss/train_set_size
        total_train_composer_notes_accuracy_array.append(train_composer_notes_accuracy)
        total_train_composer_notes_loss_array.append(train_composer_notes_loss)
        print("Train composer_notes accuracy: ", train_composer_notes_accuracy) 
        print("Train composer_notes loss: ", train_composer_notes_loss)

    if composer_decoder_at_instrument_output:
        train_composer_instrument_accuracy = total_train_composer_instrument_accuracy/train_set_size
        train_composer_instrument_loss = total_train_composer_instrument_loss/train_set_size
        total_train_composer_instrument_accuracy_array.append(train_composer_instrument_accuracy)
        total_train_composer_instrument_loss_array.append(train_composer_instrument_loss)
        print("Train composer_instrument accuracy: ", train_composer_instrument_accuracy) 
        print("Train composer_instrument loss: ", train_composer_instrument_loss)



    print("Train notes accuracy: ", total_train_accuracy)
    print("Train notes loss: ", total_train_notes_loss)

    if beta>0:
        kl_loss = total_train_loss - total_train_notes_loss * 1.0
        if include_composer_decoder: kl_loss -= train_composer_loss * composer_weight
        if meta_instrument: kl_loss -= train_meta_instrument_loss * meta_instrument_weight
        if meta_velocity: kl_loss -= train_meta_velocity_loss * meta_velocity_weight
        if meta_held_notes: kl_loss -= train_meta_held_notes_loss * meta_held_notes_weight
        if meta_next_notes: kl_loss -= train_meta_next_notes_loss * meta_next_notes_weight
        if signature_decoder: kl_loss -= train_signature_loss * signature_weight
        if composer_decoder_at_notes_output: kl_loss -= train_composer_notes_loss * composer_decoder_at_notes_weight
        if composer_decoder_at_instrument_output: kl_loss -= train_composer_instrument_loss * composer_decoder_at_instrument_weight
        #since you get the value back weighted, scale back by dividing by beta
        kl_loss = kl_loss / beta
        total_train_kl_loss_array.append(kl_loss)
        print('Train KL loss: ', kl_loss)

    print("Total train loss: ", total_train_loss)

    if e % test_step is 0:
        test()
    
    if e% save_step is 0:
        print('saving model')
        autoencoder_save_path = model_path + 'autoencoder' + 'Epoch' + str(e) + model_filetype
        #autoencoder.save(autoencoder_save_path)
        autoencoder.save_weights(autoencoder_save_path)

        encoder_save_path = model_path + 'encoder' + 'Epoch' + str(e) + model_filetype
        #encoder.save(encoder_save_path)
        encoder.save_weights(encoder_save_path)

        decoder_save_path = model_path + 'decoder' + 'Epoch' + str(e) + model_filetype
        #decoder.save(decoder_save_path)
        decoder.save_weights(decoder_save_path)
        


