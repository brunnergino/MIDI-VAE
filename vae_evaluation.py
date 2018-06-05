# ----------------------------------------------------------------------------------------------
# Import dependencies
# ----------------------------------------------------------------------------------------------

from settings import *

import sys
import math
from random import shuffle
import progressbar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
import os
import numpy as np
import _pickle as pickle
import time
import csv
from collections import defaultdict

from keras.models import load_model, model_from_yaml
from keras.utils import to_categorical
from sklearn.utils import class_weight
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib2tikz import save as tikz_save
import pretty_midi as pm
import scipy

import midi_functions as mf
import vae_definition
from vae_definition import VAE
from vae_definition import KLDivergenceLayer
import data_class
from import_midi import import_midi_from_folder


# ----------------------------------------------------------------------------------------------
# Set schedule for the evaluation
# ----------------------------------------------------------------------------------------------

harmonicity_evaluations = False
frankenstein_harmonicity_evaluations = False # runs only if harmonicity_evaluations are turned on

max_new_chosen_interpolation_songs = 42
interpolation_length = 4 #how many iterations?
how_many_songs_in_one_medley = 3
noninterpolated_samples_between_interpolation = 8 #should be at least 1, otherwise it can not interpolate

max_new_sampled_interpolation_songs = 42
interpolation_song_length = 10 #how many iterations?

latent_sweep = True
num_latent_sweep_samples = 100
num_latent_sweep_evaluation_songs = 10

chord_evaluation = True
evaluate_different_sampling_regions = True
pitch_evaluation = True
max_new_sampled_songs = 100
max_new_sampled_long_songs = 100

evaluate_autoencoding_and_stuff = True
mix_with_previous = True
switch_styles = True


# ----------------------------------------------------------------------------------------------
# Model library (Change those strings to use it)
# ----------------------------------------------------------------------------------------------


model_name = 'your_model_name/'
epoch = 410

pitches_classifier_model_path = './models/clustering/-/'
pitches_classifier_model_name = 'modelEpoch?.pickle'
pitches_classifier_model = load_model(pitches_classifier_model_path+pitches_classifier_model_name)
pitches_classifier_model_weight = 0.999 - 0.5 #subtract 0.5 since you would want to weight a random model with 0

velocity_classifier_model_path = './models/velocityclustering/1521669531-num_layers_2_maxlen_64_otns_False_lstmsize_256_trainsize_909_testsize_104_thresh_0.5_scale_False/'
velocity_classifier_model_name = 'modelEpoch?.pickle'
velocity_classifier_model = load_model(velocity_classifier_model_path+velocity_classifier_model_name)
velocity_classifier_model_weight = 0.999 - 0.5

instrument_classifier_model_path = './models/instrumentclustering/-/'
instrument_classifier_model_name = 'modelEpoch?.pickle'
instrument_classifier_model = load_model(instrument_classifier_model_path+instrument_classifier_model_name)
instrument_classifier_model_weight = 0.999 - 0.5



if test_train_set:
    set_string = 'train/'
else:
    set_string = 'test/'

model_path = 'models/autoencode/vae/' + model_name
save_folder = 'autoencode_midi/vae/' + model_name[:10] + '/' + set_string

 


if not os.path.exists(save_folder):
    os.makedirs(save_folder)   


def ensemble_prediction(Y,I,V):

    pitch_prediction = pitches_classifier_model.predict(Y)
    instrument_prediction = instrument_classifier_model.predict(I)
    velocity_prediction = velocity_classifier_model.predict(V)

    weighted_prediction = (pitch_prediction * pitches_classifier_model_weight + instrument_prediction * instrument_classifier_model_weight + velocity_prediction * velocity_classifier_model_weight) / (pitches_classifier_model_weight + instrument_classifier_model_weight + velocity_classifier_model_weight)
    return weighted_prediction

# ----------------------------------------------------------------------------------------------
# Evaluation settings
# ----------------------------------------------------------------------------------------------

model_filetype = '.pickle'


max_plots_per_song = 3

BPM = 100

shuffle = False
composer_decoder_latent_size = 10

assert(output_length > 0)

verbose = False

sample_method = 'argmax' #choice, argmax

# ----------------------------------------------------------------------------------------------
# Import and preprocess data
# ----------------------------------------------------------------------------------------------

print('loading data...')
# Get Train and test sets

if rolls:
    folder = source_folder
else:
    folder = roll_folder

V_train, V_test, D_train, D_test, T_train, T_test, I_train, I_test, Y_train, Y_test, X_train, X_test, C_train, C_test, train_paths, test_paths = import_midi_from_folder(folder)

train_set_size = len(X_train)
test_set_size = len(X_test)

print(len(train_paths))
print(len(test_paths))
print(C_test)

# ----------------------------------------------------------------------------------------------
# Simple statistics on train and test set
# ----------------------------------------------------------------------------------------------


total_train_songs_per_class = [0 for _ in range(num_classes)]
total_train_samples_per_class = [0 for _ in range(num_classes)]

total_test_songs_per_class = [0 for _ in range(num_classes)]
total_test_samples_per_class = [0 for _ in range(num_classes)]

for i, C in enumerate(C_train):
    total_train_songs_per_class[C] += 1
    total_train_samples_per_class[C] += X_train[i].shape[0]

for i, C in enumerate(C_test):
    total_test_songs_per_class[C] += 1
    total_test_samples_per_class[C] += X_test[i].shape[0]

print("Total train songs per class: ", total_train_songs_per_class)
print("Total train samples per class: ", total_train_samples_per_class)
print("Total test songs per class: ", total_test_songs_per_class)
print("Total test samples per class: ", total_test_samples_per_class)

print("Classes", classes)
print("Model name", model_name)
print("Test on train set", test_train_set)
input("Correct settings?")

# ----------------------------------------------------------------------------------------------
# Harmonicity statistics
# ----------------------------------------------------------------------------------------------

if harmonicity_evaluations:

    if frankenstein_harmonicity_evaluations:
        def spm_based_on_random_pitches(total_evaluations=1000):

            spms = np.zeros((total_evaluations, max_voices, max_voices))
            for i in range(total_evaluations):

                bar = np.zeros((1,output_length, new_num_notes))
                notes_per_step_maximum = 5
                #fill bar with random notes
                for step in range(output_length):
                    for _ in range(notes_per_step_maximum):
                        #silent every third time on average
                        silent = np.random.randint(3) == 0
                        if not silent:
                            pitch = np.random.randint(new_num_notes)
                            bar[0, step, pitch] = 1

                score_pair_matrix = data_class.get_harmonicity_scores_for_each_track_combination(bar)
                spms[i] = score_pair_matrix
            return np.nanmean(spms, axis=0)

        spm = spm_based_on_random_pitches()
        print("Harmonicity score based on random pitches :\n", spm)

        def frankenstein_spm_based_on_Y_list(Y_list, total_evaluations=1000):
            num_songs = len(Y_list)

            spms = np.zeros((total_evaluations, max_voices, max_voices))
            for i in range(total_evaluations):

                #pick max_voices different songs
                song_choices = np.random.choice(num_songs, max_voices, replace=False)

                frankenstein_bar = np.zeros((1, output_length, new_num_notes))
                for voice, song_choice in enumerate(song_choices):
                    Y = Y_list[song_choice]
                    #pick a random bar
                    num_bars = Y.shape[0]
                    bar_choice = np.random.randint(num_bars)
                    picked_bar = np.copy(Y[bar_choice])
                    if include_silent_note:
                        picked_bar = picked_bar[:, :-1]
                    #fill the frankenstein_bar
                    frankenstein_bar[0, voice::max_voices, :] = picked_bar[0::max_voices,:]

                score_pair_matrix = data_class.get_harmonicity_scores_for_each_track_combination(frankenstein_bar)
                spms[i] = score_pair_matrix
            return np.nanmean(spms, axis=0)

        for C in range(num_classes):

            indices = [i for i, x in enumerate(C_train) if x == C]
            Y_train_for_this_class = np.copy([Y_train[i] for i in indices])
            spm = frankenstein_spm_based_on_Y_list(Y_train_for_this_class)
            print("Frankenstein train spm for class " + classes[C] + ":\n", spm)

            indices = [i for i, x in enumerate(C_test) if x == C]
            Y_test_for_this_class = np.copy([Y_test[i] for i in indices])
            spm = frankenstein_spm_based_on_Y_list(Y_test_for_this_class)
            print("Frankenstein test spm for class " + classes[C] + ":\n", spm)

        spm = frankenstein_spm_based_on_Y_list(Y_train)
        print("Frankenstein train spm for whole set :\n", spm)

        spm = frankenstein_spm_based_on_Y_list(Y_test)
        print("Frankenstein test spm for whole set :\n", spm)

    spm_train = np.zeros((len(Y_train), max_voices, max_voices))
    for i, Y in enumerate(Y_train):

        bars= np.copy(Y)
        if include_silent_note:
            bars = bars[:,:,:-1] 

        score_pair_matrix = data_class.get_harmonicity_scores_for_each_track_combination(bars)
        spm_train[i] = score_pair_matrix

    spm_train_mean = np.nanmean(spm_train, axis=0)
    print("Score pair matrix train mean: \n", spm_train_mean)

    spm_train_mean_for_each_class = []
    for C in range(num_classes):
        spms_for_this_class = spm_train[np.where(np.asarray(C_train) == C)]
        m = np.nanmean(np.asarray(spms_for_this_class), axis=0)
        print("Score pair matrix for train set in class " + classes[C] + ":\n", m)
        spm_train_mean_for_each_class.append(m)

    spm_test = np.zeros((len(Y_test),max_voices, max_voices))
    for i, Y in enumerate(Y_test):
 
        bars= np.copy(Y)
        if include_silent_note:
            bars = bars[:,:,:-1] 
        score_pair_matrix = data_class.get_harmonicity_scores_for_each_track_combination(bars)
        spm_test[i] = score_pair_matrix

    spm_test_mean = np.nanmean(spm_test, axis=0)
    print("\nScore pair matrix test mean: \n", spm_test_mean)

    spm_test_mean_for_each_class = []
    for C in range(num_classes):
        spms_for_this_class = spm_test[np.where(np.asarray(C_test) == C)]
        m = np.nanmean(np.asarray(spms_for_this_class), axis=0)
        print("Score pair matrix for test set in class " + classes[C] + ":\n", m)
        spm_test_mean_for_each_class.append(m)

# ----------------------------------------------------------------------------------------------
# Instruments (midi programs) statistics
# ----------------------------------------------------------------------------------------------


programs_for_each_class = [[] for _ in range(num_classes)]
for train_song_num in range(len(Y_train)):
    C = C_train[train_song_num]
    I = I_train[train_song_num]
    programs = data_class.instrument_representation_to_programs(I, instrument_attach_method)
    for program in programs:
        if not program in programs_for_each_class[C]:
            programs_for_each_class[C].append(program)

print(programs_for_each_class)


#calculate how many programs have to be switched on average for a style change on the training set
all_programs_plus_length_for_each_class = [[] for _ in range(num_classes)]
total_programs_for_each_class = [0 for _ in range(num_classes)]
program_probability_dict_for_each_class = [dict() for _ in range(num_classes)]
for i in range(len(I_train)):
    num_samples = X_train[i].shape[0] #get the number of samples to know how many splitted songs there are for this original song
    I = I_train[i]
    C = C_train[i]
    programs = data_class.instrument_representation_to_programs(I, instrument_attach_method)
    all_programs_plus_length_for_each_class[C].append((programs, num_samples))
    total_programs_for_each_class[C] += num_samples * max_voices
    for program in programs:
        program_probability_dict_for_each_class[C][program] = program_probability_dict_for_each_class[C].get(program, 0) + num_samples

for d in program_probability_dict_for_each_class:
    print(d)

#divide by total number of programs to get a probability for each key
for C, d in enumerate(program_probability_dict_for_each_class):
    for k in d.keys():
        d[k] /= total_programs_for_each_class[C]
            

for d in program_probability_dict_for_each_class:
    print(d)

#enlist the possible instruments for each class
if instrument_attach_method == '1hot-category' or 'khot-category':
    possible_programs = list(range(0,127,8))
else:
    possible_programs = list(range(0,127))

#calculate the random probability for each class
print("Calculate how probable your instrument picks are if you pick them completely random: ")
for C, class_name in enumerate(classes):
    probabilities_for_this_class = []
    for program in possible_programs:
        probabilities_for_this_class.append(program_probability_dict_for_each_class[C].get(program, 0))
    print("Random probability for class " + class_name + ": ", np.mean(probabilities_for_this_class))
    #of course, this is the same as 1/len(possible_programs)


#calculate the instrument probability for each class
print("Calculate how probable your instrument picks are if you don't switch any instrument and stay in the same class: ")
for C, class_name in enumerate(classes):
    probability_for_this_class = 0
    for (programs, length) in all_programs_plus_length_for_each_class[C]:
        for program in programs:
            probability_for_this_class += length * program_probability_dict_for_each_class[C].get(program, 0)
    probability_for_this_class /= total_programs_for_each_class[C]
    print("Same probability for class " + class_name + ": ", probability_for_this_class)


#calculate the instrument probability for each class
print("Calculate how probable your instrument picks are in another classif you don't switch any instrument: ")
for C, class_name in enumerate(classes):
    
    for C_switch, class_name_switch in enumerate(classes):
        if C != C_switch:
            probability_for_other_class = 0
            for (programs, length) in all_programs_plus_length_for_each_class[C]:
                for program in programs:
                    probability_for_other_class += length * program_probability_dict_for_each_class[C_switch].get(program, 0)
            probability_for_other_class /= total_programs_for_each_class[C]
            print("Probability that a program-pick from class " + class_name + " is occuring class " + class_name_switch +" : ", probability_for_other_class)

for C, class_name in enumerate(classes):
    programs_plus_length_for_this_class = all_programs_plus_length_for_each_class[C]
    print(len(programs_plus_length_for_this_class))
    for C_switch, class_name_switch in enumerate(classes):
        if C_switch != C:
            print("Calculating how many instruments switches have to be made from " + class_name + " to " + class_name_switch)
            same = 0.0
            different = 0.0
            programs_plus_length_for_other_class = all_programs_plus_length_for_each_class[C_switch]
            for programs, length in programs_plus_length_for_this_class:
                for programs_switch, length_switch in programs_plus_length_for_other_class:
                    for this_program, other_program in zip(programs, programs_switch):
                        if this_program == other_program:
                            same += length * length_switch
                        else:
                            different += length * length_switch
            print("Switch percentage: ", different / (same + different))


# ----------------------------------------------------------------------------------------------
# Prepare signature vectors
# ----------------------------------------------------------------------------------------------

S_train_for_each_class = [[] for _ in range(num_classes)]
S_test_for_each_class = [[] for _ in range(num_classes)]
all_S = []
S_train = []
for train_song_num in range(len(Y_train)):
    Y = Y_train[train_song_num]
    C = C_train[train_song_num]
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
    S_train_for_each_class[C].extend(signature_vectors)

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
S_test = []
for test_song_num in range(len(Y_test)):
    Y = Y_test[test_song_num]
    C = C_test[test_song_num]
    num_samples = Y.shape[0]
    signature_vectors = np.zeros((num_samples, signature_vector_length))
    normalized_signature_vectors = np.zeros((num_samples, signature_vector_length))
    for sample in range(num_samples):
        poly_sample = data_class.monophonic_to_khot_pianoroll(Y[sample], max_voices)
        if include_silent_note:
            poly_sample = poly_sample[:,:-1]
        signature = data_class.signature_from_pianoroll(poly_sample)
        normalized_signature_vectors[sample] = signature
        signature = (signature - mean_signature) / std_signature
        normalized_signature_vectors[sample] = signature
    normalized_S_test.append(signature_vectors)
    S_test_for_each_class[C].extend(signature_vectors)
    S_test.append(signature_vectors)


normalized_S_test = np.asarray(normalized_S_test)
S_test = np.asarray(S_test)

normalized_S_train = np.asarray(normalized_S_train)
S_test = np.asarray(S_train)

S_train_for_each_class = np.asarray(S_train_for_each_class)
S_test_for_each_class = np.asarray(S_test_for_each_class)


# ----------------------------------------------------------------------------------------------
# Build VAE and load from weights
# ----------------------------------------------------------------------------------------------

#You have to create the model again with the same parameters as in training and set the weights manually
#There is an issue with storing the model with the recurrentshop extension

if do_not_sample_in_evaluation:
    e = 0.0
else:
    e = epsilon_std


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
    epsilon_std=e, 
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

autoencoder = model.autoencoder
autoencoder.load_weights(model_path+'autoencoder'+'Epoch'+str(epoch)+'.pickle', by_name=False)

encoder = model.encoder
encoder.load_weights(model_path+'encoder'+'Epoch'+str(epoch)+'.pickle', by_name=False)

decoder = model.decoder
decoder.load_weights(model_path+'decoder'+'Epoch'+str(epoch)+'.pickle', by_name=False)


print(encoder.summary())
print(decoder.summary())
print(autoencoder.summary())

if reset_states:
    autoencoder.reset_states()
    encoder.reset_states()
    decoder.reset_states()


# ----------------------------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------------------------


#spherical linear interpolation
def slerp(p0, p1, t):
    omega = arccos(dot(p0/np.linalg.norm(p0), p1/np.linalg.norm(p1)))
    so = sin(omega)
    return sin((1.0-t)*omega) / so * p0 + sin(t*omega)/so * p1

def linear_interpolation(p0, p1, t):
    return p0 * (1.0-t) + p1 * t




    

def split_song_back_to_samples(X, length):
    number_of_splits = int(X.shape[0] / length)
    splitted_songs = np.split(X, number_of_splits)
    return splitted_songs

#I_pred instrument prediction of shape (num_samples, max_voices, different_instruments)
#returns list of program numbers of length max_voices
def vote_for_programs(I_pred):
    program_voting_dict_for_each_voice = [dict() for _ in range(max_voices)]
    for instrument_feature_matrix in I_pred:
        programs = data_class.instrument_representation_to_programs(instrument_feature_matrix, instrument_attach_method)

        for voice, program in enumerate(programs):
            program_voting_dict_for_each_voice[voice][program] = program_voting_dict_for_each_voice[voice].get(program,0) + 1

    #determine mixed_programs_for_whole_song by taking the instruments for each track with the most occurence in the mixed predictions
    programs_for_whole_long_song = []
    for voice in range(max_voices):
        best_program = 0
        highest_value = 0
        for k in program_voting_dict_for_each_voice[voice].keys():
            if program_voting_dict_for_each_voice[voice][k] > highest_value:
                best_program = k 
                highest_value = program_voting_dict_for_each_voice[voice][k]
        programs_for_whole_long_song.append(best_program)

    return programs_for_whole_long_song

def prepare_for_drawing(Y, V=None):
    #use V to make a grey note if it is more silent
    newY = np.copy(Y)
    if V is not None:
        for step in range(V.shape[0]):
            
            if V[step] > velocity_threshold_such_that_it_is_a_played_note:
                velocity = (V[step] - velocity_threshold_such_that_it_is_a_played_note) * MAX_VELOCITY
                newY[step,:] *= velocity
            else:
                if step > max_voices:
                    previous_pitch = np.argmax(newY[step-max_voices])
                    current_pitch = np.argmax(newY[step])
                    if current_pitch != previous_pitch:
                        newY[step,:] = 0
                    else:
                        newY[step,:] = newY[step-max_voices,:]
                else:
                    newY[step,:] = 0

        Y_poly = data_class.monophonic_to_khot_pianoroll(newY, max_voices, set_all_nonzero_to_1=False)
    else:
        Y_poly = data_class.monophonic_to_khot_pianoroll(newY, max_voices)
    return np.transpose(Y_poly)


def restructure_song_to_fit_more_instruments(Y, I_list, V, D):

    num_samples = len(I_list)
    Y_final = np.zeros((num_samples * output_length * num_samples, Y.shape[1]))
    V_final = np.zeros((num_samples * output_length * num_samples,))
    D_final = np.zeros((num_samples * output_length * num_samples,))
    final_programs = []
    for sample, I in enumerate(I_list):
        programs = data_class.instrument_representation_to_programs(I, instrument_attach_method)
        final_programs.extend(programs)

        
        for step in range(output_length//max_voices):
            for voice in range(max_voices):
                Y_final[sample * output_length * num_samples + step * num_samples * max_voices + voice,:] = Y[sample *output_length+ step*max_voices + voice,:]
                V_final[sample * output_length * num_samples + step * num_samples * max_voices + voice] = V[sample *output_length+ step*max_voices + voice]
                D_final[sample * output_length * num_samples + step * num_samples * max_voices + voice] = D[sample *output_length + step*max_voices + voice]
    return Y_final, final_programs, V_final, D_final


# ----------------------------------------------------------------------------------------------
# Save latent train lists
# ----------------------------------------------------------------------------------------------

print("Saving latent train lists...")



train_representation_list = []
all_z = []
for train_song_num in range(len(X_train)):

    #create dataset
    song_name = train_paths[train_song_num].split('/')[-1]
    song_name = song_name.replace('mid.pickle', '')
    X = X_train[train_song_num]
    C = C_train[train_song_num] 
    I = I_train[train_song_num]
    V = V_train[train_song_num]
    D = D_train[train_song_num]

    encoder_input_list = vae_definition.prepare_encoder_input_list(X,I,V,D)
    #get the latent representation of every song part
    encoded_representation = encoder.predict(encoder_input_list, batch_size=batch_size, verbose=False)
    train_representation_list.append(encoded_representation)
    all_z.extend(encoded_representation)
    train_save_folder = save_folder
    if not test_train_set:
        train_save_folder = save_folder[:-5] + 'train/'
    if not os.path.exists(train_save_folder+ classes[C]+'/'):
        os.makedirs(train_save_folder + classes[C]+'/') 
    if save_anything: np.save(train_save_folder + classes[C]+'/'+'z_' + song_name, encoded_representation)

z_mean_train = np.mean(np.asarray(all_z))
z_std_train = np.std(np.asarray(all_z))

print("z mean train: ", z_mean_train)
print("z std train: ", z_std_train)


# ----------------------------------------------------------------------------------------------
# Generation of interpolation songs from the chosen training or test set
# ----------------------------------------------------------------------------------------------

sample_method = 'argmax'

assert(noninterpolated_samples_between_interpolation > 0)

for song_num in range(max_new_chosen_interpolation_songs):

    print("Producing chosen interpolation song ", song_num)

    medley_name = 'medley_songs_' + str(how_many_songs_in_one_medley) + '_original_' + str(noninterpolated_samples_between_interpolation) + '_bridge_' + str(interpolation_length) + '_'
    Y_list = []
    V_list = []
    D_list = []
    I_list = []

    info_dict = dict()

    previous_medley_z = None
    C = 0
    previous_latent_rep = np.zeros((1,latent_dim))
    S = np.zeros((1, signature_vector_length))


    for medley_song_num in range(how_many_songs_in_one_medley):

        if test_train_set:
            #chose random train song that is long enough
            song_num = np.random.randint(train_set_size)
            while X_train[song_num].shape[0] <= noninterpolated_samples_between_interpolation:
                song_num = np.random.randint(train_set_size)
            X = X_train[song_num]
            I = I_train[song_num] 
            C = C_train[song_num] 
            V = V_train[song_num]
            D = D_train[song_num]

            song_name = train_paths[song_num].split('/')[-1]
            song_name = song_name.replace('mid.pickle', '')
        else:
            #chose random train song that is long enough
            song_num = np.random.randint(test_set_size)
            while X_test[song_num].shape[0] <= noninterpolated_samples_between_interpolation:
                song_num = np.random.randint(test_set_size)
            X = X_test[song_num]
            I = I_test[song_num]
            C = C_test[song_num]
            V = V_test[song_num]
            D = D_test[song_num]
            song_name = test_paths[song_num].split('/')[-1]
            song_name = song_name.replace('mid.pickle', '')

        #chose random sample
        sample_num = np.random.randint(X.shape[0]) 
        if sample_num < noninterpolated_samples_between_interpolation and medley_song_num == 0:
            sample_num = noninterpolated_samples_between_interpolation
        elif sample_num >= X.shape[0] - noninterpolated_samples_between_interpolation:
            sample_num = X.shape[0] - noninterpolated_samples_between_interpolation - 1  

        medley_name += '_' + str(song_num) + '-' + str(sample_num)

        info_dict["song_name_" + str(medley_song_num)] = song_name
        info_dict["sample_num_" + str(medley_song_num)] = sample_num
        info_dict["programs_" + str(medley_song_num)] = data_class.instrument_representation_to_programs(I, instrument_attach_method)

        #calculate which samples are needed
        if medley_song_num == 0:
            sample_list = range(sample_num-noninterpolated_samples_between_interpolation,sample_num)
        else:
            sample_list = range(sample_num , sample_num + noninterpolated_samples_between_interpolation)

        X = np.copy(X[sample_list])
        V = np.copy(V[sample_list])
        D = np.copy(D[sample_list])

        if X.ndim == 2:
            X = np.expand_dims(X, axis=0)
        if V.ndim == 1:
            V = np.expand_dims(V, axis=0)
        if D.ndim == 1:
            D = np.expand_dims(D, axis=0)

        encoder_input_list = vae_definition.prepare_encoder_input_list(X,I,V,D)
        R = encoder.predict(encoder_input_list, batch_size=batch_size, verbose=False)

        if previous_medley_z is not None:

            for i in range(interpolation_length):
                z = linear_interpolation(previous_medley_z, R[0], i/float(interpolation_length))
                z = np.expand_dims(z, axis=0)
                interpolation_input_list = vae_definition.prepare_decoder_input(z, C, S, previous_latent_rep)
                decoder_outputs = decoder.predict(interpolation_input_list, batch_size=batch_size, verbose=False)
                Y, I, V, D, N = vae_definition.process_decoder_outputs(decoder_outputs, sample_method)
                Y_list.extend(Y)
                I_list.extend(I)
                V_list.extend(V)
                D_list.extend(D)

                info_dict["programs_" + str(medley_song_num) + "_interpolation_" +str(i)] = data_class.instrument_representation_to_programs(I[0], instrument_attach_method)

                previous_latent_rep = z

        for i in range(R.shape[0]):
            z = R[i]
            z = np.expand_dims(z, axis=0)
            interpolation_input_list = vae_definition.prepare_decoder_input(z, C, S, previous_latent_rep)
            decoder_outputs = decoder.predict(interpolation_input_list, batch_size=batch_size, verbose=False)
            Y, I, V, D, N = vae_definition.process_decoder_outputs(decoder_outputs, sample_method)
            Y_list.extend(Y)
            I_list.extend(I)
            V_list.extend(V)
            D_list.extend(D)

            previous_latent_rep = z

        previous_medley_z = R[-1]

    programs_for_whole_long_song = vote_for_programs(I_list)

    Y_list = np.asarray(Y_list)
    D_list = np.asarray(D_list)
    V_list = np.asarray(V_list)

    if save_anything:
        with open(save_folder + medley_name + "_info.txt", "w", encoding='utf-8') as text_file:
            for k, v in info_dict.items():
                text_file.write(k + ": %s" % v + '\n')

    if save_anything: data_class.draw_pianoroll(prepare_for_drawing(Y_list, V_list), name=medley_name, show=False, save_path=save_folder +medley_name)
    Y_all_programs, all_programs, V_all_programs, D_all_programs = restructure_song_to_fit_more_instruments(Y_list, I_list, V_list, D_list)
    if save_anything: mf.rolls_to_midi(Y_all_programs, all_programs, save_folder, medley_name, BPM, V_all_programs, D_all_programs)



# ----------------------------------------------------------------------------------------------
# Generation of random interpolation songs
# ----------------------------------------------------------------------------------------------

sample_method = 'argmax'

for song_num in range(max_new_sampled_interpolation_songs):

    print("Producing random interpolation song ", song_num)

    random_code_1 = np.random.normal(loc=0.0, scale=z_std_train, size=(1,latent_dim))
    random_code_2 = np.random.normal(loc=0.0, scale=z_std_train, size=(1,latent_dim))

    C = 0

    Y_list = []
    V_list = []
    D_list = []
    I_list = []
    previous_latent_rep = np.zeros((1,latent_dim))
    S = np.zeros((1, signature_vector_length))

    
    for i in range(interpolation_song_length+1):
        R = linear_interpolation(random_code_1, random_code_2, i/float(interpolation_song_length))
        interpolation_input_list = vae_definition.prepare_decoder_input(R, C, S, previous_latent_rep)
        decoder_outputs = decoder.predict(interpolation_input_list, batch_size=batch_size, verbose=False)
        Y, I, V, D, N = vae_definition.process_decoder_outputs(decoder_outputs, sample_method)
        Y_list.extend(Y)
        I_list.extend(I)
        V_list.extend(V)
        D_list.extend(D)

        previous_latent_rep = R

    programs_for_whole_long_song = vote_for_programs(I_list)

    Y_list = np.asarray(Y_list)
    D_list = np.asarray(D_list)
    V_list = np.asarray(V_list)



    if save_anything: data_class.draw_pianoroll(prepare_for_drawing(Y_list, V_list), name='random_interpolation_' + str(song_num) + '_length_' + str(interpolation_song_length), show=False, save_path=save_folder +'random_interpolation_' + str(song_num)+'_length_' + str(interpolation_song_length))
    if save_anything: mf.rolls_to_midi(Y_list, programs_for_whole_long_song, save_folder, 'random_interpolation_' + str(song_num) + '_length_' + str(interpolation_song_length), BPM, V_list, D_list)
    Y_all_programs, all_programs, V_all_programs, D_all_programs = restructure_song_to_fit_more_instruments(Y_list, I_list, V_list, D_list)
    if save_anything: mf.rolls_to_midi(Y_all_programs, all_programs, save_folder, 'random_interpolation_' + str(song_num) + '_length_' + str(interpolation_song_length) + '_all_programs', BPM, V_all_programs, D_all_programs)


# ----------------------------------------------------------------------------------------------
# Latent list helper functions
# ----------------------------------------------------------------------------------------------

#get points around 0 with sigma that look like this:  .  .  .  . . ... . .  .   .  . 
#range end: between 0.5 and 1.0
#evaluations_per_dimension how many samples to give back / 2
#sigma: std of normal distribution that needs to be 'sampled' from
def get_normal_distributed_values(range_end, evaluations_per_dimension, sigma, evaluate_postive_and_negative):
    values = []
    range_end = float(range_end) #make sure you have a float, otherwise the division by dimension will result in int

    cdf_values = np.linspace(0.5, 0.5 + range_end, evaluations_per_dimension)
    for cdf in cdf_values:
        x = scipy.stats.norm.ppf(cdf, loc=0.0, scale=sigma)
        if x != 0:
            if evaluate_postive_and_negative:
                values.append(-x)
            values.append(x)
        else:
            values.append(x)
    return sorted(values)

def save_to_summary(args, summary_dict):
    name, strength, probability = args
    summary_dict[name] = (strength, probability)
    
def get_strength_probability_direction_for_value_list(value_list):
    if len(value_list) > 0:
        
        #determine the order
        if np.mean(value_list[:len(value_list)//2]) > np.mean(value_list[len(value_list)//2:]):
            #descending order -> switch order
            value_list = value_list[::-1]
            direction = 'descending'
        else:
            direction = 'ascending'
    
        #calculate strength as a mean of the difference of these values
        differences_value_list = np.asarray(value_list[1:]) - np.asarray(value_list[:-1])
        strength = np.mean(differences_value_list)
        
        #calculate the probability that this 
        correct_ascending = 0
        incorrect_ascending = 0
        previous_value = value_list[0]
        for value in value_list[1:]:
            if value >= previous_value:
                correct_ascending += 1
            else:
                incorrect_ascending += 1
            previous_value = value  
        if (correct_ascending + incorrect_ascending) > 0:
            probability = correct_ascending / (correct_ascending + incorrect_ascending)
        else:
            probability = 0
    
    else:
        direction='ascending'
        strength = 0.0
        probability = 0.0
    return strength, probability, direction
    
#statistic_name: which statistic to test, can be 'mean', 'median' 'std', 'max', 'min', 'range'
def evaluate_statistic_value(splitted_list, value_name, statistic_name):
    statistic_values = []
    for value_list in splitted_list:
        if len(value_list) > 0:
            if statistic_name == 'mean':
                statistic_values.append(np.mean(value_list))
            elif statistic_name == 'median':
                statistic_values.append(np.median(value_list))
            elif statistic_name == 'std':
                statistic_values.append(np.std(value_list))
            elif statistic_name == 'max':
                statistic_values.append(np.max(value_list))
            elif statistic_name == 'min':
                statistic_values.append(np.min(value_list))
            elif statistic_name == 'range':
                statistic_values.append(np.max(value_list) - np.min(value_list))
            
    strength, probability, direction = get_strength_probability_direction_for_value_list(statistic_values)

    return (statistic_name + "_" + value_name + "_" +direction, strength, probability)



def evaluate_count_of_values(splitted_list, value_name, specific_value=None):
    count_of_values = []
    for value_list in splitted_list:
        if specific_value is None:
            count_of_values.append(len(value_list))
        else:
            count_of_values.append(value_list.count(specific_value))
            
    strength, probability, direction = get_strength_probability_direction_for_value_list(count_of_values)

    return ("total_count_of_" + value_name + "_" + direction, strength, probability)

def evaluate_change_of_values(splitted_list, value_name):
    
    previous_values = splitted_list[0]
    change_counter = 0.0
    total_counter = 0.0
    for values in splitted_list[1:]:
        for v_current, v_previous in zip(values, previous_values):
            total_counter += 1.0
            if v_current != v_previous:
                change_counter += 1.0
        previous_values = values
    
    if total_counter > 0:
        strength = change_counter / total_counter
    else:
        strength = 0.0
    probability = 1.0
    return ("total_change_of_" + value_name, strength, probability)

def run_all_statistics(list_of_lists, name, d):
    save_to_summary(evaluate_statistic_value(list_of_lists, name, 'mean'), d)  
    save_to_summary(evaluate_statistic_value(list_of_lists, name, 'median'), d)   
    save_to_summary(evaluate_statistic_value(list_of_lists, name, 'min'), d)   
    save_to_summary(evaluate_statistic_value(list_of_lists, name, 'max'), d) 
    save_to_summary(evaluate_statistic_value(list_of_lists, name, 'range'), d)   
    save_to_summary(evaluate_statistic_value(list_of_lists, name, 'std'), d) 
    
            

def evaluate_velocityroll(input_velocityroll):
    dimension_summary_dict = dict()
    assert(input_velocityroll.shape[0] % (output_length//max_voices) == 0)
    
    note_starts = np.where(input_velocityroll > velocity_threshold_such_that_it_is_a_played_note)[0]
    
    if len(note_starts) > 0:
        
        number_of_splits = int(input_velocityroll.shape[0] / output_length)
        splitted_velocityroll = np.split(input_velocityroll, number_of_splits)

        splitted_note_start_lists = []
        splitted_velocity_lists = []
        for velocityroll in splitted_velocityroll:
            note_starts = np.where(velocityroll > velocity_threshold_such_that_it_is_a_played_note)[0]
            splitted_note_start_lists.append(note_starts)
            splitted_velocity_lists.append(list(velocityroll[note_starts]))
            
        predictions = velocity_classifier_model.predict(np.expand_dims(splitted_velocityroll,2))
        prediction_list_for_class_0 = []
        for prediction in predictions:
            prediction_list_for_class_0.append([prediction[0]])
        save_to_summary(evaluate_statistic_value(prediction_list_for_class_0, 'velocitystyle', 'mean'), dimension_summary_dict)

            
        run_all_statistics(splitted_velocity_lists, 'velocity', dimension_summary_dict)
        run_all_statistics(splitted_note_start_lists, 'note_starts', dimension_summary_dict)
        
        save_to_summary(evaluate_count_of_values(splitted_note_start_lists, 'note_starts'), dimension_summary_dict)
           
    return dimension_summary_dict


def evaluate_pitchroll(input_pianoroll):
    dimension_summary_dict = dict()
    assert(input_pianoroll.shape[0] % (output_length//max_voices) == 0)
    
    total_notes = np.count_nonzero(input_pianoroll)
    
    if total_notes > 0:
        
        
        input_song = data_class.monophonic_to_khot_pianoroll(input_pianoroll, max_voices)
        input_song = np.asarray(input_song)


        number_of_splits = int(input_song.shape[0] / (output_length//max_voices))
        splitted_songs = np.split(input_song, number_of_splits)

        splitted_song_lists = []
        for song in splitted_songs:
            song_list = []
            for step in range(song.shape[0]):
                notes = list(song[step].nonzero()[0])
                for note in notes:
                    song_list.append(note)
            splitted_song_lists.append(song_list)

        run_all_statistics(splitted_song_lists, 'pitch', dimension_summary_dict)
        
        save_to_summary(evaluate_count_of_values(splitted_song_lists, 'pitch'), dimension_summary_dict)
        save_to_summary(evaluate_count_of_values(splitted_song_lists, 'specificpitch35', 35), dimension_summary_dict)
        save_to_summary(evaluate_count_of_values(splitted_song_lists, 'specificpitch39', 39), dimension_summary_dict)
        
        splitted_unrolled_songs = np.asarray(np.split(input_pianoroll, number_of_splits))
        if include_silent_note:
            splitted_unrolled_songs = np.append(splitted_unrolled_songs, np.zeros((splitted_unrolled_songs.shape[0], splitted_unrolled_songs.shape[1], 1)), axis=2)
            for sample in range(splitted_unrolled_songs.shape[0]):
                for step in range(splitted_unrolled_songs.shape[1]):
                    if np.sum(splitted_unrolled_songs[sample,step]) == 0:
                        splitted_unrolled_songs[sample,step, -1] = 1   
        predictions = pitches_classifier_model.predict(splitted_unrolled_songs)
        prediction_list_for_class_0 = []
        for prediction in predictions:
            prediction_list_for_class_0.append([prediction[0]])
        save_to_summary(evaluate_statistic_value(prediction_list_for_class_0, 'pitchstyle', 'mean'), dimension_summary_dict)
        
        
    return dimension_summary_dict

def evaluate_instrumentlist(instrument_list):
    dimension_summary_dict = dict()
    program_list = []
    for instrument_matrix in instrument_list:
        programs = data_class.instrument_representation_to_programs(instrument_matrix, instrument_attach_method)
        program_list.append(programs)
        
    predictions = instrument_classifier_model.predict(instrument_list)
    prediction_list_for_class_0 = []
    for prediction in predictions:
        prediction_list_for_class_0.append([prediction[0]])
    save_to_summary(evaluate_statistic_value(prediction_list_for_class_0, 'instrumentstyle', 'mean'), dimension_summary_dict)
    
    
    save_to_summary(evaluate_change_of_values(program_list, 'instruments'), dimension_summary_dict) 
    save_to_summary(evaluate_count_of_values(program_list, 'pianos', 0), dimension_summary_dict) #check for occurence of piano
    return dimension_summary_dict



# ----------------------------------------------------------------------------------------------
# Latent sweep
# ----------------------------------------------------------------------------------------------

#evaluate_postive_and_negative doubles the number of evaluations per dimension
def latent_sweep_over_all_dimensions(start_latent_vector, song_name='', range_end_in_stds=1.0, sigma=1.0, evaluations_per_dimension=5, evaluate_postive_and_negative=True, create_midi_for_z_song_list=None, create_midi_name_list=None):
    
    def get_sweep_output_for_values_on_dim(z, values, dim):
        Y_list = []
        I_list = []
        V_list = []
        D_list = []
        N_list = []

        for value in values:
            new_latent_vector = np.copy(z)
            new_latent_vector[:, dim] = value

            C = 0
            S = np.zeros((z.shape[0], signature_vector_length))
            sweep_input_list = vae_definition.prepare_decoder_input(new_latent_vector, C, S)

            decoder_outputs = decoder.predict(sweep_input_list, batch_size=batch_size, verbose=False)

            Y_pred, I_pred, V_pred, D_pred, N_pred = vae_definition.process_decoder_outputs(decoder_outputs, sample_method)
            
            Y_list.extend(Y_pred)
            I_list.extend(I_pred)
            V_list.extend(V_pred)
            D_list.extend(D_pred)
            N_list.extend(N_pred)

        Y_list = np.asarray(Y_list)
        I_list = np.asarray(I_list)
        V_list = np.asarray(V_list)
        D_list = np.asarray(D_list)
        N_list = np.asarray(N_list)
        
        return Y_list, I_list, V_list, D_list, N_list
    
    
    num_samples = start_latent_vector.shape[0]
    latent_size = start_latent_vector.shape[1]
    
    influence_count_for_all_dimensions_list = [defaultdict(lambda: 0.0) for _ in range(latent_size)]
    
    #values = np.linspace(-range_end, range_end, evaluations_per_dimension)
    #get points around 0 with sigma that look like this:  .  .  .  . . ... . .  .   .  . 
    range_end = scipy.stats.norm.cdf(range_end_in_stds * sigma, loc=0.0, scale=sigma) - 0.5
    values = get_normal_distributed_values(range_end, evaluations_per_dimension, sigma, evaluate_postive_and_negative)
    
    for i in range(num_samples):
        print("Evaluating sample " + str(i+1) + " of " + str(num_samples))
                                              
        z = np.copy(start_latent_vector[i])
        z = np.expand_dims(z, axis = 0)
    
        #all_dimensions_matrix = np.zeros((latent_dim*new_num_notes, output_length//max_voices*evaluations_per_dimension))
        all_dimension_summaries_list = []

        for dim in range(latent_size):

            Y_list, I_list, V_list, D_list, N_list = get_sweep_output_for_values_on_dim(z, values, dim)
            
            summary_dict = dict()
            programs = vote_for_programs(I_list)
            summary_dict.update(evaluate_pitchroll(Y_list))
            summary_dict.update(evaluate_velocityroll(V_list))
            summary_dict.update(evaluate_instrumentlist(I_list))
            all_dimension_summaries_list.append(summary_dict)
            
            for key, value in summary_dict.items():
                strength, probability = value                          
                influence_count_for_all_dimensions_list[dim][key] += strength * probability
                
        for dim, summary_dict in enumerate(all_dimension_summaries_list):
            for key in summary_dict.keys():
                if key not in best_peak_evaluations_summary_dict.keys():
                    current_strength, current_probability = summary_dict[key]
                    best_peak_evaluations_summary_dict[key] = (current_strength, current_probability, dim)
                else:
                    best_strength, best_probability, best_dimension = best_peak_evaluations_summary_dict[key]
                    current_strength, current_probability = summary_dict[key]
                    if current_strength >= best_strength and current_probability >= best_probability:
                        best_peak_evaluations_summary_dict[key] = (current_strength, current_probability, dim)
    

    for key, value_tuple in best_peak_evaluations_summary_dict.items():
        best_strength, best_probability, best_dim = value_tuple
        
        
        influence_over_all_samples_list = []
        for dim in range(latent_size):
            influence_over_all_samples_list.append(influence_count_for_all_dimensions_list[dim][key])
            
        overall_best_dim = np.argmax(influence_over_all_samples_list)
            
        plt.figure(figsize=(20.0, 10.0))
        plt.title(key + ": Overall best dim: " + str(overall_best_dim) + ". Most peaked dim: " + str(best_dim))
        plt.bar(np.arange(len(influence_over_all_samples_list)), influence_over_all_samples_list, align='center')
        plt.xlabel("Dimensions")
        plt.ylabel("Influence")
        #plt.tight_layout()
        plt.savefig(save_folder+'zsweep_'+ key +'_numsamples_' +str(num_samples) +".png") 
        tikz_save(save_folder+'zsweep_'+ key +'_numsamples_' +str(num_samples) +".tex", encoding='utf-8', show_info=False)
        
        if create_midi_for_z_song_list is not None:
            
            if create_midi_name_list is None:
                create_midi_name_list = []
                for j in range(len(create_midi_for_z_song_list)):
                    create_midi_name_list.append("Unknown_song" + str(j))
                    
            for z_list, song_name in zip(create_midi_for_z_song_list, create_midi_name_list):
     
                biggest_value = values[-1]
                Y_list_sweeped, I_list_sweeped, V_list_sweeped, D_list_sweeped, N_list_sweeped = get_sweep_output_for_values_on_dim(z_list, [biggest_value], overall_best_dim)
                programs_sweeped = vote_for_programs(I_list_sweeped)
                
                if save_anything: mf.rolls_to_midi(Y_list_sweeped, 
                                                                  programs_sweeped, 
                                                                  save_folder, 
                                                                  song_name + key + '_dim'+str(best_dim)+'_value'+str(biggest_value), 
                                                                  BPM, 
                                                                  V_list_sweeped, 
                                                                  D_list_sweeped)
                if save_anything: 
                    Y_list_original, I_list_original, V_list_original, D_list_original, N_list_original = get_sweep_output_for_values_on_dim(z_list, [0], overall_best_dim)
                    programs_original = vote_for_programs(I_list_original)
                    data_class.draw_difference_pianoroll(prepare_for_drawing(Y_list_original), 
                                                         prepare_for_drawing(Y_list_sweeped), 
                                                         name_1="Original - Programs: " + str(programs_original) , 
                                                         name_2="Sweeped - Programs: " + str(programs_original), 
                                                         show=False, 
                                                         save_path=save_folder+song_name + key + '_dim'+str(best_dim)+'_value'+str(biggest_value) +"_sweepdiff.png")

                if evaluate_postive_and_negative:
                    smallest_value = values[0]
                    Y_list_sweeped, I_list_sweeped, V_list_sweeped, D_list_sweeped, N_list_sweeped = get_sweep_output_for_values_on_dim(z_list, [smallest_value], overall_best_dim)
                    programs_sweeped = vote_for_programs(I_list_sweeped)

                    if save_anything: mf.rolls_to_midi(Y_list_sweeped, 
                                                                      programs_sweeped, 
                                                                      save_folder, 
                                                                      song_name + key + '_dim'+str(best_dim)+'_value'+str(smallest_value), 
                                                                      BPM, 
                                                                      V_list_sweeped, 
                                                                      D_list_sweeped)
                    if save_anything: 
                        Y_list_original, I_list_original, V_list_original, D_list_original, N_list_original = get_sweep_output_for_values_on_dim(z_list, [0], overall_best_dim)
                        programs_original = vote_for_programs(I_list_original)
                        data_class.draw_difference_pianoroll(prepare_for_drawing(Y_list_original), 
                                                             prepare_for_drawing(Y_list_sweeped), 
                                                             name_1="Original - Programs: " + str(programs_original) , 
                                                             name_2="Sweeped - Programs: " + str(programs_original), 
                                                             show=False, 
                                                             save_path=save_folder+song_name + key + '_dim'+str(best_dim)+'_value'+str(smallest_value) +"_sweepdiff.png")

if latent_sweep:

    best_peak_evaluations_summary_dict = dict()

    start_vector = np.random.normal(loc=0.0, scale=z_std_train, size=(num_latent_sweep_samples,latent_dim))

    if num_latent_sweep_evaluation_songs > 0:
        z_song_list = []
        z_song_name_list = []
        for _ in range(num_latent_sweep_evaluation_songs):
            train_song_index = np.random.randint(train_set_size)
            song_name = train_paths[train_song_index].split('/')[-1]
            song_name = song_name.replace('mid.pickle', '')
            z_song_list.append(train_representation_list[train_song_index])
            z_song_name_list.append(song_name)
    else:
        z_song_list = None
        z_song_name_list = None
        
    latent_sweep_over_all_dimensions(start_vector, 'Random', 
                                     range_end_in_stds=3.0, 
                                     sigma=z_std_train, 
                                     evaluate_postive_and_negative=True, 
                                     create_midi_for_z_song_list=z_song_list, 
                                     create_midi_name_list=z_song_name_list)
    
    for key, value in best_peak_evaluations_summary_dict.items():
        strength, probability, dim = value
        print(key + ": \nStrength:" + str(strength) + " Probability: " + str(probability) + " Dim: " + str(dim))

# ----------------------------------------------------------------------------------------------
# Chord evaluation
# ----------------------------------------------------------------------------------------------

if chord_evaluation:
    maj_chord_dict = dict()
    #difference from the C5
    maj_chord_dict['C'] = tuple((0,4,7))
    maj_chord_dict['C#/Db'] = tuple((1,5,8))
    maj_chord_dict['D'] = tuple((2,6,9))
    maj_chord_dict['D#/Eb'] = tuple((3,7,10))
    maj_chord_dict['E'] = tuple((4,8,11))
    maj_chord_dict['F'] = tuple((-7,-3,0))
    maj_chord_dict['F#/Gb'] = tuple((-6,-2,1))
    maj_chord_dict['G'] = tuple((-5,-1,2))
    maj_chord_dict['G#/Ab'] = tuple((-4,0,3))
    maj_chord_dict['A'] = tuple((-3,1,4))
    maj_chord_dict['A#/B'] = tuple((-2,2,5))
    maj_chord_dict['H'] = tuple((-1,3,6))

    maj_min_chord_dict = dict()
    #difference from the C5 in
    maj_min_chord_dict['C'] = tuple((0,4,7))
    maj_min_chord_dict['C#/Db'] = tuple((1,5,8))
    maj_min_chord_dict['D'] = tuple((2,6,9))
    maj_min_chord_dict['D#/Eb'] = tuple((3,7,10))
    maj_min_chord_dict['E'] = tuple((4,8,11))
    maj_min_chord_dict['F'] = tuple((-7,-3,0))
    maj_min_chord_dict['F#/Gb'] = tuple((-6,-2,1))
    maj_min_chord_dict['G'] = tuple((-5,-1,2))
    maj_min_chord_dict['G#/Ab'] = tuple((-4,0,3))
    maj_min_chord_dict['A'] = tuple((-3,1,4))
    maj_min_chord_dict['A#/B'] = tuple((-2,2,5))
    maj_min_chord_dict['H'] = tuple((-1,3,6))

    maj_min_chord_dict['Cm'] = tuple((0,3,7))
    maj_min_chord_dict['C#m/Dbm'] = tuple((1,4,8))
    maj_min_chord_dict['Dm'] = tuple((2,5,9))
    maj_min_chord_dict['D#m/Ebm'] = tuple((3,6,10))
    maj_min_chord_dict['Em'] = tuple((4,7,11))
    maj_min_chord_dict['Fm'] = tuple((-7,-4,0))
    maj_min_chord_dict['F#m/Gbm'] = tuple((-6,-3,1))
    maj_min_chord_dict['Gm'] = tuple((-5,-2,2))
    maj_min_chord_dict['G#m/Abm'] = tuple((-4,-1,3))
    maj_min_chord_dict['Am'] = tuple((-3,0,4))
    maj_min_chord_dict['A#m/Bm'] = tuple((-2,1,5))
    maj_min_chord_dict['Hm'] = tuple((-1,2,6))


    def get_input_list_for_chord_name(chord_name, octave):

        offset = 12 * octave

        chord_tuple = maj_min_chord_dict[chord_name]
        X = np.zeros((output_length, high_crop - low_crop + silent_dim))

        for step in range(output_length):
            if step % max_voices < len(chord_tuple):
                pitch = offset + chord_tuple[step % max_voices] - low_crop
                X[step, pitch] = 1
            else:  
                #add silent note if included
                if include_silent_note:
                    X[step, -1] = 1

        pitch_index = pitch - low_crop
        X[:, pitch_index] = 1

        I = np.zeros((max_voices, meta_instrument_dim))
        I[:, 0] = 1 #all piano

        V = np.ones((output_length,)) #full velocity

        D = np.ones((output_length,)) #all held
        D[0] = 0 #first not held

        X = np.expand_dims(X, axis=0)
        V = np.expand_dims(V, axis=0)
        D = np.expand_dims(D, axis=0)
        return vae_definition.prepare_encoder_input_list(X,I,V,D)


    if True:

        latent_list = []
        pitches = []
        chord_names = []

        for chord_name in list(maj_min_chord_dict.keys()):
            octave = 5
            encoder_pitch_input_list = get_input_list_for_chord_name(chord_name, octave)
            z = encoder.predict(encoder_pitch_input_list, batch_size=batch_size, verbose=False)[0]

            latent_list.append(z)
            chord_names.append(chord_name)


        X = np.asarray(latent_list)
        tsne = TSNE(n_components=2)
        X_embedded = tsne.fit_transform(X)

        fig, ax = plt.subplots()
        plt.title('Chords plot: T-sne of latent chord-songs')
        plt.xlabel('First dimension of TSNE')
        plt.xlabel('Second dimension of TSNE')

        #create legend
        #major is cm(1.0), minor 0.0
        handles = []
        cm = matplotlib.cm.get_cmap('jet')
        patch = mpatches.Patch(color=cm(0.0), label='Minor')
        handles.append(patch)
        patch = mpatches.Patch(color=cm(1.0), label='Major')
        handles.append(patch)
        plt.legend(handles=handles)

        color_list = []
        for chord_name in chord_names:
            if chord_name.endswith('m'):
                color_list.append(0.0)
            else:
                color_list.append(1.0)

        plt.scatter(X_embedded[:,0], X_embedded[:,1], c=color_list, alpha=1.0, cmap=cm)

        for i, txt in enumerate(chord_names):
            ax.annotate(str(txt), (X_embedded[i,0],X_embedded[i,1]), size=7)
        plt.tight_layout()
        plt.savefig(save_folder + 'aaa_tsne_maj_min_chords.png') 
        tikz_save(save_folder + 'aaa_tsne_maj_min_chords.tex', encoding='utf-8', show_info=False)
        print("Saved tsne maj_min_chords plot")



        X = np.asarray(latent_list)
        pca = PCA(n_components=2)
        X_embedded = pca.fit_transform(X)

        fig, ax = plt.subplots()
        plt.title('Chords plot: PCA of latent chord-songs')
        plt.xlabel('First dimension of PCA')
        plt.xlabel('Second dimension of PCA')

        #create legend
        #major is cm(1.0), minor 0.0
        handles = []
        cm = matplotlib.cm.get_cmap('jet')
        patch = mpatches.Patch(color=cm(0.0), label='Minor')
        handles.append(patch)
        patch = mpatches.Patch(color=cm(1.0), label='Major')
        handles.append(patch)
        plt.legend(handles=handles)

        color_list = []
        for chord_name in chord_names:
            if chord_name.endswith('m'):
                color_list.append(0.0)
            else:
                color_list.append(1.0)

        plt.scatter(X_embedded[:,0], X_embedded[:,1], c=color_list, alpha=1.0, cmap=cm)

        for i, txt in enumerate(chord_names):
            ax.annotate(str(txt), (X_embedded[i,0],X_embedded[i,1]), size=7)
        plt.tight_layout()
        plt.savefig(save_folder + 'aaa_pca_maj_min_chords.png') 
        tikz_save(save_folder + 'aaa_pca_maj_min_chords.tex', encoding='utf-8', show_info=False)
        print("Saved pca maj_min_chords plot")


        latent_list = []
        pitches = []
        chord_names = []

        for chord_name in list(maj_chord_dict.keys()):
            octave = 5
            encoder_pitch_input_list = get_input_list_for_chord_name(chord_name, octave)
            z = encoder.predict(encoder_pitch_input_list, batch_size=batch_size, verbose=False)[0]

            latent_list.append(z)
            chord_names.append(chord_name)


        X = np.asarray(latent_list)
        tsne = TSNE(n_components=2)
        X_embedded = tsne.fit_transform(X)

        fig, ax = plt.subplots()
        plt.title('Major Chords plot: T-sne of latent chord-songs')
        plt.xlabel('First dimension of TSNE')
        plt.xlabel('Second dimension of TSNE')

        plt.scatter(X_embedded[:,0], X_embedded[:,1], alpha=1.0, cmap=cm)


        for i, txt in enumerate(chord_names):
            ax.annotate(str(txt), (X_embedded[i,0],X_embedded[i,1]), size=7)
        plt.tight_layout()
        plt.savefig(save_folder + 'aaa_tsne_maj_chords.png') 
        tikz_save(save_folder + 'aaa_tsne_maj_chords.tex', encoding='utf-8', show_info=False)
        print("Saved tsne maj_chords plot")


        X = np.asarray(latent_list)
        pca = PCA(n_components=2)
        X_embedded = pca.fit_transform(X)

        fig, ax = plt.subplots()
        plt.title('Major Chords plot: PCA of latent chord-songs')
        plt.xlabel('First dimension of PCA')
        plt.xlabel('Second dimension of PCA')

        plt.scatter(X_embedded[:,0], X_embedded[:,1], alpha=1.0, cmap=cm)


        for i, txt in enumerate(chord_names):
            ax.annotate(str(txt), (X_embedded[i,0],X_embedded[i,1]), size=7)
        plt.tight_layout()
        plt.savefig(save_folder + 'aaa_pca_maj_chords.png')
        tikz_save(save_folder + 'aaa_pca_maj_chords.tex', encoding='utf-8', show_info=False) 
        print("Saved pca maj_chords plot")



# ----------------------------------------------------------------------------------------------
# Sampling latent with different scales and comparing to signature vector
# ----------------------------------------------------------------------------------------------


if evaluate_different_sampling_regions:

    mean, cov = data_class.get_mean_and_cov_from_vector_list(all_S)

    original_distances = []

    for s in all_S:
        distance = data_class.mahalanobis_distance(s, mean, cov)
        original_distances.append(distance)

    mean_original_distance = np.mean(original_distances)
    std_original_distance = np.std(original_distances)
    print("Mean original distance: ", mean_original_distance)
    print("Std original distance: ", std_original_distance)

    scales = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 10000.0]

    number_of_samples_per_scale = 20

    mean_distances_for_each_scale = []
    std_distances_for_each_scale = []

    for scale in scales:
        distances_of_this_scale = []

        for sample in range(number_of_samples_per_scale):
            #prepare random decoder input list
            C = 0
            S = np.zeros((1, signature_vector_length))
            R = np.random.normal(loc=0.0, scale=scale, size=(1,latent_dim))
            random_input_list = vae_definition.prepare_decoder_input(R, C, S)

            decoder_outputs = decoder.predict(random_input_list, batch_size=batch_size, verbose=False)

            Y_pred, _, _, _, _ = vae_definition.process_decoder_outputs(decoder_outputs, sample_method)
            
            splitted_songs = split_song_back_to_samples(Y_pred, output_length)
            for split_song in splitted_songs:
                poly_sample = data_class.monophonic_to_khot_pianoroll(split_song, max_voices)
                signature = data_class.signature_from_pianoroll(poly_sample)

                distance = data_class.mahalanobis_distance(signature, mean, cov)
                distances_of_this_scale.append(distance)

        mean_distance = np.mean(distances_of_this_scale)
        std_distance = np.std(distances_of_this_scale)
        print("Mean distance: ", mean_distance)
        print("Std distance: ", std_distance)

        mean_distances_for_each_scale.append(mean_distance)
        std_distances_for_each_scale.append(std_distance)


    fig, ax = plt.subplots()
    plt.plot(scales, mean_distances_for_each_scale, label='Mean Mahalanobis distance')
    plt.plot(scales, std_distances_for_each_scale, label='Std Mahalanobis distance')
    plt.title('Mahalanobis distance to train set for different sampling scales')
    plt.xlabel('Scales')
    #plt.ylabel('Mean mahalanobis distance to train set signature vectors')
    ax.set_xscale('log')
    plt.legend(loc='upper left', prop={'size': 8})
    plt.tight_layout()
    plt.savefig(save_folder + 'aaa_signature_scales.png') 
    tikz_save(save_folder + 'aaa_signature_scales.tex', encoding='utf-8', show_info=False)
    print("Saved signature scales plot")


    locs = [0, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 10000.0]

    number_of_samples_per_loc = 20

    mean_distances_for_each_loc = []
    std_distances_for_each_loc = []

    for loc in locs:
        distances_of_this_loc = []

        for sample in range(number_of_samples_per_loc):
            #prepare random decoder input list
            C = 0
            S = np.zeros((1, signature_vector_length))
            R = np.random.normal(loc=loc, scale=z_std_train, size=(1,latent_dim))
            random_input_list = vae_definition.prepare_decoder_input(R, C, S)

            decoder_outputs = decoder.predict(random_input_list, batch_size=batch_size, verbose=False)

            Y_pred, _, _, _, _ = vae_definition.process_decoder_outputs(decoder_outputs, sample_method)
            
            splitted_songs = split_song_back_to_samples(Y_pred, output_length)
            for split_song in splitted_songs:
                poly_sample = data_class.monophonic_to_khot_pianoroll(split_song, max_voices)
                signature = data_class.signature_from_pianoroll(poly_sample)

                distance = data_class.mahalanobis_distance(signature, mean, cov)
                distances_of_this_loc.append(distance)

        mean_distance = np.mean(distances_of_this_loc)
        std_distance = np.std(distances_of_this_loc)

        mean_distances_for_each_loc.append(mean_distance)
        std_distances_for_each_loc.append(std_distance)


    fig, ax = plt.subplots()
    plt.plot(locs, mean_distances_for_each_loc, label='Mean Mahalanobis distance')
    plt.plot(locs, std_distances_for_each_loc, label='Std Mahalanobis distance')
    plt.title('Mahalanobis distance to train set for different sampling locs')
    plt.xlabel('Scales')
    #plt.ylabel('Mean mahalanobis distance to train set signature vectors')
    ax.set_xscale('log')
    plt.legend(loc='upper left', prop={'size': 8})
    plt.tight_layout()
    plt.savefig(save_folder + 'aaa_signature_locs.png') 
    tikz_save(save_folder + 'aaa_signature_locs.tex', encoding='utf-8', show_info=False)
    print("Saved signature locs plot")

# ----------------------------------------------------------------------------------------------
# Pitch evaluation
# ----------------------------------------------------------------------------------------------

if pitch_evaluation:
    def pitch_to_name(pitch):
        octave = pitch // 12
        note_in_octave = pitch % 12
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'H']
        return note_names[note_in_octave] + str(octave)

    def get_input_list_for_pitch(pitch):
        X = np.zeros((output_length, high_crop - low_crop + silent_dim))
        pitch_index = pitch - low_crop
        X[:, pitch_index] = 1

        I = np.zeros((max_voices, meta_instrument_dim))
        I[:, 0] = 1 #all piano

        V = np.ones((output_length,)) #full velocity

        D = np.ones((output_length,)) #all held
        D[0] = 0 #first not held

        X = np.expand_dims(X, axis=0)
        V = np.expand_dims(V, axis=0)
        D = np.expand_dims(D, axis=0)
        return vae_definition.prepare_encoder_input_list(X,I,V,D)

    latent_list = []
    pitches = []
    pitch_names = []

    for pitch in range(low_crop, high_crop):

        encoder_pitch_input_list = get_input_list_for_pitch(pitch)
        z = encoder.predict(encoder_pitch_input_list, batch_size=batch_size, verbose=False)[0]

        pitches.append(pitch)
        latent_list.append(z)
        pitch_names.append(pitch_to_name(pitch))

    X = np.asarray(latent_list)
    tsne = TSNE(n_components=1)
    X_embedded = tsne.fit_transform(X)
    X_embedded = list(X_embedded)

    fig, ax = plt.subplots()
    plt.title('T-sne of latent pitch-songs')
    plt.xlabel('Pitches')
    plt.ylabel('Value of 1-dim T-sne')

    pitch_colors = []
    for pitch in pitches:
        pitch_colors.append((pitch-low_crop*1.0)/new_num_notes)
    #plt.scatter(X_embedded[:,0], X_embedded[:,1], c=pitch_colors, alpha=0.4, cmap=cm)
    plt.scatter(pitches, X_embedded, alpha=1.0)

    for i, txt in enumerate(pitch_names):
        ax.annotate(str(txt), (pitches[i],X_embedded[i]), size=7)
    plt.tight_layout()
    plt.savefig(save_folder + 'aaa_tsne_pitches.png') 
    tikz_save(save_folder + 'aaa_tsne_pitches.tex', encoding='utf-8', show_info=False)
    print("Saved tsne pitches plot")


    X = np.asarray(latent_list)
    pca = PCA(n_components=1)
    X_embedded = pca.fit_transform(X)
    X_embedded = list(X_embedded)

    fig, ax = plt.subplots()
    plt.title('PCA of latent pitch-songs')
    plt.xlabel('Pitch values')
    plt.ylabel('Value of 1-dim PCA')

    pitch_colors = []
    for pitch in pitches:
        pitch_colors.append((pitch-low_crop*1.0)/new_num_notes)
    #plt.scatter(X_embedded[:,0], X_embedded[:,1], c=pitch_colors, alpha=0.4, cmap=cm)
    plt.scatter(pitches, X_embedded, alpha=1.0)

    for i, txt in enumerate(pitches):
        ax.annotate(str(txt), (pitches[i],X_embedded[i]), size=7)
    plt.tight_layout()
    plt.savefig(save_folder + 'aaa_pca_pitches.png') 
    tikz_save(save_folder + 'aaa_pca_pitches.tex', encoding='utf-8', show_info=False)
    print("Saved pca pitches plot")


    X = np.asarray(latent_list)
    tsne = TSNE(n_components=2)
    X_embedded = tsne.fit_transform(X)

    fig, ax = plt.subplots()
    plt.title('Octaves plot: T-sne of latent pitch-songs')
    plt.xlabel('First dimension of TSNE')
    plt.xlabel('Second dimension of TSNE')

    pitch_colors = []
    octave_names = []
    for pitch in pitches:
        note_in_octave = pitch%12
        pitch_colors.append((note_in_octave)/12.0)
    plt.scatter(X_embedded[:,0], X_embedded[:,1], c=pitch_colors, alpha=1.0)


    for i, txt in enumerate(pitch_names):
        ax.annotate(str(txt), (X_embedded[i,0],X_embedded[i,1]), size=7)
    plt.tight_layout()
    plt.savefig(save_folder + 'aaa_tsne_octaves.png') 
    tikz_save(save_folder + 'aaa_tsne_octaves.tex', encoding='utf-8', show_info=False)
    print("Saved tsne octaves plot")


# ----------------------------------------------------------------------------------------------
# Generation of new song parts
# ----------------------------------------------------------------------------------------------
sample_method = 'choice'

for song_num in range(max_new_sampled_songs):

    #prepare random decoder input list
    C = 0
    S = np.zeros((1, signature_vector_length))
    R = np.random.normal(loc=0.0, scale=z_std_train, size=(1,latent_dim))
    random_input_list = vae_definition.prepare_decoder_input(R, C, S)

    decoder_outputs = decoder.predict(random_input_list, batch_size=batch_size, verbose=False)

    Y, I, V, D, N = vae_definition.process_decoder_outputs(decoder_outputs, sample_method)
    
    programs = data_class.instrument_representation_to_programs(I[0], instrument_attach_method)

    if save_anything: mf.rolls_to_midi(Y, programs, save_folder, 'random_'+str(song_num), BPM, V, D)

    if include_composer_decoder:

        previous_song = None
        previous_programs = None

        random_code = np.random.normal(loc=0.0, scale=z_std_train, size=(1,latent_dim))
        for C in range(num_classes):
            
            #turn the knob to one class:
            random_code[0,0:num_classes] = -1
            random_code[0,C] = 1

            S = np.zeros((1, signature_vector_length))
            R = random_code
            random_input_list = vae_definition.prepare_decoder_input(R, C, S)
            decoder_outputs = decoder.predict(random_input_list, batch_size=batch_size, verbose=False)

            Y, I, V, D, N = vae_definition.process_decoder_outputs(decoder_outputs, sample_method)

            programs = data_class.instrument_representation_to_programs(I[0], instrument_attach_method)

            if previous_song is not None:
                data_class.draw_difference_pianoroll(prepare_for_drawing(Y), prepare_for_drawing(previous_song), name_1=str(song_num) +"_" + str(C) + " Programs: " + str(programs) , name_2=str(song_num) +"_" + str(C-1) + " Programs: " + str(previous_programs), show=False, save_path=save_folder+"random_"+str(song_num) +"_" +str(C)+ "_vs_" + str(C-1) +"_switchdiff.png")

            if save_anything: mf.rolls_to_midi(Y, programs, save_folder, 'random_'+str(song_num) + "_" + str(C), BPM, V, D)

            previous_song = Y
            previous_programs = programs

# ----------------------------------------------------------------------------------------------
# Generation of new long songs
# ----------------------------------------------------------------------------------------------


long_song_length = 20 #how many iterations?

for song_num in range(max_new_sampled_long_songs):

    print("Producing random song ", song_num)

    if include_composer_decoder:

        random_code = np.random.normal(loc=0.0, scale=z_std_train, size=(1,latent_dim))

        C = 0
        R = random_code

        Y_list = []
        V_list = []
        D_list = []
        I_list = []
        previous_latent_rep = np.zeros((1,latent_dim))

        S = np.zeros((1, signature_vector_length))

        already_picked_z_indices = []
        
        for i in range(long_song_length):


            lowest_distance = np.linalg.norm(all_z[0]-R)
            best_z_index = 0
            for i, z in enumerate(all_z):
                distance = np.linalg.norm(z-R)
                if distance < lowest_distance and i not in already_picked_z_indices:
                    lowest_distance = distance
                    best_z_index = i

            already_picked_z_indices.append(best_z_index)
            closest_z = all_z[best_z_index]
            print("Closest z index : ", best_z_index)

            #e = np.random.normal(loc=0.0, scale=1.0, size=(1,latent_dim))
            e = np.random.rand()
            e = z_std_train
            R = (R + closest_z * e) / (1 + e)

            

            random_input_list = vae_definition.prepare_decoder_input(R, C, S, previous_latent_rep)
            
            decoder_outputs = decoder.predict(random_input_list, batch_size=batch_size, verbose=False)

            Y, I, V, D, N = vae_definition.process_decoder_outputs(decoder_outputs, sample_method)

            Y_list.extend(Y)
            I_list.extend(I)
            V_list.extend(V)
            D_list.extend(D)

            #use output as next input
            X = np.copy(Y)
            if include_silent_note:
                X = np.append(X, np.zeros((X.shape[0], 1)), axis=1)
                for step in range(X.shape[0]):
                    if np.sum(X[step]) == 0:
                        X[step, -1] = 1
            X = np.asarray([X])

            previous_latent_rep = R
            encoder_input_list = vae_definition.prepare_encoder_input_list(X,I[0],np.expand_dims(V, axis=0),np.expand_dims(D,axis=0))
            R = encoder.predict(encoder_input_list, batch_size=batch_size, verbose=False)

        programs_for_whole_long_song = vote_for_programs(I_list)

        Y_list = np.asarray(Y_list)
        D_list = np.asarray(D_list)
        V_list = np.asarray(V_list)

        if save_anything: mf.rolls_to_midi(Y_list, programs_for_whole_long_song, save_folder, 'random_long_temp' + str(temperature) + "_" + str(song_num), BPM, V_list, D_list)



if evaluate_autoencoding_and_stuff:

    # ----------------------------------------------------------------------------------------------
    # Setting variables to zero
    # ----------------------------------------------------------------------------------------------

    sample_method = 'argmax'

    # Test function
    total_original_notes_array = []
    total_predicted_notes_array = []
    reconstruction_accuracy_array = []
    not_predicted_notes_array = []
    new_predicted_notes_array = []
    classifier_accuracy_array = []
    composer_accuracy_array = []

    #the switched_program lists indexed by class_from, class_to
    # so switched_instruments_for_each_class[class_from][class_to] gets the instruments when a style is switched form class_form to class_to
    switched_instruments_for_each_class = [[[] for _ in range(num_classes)] for _ in range(num_classes)]

    if instrument_attach_method == '1hot-category' or 'khot-category':
        switch_instruments_matrix = np.zeros((num_classes, num_classes, 16, 16))
    else:
        switch_instruments_matrix = np.zeros((num_classes, num_classes, 128, 128))


    bar = progressbar.ProgressBar(max_value=test_set_size, redirect_stdout=False)

    previous_is_null = True
    previous_encoded_rep = None
    previous_song_name = ''

    representation_list = []

    original_signature_list = []
    generated_signature_list = []
    successful_signature_manipulations_list = []
    neutral_signature_manipulations_list = []
    unsuccessful_signature_manipulations_list = []
    most_successful_dim_list = []
    least_successful_dim_list = []

    original_signature_list_for_each_class = [[] for _ in range(num_classes)]
    autoencoded_signature_list_for_each_class = [[] for _ in range(num_classes)]
    switched_signature_list_for_each_class = [[] for _ in range(num_classes)]
    instrument_switched_signature_list_for_each_class = [[] for _ in range(num_classes)]

    original_pitches_classifier_accuracy_list = []
    autoencoded_pitches_classifier_accuracy_list = []
    switched_pitches_classifier_accuracy_list = []
    original_pitches_classifier_confidence_list = []
    autoencoded_pitches_classifier_confidence_list = []
    switched_pitches_classifier_confidence_list = []

    original_velocity_classifier_accuracy_list = []
    autoencoded_velocity_classifier_accuracy_list = []
    switched_velocity_classifier_accuracy_list = []
    original_velocity_classifier_confidence_list = []
    autoencoded_velocity_classifier_confidence_list = []
    switched_velocity_classifier_confidence_list = []

    original_instrument_classifier_accuracy_list = []
    autoencoded_instrument_classifier_accuracy_list = []
    switched_instrument_classifier_accuracy_list = []
    original_instrument_classifier_confidence_list = []
    autoencoded_instrument_classifier_confidence_list = []
    switched_instrument_classifier_confidence_list = []

    original_ensemble_classifier_accuracy_list = []
    autoencoded_ensemble_classifier_accuracy_list = []
    switched_ensemble_classifier_accuracy_list = []
    original_ensemble_classifier_confidence_list = []
    autoencoded_ensemble_classifier_confidence_list = []
    switched_ensemble_classifier_confidence_list = []

    note_start_prediction_to_original_errors_list = []
    note_start_prediction_to_prediction_errors_list = []

    harmonicity_matrix_autoencoded_list = []
    harmonicity_matrix_switched_from_class_to_class_list = [[[] for _ in range(num_classes)] for _ in range(num_classes)] #from, to


    previous_latent_list = []

    metrics_dict_for_all_songs_list = []
    ids_for_all_songs_list = []
    mean_metrics_for_all_songs_dict = defaultdict(lambda: 0.0)
    mean_metrics_for_all_songs_dict["song_name"] = "Mean"
    mean_metrics_for_all_songs_dict["class"] = "both"

    original_ensemble_classifier_accuracy_list_for_each_class = [[] for _ in range(num_classes)]
    autoencoded_ensemble_classifier_accuracy_list_for_each_class = [[] for _ in range(num_classes)]
    switched_ensemble_classifier_accuracy_list_for_each_class = [[] for _ in range(num_classes)]

    # ----------------------------------------------------------------------------------------------
    # Generation and evaluation of test data
    # ----------------------------------------------------------------------------------------------

    if test_train_set:
        l= len(X_train)

    else:
        l = len(X_test)

    print('\nTesting:')
    for song_num in range(l):

        # ----------------------------------------------------------------------------------------------
        # Prepare original data
        # ----------------------------------------------------------------------------------------------

        if test_train_set:
            song_name = train_paths[song_num].split('/')[-1]
            song_name = song_name.replace('mid.pickle', '')

            X = X_train[song_num]
            Y = Y_train[song_num]
            C = C_train[song_num]
            I = I_train[song_num]
            V = V_train[song_num]
            D = D_train[song_num]
            S = normalized_S_train[song_num]
            S_original = S_train[song_num]
            T = T_train[song_num]
        else:
            #create dataset
            song_name = test_paths[song_num].split('/')[-1]
            song_name = song_name.replace('mid.pickle', '')

            X = X_test[song_num]
            Y = Y_test[song_num]
            C = C_test[song_num]
            I = I_test[song_num]
            V = V_test[song_num]
            D = D_test[song_num]
            S = normalized_S_test[song_num]
            S_original = S_test[song_num]
            T = T_test[song_num]

        print("\nEvaluating " + song_name)

        V_flattened = []
        for sample in range(V.shape[0]):
            V_flattened.extend(V[sample])
        V_flattened = np.asarray(V_flattened)

        D_flattened = []
        for sample in range(D.shape[0]):
            D_flattened.extend(D[sample])
        D_flattened = np.asarray(D_flattened)

        num_samples = Y.shape[0]
        song = []
        for sample in range(num_samples):
            if include_silent_note:
                song.extend(Y[sample, :, :-1])
            else:
                song.extend(Y[sample])
        song = np.asarray(song)

        #save signature vectors
        original_signature_list.extend(S_original)
        original_signature_list_for_each_class[C].extend(S_original)

        #prepare programs
        programs = []
        if rolls:
            programs = data_class.instrument_representation_to_programs(I, instrument_attach_method)

        metrics_for_this_song_dict = defaultdict(lambda: 0.0)
        metrics_for_this_song_dict["song_name"] = song_name
        metrics_for_this_song_dict["class"] = classes[C]

        # ----------------------------------------------------------------------------------------------
        # Style classification on original data
        # ----------------------------------------------------------------------------------------------

        original_pitches_classifier_accuracy = 0.0
        original_pitches_classifier_confidence = 0.0
        original_velocity_classifier_accuracy = 0.0
        original_velocity_classifier_confidence = 0.0
        original_instrument_classifier_accuracy = 0.0
        original_instrument_classifier_confidence = 0.0
        original_ensemble_classifier_accuracy = 0.0
        original_ensemble_classifier_confidence = 0.0

        #calculate instrument style classifier accuracy
        instrument_classifier_input = np.asarray([I])
        instrument_classifier_prediction = instrument_classifier_model.predict(instrument_classifier_input)[0]
        #get the confidence of the style classifier
        instrument_confidence = instrument_classifier_prediction[C]
        original_instrument_classifier_confidence += instrument_confidence
        if np.argmax(instrument_classifier_prediction) == C:
            original_instrument_classifier_accuracy += 1

        for sample in range(num_samples):
            #calculate style classifier accuracy based on pitches
            pitches_classifier_input = np.asarray([Y[sample]])
            pitches_classifier_prediction = pitches_classifier_model.predict(pitches_classifier_input)[0]
            #get the confidence of the style classifier
            pitches_confidence = pitches_classifier_prediction[C]
            original_pitches_classifier_confidence += pitches_confidence
            if np.argmax(pitches_classifier_prediction) == C:
                original_pitches_classifier_accuracy += 1

            #calculate velocity style classifier accuracy
            velocity_split_song = np.copy(V[sample])
            velocity_split_song = np.expand_dims(velocity_split_song, 1)
            velocity_classifier_input = np.asarray([velocity_split_song])
            velocity_classifier_prediction = velocity_classifier_model.predict(velocity_classifier_input)[0]
            #get the confidence of the style classifier
            velocity_confidence = velocity_classifier_prediction[C]
            original_velocity_classifier_confidence += velocity_confidence
            if np.argmax(velocity_classifier_prediction) == C:
                original_velocity_classifier_accuracy += 1

            #calculate ensemble style classifier accuracy
            ensemble_classifier_prediction = ensemble_prediction(pitches_classifier_input,instrument_classifier_input,velocity_classifier_input)[0]
            #get the confidence of the style classifier
            ensemble_confidence = ensemble_classifier_prediction[C]
            original_ensemble_classifier_confidence += ensemble_confidence
            if np.argmax(ensemble_classifier_prediction) == C:
                original_ensemble_classifier_accuracy += 1

        #save style pitch classifier accuracies
        original_pitches_classifier_accuracy /= num_samples
        original_pitches_classifier_confidence /= num_samples
        original_pitches_classifier_accuracy_list.append(original_pitches_classifier_accuracy)
        original_pitches_classifier_confidence_list.append(original_pitches_classifier_confidence)
        print("Original style pitch classifier accuracy: ", original_pitches_classifier_accuracy)
        print("Original style pitch classifier confidence: ", original_pitches_classifier_confidence)
        metrics_for_this_song_dict["original_pitches_classifier_accuracy"] = original_pitches_classifier_accuracy
        mean_metrics_for_all_songs_dict["original_pitches_classifier_accuracy"] += original_pitches_classifier_accuracy
        metrics_for_this_song_dict["original_pitches_classifier_confidence"] = original_pitches_classifier_confidence
        mean_metrics_for_all_songs_dict["original_pitches_classifier_confidence"] += original_pitches_classifier_confidence

        #save style velocity classifier accuracies
        original_velocity_classifier_accuracy /= num_samples
        original_velocity_classifier_confidence /= num_samples
        original_velocity_classifier_accuracy_list.append(original_velocity_classifier_accuracy)
        original_velocity_classifier_confidence_list.append(original_velocity_classifier_confidence)
        print("Original style velocity classifier accuracy: ", original_velocity_classifier_accuracy)
        print("Original style velocity classifier confidence: ", original_velocity_classifier_confidence)
        metrics_for_this_song_dict["original_velocity_classifier_accuracy"] = original_velocity_classifier_accuracy
        mean_metrics_for_all_songs_dict["original_velocity_classifier_accuracy"] += original_velocity_classifier_accuracy
        metrics_for_this_song_dict["original_velocity_classifier_confidence"] = original_velocity_classifier_confidence
        mean_metrics_for_all_songs_dict["original_velocity_classifier_confidence"] += original_velocity_classifier_confidence


        #save style instrument classifier accuracies
        original_instrument_classifier_accuracy_list.append(original_instrument_classifier_accuracy)
        original_instrument_classifier_confidence_list.append(original_instrument_classifier_confidence)
        print("Original style instrument classifier accuracy: ", original_instrument_classifier_accuracy)
        print("Original style instrument classifier confidence: ", original_instrument_classifier_confidence)
        metrics_for_this_song_dict["original_instrument_classifier_accuracy"] = original_instrument_classifier_accuracy
        mean_metrics_for_all_songs_dict["original_instrument_classifier_accuracy"] += original_instrument_classifier_accuracy
        metrics_for_this_song_dict["original_instrument_classifier_confidence"] = original_instrument_classifier_confidence
        mean_metrics_for_all_songs_dict["original_instrument_classifier_confidence"] += original_instrument_classifier_confidence

        #save style ensemble classifier accuracies
        original_ensemble_classifier_accuracy /= num_samples
        original_ensemble_classifier_confidence /= num_samples
        original_ensemble_classifier_accuracy_list.append(original_ensemble_classifier_accuracy)
        original_ensemble_classifier_confidence_list.append(original_ensemble_classifier_confidence)
        print("Original style ensemble classifier accuracy: ", original_ensemble_classifier_accuracy)
        print("Original style ensemble classifier confidence: ", original_ensemble_classifier_confidence)
        metrics_for_this_song_dict["original_ensemble_classifier_accuracy"] = original_ensemble_classifier_accuracy
        mean_metrics_for_all_songs_dict["original_ensemble_classifier_accuracy"] += original_ensemble_classifier_accuracy
        metrics_for_this_song_dict["original_ensemble_classifier_confidence"] = original_ensemble_classifier_confidence
        mean_metrics_for_all_songs_dict["original_ensemble_classifier_confidence"] += original_ensemble_classifier_confidence

        original_ensemble_classifier_accuracy_list_for_each_class[C].append(original_ensemble_classifier_accuracy)


        # ----------------------------------------------------------------------------------------------
        # Encode data
        # ----------------------------------------------------------------------------------------------

        #get the latent representation of every song part
        encoder_input_list = vae_definition.prepare_encoder_input_list(X,I,V,D)
        encoded_representation = encoder.predict(encoder_input_list, batch_size=batch_size, verbose=False)
        representation_list.append(encoded_representation)

        #don't save train values because we already calculated and saved them previously
        if not test_train_set:
            if not os.path.exists(save_folder + classes[C]+'/'):
                os.makedirs(save_folder + classes[C]+'/') 
            if save_anything: np.save(save_folder + classes[C]+'/'+'z_' + song_name, encoded_representation)

        H = np.asarray(encoded_representation)

        # ----------------------------------------------------------------------------------------------
        # Autoencode data
        # ----------------------------------------------------------------------------------------------
        
        autoencoder_input_list, _= vae_definition.prepare_autoencoder_input_and_output_list(X,Y,C,I,V,D,S,H)

        #test the autoencoder if it can reproduce the input file
        autoencoder_outputs = autoencoder.predict(autoencoder_input_list, batch_size=batch_size, verbose=verbose)

        Y_pred, I_pred, V_pred, D_pred, N_pred = vae_definition.process_autoencoder_outputs(autoencoder_outputs, sample_method)

        #save midi
        if save_anything: mf.rolls_to_midi(Y_pred, programs, save_folder, song_name + '_autoencoded', BPM, V_pred, D_pred)
        if save_anything: mf.rolls_to_midi(np.concatenate((Y_pred, song), axis=0), programs, save_folder,song_name + '_auto+orig', BPM, np.concatenate((V_pred, V_flattened), axis=0), np.concatenate((D_pred, D_flattened), axis=0))
      
        # ----------------------------------------------------------------------------------------------
        # Calculate note start errors
        # ----------------------------------------------------------------------------------------------

        predicted_note_start_to_original_errors = 0
        predicted_note_start_to_predicted_errors = 0
        if  meta_held_notes or (meta_velocity and velocity_threshold_such_that_it_is_a_played_note > 0):
            #also include meta_velocity because it can also hold duration information
            #if meta_velocity is above a threshold, then it is a played note
            for sample in range(num_samples):
                for step in range(output_length):
                    note_vector_predicted = Y_pred[sample * output_length + step]
                    note_vector_original = Y[sample, step]
                    predicted_duration = D_pred[sample * output_length + step]
                    note_vector_predicted_is_silent = np.sum(note_vector_predicted) == 0
                    if include_silent_note:
                        note_vector_original_is_silent = note_vector_original[-1] == 1
                    else:
                        note_vector_original_is_silent = np.sum(note_vector_original) == 0

                    #these errors can be caused by meta_velocity also
                    predicted_duration_is_note_start = predicted_duration == 0
                    if note_vector_predicted_is_silent and predicted_duration_is_note_start:
                        predicted_note_start_to_predicted_errors += 1
                    if note_vector_original_is_silent and predicted_duration_is_note_start:
                        predicted_note_start_to_original_errors += 1


        predicted_note_start_to_original_errors /= num_samples * output_length
        predicted_note_start_to_predicted_errors /= num_samples * output_length
        note_start_prediction_to_original_errors_list.append(predicted_note_start_to_original_errors)
        note_start_prediction_to_prediction_errors_list.append(predicted_note_start_to_predicted_errors)
        print("Predicted note start errors compared to original: ", predicted_note_start_to_original_errors)
        print("Predicted note start errors compared to predicted: ", predicted_note_start_to_predicted_errors)
        metrics_for_this_song_dict["predicted_note_start_to_original_errors"] = predicted_note_start_to_original_errors
        mean_metrics_for_all_songs_dict["predicted_note_start_to_original_errors"] += predicted_note_start_to_original_errors
        metrics_for_this_song_dict["predicted_note_start_to_predicted_errors"] = predicted_note_start_to_predicted_errors
        mean_metrics_for_all_songs_dict["predicted_note_start_to_predicted_errors"] += predicted_note_start_to_predicted_errors
        

        
        # ----------------------------------------------------------------------------------------------
        # Style classification, signature vector and harmonicity for each autoencoded sample
        # ----------------------------------------------------------------------------------------------

        #calculate signature vectors and style classifier evaluations for generated songs
        splitted_songs = split_song_back_to_samples(Y_pred, output_length)
        current_song_generated_signature_lists = []

        autoencoded_pitches_classifier_confidence = 0.0
        autoencoded_pitches_classifier_accuracy = 0.0
        autoencoded_velocity_classifier_confidence = 0.0
        autoencoded_velocity_classifier_accuracy = 0.0
        autoencoded_instrument_classifier_confidence = 0.0
        autoencoded_instrument_classifier_accuracy = 0.0
        autoencoded_ensemble_classifier_confidence = 0.0
        autoencoded_ensemble_classifier_accuracy = 0.0

        for sample, split_song in enumerate(splitted_songs):
            split_song_with_silent_notes = np.copy(split_song)
            if include_silent_note:
                split_song_with_silent_notes = np.append(split_song_with_silent_notes, np.zeros((split_song_with_silent_notes.shape[0], 1)), axis=1)
                for step in range(split_song_with_silent_notes.shape[0]):
                    if np.sum(split_song_with_silent_notes[step]) == 0:
                        split_song_with_silent_notes[step, -1] = 1

            #calculate pitches style classifier accuracy
            pitches_classifier_input = np.asarray([split_song_with_silent_notes])
            pitches_classifier_prediction = pitches_classifier_model.predict(pitches_classifier_input)[0]
            #get the confidence of the style classifier
            pitches_confidence = pitches_classifier_prediction[C]
            autoencoded_pitches_classifier_confidence += pitches_confidence
            if np.argmax(pitches_classifier_prediction) == C:
                autoencoded_pitches_classifier_accuracy += 1

            if meta_velocity:
                #calculate velocity style classifier accuracy
                velocity_split_song = np.copy(V_pred[sample*output_length:(sample+1)*output_length])
                velocity_split_song = np.expand_dims(velocity_split_song, 1)
                velocity_classifier_input = np.asarray([velocity_split_song])
                velocity_classifier_prediction = velocity_classifier_model.predict(velocity_classifier_input)[0]
                #get the confidence of the style classifier
                velocity_confidence = velocity_classifier_prediction[C]
                autoencoded_velocity_classifier_confidence += velocity_confidence
                if np.argmax(velocity_classifier_prediction) == C:
                    autoencoded_velocity_classifier_accuracy += 1

            if meta_instrument:
                #calculate instrument style classifier accuracy
                instrument_classifier_input = np.asarray([I_pred[sample]])
                instrument_classifier_prediction = instrument_classifier_model.predict(instrument_classifier_input)[0]
                #get the confidence of the style classifier
                instrument_confidence = instrument_classifier_prediction[C]
                autoencoded_instrument_classifier_confidence += instrument_confidence
                if np.argmax(instrument_classifier_prediction) == C:
                    autoencoded_instrument_classifier_accuracy += 1

            if meta_velocity and meta_instrument:
                #calculate style classifier accuracy
                ensemble_classifier_prediction = ensemble_prediction(pitches_classifier_input,instrument_classifier_input,velocity_classifier_input)[0]
                #get the confidence of the style classifier
                ensemble_confidence = ensemble_classifier_prediction[C]
                autoencoded_ensemble_classifier_confidence += ensemble_confidence
                if np.argmax(ensemble_classifier_prediction) == C:
                    autoencoded_ensemble_classifier_accuracy += 1


            harmonicity_matrix_autoencoded_list.append(data_class.get_harmonicity_scores_for_each_track_combination(split_song))

            poly_sample = data_class.monophonic_to_khot_pianoroll(split_song, max_voices)
            signature = data_class.signature_from_pianoroll(poly_sample)
            generated_signature_list.append(signature)
            current_song_generated_signature_lists.append(signature)
            autoencoded_signature_list_for_each_class[C].append(signature)

        #save style pitches classifier accuracies
        autoencoded_pitches_classifier_accuracy /= len(splitted_songs)
        autoencoded_pitches_classifier_confidence /= len(splitted_songs)
        autoencoded_pitches_classifier_accuracy_list.append(autoencoded_pitches_classifier_accuracy)
        autoencoded_pitches_classifier_confidence_list.append(autoencoded_pitches_classifier_confidence)
        print("Autoencoded style pitch classifier accuracy: ", autoencoded_pitches_classifier_accuracy)
        print("Autoencoded style pitch classifier confidence: ", autoencoded_pitches_classifier_confidence)
        metrics_for_this_song_dict["autoencoded_pitches_classifier_accuracy"] = autoencoded_pitches_classifier_accuracy
        mean_metrics_for_all_songs_dict["autoencoded_pitches_classifier_accuracy"] += autoencoded_pitches_classifier_accuracy
        metrics_for_this_song_dict["autoencoded_pitches_classifier_confidence"] = autoencoded_pitches_classifier_confidence
        mean_metrics_for_all_songs_dict["autoencoded_pitches_classifier_confidence"] += autoencoded_pitches_classifier_confidence

        if meta_velocity:
            #save style velocity classifier accuracies
            autoencoded_velocity_classifier_accuracy /= len(splitted_songs)
            autoencoded_velocity_classifier_confidence /= len(splitted_songs)
            autoencoded_velocity_classifier_accuracy_list.append(autoencoded_velocity_classifier_accuracy)
            autoencoded_velocity_classifier_confidence_list.append(autoencoded_velocity_classifier_confidence)
            print("Autoencoded style velocity classifier accuracy: ", autoencoded_velocity_classifier_accuracy)
            print("Autoencoded style velocity classifier confidence: ", autoencoded_velocity_classifier_confidence)
            metrics_for_this_song_dict["autoencoded_velocity_classifier_accuracy"] = autoencoded_velocity_classifier_accuracy
            mean_metrics_for_all_songs_dict["autoencoded_velocity_classifier_accuracy"] += autoencoded_velocity_classifier_accuracy
            metrics_for_this_song_dict["autoencoded_velocity_classifier_confidence"] = autoencoded_velocity_classifier_confidence
            mean_metrics_for_all_songs_dict["autoencoded_velocity_classifier_confidence"] += autoencoded_velocity_classifier_confidence

        if meta_instrument:
            #save style instrument classifier accuracies
            autoencoded_instrument_classifier_accuracy /= len(splitted_songs)
            autoencoded_instrument_classifier_confidence /= len(splitted_songs)
            autoencoded_instrument_classifier_accuracy_list.append(autoencoded_instrument_classifier_accuracy)
            autoencoded_instrument_classifier_confidence_list.append(autoencoded_instrument_classifier_confidence)
            print("Autoencoded style instrument classifier accuracy: ", autoencoded_instrument_classifier_accuracy)
            print("Autoencoded style instrument classifier confidence: ", autoencoded_instrument_classifier_confidence)
            metrics_for_this_song_dict["autoencoded_instrument_classifier_accuracy"] = autoencoded_instrument_classifier_accuracy
            mean_metrics_for_all_songs_dict["autoencoded_instrument_classifier_accuracy"] += autoencoded_instrument_classifier_accuracy
            metrics_for_this_song_dict["autoencoded_instrument_classifier_confidence"] = autoencoded_instrument_classifier_confidence
            mean_metrics_for_all_songs_dict["autoencoded_instrument_classifier_confidence"] += autoencoded_instrument_classifier_confidence

        if meta_velocity and meta_instrument:
            #save style ensemble classifier accuracies
            autoencoded_ensemble_classifier_accuracy /= len(splitted_songs)
            autoencoded_ensemble_classifier_confidence /= len(splitted_songs)
            autoencoded_ensemble_classifier_accuracy_list.append(autoencoded_ensemble_classifier_accuracy)
            autoencoded_ensemble_classifier_confidence_list.append(autoencoded_ensemble_classifier_confidence)
            print("Autoencoded style ensemble classifier accuracy: ", autoencoded_ensemble_classifier_accuracy)
            print("Autoencoded style ensemble classifier confidence: ", autoencoded_ensemble_classifier_confidence)
            metrics_for_this_song_dict["autoencoded_ensemble_classifier_accuracy"] = autoencoded_ensemble_classifier_accuracy
            mean_metrics_for_all_songs_dict["autoencoded_ensemble_classifier_accuracy"] += autoencoded_ensemble_classifier_accuracy
            metrics_for_this_song_dict["autoencoded_ensemble_classifier_confidence"] = autoencoded_ensemble_classifier_confidence
            mean_metrics_for_all_songs_dict["autoencoded_ensemble_classifier_confidence"] += autoencoded_ensemble_classifier_confidence

            autoencoded_ensemble_classifier_accuracy_list_for_each_class[C].append(autoencoded_ensemble_classifier_accuracy)


        # ----------------------------------------------------------------------------------------------
        # Calculate statistics on autoencoded songs
        # ----------------------------------------------------------------------------------------------

        difference_song = song * 2 + Y_pred
        unique, counts = np.unique(difference_song, return_counts=True)
        difference_statistics = dict(zip(unique, counts))
        total_original_notes = np.count_nonzero(song)
        total_predicted_notes = np.count_nonzero(Y_pred)
        if 3 in difference_statistics.keys():
            correct_predicted_notes = difference_statistics[3]
        else:
            correct_predicted_notes = 0
        if 2 in difference_statistics.keys():
            not_predicted_notes = difference_statistics[2]
        else:
            not_predicted_notes = 0
        if 1 in difference_statistics.keys():
            new_predicted_notes = difference_statistics[1]
        else:
            new_predicted_notes = 0
        pitch_reconstruction_accuracy = correct_predicted_notes/total_original_notes
        total_original_notes_array.append(total_original_notes)
        total_predicted_notes_array.append(total_predicted_notes)
        reconstruction_accuracy_array.append(pitch_reconstruction_accuracy)
        not_predicted_notes_array.append(not_predicted_notes)
        new_predicted_notes_array.append(new_predicted_notes)
        print("Total original notes: ", total_original_notes)
        print("Total predicted notes: ", total_predicted_notes)
        print("Pitch Reconstruction accuracy: ",  pitch_reconstruction_accuracy)
        if total_original_notes > 0:
            print("Not predicted notes/Original notes: ", not_predicted_notes/total_original_notes)
        if total_predicted_notes > 0:
            print("New predicted notes/Predicted notes: ", new_predicted_notes/total_predicted_notes)
        metrics_for_this_song_dict["total_original_notes"] = total_original_notes
        mean_metrics_for_all_songs_dict["total_original_notes"] += total_original_notes
        metrics_for_this_song_dict["total_predicted_notes"] = total_predicted_notes
        mean_metrics_for_all_songs_dict["total_predicted_notes"] += total_predicted_notes
        metrics_for_this_song_dict["pitch_reconstruction_accuracy"] = pitch_reconstruction_accuracy
        mean_metrics_for_all_songs_dict["pitch_reconstruction_accuracy"] += pitch_reconstruction_accuracy


        # ----------------------------------------------------------------------------------------------
        # Create mix by interpolating with previous
        # ----------------------------------------------------------------------------------------------
        
        if mix_with_previous:
            #generate mix with previous song
            if len(previous_latent_list) > 0:
                
                if encoded_representation.shape[0] <= previous_latent_list.shape[0]:
                    interpolated_representation = (encoded_representation + previous_latent_list[:encoded_representation.shape[0]]) / 2
                else:
                    interpolated_representation = (encoded_representation[:previous_latent_list.shape[0]] + previous_latent_list) / 2

                mix_length = interpolated_representation.shape[0]
                S_mix = np.zeros((mix_length, signature_vector_length))

                mix_input_list = vae_definition.prepare_decoder_input(interpolated_representation, C, S_mix)

                decoder_outputs = decoder.predict(mix_input_list, batch_size=batch_size, verbose=False)
                        
                Y_pred, I_pred, V_pred, D_pred, N_pred = vae_definition.process_decoder_outputs(decoder_outputs, sample_method)

                mixed_programs = vote_for_programs(I_pred)

                if save_anything: mf.rolls_to_midi(Y_pred, mixed_programs, save_folder, song_name + '_x_' + previous_song_name + '_mixed', BPM, V_pred, D_pred)

        # ----------------------------------------------------------------------------------------------
        # Switch style component and evaluate
        # ----------------------------------------------------------------------------------------------

        if switch_styles:
            #switch composer (or style)
            if include_composer_decoder:
                for C_switch, class_name_switch in enumerate(classes):
                    if C != C_switch:
                
                        switched_signature_list = []
                        Y_list_switched = []
                        I_list_switched = []
                        V_list_switched = []
                        D_list_switched = []

                        previous_switched_rep = np.zeros((1,latent_dim))

                        switched_pitches_classifier_accuracy = 0.0
                        switched_pitches_classifier_confidence = 0.0
                        switched_velocity_classifier_accuracy = 0.0
                        switched_velocity_classifier_confidence = 0.0
                        switched_instrument_classifier_accuracy = 0.0
                        switched_instrument_classifier_confidence = 0.0
                        switched_ensemble_classifier_accuracy = 0.0
                        switched_ensemble_classifier_confidence = 0.0

                        for i in range(len(encoded_representation)):

                            #switch the style
                            original_rep = encoded_representation[i]
                            switched_rep = np.copy(original_rep)
                            switched_rep[C] = original_rep[C_switch]
                            switched_rep[C_switch] = original_rep[C]
                            switched_rep = np.asarray([switched_rep])

                            #run the switched z into the decoder
                            adapted_input_list = vae_definition.prepare_decoder_input(switched_rep, C_switch, S[i], previous_switched_rep)
                            decoder_outputs = decoder.predict(adapted_input_list, batch_size=batch_size, verbose=False)
                            Y_switched, I_switched, V_switched, D_switched, N_switched = vae_definition.process_decoder_outputs(decoder_outputs, sample_method)

                            #save all outputs to form a long switched song
                            Y_list_switched.extend(Y_switched)
                            I_list_switched.extend(I_switched)
                            V_list_switched.extend(V_switched)
                            D_list_switched.extend(D_switched)

                            pitches_classifier_input = np.copy(Y_switched)
                            if include_silent_note:
                                pitches_classifier_input = np.append(pitches_classifier_input, np.zeros((pitches_classifier_input.shape[0], 1)), axis=1)
                                for step in range(pitches_classifier_input.shape[0]):
                                    if np.sum(pitches_classifier_input[step]) == 0:
                                        pitches_classifier_input[step, -1] = 1
                            pitches_classifier_input = np.asarray([pitches_classifier_input])
                            pitches_classifier_prediction = pitches_classifier_model.predict(pitches_classifier_input)[0]
                            #get the confidence of the classifier, if this switched song is from the this class
                            switched_pitches_confidence = pitches_classifier_prediction[C]
                            switched_pitches_classifier_confidence += switched_pitches_confidence
                            if np.argmax(pitches_classifier_prediction) == C:
                                switched_pitches_classifier_accuracy += 1

                            if meta_velocity:
                                #calculate velocity style classifier accuracy
                                velocity_split_song = np.copy(V_switched)
                                velocity_split_song = np.expand_dims(velocity_split_song, 1)
                                velocity_classifier_input = np.asarray([velocity_split_song])
                                velocity_classifier_prediction = velocity_classifier_model.predict(velocity_classifier_input)[0]
                                #get the confidence of the style classifier
                                switched_velocity_confidence = velocity_classifier_prediction[C]
                                switched_velocity_classifier_confidence += switched_velocity_confidence
                                if np.argmax(velocity_classifier_prediction) == C:
                                    switched_velocity_classifier_accuracy += 1

                            if meta_instrument:
                                #calculate instrument style classifier accuracy
                                instrument_classifier_input = I_switched
                                instrument_classifier_prediction = instrument_classifier_model.predict(instrument_classifier_input)[0]
                                #get the confidence of the style classifier
                                switched_instrument_confidence = instrument_classifier_prediction[C]
                                switched_instrument_classifier_confidence += switched_instrument_confidence
                                if np.argmax(instrument_classifier_prediction) == C:
                                    switched_instrument_classifier_accuracy += 1

                            if meta_velocity and meta_instrument:
                                #calculate ensemble style classifier accuracy
                                ensemble_classifier_prediction = ensemble_prediction(pitches_classifier_input, instrument_classifier_input, velocity_classifier_input)[0]
                                #get the confidence of the style classifier
                                switched_ensemble_confidence = ensemble_classifier_prediction[C]
                                switched_ensemble_classifier_confidence += switched_ensemble_confidence
                                if np.argmax(ensemble_classifier_prediction) == C:
                                    switched_ensemble_classifier_accuracy += 1

                            #calculate harmonicity matrix
                            harmonicity_matrix_switched_from_class_to_class_list[C][C_switch].append(data_class.get_harmonicity_scores_for_each_track_combination(Y_switched))

                            #calculate signature vectors
                            poly_sample = data_class.monophonic_to_khot_pianoroll(Y_switched, max_voices)
                            signature = data_class.signature_from_pianoroll(poly_sample)
                            switched_signature_list_for_each_class[C_switch].append(signature)
                            switched_signature_list.append(signature)

                            #evaluate switched programs
                            switched_programs = data_class.instrument_representation_to_programs(I_switched[0], instrument_attach_method)
                            switched_instruments_for_each_class[C][C_switch].append(switched_programs)
                            
                            #prepare for next loop step
                            previous_switched_rep = switched_rep

                        #save style pitch classifier accuracies
                        switched_pitches_classifier_accuracy /= len(encoded_representation)
                        switched_pitches_classifier_confidence /= len(encoded_representation)
                        switched_pitches_classifier_accuracy_list.append(switched_pitches_classifier_accuracy)
                        switched_pitches_classifier_confidence_list.append(switched_pitches_classifier_confidence)
                        print("Switched style pitch classifier accuracy: ", switched_pitches_classifier_accuracy)
                        print("Switched style pitch classifier confidence: ", switched_pitches_classifier_confidence)
                        metrics_for_this_song_dict["switched_pitches_classifier_accuracy"] = switched_pitches_classifier_accuracy
                        mean_metrics_for_all_songs_dict["switched_pitches_classifier_accuracy"] += switched_pitches_classifier_accuracy
                        metrics_for_this_song_dict["switched_pitches_classifier_confidence"] = switched_pitches_classifier_confidence
                        mean_metrics_for_all_songs_dict["switched_pitches_classifier_confidence"] += switched_pitches_classifier_confidence
                        
                        if meta_velocity:
                            #save style velocity classifier accuracies
                            switched_velocity_classifier_accuracy /= len(encoded_representation)
                            switched_velocity_classifier_confidence /= len(encoded_representation)
                            switched_velocity_classifier_accuracy_list.append(switched_velocity_classifier_accuracy)
                            switched_velocity_classifier_confidence_list.append(switched_velocity_classifier_confidence)
                            print("Switched style velocity classifier accuracy: ", switched_velocity_classifier_accuracy)
                            print("Switched style velocity classifier confidence: ", switched_velocity_classifier_confidence)
                            metrics_for_this_song_dict["switched_velocity_classifier_accuracy"] = switched_velocity_classifier_accuracy
                            mean_metrics_for_all_songs_dict["switched_velocity_classifier_accuracy"] += switched_velocity_classifier_accuracy
                            metrics_for_this_song_dict["switched_velocity_classifier_confidence"] = switched_velocity_classifier_confidence
                            mean_metrics_for_all_songs_dict["switched_velocity_classifier_confidence"] += switched_velocity_classifier_confidence


                        if meta_instrument:
                            #save style instrument classifier accuracies
                            switched_instrument_classifier_accuracy /= len(encoded_representation)
                            switched_instrument_classifier_confidence /= len(encoded_representation)
                            switched_instrument_classifier_accuracy_list.append(switched_instrument_classifier_accuracy)
                            switched_instrument_classifier_confidence_list.append(switched_instrument_classifier_confidence)
                            print("Switched style instrument classifier accuracy: ", switched_instrument_classifier_accuracy)
                            print("Switched style instrument classifier confidence: ", switched_instrument_classifier_confidence)
                            metrics_for_this_song_dict["switched_instrument_classifier_accuracy"] = switched_instrument_classifier_accuracy
                            mean_metrics_for_all_songs_dict["switched_instrument_classifier_accuracy"] += switched_instrument_classifier_accuracy
                            metrics_for_this_song_dict["switched_instrument_classifier_confidence"] = switched_instrument_classifier_confidence
                            mean_metrics_for_all_songs_dict["switched_instrument_classifier_confidence"] += switched_instrument_classifier_confidence

                        if meta_velocity and meta_instrument:
                            #save style ensemble classifier accuracies
                            switched_ensemble_classifier_accuracy /= len(encoded_representation)
                            switched_ensemble_classifier_confidence /= len(encoded_representation)
                            switched_ensemble_classifier_accuracy_list.append(switched_ensemble_classifier_accuracy)
                            switched_ensemble_classifier_confidence_list.append(switched_ensemble_classifier_confidence)
                            print("Switched style ensemble classifier accuracy: ", switched_ensemble_classifier_accuracy)
                            print("Switched style ensemble classifier confidence: ", switched_ensemble_classifier_confidence)
                            metrics_for_this_song_dict["switched_ensemble_classifier_accuracy"] = switched_ensemble_classifier_accuracy
                            mean_metrics_for_all_songs_dict["switched_ensemble_classifier_accuracy"] += switched_ensemble_classifier_accuracy
                            metrics_for_this_song_dict["switched_ensemble_classifier_confidence"] = switched_ensemble_classifier_confidence
                            mean_metrics_for_all_songs_dict["switched_ensemble_classifier_confidence"] += switched_ensemble_classifier_confidence

                            switched_ensemble_classifier_accuracy_list_for_each_class[C].append(switched_ensemble_classifier_accuracy)

                        switched_programs_for_whole_song = vote_for_programs(I_list_switched)

                        for program, switched_program in zip(programs, switched_programs_for_whole_song):
                            if instrument_attach_method == '1hot-category' or 'khot-category':
                                switch_instruments_matrix[C, C_switch, program//8, switched_program//8] += 1
                            else:
                                switch_instruments_matrix[C, C_switch, program, switched_program] += 1

                        if meta_instrument and switched_programs_for_whole_song != programs:
                            switch_string = 'SI_'
                            instrument_switched_signature_list_for_each_class[C].extend(switched_signature_list)
                        else:
                            switch_string = ''
                            switched_programs_for_whole_song = programs

                        Y_list_switched = np.asarray(Y_list_switched)
                        V_list_switched = np.asarray(V_list_switched)
                        D_list_switched = np.asarray(D_list_switched)

                        if save_anything: mf.rolls_to_midi(Y_list_switched, switched_programs_for_whole_song, save_folder, song_name + '_fullswitch_' + switch_string + str(C) + "to" +str(C_switch), BPM, V_list_switched, D_list_switched)

        # ----------------------------------------------------------------------------------------------
        # Prepare for next round and save evaluation progress in pickle files
        # ----------------------------------------------------------------------------------------------
        
        #make data ready for next iteration
        previous_song_name = song_name
        previous_latent_list = encoded_representation  
        previous_programs = programs

        metrics_dict_for_all_songs_list.append(metrics_for_this_song_dict)

    # ----------------------------------------------------------------------------------------------
    # Store evaluation array that are not saved in the csv
    # ----------------------------------------------------------------------------------------------

    if save_anything: pickle.dump(total_original_notes_array,open(save_folder+'aaa_total_original_notes_array.pickle', 'wb'))
    if save_anything: pickle.dump(reconstruction_accuracy_array,open(save_folder+'aaa_reconstruction_accuracy_array.pickle', 'wb'))
    if save_anything: pickle.dump(total_predicted_notes_array,open(save_folder+'aaa_total_predicted_notes_array.pickle', 'wb'))
    if save_anything: pickle.dump(new_predicted_notes_array,open(save_folder+'aaa_new_predicted_notes_array.pickle', 'wb'))
    if save_anything: pickle.dump(not_predicted_notes_array,open(save_folder+'aaa_not_predicted_notes_array.pickle', 'wb'))
    if save_anything: pickle.dump(classifier_accuracy_array,open(save_folder+'aaa_classifier_accuracy_array.pickle', 'wb'))
    if save_anything: pickle.dump(composer_accuracy_array,open(save_folder+'aaa_composer_accuracy_array.pickle', 'wb'))

    if save_anything: pickle.dump(switched_instruments_for_each_class,open(save_folder+'aaa_switched_instruments_for_each_class.pickle', 'wb'))
    if save_anything: pickle.dump(original_signature_list_for_each_class,open(save_folder+'aaa_original_signature_list_for_each_class.pickle', 'wb'))
    if save_anything: pickle.dump(autoencoded_signature_list_for_each_class,open(save_folder+'aaa_autoencoded_signature_list_for_each_classs.pickle', 'wb'))
    if save_anything: pickle.dump(switched_signature_list_for_each_class,open(save_folder+'aaa_switched_signature_list_for_each_class.pickle', 'wb'))
    if save_anything: pickle.dump(instrument_switched_signature_list_for_each_class,open(save_folder+'aaa_instrument_switched_signature_list_for_each_class.pickle', 'wb'))

    if save_anything: pickle.dump(note_start_prediction_to_original_errors_list,open(save_folder+'aaa_note_start_prediction_to_original_errors_lists.pickle', 'wb'))
    if save_anything: pickle.dump(note_start_prediction_to_prediction_errors_list,open(save_folder+'aaa_note_start_prediction_to_prediction_errors_list.pickle', 'wb'))

    if save_anything: pickle.dump(harmonicity_matrix_autoencoded_list,open(save_folder+'aaa_harmonicity_matrix_autoencoded_list.pickle', 'wb'))
    if save_anything: pickle.dump(instrument_switched_signature_list_for_each_class,open(save_folder+'aaa_harmonicity_matrix_switched_from_class_to_class_list.pickle', 'wb'))

    if save_anything: pickle.dump(original_ensemble_classifier_accuracy_list_for_each_class,open(save_folder+'aaa_original_ensemble_classifier_accuracy_list_for_each_class.pickle', 'wb'))
    if save_anything: pickle.dump(autoencoded_ensemble_classifier_accuracy_list_for_each_class,open(save_folder+'aaa_autoencoded_ensemble_classifier_accuracy_list_for_each_class.pickle', 'wb'))
    if save_anything: pickle.dump(switched_ensemble_classifier_accuracy_list_for_each_class,open(save_folder+'aaa_switched_ensemble_classifier_accuracy_list_for_each_class.pickle', 'wb'))

    if save_anything: pickle.dump(all_programs_plus_length_for_each_class,open(save_folder+'aaa_all_programs_plus_length_for_each_class.pickle', 'wb'))
    if save_anything: pickle.dump(switched_instruments_for_each_class,open(save_folder+'aaa_switched_instruments_for_each_class.pickle', 'wb'))




    # ----------------------------------------------------------------------------------------------
    # Print statistics
    # ----------------------------------------------------------------------------------------------
    print("\n---------------------\n")

    if include_composer_decoder:
        print("Pitch classifier prediction")
        print("Original mean pitches accuracy: ", np.mean(original_pitches_classifier_accuracy_list))
        print("Autoencoded mean pitches accuracy: ", np.mean(autoencoded_pitches_classifier_accuracy_list))
        if switch_styles: print("Switched mean pitches accuracy: ", np.mean(switched_pitches_classifier_accuracy_list))

        print("Original mean pitches confidence: ", np.mean(original_pitches_classifier_confidence_list))
        print("Autoencoded mean pitches confidence: ", np.mean(autoencoded_pitches_classifier_confidence_list))
        if switch_styles: print("Switched mean pitches confidence: ", np.mean(switched_pitches_classifier_confidence_list))

    if meta_velocity:
        print("Velocity classifier prediction")
        print("Original mean velocity accuracy: ", np.mean(original_velocity_classifier_accuracy_list))
        print("Autoencoded mean velocity accuracy: ", np.mean(autoencoded_velocity_classifier_accuracy_list))
        if switch_styles: print("Switched mean velocity accuracy: ", np.mean(switched_velocity_classifier_accuracy_list))

        print("Original mean velocity confidence: ", np.mean(original_velocity_classifier_confidence_list))
        print("Autoencoded mean velocity confidence: ", np.mean(autoencoded_velocity_classifier_confidence_list))
        if switch_styles: print("Switched mean velocity confidence: ", np.mean(switched_velocity_classifier_confidence_list))

    if meta_instrument:
        print("Instrument classifier prediction")
        print("Original mean instrument accuracy: ", np.mean(original_instrument_classifier_accuracy_list))
        print("Autoencoded mean instrument accuracy: ", np.mean(autoencoded_instrument_classifier_accuracy_list))
        if switch_styles: print("Switched mean instrument accuracy: ", np.mean(switched_instrument_classifier_accuracy_list))

        print("Original mean instrument confidence: ", np.mean(original_instrument_classifier_confidence_list))
        print("Autoencoded mean instrument confidence: ", np.mean(autoencoded_instrument_classifier_confidence_list))
        if switch_styles: print("Switched mean instrument confidence: ", np.mean(switched_instrument_classifier_confidence_list))

    if meta_velocity and meta_instrument:
        print("Ensemble classifier prediction")
        print("Original mean ensemble accuracy: ", np.mean(original_ensemble_classifier_accuracy_list))
        print("Autoencoded mean ensemble accuracy: ", np.mean(autoencoded_ensemble_classifier_accuracy_list))
        if switch_styles: print("Switched mean ensemble accuracy: ", np.mean(switched_ensemble_classifier_accuracy_list))

        print("Original mean ensemble confidence: ", np.mean(original_ensemble_classifier_confidence_list))
        print("Autoencoded mean ensemble confidence: ", np.mean(autoencoded_ensemble_classifier_confidence_list))
        if switch_styles: print("Switched mean ensemble confidence: ", np.mean(switched_ensemble_classifier_confidence_list))


    if meta_velocity or meta_held_notes:
        print("Mean note start errors compared to original notes: ", np.mean(note_start_prediction_to_original_errors_list))
        print("Mean note start errors compared to predicted notes: ", np.mean(note_start_prediction_to_prediction_errors_list))


    harmonicity_matrix_autoencoded_list = np.asarray(harmonicity_matrix_autoencoded_list)
    print("Autoencoded harmonicity matrix:\n", np.nanmean(harmonicity_matrix_autoencoded_list, axis=0))

    if switch_styles:
        if meta_instrument and include_composer_decoder:
            for C, class_name in enumerate(classes):

                print("Class " + class_name)
                print("Original ensemble accuracy ", np.mean(original_ensemble_classifier_accuracy_list_for_each_class[C]))
                print("Autoencoded ensemble accuracy ", np.mean(autoencoded_ensemble_classifier_accuracy_list_for_each_class[C]))
                print("Switched ensemble accuracy ", np.mean(switched_ensemble_classifier_accuracy_list_for_each_class[C]))

                for C_switch, class_name_switch in enumerate(classes):
                    if C != C_switch:
                        print("Evaluating instrument switch from " + class_name + " to " + class_name_switch)

                        switched_instrument_probability_in_this_class = 0
                        switched_instrument_probability_in_switched_class = 0

                        total_programs = 0.0
                        for programs in switched_instruments_for_each_class[C][C_switch]:
                            for program in programs:
                                total_programs += 1
                                switched_instrument_probability_in_this_class += program_probability_dict_for_each_class[C].get(program, 0)
                                switched_instrument_probability_in_switched_class += program_probability_dict_for_each_class[C_switch].get(program, 0)

                        switched_instrument_probability_in_this_class /= total_programs
                        switched_instrument_probability_in_switched_class /= total_programs

                        print("Probability of instruments that they are from " + class_name + " set: ", switched_instrument_probability_in_this_class)
                        print("Probability of instruments that they are from " + class_name_switch + " set: ", switched_instrument_probability_in_switched_class)


                        harmonicity_matrix_for_this_switch = np.asarray(harmonicity_matrix_switched_from_class_to_class_list[C][C_switch])
                        print("Harmonicity between voices of switched songs:\n", np.nanmean(harmonicity_matrix_for_this_switch, axis=0))


                        print("Calculating how many instruments switches have to be made from switched " + class_name + " to original " + class_name_switch)
                        same = 0.0
                        different = 0.0
                        programs_plus_length_for_other_class = all_programs_plus_length_for_each_class[C_switch]
                        for programs in switched_instruments_for_each_class[C][C_switch]:
                            for programs_switch, length_switch in programs_plus_length_for_other_class:
                                for this_program, other_program in zip(programs, programs_switch):
                                    if this_program == other_program:
                                        same += length_switch
                                    else:
                                        different += length_switch
                        print("Switch percentage unswitched to other class: ", different / (same + different))


            for C, class_name in enumerate(classes):
                for C_switch, class_name_switch in enumerate(classes):
                    confusion_matrix = switch_instruments_matrix[C, C_switch]
                    not_switched_instruments_count = np.sum(np.diag(confusion_matrix))
                    total_instruments_count = np.sum(confusion_matrix)
                    if total_instruments_count > 0:
                        switched_instruments_count = total_instruments_count - not_switched_instruments_count
                        confusion_matrix = confusion_matrix/confusion_matrix.sum(axis=1, keepdims=True)
                        confusion_matrix = confusion_matrix/total_instruments_count
                        plt.figure()
                        plt.imshow(confusion_matrix, interpolation='nearest')
                        plt.title(classes[C] + ' switched to ' + classes[C_switch]+ ': Switched instruments: %6.2f %%' % (switched_instruments_count/total_instruments_count*100.))
                        plt.ylabel('Original instrument')
                        plt.xlabel('Switched instrument')
                        if instrument_attach_method == '1hot-category' or 'khot-category':
                            plt.xticks(np.arange(0,16), instrument_category_names, rotation='vertical')
                            plt.yticks(np.arange(0,16), instrument_category_names)
                        else:
                            plt.xticks(np.arange(0,128), instrument_names, rotation='vertical')
                            plt.yticks(np.arange(0,128), instrument_names)
                        
                        plt.colorbar()
                        plt.tight_layout()
                        if save_anything: 
                            plt.savefig(save_folder+'aaa_switch_matrix_total_normalized_' + classes[C] + '_to_' + classes[C_switch] +'.png')
                            tikz_save(save_folder+'aaa_switch_matrix_total_normalized_' + classes[C] + '_to_' + classes[C_switch] +'.tex', encoding='utf-8', show_info=False)
                        plt.close()


                        switched_instruments_count = total_instruments_count - not_switched_instruments_count
                        confusion_matrix = switch_instruments_matrix[C, C_switch]
                        confusion_matrix = confusion_matrix/confusion_matrix.sum(axis=1, keepdims=True)
                        #confusion_matrix = confusion_matrix/total_instruments_count
                        plt.figure()
                        plt.imshow(confusion_matrix, interpolation='nearest')
                        plt.title(classes[C] + ' switched to ' + classes[C_switch]+ ': Switched instruments: %6.2f %%' % (switched_instruments_count/total_instruments_count*100.))
                        plt.ylabel('Original instrument')
                        plt.xlabel('Switched instrument')
                        if instrument_attach_method == '1hot-category' or 'khot-category':
                            plt.xticks(np.arange(0,16), instrument_category_names, rotation='vertical')
                            plt.yticks(np.arange(0,16), instrument_category_names)
                        else:
                            plt.xticks(np.arange(0,128), instrument_names, rotation='vertical')
                            plt.yticks(np.arange(0,128), instrument_names)
                        
                        plt.colorbar()
                        plt.tight_layout()
                        if save_anything: 
                            plt.savefig(save_folder+'aaa_switch_matrix_row_normalized' + classes[C] + '_to_' + classes[C_switch] + '.png')
                            tikz_save(save_folder+'aaa_switch_matrix_row_normalized' + classes[C] +'_to_' + classes[C_switch] + '.tex', encoding='utf-8', show_info=False)
                        plt.close()

    for C in range(num_classes):
        original_class = classes[C]
        print("Signature train set: " + classes[C])
        S_train_for_this_class = S_train_for_each_class[C]
        mean, cov = data_class.get_mean_and_cov_from_vector_list(S_train_for_this_class)

        for other_class in range(num_classes):

            train_distances = []
            for s in S_train_for_each_class[other_class]:
                distance = data_class.mahalanobis_distance(s, mean, cov)
                train_distances.append(distance)
            print("Mean (+std) distance of original train songs from " + classes[other_class] + " to train " + original_class + ": %4.4f (%4.4f)" % (np.mean(train_distances), np.std(train_distances)))

            original_distances = []
            for s in original_signature_list_for_each_class[other_class]:
                distance = data_class.mahalanobis_distance(s, mean, cov)
                original_distances.append(distance)
            print("Mean (+std) distance of original test songs from " + classes[other_class] + " to train " + original_class + ": %4.4f (%4.4f)" % (np.mean(original_distances), np.std(original_distances)))

            autoencoded_distances = []
            for s in autoencoded_signature_list_for_each_class[other_class]:
                distance = data_class.mahalanobis_distance(s, mean, cov)
                autoencoded_distances.append(distance)
            print("Mean (+std) distance of autoencoded test songs from " + classes[other_class] + " to train " + original_class + ": %4.4f (%4.4f)" % (np.mean(autoencoded_distances), np.std(autoencoded_distances)))

            if switch_styles:

                switched_distances = []
                for s in switched_signature_list_for_each_class[other_class]:
                    distance = data_class.mahalanobis_distance(s, mean, cov)
                    switched_distances.append(distance)
                print("Mean (+std) distance of switched test songs from " + classes[other_class] + " to train " + original_class + ": %4.4f (%4.4f)" % (np.mean(switched_distances), np.std(switched_distances)))

                instrument_switched_distances = []
                for s in instrument_switched_signature_list_for_each_class[other_class]:
                    distance = data_class.mahalanobis_distance(s, mean, cov)
                    instrument_switched_distances.append(distance)
                print("Mean (+std) distance of instrument switched test songs from " + classes[other_class] + " to train " + original_class + ": %4.4f (%4.4f)" % (np.mean(instrument_switched_distances), np.std(instrument_switched_distances)))



    if append_signature_vector_to_latent:
        print("Mean successful signature manipulations: ", np.mean(successful_signature_manipulations_list))
        print("Mean unsuccessful signature manipulations: ", np.mean(unsuccessful_signature_manipulations_list))
        print("Mean neutral signature manipulations: ", np.mean(neutral_signature_manipulations_list))
        print("Most successful dim list", most_successful_dim_list)
        print("Least successful dim list", least_successful_dim_list)

    #print statistics
    print("Mean total original notes: ", np.mean(total_original_notes_array))
    print("Mean total predicted notes: ", np.mean(total_predicted_notes_array))
    print("Mean reconstruction accuracy: ", np.mean(reconstruction_accuracy_array))
    print("Std reconstruction accuracy: ", np.std(reconstruction_accuracy_array))
    print("Not predicted notes: ", np.mean(not_predicted_notes_array))
    print("New predicted notes: ", np.mean(new_predicted_notes_array))


    # ----------------------------------------------------------------------------------------------
    # Save statistics to csv
    # ----------------------------------------------------------------------------------------------

    #divide every value (except the song_name) by the number of tested songs
    for k in list(mean_metrics_for_all_songs_dict.keys()):
        if k != "song_name" and k != "class":
            mean_metrics_for_all_songs_dict[k] /= l

    metrics_dict_for_all_songs_list.append(mean_metrics_for_all_songs_dict)


    if test_train_set:
        train_or_test_string = 'train'
    else:
        train_or_test_string = 'test'

    with open(save_folder + model_name[:10] + 'beta_' + str(beta) + 'epsstd_' + str(epsilon_std) + '_' + train_or_test_string +'.csv','w', newline='') as f:
        w = csv.writer(f)

        #Write header with all keys of the dict
        w.writerow(metrics_dict_for_all_songs_list[0].keys())

        #write the evaluations for every song on each row
        for metric_dict in metrics_dict_for_all_songs_list:
            w.writerow(metric_dict.values())

