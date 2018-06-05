from settings import *
import numpy as np
import _pickle as pickle
import os
import midi_functions as mf

import matplotlib.patches as mpatches
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import colors

from matplotlib2tikz import save as tikz_save


# ----------------------------------------------------------------------------------------------
# Harmonicity evaluation
#found on https://arxiv.org/pdf/1709.06298.pdf MuseGAN
#originally from https://www.researchgate.net/publication/200806168_Detecting_harmonic_change_in_musical_audio

#code modified from
#https://github.com/salu133445/musegan
# ----------------------------------------------------------------------------------------------


def get_tonal_matrix(r1=1.0, r2=1.0, r3=0.5):
    tm = np.empty((6, 12), dtype=np.float32)
    tm[0, :] = r1*np.sin(np.arange(12)*(7./6.)*np.pi)
    tm[1, :] = r1*np.cos(np.arange(12)*(7./6.)*np.pi)
    tm[2, :] = r2*np.sin(np.arange(12)*(3./2.)*np.pi)
    tm[3, :] = r2*np.cos(np.arange(12)*(3./2.)*np.pi)
    tm[4, :] = r3*np.sin(np.arange(12)*(2./3.)*np.pi)
    tm[5, :] = r3*np.cos(np.arange(12)*(2./3.)*np.pi)
    return tm

#returns nan if one of the chromas is empty
def tonal_dist(beat_chroma1, beat_chroma2):

    #skip empty bars
    if np.sum(beat_chroma1) == 0 or np.sum(beat_chroma1) == 0:
        return np.nan

    beat_chroma1 = beat_chroma1 / np.sum(beat_chroma1)
    beat_chroma2 = beat_chroma2 / np.sum(beat_chroma2)
    tonal_matrix = get_tonal_matrix()
    c1 = np.matmul(tonal_matrix, beat_chroma1)
    c2 = np.matmul(tonal_matrix, beat_chroma2)
    return np.linalg.norm(c1-c2)


def to_chroma(track):
    chroma = track.reshape(track.shape[0], 12, -1).sum(axis=2)
    return chroma

#use same resolutition in terms of bars.
#museGAN used resolution = 24 with a bar-length of 96, so they had a resolution of a fourth of a bar
def metrics_harmonicity(chroma1, chroma2, resolution=SMALLEST_NOTE//4):
    score_list = []
    for r in range(chroma1.shape[0]//resolution):
        chr1 = np.sum(chroma1[resolution*r: resolution*(r+1)], axis=0)
        chr2 = np.sum(chroma2[resolution*r: resolution*(r+1)], axis=0)
        dist = tonal_dist(chr1, chr2)
        score_list.append(tonal_dist(chr1, chr2))
    return np.nanmean(score_list)

def get_harmonicity_scores_for_each_track_combination(unrolled_pianoroll):

    if unrolled_pianoroll.ndim > 2:
        length = unrolled_pianoroll.shape[0]
        spm = np.empty((length, max_voices, max_voices))
        
        for i in range(length):
            spm[i] = get_harmonicity_scores_for_each_track_combination(unrolled_pianoroll[i])
        return np.nanmean(spm, axis=0)

    score_pair_matrix = np.zeros((max_voices, max_voices))

    chromas = []
    for voice in range(max_voices):
        track = np.copy(unrolled_pianoroll[voice::max_voices])
        chroma = to_chroma(track)
        chromas.append(chroma)

    for voice_1 in range(max_voices):
        for voice_2 in range(voice_1):
            score_pair_matrix[voice_1, voice_2] = metrics_harmonicity(chromas[voice_1], chromas[voice_2])
            score_pair_matrix[voice_2, voice_1] = score_pair_matrix[voice_1, voice_2]

    return score_pair_matrix


# ----------------------------------------------------------------------------------------------
# Signature vector evaluation
# ----------------------------------------------------------------------------------------------


def get_statistics_on_list(l, scale=1.0):
    stats = []

    highest = 0
    lowest = 0
    mean = 0
    std = 0
    if len(l) > 0:
        highest = np.max(l)
        lowest = np.min(l)
        mean = np.mean(l)
        std = np.std(l)

    stats.append(highest / scale)
    stats.append(lowest / scale)
    stats.append(mean / scale)
    stats.append(std / scale)
    return stats


def signature_from_index(song):
    signature_list = []
    
    #preprocessing of statistics
    polyphonic_count = 0
    previous_notes = ()
    all_notes_flattened_list = []
    pitch_interval_range_list = []
    duration_list = []
    held_notes = []
    held_notes_how_long = []
    for notes in song:

        #update the held_notes
        for note in held_notes:
            index = held_notes.index(note)
            if note not in notes:
                duration_list.append(held_notes_how_long[index])
                del held_notes[index]
                del held_notes_how_long[index]

        for note in notes:
            all_notes_flattened_list.append(note)
            if note in held_notes:
                held_notes_how_long[held_notes.index(note)] += 1
            else:
                held_notes.append(note)
                held_notes_how_long.append(1)

        #intervals between two consecutive notes (with or without being separated by a period of silence)
        #the notes may not be aligned -> find notes which are close to before in terms of absolute difference
        if len(notes) != len(previous_notes) and len(notes) != 0 and len(previous_notes) != 0:
            #hard case: found the notes that can be called 'consecutive'
            if len(notes) < len(previous_notes):
                shorter_list = notes
                longer_list = previous_notes
            else:
                shorter_list = previous_notes
                longer_list = notes
            shortest_distance_to_other_notes = []
            for pitch in longer_list:
                shortest_distance = 9999
                for other_pitch in shorter_list:
                    dist = abs(pitch-other_pitch)
                    if dist < shortest_distance:
                        shortest_distance = dist
                shortest_distance_to_other_notes.append(shortest_distance)
            truncated_list = []
            for index in np.argsort(shortest_distance_to_other_notes)[:len(shorter_list)]:
                truncated_list.append(longer_list[index])
            #only take those values of the longer list, which have the shortest distance to the shorter notes
            zip_pitch_list = zip(sorted(shorter_list), sorted(truncated_list))

        else:
            #easy case: just compare the sorted notes
            zip_pitch_list = zip(sorted(notes), sorted(previous_notes))
        for (note_1, note_2) in zip_pitch_list:
            pitch_interval_range_list.append(abs(note_1-note_2)) #original: abs, but 2 notes upwards and downwards shouldn't be the same, future improvement



        if len(notes) > 1:
            polyphonic_count += 1
        #if there is silence, ignore this step
        if len(notes) > 0:
            previous_notes = notes
        else:
            duration_list.extend(held_notes_how_long)
            held_notes = []
            held_notes_how_long = []
    

    #number of total (held or played right after) notes divided by length
    signature_list.append(len(duration_list) / len(song))

    #occupation rate in piano_roll
    signature_list.append(len(all_notes_flattened_list) / len(song))

    #polyphonic rate
    signature_list.append(polyphonic_count / len(song))

    #pitch range descriptors
    signature_list.extend(get_statistics_on_list(all_notes_flattened_list, scale=127))

    #pitch interval range
    signature_list.extend(get_statistics_on_list(pitch_interval_range_list, scale=127))

    #duration range
    signature_list.extend(get_statistics_on_list(duration_list, scale=1.0))

    return signature_list

def signature_from_pianoroll(pianoroll):
    song = []
    for step in pianoroll:
        indices = step.nonzero()[0]
        #add low_crop because pianoroll is shifted
        indices = [x + low_crop for x in indices]
        song.append(tuple(indices))
    return signature_from_index(song)

def signature_form_unrolled_pianoroll(pianoroll, voices, include_silent_note):
    poly_sample = monophonic_to_khot_pianoroll(pianoroll, max_voices)
    if include_silent_note:
        poly_sample = poly_sample[:,:-1]
    return signature_from_pianoroll(poly_sample)



def mahalanobis_distance(x, mean, cov):
    cov_I = np.linalg.pinv(cov)
    diff = x - mean
    return np.sqrt(np.dot(np.dot(diff, cov_I), diff.T))

def get_mean_and_cov_from_vector_list(vector_list):
    mean = np.mean(vector_list, axis=0)
    cov = np.cov(np.transpose(vector_list))
    return mean, cov


# ----------------------------------------------------------------------------------------------
# Pianoroll manipulations
# ----------------------------------------------------------------------------------------------


def monophonic_to_khot_pianoroll(pianoroll, max_voices, set_all_nonzero_to_1=True):
    assert(max_voices > 1)
    polyphonic_X = np.zeros((pianoroll.shape[0]//max_voices, pianoroll.shape[1]))
    for step in range(pianoroll.shape[0]):
        polyphonic_X[step//max_voices] += pianoroll[step]

    #it may happen, that a note is now higher than 1 in polyphonic X
    #set all nonzero indices to 1
    if set_all_nonzero_to_1:
        nonzero_indices = np.nonzero(polyphonic_X)
        polyphonic_X[nonzero_indices] = 1
    return polyphonic_X


# ----------------------------------------------------------------------------------------------
# Draw plots from pianoroll
# ----------------------------------------------------------------------------------------------


def draw_mixture_pianoroll(song_1, song_2, mixture_song, name_1='Song 1', name_2='Song 2', mixture_name='Mixture', show=False, save_path=''):

    if song_1.shape!=song_2.shape or song_1.shape!=mixture_song.shape:
        print("Shape mismatch. Not drawing a plot.")
        return

    draw_matrix = song_1 + song_2 * 2 + mixture_song * 4

    cm = matplotlib.cm.get_cmap('jet')
    song_1_color = cm(1/7)
    song_2_color = cm(2/7)
    song_1_song_2_color = cm(3/7)
    mixture_color = cm(4/7)
    song_1_mixture_color = cm(5/7)
    song_2_mixture_color = cm(6/7)
    song_1_song_2_mixture_color = cm(1.0)

    song_1_patch = mpatches.Patch(color=song_1_color, label=name_1)
    song_2_patch = mpatches.Patch(color=song_2_color, label=name_2)
    song_1_song_2_patch = mpatches.Patch(color=song_1_song_2_color, label=name_1 + " & " + name_2)
    mixture_patch = mpatches.Patch(color=mixture_color, label=mixture_name)
    song_1_mixture_patch = mpatches.Patch(color=song_1_mixture_color, label=name_1 + " & " + mixture_name)
    song_2_mixture_patch = mpatches.Patch(color=song_2_mixture_color, label=name_2 + " & " + mixture_name)
    song_1_song_2_mixture_patch = mpatches.Patch(color=song_1_song_2_mixture_color, label=name_1 + " & " + name_2 + " & " + mixture_name)

    plt.figure(figsize=(20.0, 10.0))
    plt.title('Mixture-Pitch-plot of ' + name_1 + ' and ' + name_2, fontsize=10)
    plt.legend(handles=[song_1_patch, song_2_patch, song_1_song_2_patch, mixture_patch, song_1_mixture_patch, song_2_mixture_patch, song_1_song_2_mixture_patch], loc='upper right', prop={'size': 8})

    plt.pcolor(draw_matrix, cmap='jet', vmin=-7, vmax=7)
    if show:
        plt.show()
    if len(save_path) > 0:
        plt.savefig(save_path)
        tikz_save(save_path + ".tex", encoding='utf-8', show_info=False)  
    plt.close()


def draw_difference_pianoroll(original, predicted, name_1='Original', name_2='Predicted', show=False, save_path=''):

    if original.shape!=predicted.shape:
        print("Shape mismatch. Not drawing a plot.")
        return

    draw_matrix = original + 2 * predicted
    
    cm = colors.ListedColormap(['white', 'blue', 'red', 'black'])
    bounds=[0,1,2,3,4]
    n = colors.BoundaryNorm(bounds, cm.N)

    original_color = cm(1/3)
    predicted_color = cm(2/3)
    both_color = cm(1.0)

    original_patch = mpatches.Patch(color=original_color, label=name_1)
    predicted_patch = mpatches.Patch(color=predicted_color, label=name_2)
    both_patch = mpatches.Patch(color=both_color, label='Notes in both songs')

    plt.figure(figsize=(20.0, 10.0))
    plt.title('Difference-Pitch-plot of ' + name_1 + ' and ' + name_2, fontsize=10)
    plt.legend(handles=[original_patch, predicted_patch, both_patch], loc='upper right', prop={'size': 8})

    plt.pcolor(draw_matrix, cmap=cm, vmin=0, vmax=3, norm=n)
    if show:
        plt.show()
    if len(save_path) > 0:
        plt.savefig(save_path)
        tikz_save(save_path + ".tex", encoding='utf-8', show_info=False)
        
    plt.close()



def draw_pianoroll(pianoroll, name='Notes', show=False, save_path=''):

    cm = matplotlib.cm.get_cmap('Greys')
    notes_color = cm(1.0)

    notes_patch = mpatches.Patch(color=notes_color, label=name)

    plt.figure(figsize=(20.0, 10.0))
    plt.title('Pianoroll Pitch-plot of ' + name, fontsize=10)
    plt.legend(handles=[notes_patch], loc='upper right', prop={'size': 8})

    plt.pcolor(pianoroll, cmap='Greys', vmin=0, vmax=np.max(pianoroll))
    if show:
        plt.show()
    if len(save_path) > 0:
        plt.savefig(save_path)
        tikz_save(save_path + ".tex", encoding='utf-8', show_info=False)
    plt.close()

def instrument_representation_to_programs(I, instrument_attach_method):
    programs = []
    for instrument_vector in I:
        if instrument_attach_method == '1hot-category':
            index = np.argmax(instrument_vector)
            programs.append(index * 8)
        elif instrument_attach_method == 'khot-category':
            nz = np.nonzero(instrument_vector)[0]
            index = 0
            for exponent in nz:
                index += 2^exponent
            programs.append(index * 8)
        elif instrument_attach_method == '1hot-instrument':
            index = np.argmax(instrument_vector)
            programs.append(index)
        elif instrument_attach_method == 'khot-instrument':
            nz = np.nonzero(instrument_vector)[0]
            index = 0
            for exponent in nz:
                index += 2^exponent
            programs.append(index)
    return programs


