#Script to import the files
from settings import *
import pretty_midi as pm
import midi_functions as mf
import os
import sys
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

print_anything = False

def load_rolls(path, name):

    #try loading the midi file
    #if it fails, return all None objects
    try:
        mid = pm.PrettyMIDI(path + name)
    except (ValueError, EOFError, IndexError, OSError, KeyError, ZeroDivisionError, AttributeError) as e:
        exception_str = 'Unexpected error in ' + name  + ':\n', e, sys.exc_info()[0]
        print(exception_str)
        return None, None, None, None, None, None

    if print_anything: print("Time signature changes: ", mid.time_signature_changes)

    #determine start and end of the song
    #if there are tempo changes in the song, only take the longest part where the tempo is steady
    #this cuts of silent starts and extended ends
    #this also makes sure that the start of the bars are aligned through the song
    tempo_change_times, tempo_change_bpm = mid.get_tempo_changes()
    song_start = 0
    song_end = mid.get_end_time()
    #there will always be at least one tempo change to set the first tempo
    #but if there are more than one tempo changes, that means that the tempos are changed
    if len(tempo_change_times) > 1:
        longest_part = 0
        longest_part_start_time = 0
        longest_part_end_time = song_end
        longest_part_tempo = 0
        for i, tempo_change_time in enumerate(tempo_change_times):
            if i == len(tempo_change_times) - 1:
                end_time = song_end
            else:
                end_time = tempo_change_times[i+1]
            current_part_length = end_time - tempo_change_time
            if current_part_length > longest_part:
                longest_part = current_part_length
                longest_part_start_time = tempo_change_time
                longest_part_end_time = end_time
                longest_part_tempo = tempo_change_bpm[i]
        song_start = longest_part_start_time
        song_end = longest_part_end_time
        tempo = longest_part_tempo
    else:
        tempo = tempo_change_bpm[0]

    #cut off the notes that are not in the longest part where the tempo is steady
    for instrument in mid.instruments:
        new_notes = [] #list for the notes that survive the cutting
        for note in instrument.notes:
            #check if it is in the given range of the longest part where the tempo is steady
            if note.start >= song_start and note.end <= song_end:
                #adjust to new times
                note.start -= song_start
                note.end -= song_start
                new_notes.append(note)
        instrument.notes = new_notes

    #(descending) order the piano_rolls according to the number of notes per track
    number_of_notes = []
    piano_rolls = [i.get_piano_roll(fs=100) for i in mid.instruments]
    for piano_roll in piano_rolls:
        number_of_notes.append(np.count_nonzero(piano_roll))
    permutation = np.argsort(number_of_notes)[::-1]
    mid.instruments = [mid.instruments[i] for i in permutation]


    if print_anything: print("Song start: ", song_start)
    if print_anything: print("Song end: ", song_end)
    if print_anything: print("Tempo: ", tempo)


    quarter_note_length = 1. / (tempo/60.)
    #fs is is the frequency for the song at what rate notes are picked
    #the song will by sampled by (0, song_length_in_seconds, 1./fs)
    #fs should be the inverse of the length of the note, that is to be sampled
    #the value should be in beats per seconds, where beats can be quarter notes or whatever...
    fs = 1. / (quarter_note_length * 4. / SMALLEST_NOTE)

    if print_anything: print("fs: ", fs)
    total_ticks = math.ceil(song_end * fs)
    if print_anything: print("Total ticks: ", total_ticks)
    
    #assemble piano_rolls, velocity_rolls and held_note_rolls
    piano_rolls = []
    velocity_rolls = []
    held_note_rolls = []
    max_concurrent_notes_per_track_list = []
    for instrument in mid.instruments:
        piano_roll = np.zeros((total_ticks, 128))

        #counts how many notes are played at maximum for this instrument at any given tick
        #this is used to determine the depth of the velocity_roll and held_note_roll
        concurrent_notes_count = np.zeros((total_ticks,))
        
        #keys is a tuple of the form (tick_start_of_the_note, pitch)
        #this uniquely identifies a note since there can be no two notes playing on the same pitch for the same instrument
        note_to_velocity_dict = dict()

        #keys is a tuple of the form (tick_start_of_the_note, pitch)
        #this uniquely identifies a note since there can be no two notes playing on the same pitch for the same instrument
        note_to_duration_dict = dict()

        for note in instrument.notes:
            note_tick_start = note.start * fs
            note_tick_end = note.end * fs
            absolute_start = int(round(note_tick_start))
            absolute_end = int(round(note_tick_end))
            decimal = note_tick_start - absolute_start
            #see if it starts at a tick or not
            #if it doesn't start at a tick (decimal > 10e-3) but is longer than one tick, include it anyways
            if decimal < 10e-3 or absolute_end-absolute_start >= 1:
                piano_roll[absolute_start:absolute_end, note.pitch] = 1
                concurrent_notes_count[absolute_start:absolute_end] += 1

                #save information of velocity and duration for later use
                #this can not be done right now because there might be no ordering in the notes
                note_to_velocity_dict[(absolute_start, note.pitch)] = note.velocity
                note_to_duration_dict[(absolute_start, note.pitch)] = absolute_end - absolute_start
                

        max_concurrent_notes = int(np.max(concurrent_notes_count))
        max_concurrent_notes_per_track_list.append(max_concurrent_notes)
        if print_anything: print("Max concurrent notes: ", max_concurrent_notes)
            
        velocity_roll = np.zeros((total_ticks, max_concurrent_notes))
        held_note_roll = np.zeros((total_ticks, max_concurrent_notes))

        for step, note_vector in enumerate(piano_roll):
            pitches = list(note_vector.nonzero()[0])
            sorted_pitches_from_highest_to_lowest = sorted(pitches)[::-1]
            for voice_number, pitch in enumerate(sorted_pitches_from_highest_to_lowest):
                if (step, pitch) in note_to_velocity_dict.keys():
                    velocity_roll[step, voice_number] = note_to_velocity_dict[(step, pitch)]
                if (step, pitch) not in note_to_duration_dict.keys():
                    #if the note is in the dictionary, it means that it is the start of the note
                    #if its not the start of a note, it means it is held
                    held_note_roll[step, voice_number] = 1

        piano_rolls.append(piano_roll)
        velocity_rolls.append(velocity_roll)
        held_note_rolls.append(held_note_roll)

    #get the program numbers for each instrument
    #program numbers are between 0 and 127 and have a 1:1 mapping to the instruments described in settings file
    programs = [i.program for i in mid.instruments]

    #we may want to override the maximal_number_of_voices_per_track if the following tracks are all silent
    #it makes no sense to exclude voices from the first instrument and then just have a song with 1 voice
    if print_anything: print(max_concurrent_notes_per_track_list)
    override_max_notes_per_track_list = [MAXIMAL_NUMBER_OF_VOICES_PER_TRACK for _ in max_concurrent_notes_per_track_list]
    silent_tracks_if_we_dont_override = max_voices - sum([min(MAXIMAL_NUMBER_OF_VOICES_PER_TRACK, x) if x > 0 else 0 for x in max_concurrent_notes_per_track_list[:max_voices]])
    
    if print_anything:print("Silent tracks if no override: ", silent_tracks_if_we_dont_override)
    for voice in range(min(max_voices, len(max_concurrent_notes_per_track_list))):
        if silent_tracks_if_we_dont_override > 0 and max_concurrent_notes_per_track_list[voice] > MAXIMAL_NUMBER_OF_VOICES_PER_TRACK:
            additional_voices = min(silent_tracks_if_we_dont_override, max_concurrent_notes_per_track_list[voice] - MAXIMAL_NUMBER_OF_VOICES_PER_TRACK)
            override_max_notes_per_track_list[voice] += additional_voices
            silent_tracks_if_we_dont_override -= additional_voices
    if print_anything: print("Override programs: ", override_max_notes_per_track_list)


        
    #chose the most important piano_rolls
    #each of them will be monophonic
    chosen_piano_rolls = []
    chosen_velocity_rolls = []
    chosen_held_note_rolls = []
    chosen_programs = []
    max_song_length = 0

    #go through all pianorolls in the descending order of the total notes they have
    for piano_roll, velocity_roll, held_note_roll, program, max_concurrent_notes, override_max_notes_per_track in zip(piano_rolls,  velocity_rolls, held_note_rolls, programs, max_concurrent_notes_per_track_list, override_max_notes_per_track_list):
        #see if there is actually a note played in that pianoroll
        if max_concurrent_notes > 0:

            #skip if you only want monophonic instruments and there are more than 1 notes played at the same time
            if include_only_monophonic_instruments:
                if max_concurrent_notes > 1:
                    if print_anything: print("Skipping this piano roll since it's polyphonic. Program number ", program)
                    continue
                else:
                    if print_anything: print("Adding monophonic program number: ", program)

                monophonic_piano_roll = piano_roll

                #append them to the chosen ones
                if len(chosen_piano_rolls) < max_voices:
                    chosen_piano_rolls.append(monophonic_piano_roll)
                    chosen_velocity_rolls.append(velocity_roll)
                    chosen_held_note_rolls.append()
                    chosen_programs.append(program)
                    if monophonic_piano_roll.shape[0] > max_song_length:
                        max_song_length = monophonic_piano_roll.shape[0]
                else:
                    break

            else:

                #limit the number of voices per track by the minimum of the actual concurrent voices per track or the maximal allowed in the settings file
                for voice in range(min(max_concurrent_notes, max(MAXIMAL_NUMBER_OF_VOICES_PER_TRACK, override_max_notes_per_track))):
                    #Take the highest note for voice 0, second highest for voice 1 and so on...
                    monophonic_piano_roll = np.zeros(piano_roll.shape)
                    for step in range(piano_roll.shape[0]):
                        #sort all the notes from highest to lowest
                        notes = np.nonzero(piano_roll[step,:])[0][::-1]
                        if len(notes) > voice:
                            monophonic_piano_roll[step, notes[voice]] = 1

                    #append them to the chosen ones
                    if len(chosen_piano_rolls) < max_voices:
                        chosen_piano_rolls.append(monophonic_piano_roll)
                        chosen_velocity_rolls.append(velocity_roll[:, voice])
                        chosen_held_note_rolls.append(held_note_roll[:, voice])
                        chosen_programs.append(program)
                        if monophonic_piano_roll.shape[0] > max_song_length:
                            max_song_length = monophonic_piano_roll.shape[0]
                    else:
                        break
                if len(chosen_piano_rolls) == max_voices:
                    break

    assert(len(chosen_piano_rolls) == len(chosen_velocity_rolls))
    assert(len(chosen_piano_rolls) == len(chosen_held_note_rolls))
    assert(len(chosen_piano_rolls) == len(chosen_programs))

            
    #do the unrolling and prepare for model input
    if len(chosen_piano_rolls) > 0:

        song_length = max_song_length * max_voices

        #prepare Y
        #Y will be the target notes
        Y = np.zeros((song_length, chosen_piano_rolls[0].shape[1]))
        #unroll the pianoroll into one matrix
        for i, piano_roll in enumerate(chosen_piano_rolls):
            for step in range(piano_roll.shape[0]):
                Y[i + step*max_voices,:] += piano_roll[step,:]
        #assert that there is always at most one note played
        for step in range(Y.shape[0]):
            assert(np.sum(Y[step,:]) <= 1)
        #cut off pitch values which are very uncommon
        #this reduces the feature space significantly
        Y = Y[:,low_crop:high_crop]
        #append silent note if desired
        #the silent note will always be at the last note
        if include_silent_note:
            Y = np.append(Y, np.zeros((Y.shape[0], 1)), axis=1)
            for step in range(Y.shape[0]):
                if np.sum(Y[step]) == 0:
                    Y[step, -1] = 1
            #assert that there is now a 1 at every step
            for step in range(Y.shape[0]):
                assert(np.sum(Y[step,:]) == 1)

        #unroll the velocity roll
        #V will only have shape (song_length,) and it's values will be between 0 and 1 (divide by MAX_VELOCITY)
        V = np.zeros((song_length,))
        for i, velocity_roll in enumerate(chosen_velocity_rolls):
            for step in range(velocity_roll.shape[0]):
                if velocity_roll[step] > 0:
                    velocity = velocity_threshold_such_that_it_is_a_played_note + (velocity_roll[step] / MAX_VELOCITY) * (1.0 - velocity_threshold_such_that_it_is_a_played_note) 
                    # a note is therefore at least 0.1*max_velocity loud
                    # but this is good, since we can now more clearly distinguish between silent or played notes
                    assert(velocity <= 1.0)
                    V[i + step*max_voices] = velocity


        #unroll the held_note_rolls
        #D will only have shape (song_length,) and it's values will be  0 or 1 (1 if held)
        #it's name is D for Duration to not have a name clash with the history (H)
        D = np.zeros((song_length,))
        for i, held_note_roll in enumerate(chosen_held_note_rolls):
            for step in range(held_note_roll.shape[0]):
                D[i + step*max_voices] = held_note_roll[step]

        instrument_feature_matrix = mf.programs_to_instrument_matrix(chosen_programs, instrument_attach_method, max_voices)

        if attach_instruments:
            instrument_feature_matrix = np.transpose(np.tile(np.transpose(instrument_feature_matrix), song_length//max_voices))
            Y = np.append(Y, instrument_feature_matrix, axis=1)

        if song_completion:
            #only take voice 1 (jump by max_voices)
            X = Y[::max_voices,:]
        else:
            X = Y

        if save_preprocessed_midi: mf.rolls_to_midi(Y, chosen_programs,'preprocess_midi_data/' + t+ '/', name, tempo, V, D)


        #split the song into chunks of size output_length or input_length
        #pad them with silent notes if necessary
        if input_length > 0:

            #split X
            padding_length = input_length - (X.shape[0] % input_length)
            if input_length == padding_length:
                padding_length = 0
            #pad to the right..
            X = np.pad(X, ((0,padding_length),(0, 0)), 'constant', constant_values=(0, 0))
            if include_silent_note:
                X[-padding_length:,-1] = 1
            number_of_splits = X.shape[0] // input_length
            X = np.split(X, number_of_splits)
            X = np.asarray(X)

        if output_length > 0:
            #split Y
            padding_length = output_length - (Y.shape[0] % output_length)
            if output_length == padding_length:
                padding_length = 0

            #pad to the right..
            Y = np.pad(Y, ((0,padding_length),(0, 0)), 'constant', constant_values=(0, 0))
            if include_silent_note:
                Y[-padding_length:,-1] = 1
            number_of_splits = Y.shape[0] // output_length
            Y = np.split(Y, number_of_splits)
            Y = np.asarray(Y)

            #split V
            #pad to the right with zeros..
            V = np.pad(V, (0,padding_length), 'constant', constant_values=0)
            number_of_splits = V.shape[0] // output_length
            V = np.split(V, number_of_splits)
            V = np.asarray(V)

            #split D
            #pad to the right with zeros..
            D = np.pad(D, (0,padding_length), 'constant', constant_values=0)
            number_of_splits = D.shape[0] // output_length
            D = np.split(D, number_of_splits)
            D = np.asarray(D)


        return X, Y, instrument_feature_matrix, tempo, V, D
    else:
        return None, None, None, None, None, None

def import_midi_from_folder(folder):

    #take the shortcut if allowed
    if load_from_pickle_instead_of_midi:
        V_train = pickle.load(open(pickle_load_path + "V_train.pickle",'rb'))
        V_test = pickle.load(open(pickle_load_path + "V_test.pickle",'rb')) 
        D_train = pickle.load(open(pickle_load_path + "D_train.pickle",'rb')) 
        D_test = pickle.load(open(pickle_load_path + "D_test.pickle",'rb')) 
        T_train = pickle.load(open(pickle_load_path + "T_train.pickle",'rb'))
        T_test = pickle.load(open(pickle_load_path + "T_test.pickle",'rb'))
        I_train = pickle.load(open(pickle_load_path + "I_train.pickle",'rb'))
        I_test = pickle.load(open(pickle_load_path + "I_test.pickle",'rb'))
        Y_train = pickle.load(open(pickle_load_path + "Y_train.pickle",'rb'))
        Y_test = pickle.load(open(pickle_load_path + "Y_test.pickle",'rb'))
        X_train = pickle.load(open(pickle_load_path + "X_train.pickle",'rb'))
        X_test = pickle.load(open(pickle_load_path + "X_test.pickle",'rb'))
        c_train = pickle.load(open(pickle_load_path + "c_train.pickle",'rb'))
        c_test = pickle.load(open(pickle_load_path + "c_test.pickle",'rb'))
        train_paths = pickle.load(open(pickle_load_path + "train_paths.pickle",'rb'))
        test_paths = pickle.load(open(pickle_load_path + "test_paths.pickle",'rb'))

        return V_train, V_test, D_train, D_test, T_train, T_test, I_train, I_test, Y_train, Y_test, X_train, X_test, c_train, c_test, train_paths, test_paths

    X_list = []
    Y_list = []
    paths = []
    c_classes = []
    I_list = []
    T_list = []
    V_list = []
    D_list = []
    no_imported = 0
    for path, subdirs, files in os.walk(folder):
        for name in files:
            if no_imported >= max_songs:
                break
            _path = path.replace('\\', '/') + '/'
            _name = name.replace('\\', '/')

            if _name.endswith('.mid') or _name.endswith('.midi'):

                shortpath = _path[len(folder):]
                found = False
                for i, c in enumerate(classes):
                    if c.lower() in shortpath.lower():
                        found = True
                        print("Importing " + c + " song called " + _name)
                        C = i
                        if not only_unknown:

                            X, Y, I, T, V, D = load_rolls(_path, _name)
    
                            if X is not None and Y is not None:
                                X_list.append(X)
                                Y_list.append(Y)
                                I_list.append(I)
                                T_list.append(T)
                                V_list.append(V)
                                D_list.append(D)
                                paths.append(_path + _name)
                                c_classes.append(C)
                                no_imported += 1
                        break
                if not found:
                    #assign new category for all the files with no proper title
                    if include_unknown:
                        C = num_classes -1
                        print("Importing unknown song ", _name)

                        X, Y, I, T, V, D = load_rolls(_path, _name)

                        if X is not None and Y is not None:
                            X_list.append(X)
                            Y_list.append(Y)
                            I_list.append(I)
                            T_list.append(T)
                            V_list.append(V)
                            D_list.append(D)
                            paths.append(_path + _name)
                            c_classes.append(C)
                            no_imported += 1
            if no_imported >= max_songs:
                break

     
    if print_anything: print(len(X_list))
    assert(len(X_list) == len(paths))
    assert(len(X_list) == len(c_classes))
    assert(len(X_list) == len(I_list))
    assert(len(X_list) == len(T_list))
    assert(len(X_list) == len(D_list))
    assert(len(X_list) == len(V_list))

    unique, counts = np.unique(c_classes, return_counts=True)
    if print_anything: print(dict(zip(unique, counts)))


    if split_equally_to_train_and_test:

        V_train, V_test, D_train, D_test, T_train, T_test, I_train, I_test, Y_train, Y_test, X_train, X_test, c_train, c_test, train_paths, test_paths = train_test_split(V_list, D_list, T_list, I_list, Y_list, X_list, c_classes ,paths, test_size=test_fraction, random_state=42, stratify=c_classes)

        train_set_size = len(X_train)
        test_set_size = len(X_test)  

    else:

        if shuffle:
            permutation = np.random.permutation(len(X_list))
        else:
            #arrange the songs such that they get evenly distributed into 'parts' chunks
            chunks = num_classes
            permutation = []
            for i in range(int(len(X_array)/chunks)):
                permutation.extend(range(i,len(X_array), int(len(X_array)/chunks)))

        paths = [paths[i] for i in permutation]
        X_list = [X_list[i] for i in permutation]
        Y_list = [Y_list[i] for i in permutation]
        I_list = [I_list[i] for i in permutation]
        T_list = [T_list[i] for i in permutation]
        V_list = [V_list[i] for i in permutation]
        D_list = [D_list[i] for i in permutation]
        c_classes = [c_classes[i] for i in permutation]


        c_train = c_classes[:train_set_size]
        c_test = c_classes[train_set_size: train_set_size + test_set_size]

        train_paths = paths[:train_set_size]
        test_paths = paths[train_set_size: train_set_size + test_set_size]

        X_train = X_list[:train_set_size] 
        X_test = X_list[train_set_size: train_set_size + test_set_size]

        Y_train = Y_list[:train_set_size] 
        Y_test = Y_list[train_set_size: train_set_size + test_set_size]

        I_train = I_list[:train_set_size] 
        I_test = I_list[train_set_size: train_set_size + test_set_size]

        T_train = T_list[:train_set_size] 
        T_test = T_list[train_set_size: train_set_size + test_set_size]

        V_train = V_list[:train_set_size] 
        V_test = V_list[train_set_size: train_set_size + test_set_size]

        D_train = D_list[:train_set_size] 
        D_test = D_list[train_set_size: train_set_size + test_set_size]


    #make sure that there are as many output_length mini-songs for all classes (amount of splits after the division by output_length)
    #round up because songs will be padded with zeros to be divisible by output_length
    if equal_mini_songs:
        splits_per_class = np.zeros((num_classes,))
        for i, song in enumerate(X_train):
            c = c_train[i]
            splits_per_class[c] += math.ceil(len(song)/(output_length//max_voices))

        amount_of_splits = min(splits_per_class)
        amount_of_splits = int(amount_of_splits * smaller_training_set_factor)
        if print_anything: print(splits_per_class)
        if print_anything: print(amount_of_splits)

        c_train_new = []
        X_train_new = []
        Y_train_new = []
        I_train_new = []
        T_train_new = []
        V_train_new = []
        D_train_new = []
        train_paths_new = []
        splits_per_class_new = np.zeros((num_classes,))
        for i, song in enumerate(X_train):
            c = c_train[i]
            if splits_per_class_new[c] + math.ceil(len(song)/(output_length//max_voices)) <= amount_of_splits:
                c_train_new.append(c_train[i])
                X_train_new.append(X_train[i])
                Y_train_new.append(Y_train[i])
                I_train_new.append(I_train[i])
                T_train_new.append(T_train[i])
                V_train_new.append(V_train[i])
                D_train_new.append(D_train[i])
                train_paths_new.append(train_paths[i])
                splits_per_class_new[c] += math.ceil(len(song)/(output_length//max_voices))
        if print_anything: print(splits_per_class_new)


        c_train = c_train_new
        X_train = X_train_new
        Y_train = Y_train_new
        I_train = I_train_new
        T_train = T_train_new
        V_train = V_train_new
        D_train = D_train_new
        train_paths = train_paths_new

    if save_imported_midi_as_pickle:
        pickle.dump(V_train,open(pickle_store_folder+'V_train.pickle', 'wb'))
        pickle.dump(V_test,open(pickle_store_folder+'V_test.pickle', 'wb'))

        pickle.dump(D_train,open(pickle_store_folder+'D_train.pickle', 'wb'))
        pickle.dump(D_test,open(pickle_store_folder+'D_test.pickle', 'wb'))

        pickle.dump(T_train,open(pickle_store_folder+'T_train.pickle', 'wb'))
        pickle.dump(T_test,open(pickle_store_folder+'T_test.pickle', 'wb'))

        pickle.dump(I_train,open(pickle_store_folder+'I_train.pickle', 'wb'))
        pickle.dump(I_test,open(pickle_store_folder+'I_test.pickle', 'wb'))

        pickle.dump(Y_train,open(pickle_store_folder+'Y_train.pickle', 'wb'))
        pickle.dump(Y_test,open(pickle_store_folder+'Y_test.pickle', 'wb'))

        pickle.dump(X_train,open(pickle_store_folder+'X_train.pickle', 'wb'))
        pickle.dump(X_test,open(pickle_store_folder+'X_test.pickle', 'wb'))

        pickle.dump(c_train,open(pickle_store_folder+'c_train.pickle', 'wb'))
        pickle.dump(c_test,open(pickle_store_folder+'c_test.pickle', 'wb'))

        pickle.dump(train_paths,open(pickle_store_folder+'train_paths.pickle', 'wb'))
        pickle.dump(test_paths,open(pickle_store_folder+'test_paths.pickle', 'wb'))


    return V_train, V_test, D_train, D_test, T_train, T_test, I_train, I_test, Y_train, Y_test, X_train, X_test, c_train, c_test, train_paths, test_paths

         