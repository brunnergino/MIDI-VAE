from keras import objectives, backend as K
from keras.layers import Bidirectional, Dense, Embedding, Input, Lambda, LSTM, RepeatVector, TimeDistributed, Add, GRU, SimpleRNN
from keras.models import Model
from keras.layers import Layer
import keras
from recurrentshop import *
from recurrentshop.cells import LSTMCell, GRUCell, SimpleRNNCell
from keras.layers.merge import Concatenate
from keras.utils import to_categorical

import data_class
from settings import *


class KLDivergenceLayer(Layer):

    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, beta=1.0, prior_mean=0.0, prior_std=1.0, *args, **kwargs):
        self.is_placeholder = True

        self.beta = beta
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs
        prior_log_var = K.log(self.prior_std) * 2
        prior_var = K.square(self.prior_std)
        #kl_batch = self.beta *( - .5 * K.sum(1 + log_var - K.square(mu) - K.exp(log_var), axis=1))
        kl_batch = self.beta * ( - 0.5 * K.sum(1 + log_var - prior_log_var - ((K.square(mu - self.prior_mean) + K.exp(log_var)) / prior_var), axis=1))
        self.add_loss(K.mean(kl_batch), inputs=inputs)
        return inputs

class VAE(object):
    def create(self, 
        input_dim=64, 
        output_dim=64, 
        use_embedding=False, 
        embedding_dim=0, 
        input_length=16, 
        output_length=16, 
        latent_rep_size=256, 
        vae_loss = 'categorical_crossentropy',
        optimizer='Adam', 
        activation='sigmoid', 
        lstm_activation='tanh', 
        lstm_state_activation='tanh', 
        epsilon_std=1.0, 
        epsilon_factor=0.0,
        include_composer_decoder=False,
        num_composers=0, 
        composer_weight=1.0, 
        lstm_size=256, 
        cell_type='LSTM', 
        num_layers_encoder=1, 
        num_layers_decoder=1, 
        bidirectional=False, 
        decode=True, 
        teacher_force=False, 
        learning_rate=0.001, 
        split_lstm_vector=True, 
        history=True, 
        beta=0.01, 
        prior_mean=0.0,
        prior_std=1.0,
        decoder_additional_input=False, 
        decoder_additional_input_dim=0, 
        extra_layer=False, 
        meta_instrument=False, 
        meta_instrument_dim=0, 
        meta_instrument_length=0, 
        meta_instrument_activation='sigmoid', 
        meta_instrument_weight=1.0,
        signature_decoder=False,
        signature_dim=0,
        signature_activation='sigmoid',
        signature_weight=1.0,
        composer_decoder_at_notes_output=False,
        composer_decoder_at_notes_weight=1.0,
        composer_decoder_at_notes_activation='softmax',
        composer_decoder_at_instrument_output=False,
        composer_decoder_at_instrument_weight=1.0,
        composer_decoder_at_instrument_activation='softmax',
        meta_velocity=False,
        meta_velocity_length=0, 
        meta_velocity_activation='sigmoid', 
        meta_velocity_weight=1.0,
        meta_held_notes=False,
        meta_held_notes_length=0, 
        meta_held_notes_activation='softmax', 
        meta_held_notes_weight=1.0,
        meta_next_notes=False,
        meta_next_notes_output_length=16,
        meta_next_notes_weight=1.0,
        meta_next_notes_teacher_force=False,
        activation_before_splitting='tanh'
        ):
        self.encoder = None
        self.decoder = None
        self.composer_decoder = None
        self.autoencoder = None
        self.signature_decoder = None

        self.input_dim=input_dim
        self.output_dim = output_dim
        
        self.decode = decode
        self.input_length = input_length
        self.output_length = output_length
        self.latent_rep_size = latent_rep_size
        self.vae_loss = vae_loss
        self.activation = activation
        self.lstm_activation = lstm_activation
        self.lstm_state_activation = lstm_state_activation
        self.include_composer_decoder = include_composer_decoder
        self.num_composers = num_composers
        self.composer_weight = composer_weight
        self.lstm_size = lstm_size
        self.cell_type = cell_type
        self.num_layers_encoder = num_layers_encoder
        self.num_layers_decoder = num_layers_decoder
        self.bidirectional = bidirectional
        self.teacher_force = teacher_force
        self.use_embedding = use_embedding
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.split_lstm_vector = split_lstm_vector
        self.history = history
        self.beta = beta
        self.prior_mean=prior_mean
        self.prior_std=prior_std
        self.decoder_additional_input = decoder_additional_input
        self.decoder_additional_input_dim = decoder_additional_input_dim
        self.epsilon_std = epsilon_std
        self.epsilon_factor = epsilon_factor
        self.extra_layer = extra_layer
        self.meta_instrument= meta_instrument
        self.meta_instrument_dim= meta_instrument_dim
        self.meta_instrument_length = meta_instrument_length
        self.meta_instrument_activation = meta_instrument_activation
        self.meta_instrument_weight = meta_instrument_weight
        self.meta_velocity=meta_velocity
        self.meta_velocity_length=meta_velocity_length 
        self.meta_velocity_activation=meta_velocity_activation
        self.meta_velocity_weight=meta_velocity_weight
        self.meta_held_notes=meta_held_notes
        self.meta_held_notes_length=meta_held_notes_length 
        self.meta_held_notes_activation=meta_held_notes_activation
        self.meta_held_notes_weight=meta_held_notes_weight
        self.meta_next_notes=meta_next_notes
        self.meta_next_notes_output_length=meta_next_notes_output_length
        self.meta_next_notes_weight=meta_next_notes_weight
        self.meta_next_notes_teacher_force=meta_next_notes_teacher_force
        
        self.signature_decoder = signature_decoder
        self.signature_dim = signature_dim
        self.signature_activation = signature_activation
        self.signature_weight = signature_weight

        self.composer_decoder_at_notes_output = composer_decoder_at_notes_output
        self.composer_decoder_at_notes_weight = composer_decoder_at_notes_weight
        self.composer_decoder_at_notes_activation = composer_decoder_at_notes_activation
        self.composer_decoder_at_instrument_output = composer_decoder_at_instrument_output
        self.composer_decoder_at_instrument_weight = composer_decoder_at_instrument_weight
        self.composer_decoder_at_instrument_activation = composer_decoder_at_instrument_activation

        self.activation_before_splitting = activation_before_splitting

        if optimizer == 'RMSprop': self.optimizer = keras.optimizers.RMSprop(lr=learning_rate)
        if optimizer == 'Adam': self.optimizer = keras.optimizers.Adam(lr=learning_rate)

        assert(self.num_layers_encoder > 0)
        assert(self.num_layers_decoder > 0)
        assert(self.input_length > 0)
        assert(self.output_length > 0)
        assert(self.lstm_size > 0)
        assert(self.latent_rep_size > 0)
        assert(self.beta > 0)
        if self.use_embedding:
            assert(embedding_dim > 0)
        if self.meta_instrument:
            assert(meta_instrument_dim > 0)
            assert(meta_instrument_weight > 0)
        if self.signature_decoder:
            assert(self.signature_dim > 0)
            assert(self.signature_weight > 0)

        if self.composer_decoder_at_notes_output:
            assert(composer_decoder_at_notes_weight > 0)
        if self.composer_decoder_at_instrument_output:
            assert(meta_instrument)
            assert(composer_decoder_at_instrument_weight > 0)

        if self.meta_velocity:
            assert(meta_velocity_weight > 0)
            assert(meta_velocity_length > 0)

        if self.meta_held_notes:
            assert(meta_held_notes_weight > 0)
            assert(meta_held_notes_length > 0)
        if self.meta_next_notes:
            assert(meta_next_notes_weight > 0)
            assert(meta_next_notes_output_length > 0)


        
        if self.use_embedding:
            input_x = Input(shape=(self.input_length,), name='embedding_input')
            x = Embedding(self.input_dim, self.embedding_dim)(input_x)
        else:
            input_x = Input(shape=(self.input_length,self.input_dim), name='notes_input')
            x = input_x

        encoder_input_list = [input_x]
        if self.meta_instrument:
            if self.meta_instrument_length > 0:
                meta_instrument_input = Input(shape=(self.meta_instrument_length, self.meta_instrument_dim), name='meta_instrument_input')
            else:
                meta_instrument_input = Input(shape=(self.meta_instrument_dim,), name='meta_instrument_input')
            encoder_input_list.append(meta_instrument_input)
        else:
            meta_instrument_input = None

        if self.meta_velocity:
            meta_velocity_input = Input(shape=(self.meta_velocity_length,1), name='meta_velocity_input')
            encoder_input_list.append(meta_velocity_input)
        else:
            meta_velocity_input = None

        if self.meta_held_notes:
            meta_held_notes_input = Input(shape=(self.meta_held_notes_length,2), name='meta_held_notes_input')
            encoder_input_list.append(meta_held_notes_input)
        else:
            meta_held_notes_input = None

        encoded = self._build_encoder(x, meta_instrument_input, meta_velocity_input, meta_held_notes_input)
        self.encoder = Model(inputs=encoder_input_list, outputs=encoded)

        
        encoded_input = Input(shape=(self.latent_rep_size,), name='encoded_input')
        
        
        if self.use_embedding:
            input_decoder_x = Input(shape=(self.output_dim,), name='embedding_input_decoder_start')
            #decoder_x = Embedding(self.output_dim, self.output_dim, input_length=1)(input_decoder_x)
            decoder_x = input_decoder_x
        else:
            input_decoder_x = Input(shape=(self.output_dim,), name='input_decoder_start')
            decoder_x = input_decoder_x

        autoencoder_decoder_input_list = [input_decoder_x, encoded]
        decoder_input_list = [input_decoder_x, encoded_input]
        autoencoder_input_list = [input_x, input_decoder_x]
        autoencoder_output_list = []

        
        if self.teacher_force:
            ground_truth_input = Input(shape=(self.output_length, self.output_dim), name='ground_truth_input')
            decoder_input_list.append(ground_truth_input)
            autoencoder_decoder_input_list.append(ground_truth_input)
            autoencoder_input_list.append(ground_truth_input)
        else:
            ground_truth_input = None

        if self.history:
            history_input = Input(shape=(self.latent_rep_size,), name='history_input')
            decoder_input_list.append(history_input)
            autoencoder_decoder_input_list.append(history_input)
            autoencoder_input_list.append(history_input)
        else:
            history_input = None

        if decoder_additional_input:
            decoder_additional_input_layer = Input(shape=(decoder_additional_input_dim,), name='decoder_additional_input')
            decoder_input_list.append(decoder_additional_input_layer )
            autoencoder_decoder_input_list.append(decoder_additional_input_layer )
            autoencoder_input_list.append(decoder_additional_input_layer )
        else:
            decoder_additional_input_layer  = False

        if self.meta_instrument:
            input_decoder_meta_instrument_start = Input(shape=(self.meta_instrument_dim,), name='input_decoder_meta_instrument_start')
            decoder_input_list.append(input_decoder_meta_instrument_start)
            autoencoder_decoder_input_list.append(input_decoder_meta_instrument_start)
            autoencoder_input_list.append(input_decoder_meta_instrument_start)
            autoencoder_input_list.append(meta_instrument_input)
        else:
            input_decoder_meta_instrument_start = None

        if self.meta_velocity:
            input_decoder_meta_velocity_start = Input(shape=(1,), name='input_decoder_meta_velocity_start')
            decoder_input_list.append(input_decoder_meta_velocity_start)
            autoencoder_decoder_input_list.append(input_decoder_meta_velocity_start)
            autoencoder_input_list.append(input_decoder_meta_velocity_start)
            autoencoder_input_list.append(meta_velocity_input)
        else:
            input_decoder_meta_velocity_start = None

        if self.meta_held_notes:
            input_decoder_meta_held_notes_start = Input(shape=(2,), name='input_decoder_meta_held_notes_start')
            decoder_input_list.append(input_decoder_meta_held_notes_start)
            autoencoder_decoder_input_list.append(input_decoder_meta_held_notes_start)
            autoencoder_input_list.append(input_decoder_meta_held_notes_start)
            autoencoder_input_list.append(meta_held_notes_input)
        else:
            input_decoder_meta_held_notes_start = None

        if self.meta_next_notes:
            input_decoder_meta_next_notes_start = Input(shape=(self.output_dim,), name='input_decoder_meta_next_notes_start')
            decoder_input_list.append(input_decoder_meta_next_notes_start)
            autoencoder_input_list.append(input_decoder_meta_next_notes_start)
            autoencoder_decoder_input_list.append(input_decoder_meta_next_notes_start)
            if self.meta_next_notes_teacher_force:
                meta_next_notes_ground_truth_input = Input(shape=(self.meta_next_notes_output_length, self.output_dim), name='meta_next_notes_ground_truth_input')
                decoder_input_list.append(meta_next_notes_ground_truth_input)
                autoencoder_decoder_input_list.append(meta_next_notes_ground_truth_input)
                autoencoder_input_list.append(meta_next_notes_ground_truth_input)
            else:
                meta_next_notes_ground_truth_input = None
        else:
            input_decoder_meta_next_notes_start = None
            meta_next_notes_ground_truth_input = None


        decoded, meta_instrument_output, meta_velocity_output, meta_held_notes_output, meta_next_notes_output = self._build_decoder(decoder_x, encoded_input, ground_truth_input, history_input, decoder_additional_input_layer, input_decoder_meta_instrument_start, input_decoder_meta_velocity_start, input_decoder_meta_held_notes_start, input_decoder_meta_next_notes_start, meta_next_notes_ground_truth_input)

        loss_list = []
        loss_weights_list = []
        sample_weight_modes = []

        loss_weights_list.append(1.0)
        sample_weight_modes.append('temporal')
        loss_list.append(self.vae_loss)
        metrics_list = ['accuracy']

        if self.meta_instrument or self.meta_velocity or self.meta_held_notes or self.meta_next_notes:
            decoder_output = [decoded]
            if self.meta_instrument:
                decoder_output.append(meta_instrument_output)
            if self.meta_velocity:
                decoder_output.append(meta_velocity_output)
            if self.meta_held_notes:
                decoder_output.append(meta_held_notes_output)
            if self.meta_next_notes:
                decoder_output.append(meta_next_notes_output)
        else:
            decoder_output = decoded


        self.decoder = Model(inputs=decoder_input_list, outputs=decoder_output, name='decoder')

        decoder_final_output = self.decoder(autoencoder_decoder_input_list)


        if isinstance(decoder_final_output, list):
            autoencoder_output_list.extend(decoder_final_output)
        else:
            autoencoder_output_list.append(decoder_final_output)

        if self.meta_instrument:
            #dont append meta_instrument since it is already appended previously to the autoencoder output
            loss_list.append('categorical_crossentropy')
            loss_weights_list.append(self.meta_instrument_weight)
            sample_weight_modes.append('None')

        if self.meta_velocity:

            #dont append meta_velocity since it is already appended previously to the autoencoder output
            loss_list.append('mse')
            loss_weights_list.append(self.meta_velocity_weight)
            sample_weight_modes.append('None')

        if self.meta_held_notes:
            #dont append meta_held_notes since it is already appended previously to the autoencoder output
            loss_list.append('categorical_crossentropy')
            loss_weights_list.append(self.meta_held_notes_weight)
            sample_weight_modes.append('None')

        if self.meta_next_notes:
            #dont append meta_next_notes since it is already appended previously to the autoencoder output
            loss_list.append('categorical_crossentropy')
            loss_weights_list.append(self.meta_next_notes_weight)
            sample_weight_modes.append('None')

        
        if self.include_composer_decoder:
            predicted_composer = self._build_composer_decoder(encoded_input)
            self.composer_decoder = Model(encoded_input, predicted_composer, name='composer_decoder')
            autoencoder_output_list.append(self.composer_decoder(encoded))
            loss_list.append('categorical_crossentropy')
            loss_weights_list.append(self.composer_weight)
            sample_weight_modes.append('None')

        if self.signature_decoder:

            predicted_signature = self._build_signature_decoder(encoded_input)
            self.signature_decoder = Model(encoded_input, predicted_signature, name='signature_decoder')
            autoencoder_output_list.append(self.signature_decoder(encoded))
            loss_list.append('mse')
            loss_weights_list.append(self.signature_weight)
            sample_weight_modes.append('None')


        if self.composer_decoder_at_notes_output:

            notes_composer_decoder_input = Input(shape=(self.output_length,self.output_dim), name='notes_composer_decoder_input')
            predicted_composer_2 = self._build_composer_decoder_at_notes_output(notes_composer_decoder_input)
            self.composer_decoder_2 = Model(notes_composer_decoder_input, predicted_composer_2, name='composer_decoder_at_notes')
            if not meta_instrument and not meta_velocity and not meta_held_notes and not meta_next_notes:
                autoencoder_output_list.append(self.composer_decoder_2(decoder_final_output))
            else:
                autoencoder_output_list.append(self.composer_decoder_2(decoder_final_output[0]))
            loss_list.append('categorical_crossentropy')
            loss_weights_list.append(self.composer_decoder_at_notes_weight)
            sample_weight_modes.append('None')

        if self.composer_decoder_at_instrument_output:
            if self.meta_instrument_length > 0:
                meta_instrument_composer_decoder_input = Input(shape=(self.meta_instrument_length, self.meta_instrument_dim), name='meta_instrument_composer_decoder_input')
            else:
                meta_instrument_composer_decoder_input = Input(shape=(self.meta_instrument_dim,), name='meta_instrument_composer_decoder_input')
            predicted_composer_3 = self._build_composer_decoder_at_instrument_output(meta_instrument_composer_decoder_input)
            self.composer_decoder_3 = Model(meta_instrument_composer_decoder_input, predicted_composer_3, name='composer_decoder_at_instruments')
            autoencoder_output_list.append(self.composer_decoder_3(decoder_final_output[1]))
            loss_list.append('categorical_crossentropy')
            loss_weights_list.append(self.composer_decoder_at_instrument_weight)
            sample_weight_modes.append('None')

            

        self.autoencoder = Model(inputs=autoencoder_input_list, outputs=autoencoder_output_list, name='autoencoder')
        self.autoencoder.compile(optimizer=self.optimizer,
                                 loss=loss_list,
                                 loss_weights=loss_weights_list,
                                 sample_weight_mode=sample_weight_modes,
                                 metrics=metrics_list)

    def _build_encoder(self, x, meta_instrument_input=None, meta_velocity_input=None, meta_held_notes_input=None):
        h = x
        if self.bidirectional:

            for layer_no in range(1,self.num_layers_encoder-1):
                if self.cell_type == 'SimpleRNN': h = Bidirectional(SimpleRNN(self.lstm_size, return_sequences=True, activation=self.lstm_activation, name='rnn_' + str(layer_no)), merge_mode='concat')(h)
                if self.cell_type == 'LSTM': h = Bidirectional(LSTM(self.lstm_size, return_sequences=True, activation=self.lstm_activation, name='lstm_' + str(layer_no)), merge_mode='concat')(h)
                if self.cell_type == 'GRU': h = Bidirectional(GRU(self.lstm_size, return_sequences=True, activation=self.lstm_activation, name='gru_' + str(layer_no)), merge_mode='concat')(h)
            if self.cell_type == 'SimpleRNN': h = SimpleRNN(self.lstm_size, return_sequences=False, activation=self.lstm_activation, name='rnn_' + str(self.num_layers_encoder))(h)
            if self.cell_type == 'LSTM': h = LSTM(self.lstm_size, return_sequences=False, activation=self.lstm_activation, name='lstm_' + str(self.num_layers_encoder))(h)
            if self.cell_type == 'GRU': h = GRU(self.lstm_size, return_sequences=False, activation=self.lstm_activation, name='gru_' + str(self.num_layers_encoder))(h)
        else:
            for layer_no in range(1, self.num_layers_encoder):
                if self.cell_type == 'SimpleRNN': h = SimpleRNN(self.lstm_size, return_sequences=True, activation=self.lstm_activation, name='rnn_' + str(layer_no))(h)
                if self.cell_type == 'LSTM': h = LSTM(self.lstm_size, return_sequences=True, activation=self.lstm_activation, name='lstm_' + str(layer_no))(h)
                if self.cell_type == 'GRU': h = GRU(self.lstm_size, return_sequences=True, activation=self.lstm_activation, name='gru_' + str(layer_no))(h)
            if self.cell_type == 'SimpleRNN': h = SimpleRNN(self.lstm_size, return_sequences=False, activation=self.lstm_activation, name='rnn_' +str(self.num_layers_encoder))(h)
            if self.cell_type == 'LSTM': h = LSTM(self.lstm_size, return_sequences=False, activation=self.lstm_activation, name='lstm_' +str(self.num_layers_encoder))(h)
            if self.cell_type == 'GRU': h = GRU(self.lstm_size, return_sequences=False, activation=self.lstm_activation, name='gru_' +str(self.num_layers_encoder))(h)
        #h = Dense(self.lstm_size, activation='relu', name='dense_1')(h)

        if self.meta_instrument:
            if self.cell_type == 'SimpleRNN': m = SimpleRNN(self.lstm_size, return_sequences=False, activation=self.lstm_activation, name='rnn_meta_instrument')(meta_instrument_input)
            if self.cell_type == 'LSTM': m = LSTM(self.lstm_size, return_sequences=False, activation=self.lstm_activation, name='lstm_meta_instrument')(meta_instrument_input)
            if self.cell_type == 'GRU': m = GRU(self.lstm_size, return_sequences=False, activation=self.lstm_activation, name='gru_meta_instrument')(meta_instrument_input)
            h = Concatenate(name='concatenated_instrument_and_notes_layer')([h, m])
            
        if self.meta_velocity:
            if self.cell_type == 'SimpleRNN': m = SimpleRNN(self.lstm_size, return_sequences=False, activation=self.lstm_activation, name='rnn_meta_velocity')(meta_velocity_input)
            if self.cell_type == 'LSTM': m = LSTM(self.lstm_size, return_sequences=False, activation=self.lstm_activation, name='lstm_meta_velocity')(meta_velocity_input)
            if self.cell_type == 'GRU': m = GRU(self.lstm_size, return_sequences=False, activation=self.lstm_activation, name='gru_meta_velocity')(meta_velocity_input)
            h = Concatenate(name='concatenated_velocity_and_rest_layer')([h, m])
            
        if self.meta_held_notes:
            if self.cell_type == 'SimpleRNN': m = SimpleRNN(self.lstm_size, return_sequences=False, activation=self.lstm_activation, name='rnn_meta_held_notes')(meta_held_notes_input)
            if self.cell_type == 'LSTM': m = LSTM(self.lstm_size, return_sequences=False, activation=self.lstm_activation, name='lstm_meta_held_notes')(meta_held_notes_input)
            if self.cell_type == 'GRU': m = GRU(self.lstm_size, return_sequences=False, activation=self.lstm_activation, name='gru_meta_held_notes')(meta_held_notes_input)
            h = Concatenate(name='concatenated_meta_held_notes_and_rest_layer')([h, m])

        #use a dense layer to pack all the meta information + notes together
        if self.meta_instrument or self.meta_velocity or self.meta_instrument:
            h = Dense(self.lstm_size, name='extra_instrument_after_concat_layer', activation=self.activation_before_splitting, kernel_initializer='glorot_uniform')(h)

        if self.extra_layer:
            h = Dense(self.lstm_size, name='extra_layer', activation=self.activation_before_splitting, kernel_initializer='glorot_uniform')(h)

        if self.split_lstm_vector:
            half_size = int(self.lstm_size/2)
            h_1 = Lambda(lambda x : x[:,:half_size], output_shape=(half_size,))(h)
            h_2 = Lambda(lambda x : x[:,half_size:], output_shape=(self.lstm_size-half_size,))(h)

        else:
            h_1 = h
            h_2 = h

        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(shape=(batch_size, self.latent_rep_size), mean=0., stddev=self.epsilon_std)
            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon 

        #s_3 = (s_1^2 * s_2^2) / (s_1^2 + s_2^2)
        #tf.contrib.distributions.MultivariateNormalDiag(mean=0.0, stddev=0.05, seed=None)
        z_mean = Dense(self.latent_rep_size, name='z_mean', activation='linear', kernel_initializer='glorot_uniform')(h_1)
        z_log_var = Dense(self.latent_rep_size, name='z_log_var', activation='linear', kernel_initializer='glorot_uniform')(h_2)
        
        if epsilon_factor > 0:
            e = Input(shape=(1,), tensor=K.constant(self.epsilon_factor))
            scaled_z_log_var = Add()[z_log_var, e]
            z_mean, scaled_z_log_var = KLDivergenceLayer(beta=self.beta, prior_mean=self.prior_mean, prior_std=self.prior_std, name='kl_layer')([z_mean, scaled_z_log_var])
        else:
            z_mean, z_log_var = KLDivergenceLayer(beta=self.beta, prior_mean=self.prior_mean, prior_std=self.prior_std, name='kl_layer')([z_mean, z_log_var])
        z = Lambda(sampling, output_shape=(self.latent_rep_size,), name='lambda')([z_mean, z_log_var])
        return (z)


    def _build_decoder(self, input_layer, encoded, ground_truth, history_input, decoder_additional_input_layer, input_decoder_meta_instrument_start, input_decoder_meta_velocity_start, input_decoder_meta_held_notes_start, input_decoder_meta_next_notes_start, meta_next_notes_ground_truth_input):

        input_states = []
        for layer_no in range(0,self.num_layers_decoder):

            state_c = Input((self.lstm_size,))
            input_states.append(state_c)

            if self.cell_type == 'LSTM':
                state_h = Input((self.lstm_size,))
                input_states.append(state_h)
        
        final_states = []
        lstm_input = input_layer
        for layer_no in range(0,self.num_layers_decoder):
            if self.cell_type == 'SimpleRNN': lstm_output, state1_t = SimpleRNNCell(self.lstm_size)([lstm_input, input_states[layer_no]])
            if self.cell_type == 'LSTM': lstm_output, state1_t, state2_t = LSTMCell(self.lstm_size)([lstm_input, input_states[layer_no*2], input_states[layer_no*2+1]])
            if self.cell_type == 'GRU': lstm_output, state1_t = GRUCell(self.lstm_size)([lstm_input, input_states[layer_no]])
            lstm_input = lstm_output
            final_states.append(state1_t)
            if self.cell_type == 'LSTM':
                final_states.append(state2_t)

        output = Dense(self.output_dim, activation=self.activation)(lstm_output)
        # use learn_mode = 'join', test_mode = 'viterbi', sparse_target = True (label indice output)
        
        readout_input_sequence = Input((self.output_length,self.output_dim))
        rnn = RecurrentModel(input_layer, output, initial_states=input_states, final_states=final_states, readout_input=readout_input_sequence, teacher_force=self.teacher_force, decode=self.decode, output_length=self.output_length, return_states=False, state_initializer=None, name='notes')

        if self.history:
            new_encoded = Concatenate()([encoded, history_input])
        else:
            new_encoded = encoded

        if self.decoder_additional_input:
            new_encoded = Concatenate()([new_encoded, decoder_additional_input_layer])
        else:
            new_encoded = new_encoded

        initial_states = []
        bias_initializer = 'zeros'
        for layer_no in range(0,self.num_layers_decoder):
        
            
            encoded_c = Dense(self.lstm_size, activation=self.lstm_state_activation, bias_initializer=bias_initializer)(new_encoded)
            initial_states.append(encoded_c)

            if self.cell_type == 'LSTM':
                encoded_h = Dense(self.lstm_size, activation=self.lstm_state_activation, bias_initializer=bias_initializer)(new_encoded)
                initial_states.append(encoded_h)
            
        decoded = rnn(input_layer, initial_state=initial_states, initial_readout=input_layer, ground_truth=ground_truth)

        if self.meta_instrument:
            input_states = []
            
            state_c = Input((self.lstm_size,))
            input_states.append(state_c)

            if self.cell_type == 'LSTM':
                state_h = Input((self.lstm_size,))
                input_states.append(state_h)
            
            final_states = []
            lstm_input = input_decoder_meta_instrument_start
            if self.cell_type == 'SimpleRNN': lstm_output, state1_t = SimpleRNNCell(self.lstm_size)([lstm_input, input_states[0]])
            if self.cell_type == 'LSTM': lstm_output, state1_t, state2_t = LSTMCell(self.lstm_size)([lstm_input, input_states[0], input_states[1]])
            if self.cell_type == 'GRU': lstm_output, state1_t = GRUCell(self.lstm_size)([lstm_input, input_states[0]])
            lstm_input = lstm_output
            final_states.append(state1_t)
            if self.cell_type == 'LSTM':
                final_states.append(state2_t)

            readout_input_sequence = Input((self.meta_instrument_length,self.meta_instrument_dim))
            output = Dense(self.meta_instrument_dim, activation=self.meta_instrument_activation)(lstm_output)
            rnn = RecurrentModel(input_decoder_meta_instrument_start, output, initial_states=input_states, final_states=final_states, readout_input=readout_input_sequence, teacher_force=False, decode=self.decode, output_length=self.meta_instrument_length, return_states=False, state_initializer=None, name='meta_instrument')
            
            initial_states = []
            bias_initializer = 'zeros'

            encoded_c = Dense(self.lstm_size, activation=self.lstm_state_activation, bias_initializer=bias_initializer)(new_encoded)
            initial_states.append(encoded_c)

            if self.cell_type == 'LSTM':
                encoded_h = Dense(self.lstm_size, activation=self.lstm_state_activation, bias_initializer=bias_initializer)(new_encoded)
                initial_states.append(encoded_h)
            meta_instrument_output = rnn(input_decoder_meta_instrument_start, initial_state=initial_states, initial_readout=input_decoder_meta_instrument_start)
        else:
            meta_instrument_output = None


        if self.meta_velocity:
            input_states = []
            
            state_c = Input((self.lstm_size,))
            input_states.append(state_c)

            if self.cell_type == 'LSTM':
                state_h = Input((self.lstm_size,))
                input_states.append(state_h)
            
            final_states = []
            lstm_input = input_decoder_meta_velocity_start
            if self.cell_type == 'SimpleRNN': lstm_output, state1_t = SimpleRNNCell(self.lstm_size)([lstm_input, input_states[0]])
            if self.cell_type == 'LSTM': lstm_output, state1_t, state2_t = LSTMCell(self.lstm_size)([lstm_input, input_states[0], input_states[1]])
            if self.cell_type == 'GRU': lstm_output, state1_t = GRUCell(self.lstm_size)([lstm_input, input_states[0]])
            lstm_input = lstm_output
            final_states.append(state1_t)
            if self.cell_type == 'LSTM':
                final_states.append(state2_t)

            readout_input_sequence = Input((self.meta_velocity_length,1))
            output = Dense(1, activation=self.meta_velocity_activation)(lstm_output)
            rnn = RecurrentModel(input_decoder_meta_velocity_start, output, initial_states=input_states, final_states=final_states, readout_input=readout_input_sequence, teacher_force=False, decode=self.decode, output_length=self.meta_velocity_length, return_states=False, state_initializer=None, name='meta_velocity')
            
            initial_states = []
            bias_initializer = 'zeros'

            encoded_c = Dense(self.lstm_size, activation=self.lstm_state_activation, bias_initializer=bias_initializer)(new_encoded)
            initial_states.append(encoded_c)

            if self.cell_type == 'LSTM':
                encoded_h = Dense(self.lstm_size, activation=self.lstm_state_activation, bias_initializer=bias_initializer)(new_encoded)
                initial_states.append(encoded_h)
            meta_velocity_output = rnn(input_decoder_meta_velocity_start, initial_state=initial_states, initial_readout=input_decoder_meta_velocity_start)
        else:
            meta_velocity_output = None


        if self.meta_held_notes:
            input_states = []
            
            state_c = Input((self.lstm_size,))
            input_states.append(state_c)

            if self.cell_type == 'LSTM':
                state_h = Input((self.lstm_size,))
                input_states.append(state_h)
            
            final_states = []
            lstm_input = input_decoder_meta_held_notes_start
            if self.cell_type == 'SimpleRNN': lstm_output, state1_t = SimpleRNNCell(self.lstm_size)([lstm_input, input_states[0]])
            if self.cell_type == 'LSTM': lstm_output, state1_t, state2_t = LSTMCell(self.lstm_size)([lstm_input, input_states[0], input_states[1]])
            if self.cell_type == 'GRU': lstm_output, state1_t = GRUCell(self.lstm_size)([lstm_input, input_states[0]])
            lstm_input = lstm_output
            final_states.append(state1_t)
            if self.cell_type == 'LSTM':
                final_states.append(state2_t)

            readout_input_sequence = Input((self.meta_held_notes_length,2))
            output = Dense(2, activation=self.meta_held_notes_activation)(lstm_output)
            rnn = RecurrentModel(input_decoder_meta_held_notes_start, output, initial_states=input_states, final_states=final_states, readout_input=readout_input_sequence, teacher_force=False, decode=self.decode, output_length=self.meta_held_notes_length, return_states=False, state_initializer=None, name='meta_held_notes')
            
            initial_states = []
            bias_initializer = 'zeros'

            encoded_c = Dense(self.lstm_size, activation=self.lstm_state_activation, bias_initializer=bias_initializer)(new_encoded)
            initial_states.append(encoded_c)

            if self.cell_type == 'LSTM':
                encoded_h = Dense(self.lstm_size, activation=self.lstm_state_activation, bias_initializer=bias_initializer)(new_encoded)
                initial_states.append(encoded_h)
            meta_held_notes_output = rnn(input_decoder_meta_held_notes_start, initial_state=initial_states, initial_readout=input_decoder_meta_held_notes_start)
        else:
            meta_held_notes_output = None

        if self.meta_next_notes:
            input_states = []
            for layer_no in range(0,self.num_layers_decoder):
            
                state_c = Input((self.lstm_size,))
                input_states.append(state_c)

                if self.cell_type == 'LSTM':
                    state_h = Input((self.lstm_size,))
                    input_states.append(state_h)
            
            final_states = []
            lstm_input = input_decoder_meta_next_notes_start
            for layer_no in range(0,self.num_layers_decoder):
                if self.cell_type == 'SimpleRNN': lstm_output, state1_t = SimpleRNNCell(self.lstm_size)([lstm_input, input_states[layer_no]])
                if self.cell_type == 'LSTM': lstm_output, state1_t, state2_t = LSTMCell(self.lstm_size)([lstm_input, input_states[layer_no*2], input_states[layer_no*2+1]])
                if self.cell_type == 'GRU': lstm_output, state1_t = GRUCell(self.lstm_size)([lstm_input, input_states[layer_no]])
                lstm_input = lstm_output
                final_states.append(state1_t)
                if self.cell_type == 'LSTM':
                    final_states.append(state2_t)

            output = Dense(self.output_dim, activation=self.activation)(lstm_output)
            
            readout_input_sequence = Input((self.meta_next_notes_output_length,self.output_dim))
            rnn = RecurrentModel(input_decoder_meta_next_notes_start, output, initial_states=input_states, final_states=final_states, readout_input=readout_input_sequence, teacher_force=self.meta_next_notes_teacher_force, decode=self.decode, output_length=self.meta_next_notes_output_length, return_states=False, state_initializer=None, name='next_notes')
            
            initial_states = []
            bias_initializer = 'zeros'

            for layer_no in range(0,self.num_layers_decoder):
             
                encoded_c = Dense(self.lstm_size, activation=self.lstm_state_activation, bias_initializer=bias_initializer)(new_encoded)
                initial_states.append(encoded_c)

                if self.cell_type == 'LSTM':
                    encoded_h = Dense(self.lstm_size, activation=self.lstm_state_activation, bias_initializer=bias_initializer)(new_encoded)
                    initial_states.append(encoded_h)
                
            meta_next_notes_output = rnn(input_decoder_meta_next_notes_start, initial_state=initial_states, initial_readout=input_decoder_meta_next_notes_start, ground_truth=meta_next_notes_ground_truth_input)
        else:
            meta_next_notes_output = None

        return decoded, meta_instrument_output, meta_velocity_output, meta_held_notes_output, meta_next_notes_output

    def _build_composer_decoder(self, encoded_rep):
        composer_latent_length = self.num_composers
        h = Lambda(lambda x : x[:,:composer_latent_length], output_shape=(composer_latent_length,))(encoded_rep)
        composer_prediction = Activation('softmax')(h)
        return composer_prediction


    def _build_signature_decoder(self, encoded_rep):
        decoder_latent_length = self.signature_dim
        #add additional offset if the composer already is attached to the first dimensions
        offset = 0
        if self.composer_decoder:
            offset += self.num_composers
        h = Lambda(lambda x : x[:,offset:offset+decoder_latent_length], output_shape=(decoder_latent_length,))(encoded_rep)
        signature_prediction = Activation(self.signature_activation)(h)
        return signature_prediction

    def _build_composer_decoder_at_notes_output(self, composer_notes_input):

        if self.cell_type == 'SimpleRNN': composer_notes_decoder_prediction = SimpleRNN(self.lstm_size, return_sequences=False, activation=self.lstm_activation, name='rnn_composer_decoder_at_notes')(composer_notes_input)
        if self.cell_type == 'LSTM': composer_notes_decoder_prediction = LSTM(self.lstm_size, return_sequences=False, activation=self.lstm_activation, name='lstm_composer_decoder_at_notes')(composer_notes_input)
        if self.cell_type == 'GRU': composer_notes_decoder_prediction = GRU(self.lstm_size, return_sequences=False, activation=self.lstm_activation, name='gru_composer_decoder_at_notes')(composer_notes_input)
        composer_notes_decoder_prediction = Dense(self.num_composers, activation=self.composer_decoder_at_notes_activation)(composer_notes_decoder_prediction)
        return composer_notes_decoder_prediction

    def _build_composer_decoder_at_instrument_output(self, composer_instrument_input):

        if self.cell_type == 'SimpleRNN': composer_instrument_decoder_prediction = SimpleRNN(self.lstm_size, return_sequences=False, activation=self.lstm_activation, name='rnn_composer_decoder_at_instrument')(composer_instrument_input)
        if self.cell_type == 'LSTM': composer_instrument_decoder_prediction = LSTM(self.lstm_size, return_sequences=False, activation=self.lstm_activation, name='lstm_composer_decoder_at_instrument')(composer_instrument_input)
        if self.cell_type == 'GRU': composer_instrument_decoder_prediction = GRU(self.lstm_size, return_sequences=False, activation=self.lstm_activation, name='gru_composer_decoder_at_instrument')(composer_instrument_input)
        composer_instrument_decoder_prediction = Dense(self.num_composers, activation=self.composer_decoder_at_instrument_activation)(composer_instrument_decoder_prediction)
        return composer_instrument_decoder_prediction



# prerpares encoder input  for a song that is already split
# X: input pitches of shape (num_samples, input_length, different_pitches)
# I: instruments for each voice of shape (max_voices, different_instruments)
# V: velocity information of shape (num_samples, output_length==input_length), values between 0 and 1 when there is no silent note, 1 denotes MAX_VELOCITY
# D: duration information of shape (num_samples, output_length==input_length), values are 1 if a note is held
def prepare_encoder_input_list(X,I,V,D):
    num_samples = X.shape[0]

    #transform duration into categorical
    D_cat = np.zeros((D.shape[0], D.shape[1], 2))
    for sample in range(num_samples):
        for step in range(output_length):
            if D[sample,step] == 0:
                D_cat[sample, step, 0] = 1
            else:
                D_cat[sample, step, 1] = 1
    D = D_cat

    V = np.copy(V) #make a deep copy since it may be changed if you combine_velocity_and_held_notes
    V = np.expand_dims(V, 2)
    if combine_velocity_and_held_notes:
        for sample in range(num_samples):
            for step in range(output_length):
                if D[sample,step, 1] == 1:
                    #assert, that it is held and therefore has no velocity since its not hit
                    assert(V[sample, step,0] == 0)
                    V[sample,step,0] = 1
    

    #tile the meta_instrument as well for every sample
    I = np.tile(np.expand_dims(I, axis=0), (num_samples,1,1))

    if meta_instrument or meta_velocity or meta_held_notes:
        encoder_input_list = [X]
        if meta_instrument:
            encoder_input_list.append(I)
        if meta_velocity:
            encoder_input_list.append(V)
        if meta_held_notes:
            encoder_input_list.append(D)

        return encoder_input_list
    else:
        return X


# prerpares autoencoder input and output for a song that is already split
# R: latent list of shape (num_samples, latent_dim)
# C: class of epoch, integer in range(num_classes)
# S: normalized signature vector of shape (num_samples, signature_dim)
# H: history list of shape (num_samples, latent_dim), if None will form automatically from R by rolling once
def prepare_decoder_input(R,C,S,H=None):

    num_samples = R.shape[0]

    Y_start = np.zeros((num_samples, output_dim))
    input_list = [Y_start, R]

    if teacher_force:
        empty_Y = np.zeros((num_samples, input_length, output_dim))
        input_list.append(empty_Y)

    if history:
        if H is not None:
            input_list.append(H)
        else:
            history_list = np.zeros(R.shape)
            history_list[1:] = R[:-1]
            input_list.append(history_list)

    if decoder_additional_input:
        decoder_additional_input_list = []
        if decoder_input_composer:
            decoder_additional_input_list.extend(C)
        
        if append_signature_vector_to_latent:
            if len(decoder_additional_input_list) > 0:
                decoder_additional_input_list = np.asarray(decoder_additional_input_list)
                decoder_additional_input_list = np.append(decoder_additional_input_list, S, axis=1)
            else:
                decoder_additional_input_list.extend(S)
        decoder_additional_input_list = np.asarray(decoder_additional_input_list)
        input_list.append(decoder_additional_input_list)

    if meta_instrument:
        meta_instrument_start = np.zeros((num_samples, meta_instrument_dim))
        input_list.append(meta_instrument_start)

    if meta_velocity:
        meta_velocity_start = np.zeros((num_samples,))
        input_list.append(meta_velocity_start)

    if meta_held_notes:
        meta_held_notes_start = np.zeros((num_samples,2))
        input_list.append(meta_held_notes_start)

    if meta_next_notes:
        meta_next_notes_start = np.zeros((num_samples,output_dim))
        input_list.append(meta_next_notes_start)

    return input_list





# prerpares autoencoder input and output for a song that is already split
# X: input pitches of shape (num_samples, input_length, different_pitches)
# Y: ouput pitches of shape (num_samples, ouput_length, different_pitches)
# C: class of epoch, integer in range(num_classes)
# I: instruments for each voice of shape (max_voices, different_instruments)
# V: velocity information of shape (num_samples, output_length==input_length), values between 0 and 1 when there is no silent note, 1 denotes MAX_VELOCITY
# D: duration information of shape (num_samples, output_length==input_length), values are 1 if a note is held
# S: normalized signature vector of shape (num_samples, signature_dim)
# H: history list of shape (num_samples, latent_dim)
def prepare_autoencoder_input_and_output_list(X,Y,C,I,V,D,S,H, return_sample_weight=False):

    num_samples = X.shape[0]

    #transform duration into categorical
    D_cat = np.zeros((D.shape[0], D.shape[1], 2))
    for sample in range(num_samples):
        for step in range(output_length):
            if D[sample,step] == 0:
                D_cat[sample, step, 0] = 1
            else:
                D_cat[sample, step, 1] = 1
    D = D_cat

    V = np.copy(V) #make a deep copy since it may be changed if you combine_velocity_and_held_notes
    V = np.expand_dims(V, 2)
    if combine_velocity_and_held_notes:
        for sample in range(num_samples):
            for step in range(output_length):
                if D[sample,step, 1] == 1:
                    #assert, that it is held and therefore has no velocity since its not hit
                    assert(V[sample, step,0] == 0)
                    V[sample,step,0] = 1
    
    #since we have to predict the last 
    if meta_next_notes:
        N = Y[1:] #next input
        X = X[:-1]
        Y = Y[:-1]
        V = V[:-1]
        D = D[:-1]
        S = S[:-1]
        H = H[:-1]
        num_samples = X.shape[0]

    #create start symbol for every sample
    Y_start = np.zeros((num_samples, Y.shape[2]))

    #transform C into categorical format as well and duplicate it by num_samples
    C = np.asarray([to_categorical(C, num_classes=num_classes)]*num_samples).squeeze()

    #tile the meta_instrument as well for every sample
    meta_instrument_input = np.tile(np.expand_dims(I, axis=0), (num_samples,1,1))

    input_list = [X,Y_start]

    output_list = [Y]

    if return_sample_weight:
        #weight matrix for every sample and steps
        sample_weight = np.ones((num_samples,output_length))
        if include_silent_note:
            #set the weight to silent_weight for every sample where a silent note is played
            sample_weight[np.where(Y[:,:,-1]==1)] = silent_weight

        if include_composer_decoder:
            sample_weight_composer_decoder = np.ones((num_samples,))
            if isinstance(sample_weight, list):
                sample_weight.append(sample_weight_composer_decoder)
            else:
                sample_weight = [sample_weight, sample_weight_composer_decoder]

        if signature_decoder:
            sample_weight_signature_decoder = np.ones((num_samples,))
            if isinstance(sample_weight, list):
                sample_weight.append(sample_weight_signature_decoder)
            else:
                sample_weight = [sample_weight, sample_weight_signature_decoder]

        if composer_decoder_at_notes_output:
            sample_weight_composer_notes_decoder = np.ones((num_samples,))
            if isinstance(sample_weight, list):
                sample_weight.append(sample_weight_composer_notes_decoder)
            else:
                sample_weight = [sample_weight, sample_weight_composer_notes_decoder]

        if composer_decoder_at_instrument_output:
            sample_weight_composer_instrument_decoder = np.ones((num_samples,))
            if isinstance(sample_weight, list):
                sample_weight.append(sample_weight_composer_instrument_decoder)
            else:
                sample_weight = [sample_weight, sample_weight_composer_instrument_decoder]

    if teacher_force:
        input_list.append(Y)

    if history:
        input_list.append(H)
    if decoder_additional_input:
        decoder_additional_input_list = []
        if decoder_input_composer:
            decoder_additional_input_list.extend(C)
        if append_signature_vector_to_latent:
            
            if len(decoder_additional_input_list) > 0:
                decoder_additional_input_list = np.asarray(decoder_additional_input_list)
                decoder_additional_input_list = np.append(decoder_additional_input_list, S, axis=1)
            else:
                decoder_additional_input_list.extend(S)
        decoder_additional_input_list = np.asarray(decoder_additional_input_list)
        input_list.append(decoder_additional_input_list)

    if meta_instrument:
        meta_instrument_start = np.zeros((num_samples, meta_instrument_dim))
        input_list.append(meta_instrument_start)
        input_list.append(meta_instrument_input)
        output_list.append(meta_instrument_input)
        if return_sample_weight:
            sample_weight_meta_instrument = np.ones((num_samples,))
            if isinstance(sample_weight, list):
                sample_weight.append(sample_weight_meta_instrument)
            else:
                sample_weight = [sample_weight, sample_weight_meta_instrument]

    if meta_velocity:
        meta_velocity_start = np.zeros((num_samples,))
        input_list.append(meta_velocity_start)
        input_list.append(V)
        output_list.append(V)
        if return_sample_weight:
            sample_weight_meta_velocity = np.ones((num_samples,))
            if isinstance(sample_weight, list):
                sample_weight.append(sample_weight_meta_velocity)
            else:
                sample_weight = [sample_weight, sample_weight_meta_velocity]

    if meta_held_notes:
        meta_held_notes_start = np.zeros((num_samples,2))
        input_list.append(meta_held_notes_start)
        input_list.append(D)
        output_list.append(D)
        if return_sample_weight:
            sample_weight_meta_held_notes = np.ones((num_samples,))
            if isinstance(sample_weight, list):
                sample_weight.append(sample_weight_meta_held_notes)
            else:
                sample_weight = [sample_weight, sample_weight_meta_held_notes]

    if meta_next_notes:
        meta_next_notes_start = np.zeros((num_samples,output_dim))
        input_list.append(meta_next_notes_start)
        output_list.append(N)
        if return_sample_weight:
            #set the weight to silent_weight for every sample where a silent note is played
            sample_weight_meta_next_notes = np.ones((num_samples,))
            if isinstance(sample_weight, list):
                sample_weight.append(sample_weight_meta_next_notes)
            else:
                sample_weight = [sample_weight, sample_weight_meta_next_notes]

    if include_composer_decoder:
        output_list.append(C)

    if signature_decoder:
        output_list.append(S)

    if composer_decoder_at_notes_output:
        output_list.append(C)

    if composer_decoder_at_instrument_output:
        output_list.append(C)

    if return_sample_weight:
        return input_list, output_list, sample_weight
    else:
        return input_list, output_list

#samples a vector by giving an index back that was chosen
def sample_vector(vector, sample_method):
    if np.sum(vector) > 0:
        if sample_method == 'argmax':
            max_index = np.argmax(vector)

        if sample_method == 'choice':
            vector = vector/(np.sum(vector)*1.0)

            vector = np.log(vector) / temperature
            vector = np.exp(vector) / np.sum(np.exp(vector))

            #give it number_of_tries to find a note that is above the cutoff_sample_threshold
            for _ in range(number_of_tries):
                max_index = np.random.choice(len(vector), p=vector)

                if vector[max_index] > cutoff_sample_threshold:
                    break
    else:
        max_index = 0
    return max_index



def sample_notes_prediction(Y, sample_method):
    assert(Y.ndim == 2 or Y.ndim == 3)
    if Y.ndim == 2:
        input_song = Y
    if Y.ndim == 3:
        input_song = []
        for i in range(Y.shape[0]):
            song_part = Y[i]
            input_song.extend(song_part)


    input_song = np.asarray(input_song)

    output_song = np.zeros((input_song.shape[0], high_crop-low_crop))

    for i, step in enumerate(input_song):

        max_index = sample_vector(step, sample_method)

        if include_silent_note and max_index == len(step)-1:
            continue 

        output_song[i, max_index] = 1

    return output_song

def sample_instrument_prediction(I, sample_method):
    if I.ndim > 1:
        output_list = []
        for i in range(I.shape[0]):
            output_list.append(sample_instrument_prediction(I[i], sample_method))
        return np.asarray(output_list)
    else:
        max_index = sample_vector(I, sample_method)
        ret_X = np.zeros(I.shape)
        ret_X[max_index] = 1
        return ret_X

def sample_held_notes_prediction(D, sample_method):
    if D.ndim > 1:
        output_list = []
        for i in range(D.shape[0]):
            pred = sample_held_notes_prediction(D[i], sample_method)
            if isinstance(pred, int):
                output_list.append(pred)
            else:
                output_list.extend(pred)
        return np.asarray(output_list)
    else:
        #go back from categorical format to list format
        max_index = sample_vector(D, sample_method)
        return int(max_index)
        
#returns:
# Y: ouput pitches of shape (num_samples, ouput_length, different_pitches)
# I: instruments for each voice of shape (max_voices, different_instruments)
# V: velocity information of shape (num_samples, output_length==input_length), values between 0 and 1 when there is no silent note, 1 denotes MAX_VELOCITY
# D: duration information of shape (num_samples, output_length==input_length), values are 1 if a note is held
# N: next note pitches of shape (num_samples, output_length, different_pitches)
# sample method: 'argmax' or 'choice'
def process_decoder_outputs(decoder_outputs, sample_method):

    #initalize to None
    Y = None
    I = None
    V = None
    D = None
    N = None

    if isinstance(decoder_outputs, list):
        Y = decoder_outputs[0]
        num_samples = Y.shape[0]
        Y = sample_notes_prediction(Y, sample_method)
        count = 1
        if meta_instrument or meta_velocity or meta_held_notes_output or meta_next_notes:
            meta_instrument_pred = decoder_outputs[count]
            I = sample_instrument_prediction(meta_instrument_pred, sample_method)
            count += 1

        if meta_velocity:
            #set to 0 if silent note or if too low if not meta_held_notes
            meta_velocity_pred = decoder_outputs[count]
            V = np.zeros((Y.shape[0],))
            for sample in range(meta_velocity_pred.shape[0]):
                V[sample*output_length:(sample+1)*output_length] = meta_velocity_pred[sample,:,0]
            for step in range(V.shape[0]):
                #if silent
                if np.sum(Y[step]) == 0:
                    V[step] = 0

            if override_sampled_pitches_based_on_velocity_info:

                for voice in range(max_voices):
                    previous_pitch = -1
                    previous_velocity = 0.0
                    voice_pitch_roll = Y[voice::max_voices]
                    voice_velocity_roll = V[voice::max_voices]
                    for i, (note_vector, velocity) in enumerate(zip(voice_pitch_roll, voice_velocity_roll)):
                        
                        pitch_is_silent = np.sum(note_vector) == 0
                        if pitch_is_silent:
                            pitch = -1
                        else:
                            pitch = np.argmax(note_vector)

                        velocity_is_silent = velocity < velocity_threshold_such_that_it_is_a_played_note
                        
                        if velocity_is_silent:

                            if not pitch_is_silent and previous_pitch > 0 and previous_pitch != pitch:
                                #play it as loud as previous note
                                V[i*max_voices + voice] = previous_velocity
                        else:
                            if pitch_is_silent:
                                #set velocity to 0, since there is no pitch played
                                V[i*max_voices + voice] = 0

                        previous_pitch = pitch
                        if not velocity_is_silent:
                            previous_velocity = velocity
            count += 1

        if meta_held_notes:
            meta_held_notes_pred = decoder_outputs[count]
            D = sample_held_notes_prediction(meta_held_notes_pred, sample_method)
            count += 1

        if meta_next_notes:
            meta_next_notes_pred = decoder_outputs[count]
            N = sample_notes_prediction(meta_next_notes_pred, sample_method)
            count += 1
    else:
        Y = sample_notes_prediction(decoder_outputs, sample_method)

    length_of_song = Y.shape[0]
    
    # set default outputs if not in output of model
    if I is None: 
        I = np.zeros((length_of_song//output_length,max_voices, meta_instrument_dim))
        I[:,0] = 1 # all piano
    if V is None: 
        #set all to half of the max_velocity
        V = np.ones((length_of_song,)) * (velocity_threshold_such_that_it_is_a_played_note + (1.-velocity_threshold_such_that_it_is_a_played_note)*0.5)
    if D is None: 
        D = np.ones((length_of_song,))
        if meta_velocity:
            #if there is velocity information (for every played note) but no held notes:
            #then you can figure out which notes are played or held
            for step in range(length_of_song):
                if V[step] > velocity_threshold_such_that_it_is_a_played_note:
                    D[step] = 0
    if N is None: 
        N = np.zeros(Y.shape) #all silent

    return Y, I, V, D, N

#returns:
# Y: ouput pitches of shape (num_samples, ouput_length, different_pitches)
# I: instruments for each voice of shape (max_voices, different_instruments)
# V: velocity information of shape (num_samples, output_length==input_length), values between 0 and 1 when there is no silent note, 1 denotes MAX_VELOCITY
# D: duration information of shape (num_samples, output_length==input_length), values are 1 if a note is held
# N: next note pitches of shape (num_samples, output_length, different_pitches)
# sample method: 'argmax' or 'choice'
def process_autoencoder_outputs(autoencoder_outputs, sample_method):
    return process_decoder_outputs(autoencoder_outputs, sample_method)




    
