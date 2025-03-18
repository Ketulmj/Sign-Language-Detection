import json
import os
import random
import keras
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Input, LSTM, Dense, Bidirectional
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import joblib
import config
import tensorflow as tf

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Enable mixed precision training
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("GPU is configured successfully with mixed precision training")
    except RuntimeError as e:
        print("GPU configuration error:", e)
else:
    print("No GPU found. Running on CPU")


class VideoDescriptionTrain(object):
    """
    Initialize the parameters for the model
    """
    def __init__(self, config):
        self.train_path = config.train_path
        self.test_path = config.test_path
        self.max_length = config.max_length
        self.batch_size = config.batch_size
        self.lr = config.learning_rate
        self.epochs = config.epochs
        self.latent_dim = config.latent_dim
        self.validation_split = config.validation_split
        self.num_encoder_tokens = config.num_encoder_tokens
        self.num_decoder_tokens = config.num_decoder_tokens
        self.time_steps_encoder = config.time_steps_encoder
        self.time_steps_decoder = config.max_length - 1
        self.x_data = {}

        # processed data
        self.tokenizer = None

        # models
        self.encoder_model = None
        self.decoder_model = None
        self.inf_encoder_model = None
        self.inf_decoder_model = None
        self.save_model_path = config.save_model_path

    def preprocessing(self):
        """
        Preprocessing the data
        dumps values of the json file into a list
        """
        TRAIN_LABEL_PATH = os.path.join(self.train_path, 'training_label.json')
        with open(TRAIN_LABEL_PATH) as data_file:
            y_data = json.load(data_file)
        train_list = []
        vocab_list = []
        for y in y_data:
            for caption in y['caption']:
                # Normalize and clean caption text
                caption = caption.lower().strip()
                caption = "<bos> " + caption + " <eos>"
                # More flexible length filtering
                if len(caption.split()) > 15 or len(caption.split()) < 4:
                    continue
                else:
                    train_list.append([caption, y['id'].split('-')[1]])

        random.shuffle(train_list)
        training_list = train_list[int(len(train_list) * self.validation_split):]
        validation_list = train_list[:int(len(train_list) * self.validation_split)]
        for train in training_list:
            vocab_list.append(train[0])
        self.tokenizer = Tokenizer(num_words=self.num_decoder_tokens)
        self.tokenizer.fit_on_texts(vocab_list)

        TRAIN_FEATURE_DIR = os.path.join(self.train_path, 'feat')
        for filename in os.listdir(TRAIN_FEATURE_DIR):
            f = np.load(os.path.join(TRAIN_FEATURE_DIR, filename), allow_pickle=True)
            self.x_data[filename[:-4].split('.')[0]] = f
        return training_list, validation_list

    def load_dataset(self, training_list):
        """
        Loads the dataset in batches for training using tf.data.Dataset
        :return: TensorFlow dataset
        """
        import tensorflow as tf

        def generator():
            encoder_input_data = []
            decoder_input_data = []
            decoder_target_data = []
            videoId = []
            videoSeq = []
            for idx, cap in enumerate(training_list):
                caption = cap[0]
                videoId.append(cap[1])
                videoSeq.append(caption)

            train_sequences = self.tokenizer.texts_to_sequences(videoSeq)
            train_sequences = pad_sequences(train_sequences, padding='post', truncating='post',
                                            maxlen=self.max_length)
            file_size = len(train_sequences)
            n = 0
            for idx in range(0, file_size):
                n += 1
                encoder_input_data.append(self.x_data.get(videoId[idx]))
                y = to_categorical(train_sequences[idx], self.num_decoder_tokens)
                decoder_input_data.append(y[:-1])
                decoder_target_data.append(y[1:])
                if n == self.batch_size:
                    encoder_input = np.array(encoder_input_data)
                    decoder_input = np.array(decoder_input_data)
                    decoder_target = np.array(decoder_target_data)
                    encoder_input_data = []
                    decoder_input_data = []
                    decoder_target_data = []
                    n = 0
                    yield {
                        'encoder_inputs': encoder_input,
                        'decoder_inputs': decoder_input
                    }, decoder_target

        output_signature = (
            {
                'encoder_inputs': tf.TensorSpec(shape=(self.batch_size, self.time_steps_encoder, self.num_encoder_tokens), dtype=tf.float32),
                'decoder_inputs': tf.TensorSpec(shape=(self.batch_size, self.max_length-1, self.num_decoder_tokens), dtype=tf.float32)
            },
            tf.TensorSpec(shape=(self.batch_size, self.max_length-1, self.num_decoder_tokens), dtype=tf.float32)
        )
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature
        )
        dataset = (dataset.repeat()  # Ensure dataset repeats indefinitely
                       .prefetch(buffer_size=tf.data.AUTOTUNE))
        return dataset

    def train_model(self):
        """
        an encoder decoder sequence to sequence model
        reference : https://arxiv.org/abs/1505.00487
        """
        encoder_inputs = Input(shape=(config.time_steps_encoder, config.num_encoder_tokens), name="encoder_inputs")
        encoder = Bidirectional(LSTM(config.latent_dim, return_sequences=True, return_state=True, name='encoder_lstm'))
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_inputs)
        state_h = keras.layers.Concatenate()([forward_h, backward_h])
        state_c = keras.layers.Concatenate()([forward_c, backward_c])
        encoder_states = [state_h, state_c]
        
        # Enhanced Attention mechanism with batch normalization
        attention = Dense(config.latent_dim, activation='tanh')(encoder_outputs)
        attention = keras.layers.BatchNormalization()(attention)
        attention = Dense(1)(attention)
        attention = keras.layers.Flatten()(attention)
        attention = keras.layers.Activation('softmax')(attention)
        attention = keras.layers.RepeatVector(config.latent_dim * 2)(attention)
        attention = keras.layers.Permute([2, 1])(attention)
        context = keras.layers.multiply([encoder_outputs, attention])
        context = keras.layers.BatchNormalization()(context)
        context = keras.layers.Lambda(lambda x: keras.backend.sum(x, axis=1), output_shape=(config.latent_dim * 2,))(context)

        decoder_inputs = Input(shape=(self.max_length - 1, self.num_decoder_tokens), name="decoder_inputs")
        decoder_lstm = LSTM(config.latent_dim * 2, return_sequences=True, return_state=True, name='decoder_lstm')
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax', name='decoder_softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        training_list, validation_list = self.preprocessing()

        train_dataset = self.load_dataset(training_list)
        valid_dataset = self.load_dataset(validation_list)

        early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='min')
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=5,
            verbose=1,
            mode="auto",
            min_lr=1e-6
        )

        # Run training with gradient clipping
        optimizer = keras.optimizers.Adam(learning_rate=self.lr, clipnorm=1.0)
        model.compile(
            metrics=['accuracy'],
            optimizer=optimizer,
            loss='categorical_crossentropy'
        )

        validation_steps = len(validation_list)//self.batch_size
        steps_per_epoch = len(training_list)//self.batch_size

        model.fit(
            train_dataset,
            validation_data=valid_dataset,
            validation_steps=validation_steps,
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=[early_stopping, reduce_lr]
        )

        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)

        self.encoder_model = Model(encoder_inputs, encoder_states)
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)
        # self.encoder_model.summary()

        # saving the models
        self.encoder_model.save(os.path.join(self.save_model_path, 'encoder_model.keras'))
        self.decoder_model.save(os.path.join(self.save_model_path, 'decoder_model.keras'))
        self.decoder_model.save_weights(os.path.join(self.save_model_path, 'decoder_model.weights.h5'))
        with open(os.path.join(self.save_model_path, 'tokenizer' + str(self.num_decoder_tokens)), 'wb') as file:
            joblib.dump(self.tokenizer, file)

if __name__ == "__main__":
    video_to_text = VideoDescriptionTrain(config)
    video_to_text.train_model()
