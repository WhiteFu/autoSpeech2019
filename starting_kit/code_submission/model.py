import librosa
from sklearn.utils import shuffle
import json
import tensorflow as tf
import numpy as np
from tensorflow.python.keras import models
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D, LSTM
from tensorflow.python.keras.layers import MaxPooling2D,BatchNormalization,GlobalAveragePooling2D,GlobalMaxPooling2D
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.callbacks import ModelCheckpoint, Callback
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from keras.backend.tensorflow_backend import set_session
from keras.layers import add
from sklearn.preprocessing import StandardScaler
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

# class Metrics(tf.keras.callbacks.Callback):
#     def on_train_begin(self, logs={}):
#         self.val_f1s = []
#         self.val_recalls = []
#         self.val_precisions = []

#     def on_epoch_end(self, epoch, logs={}):
#         val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
#         val_targ = self.validation_data[1]
#         _val_f1 = f1_score(val_targ, val_predict)
#         _val_recall = recall_score(val_targ, val_predict)
#         _val_precision = precision_score(val_targ, val_predict)
#         self.val_f1s.append(_val_f1)
#         self.val_recalls.append(_val_recall)
#         self.val_precisions.append(_val_precision)
#         print('- val_f1: %.4f - val_precision: %.4f - val_recall: %.4f'%(_val_f1, _val_precision, _val_recall))
#         return


def windows(data, window_size):
    start = 0
    i = 0
    max_i = 1000
    while (start < len(data)) and (i < max_i):
        yield int(start), int(start + window_size)
        start += window_size
        i += 1

def extract_mfcc(data,sr=16000):
    results = []
    for d in data:
        r = librosa.feature.mfcc(d,sr=16000,n_mfcc=30)
        r = r.transpose()
        results.append(r)
    return results


def extract_features1(data, sr=16000, bands=60, frames=11):
    log_specgrams = []
    for d in data:
        d = librosa.effects.trim(d, top_db=23, frame_length=512, hop_length=128)[0]
        melspec = librosa.feature.melspectrogram(d, n_mels=bands)
        logspec = librosa.core.amplitude_to_db(melspec).T
        log_specgrams.append(logspec)
    return log_specgrams

def extract_features2(data, sr=16000):
    results = []
    for d in data:
        d = librosa.effects.trim(d)[0]
        stft = np.abs(librosa.stft(d))
        mfccs = np.array(librosa.feature.mfcc(y=d, sr=sr,n_mfcc=30).T)
        chroma = np.array(librosa.feature.chroma_stft(S=stft, sr=sr).T)
        zcr = np.array(librosa.feature.zero_crossing_rate(d).T)
        rolloff = np.array(librosa.feature.spectral_rolloff(y=d, sr=sr).T)
        spec_bw = np.array(librosa.feature.spectral_bandwidth(y=d, sr=sr).T)
        # spec_cent = np.array(librosa.feature.spectral_centroid(y=d, sr=sr).T)
        # mel = librosa.feature.melspectrogram(d, sr=sr, n_mels=60)
        # mel = np.array(librosa.core.amplitude_to_db(mel).T)
        # contrast = np.array(librosa.feature.spectral_contrast(S=stft, sr=sr).T)
        # tonnetz = np.array(librosa.feature.tonnetz(y=librosa.effects.harmonic(d), sr=sr).T)
        #r = np.hstack([mfccs, chroma, zcr, rolloff, spec_bw, spec_cent])
        r = np.hstack([mfccs, chroma, zcr, rolloff, spec_bw])
        results.append(r)
    
    return results



def pad_seq(data,pad_len):
    return sequence.pad_sequences(data,maxlen=pad_len,dtype='float32',padding='post')

# onhot encode to category
def ohe2cat(label):
    return np.argmax(label, axis=1)

# def Conv2D(x, filters, kernel_size, strides=(1, 1), padding='same', name=None):
#     if name:
#         bn_name = name + '_bn'
#         conv_name = name + '_conv'
#     else:
#         bn_name = None
#         conv_name = None
#     x = Conv2D(filters, kernel_size, strides=strides, padding=padding, activation='relu', name=conv_name)(x)
#     #x = BatchNormalization(name=bn_name)(x)
#     return x

# def Conv2D_BN(x, filters, kernel_size, strides=(1, 1), padding='same', name=None):
#     if name:
#         bn_name = name + '_bn'
#         conv_name = name + '_conv'
#     else:
#         bn_name = None
#         conv_name = None
#     x = Conv2D(filters, kernel_size, strides=strides, padding=padding, activation='relu', name=conv_name)(x)
#     x = BatchNormalization(name=bn_name)(x)
#     return x

# def identity_block_BN(input_tensor, filters, kernel_size, strides=(1, 1), is_conv_shortcuts=False):
#     """
#     :param input_tensor:
#     :param filters:
#     :param kernel_size:
#     :param strides:
#     :param is_conv_shortcuts: 直接连接或者投影连接
#     :return:
#     """
#     x = Conv2D_BN(input_tensor, filters, kernel_size, strides=strides, padding='same')
#     x = Conv2D_BN(x, filters, kernel_size, padding='same')
#     if is_conv_shortcuts:
#         shortcut = Conv2D_BN(input_tensor, filters, kernel_size, strides=strides, padding='same')
#         x = add([x, shortcut])
#     else:
#         x = add([x, input_tensor])
#     return x

def cnn_model(input_shape,num_class,max_layer_num=5):
        model = Sequential()
        #min_size = min(input_shape[:2])
        model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='elu', input_shape=input_shape, padding='same'))
        #model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
        #model.add(Dropout(0.5))
        model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape, padding='same'))
        #model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
        #model.add(Dropout(0.5))
        model.add(Conv2D(128, kernel_size=(5, 5), activation='elu', padding='same'))
        #model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
        model.add(Conv2D(128, kernel_size=(5, 5), activation='elu', padding='same'))
       # model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
        #model.add(Dropout(0.5))
        model.add(Conv2D(256, kernel_size=(5, 5), activation='elu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(GlobalAveragePooling2D())
        model.add(Dense(num_class))
        model.add(Activation('softmax'))



        return model
                
# def hybird_model(input_shape,num_class,max_layer_num=5):

class Model(object):

    def __init__(self, metadata, train_output_path="./", test_input_path="./"):
        """ Initialization for model
        :param metadata: a dict formed like:
            {"class_num": 7,
             "train_num": 428,
             "test_num": 107,
             "time_budget": 1800}
        """
        self.done_training = False
        self.metadata = metadata
        self.train_output_path = train_output_path
        self.test_input_path = test_input_path
        # self.std = StandardScaler()

    def train(self, train_dataset, remaining_time_budget=None):
        """model training on train_dataset.
        
        :param train_dataset: tuple, (x_train, y_train)
            train_x: list of vectors, input train speech raw data.
            train_y: A `numpy.ndarray` matrix of shape (sample_count, class_num).
                     here `sample_count` is the number of examples in this dataset as train
                     set and `class_num` is the same as the class_num in metadata. The
                     values should be binary.
        :param remaining_time_budget:
        """
        if self.done_training:
            return
        train_x, train_y = train_dataset
        train_x, train_y = shuffle(train_x, train_y)

        #extract train feature
        fea_x = extract_mfcc(train_x)
        max_len = max([len(_) for _ in fea_x])
        fea_x = pad_seq(fea_x, max_len)
       
        num_class = self.metadata['class_num']
        X=fea_x[:,:,:, np.newaxis]
        y=train_y
        
        model = cnn_model(X.shape[1:],num_class)
 

        optimizer = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
        #optimizer = tf.keras.optimizers.SGD(lr=0.001,decay=1e-6)
        # model.compile(loss = 'sparse_categorical_crossentropy',
        #              optimizer = optimizer,
        #              metrics= ['accuracy'])
        model.compile(loss = 'sparse_categorical_crossentropy',
                     optimizer = optimizer,
                     metrics=['accuracy'])
        model.summary()

        # metrics = Metrics()
        checkpoint_path = self.train_output_path + '/model.h5'
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks = tf.keras.callbacks.EarlyStopping(
                    monitor='val_acc', patience=20)
        history = model.fit(X,ohe2cat(y),
                    epochs=200,
                    validation_split=0.2,
                    callbacks=[callbacks,checkpoint],
                    verbose=1,  # Logs once per epoch.
                    batch_size=64,
                    shuffle=True)
        # #绘制指标值
        # plt.plot(history.history['acc'],'b--')
        # plt.plot(history.history['val_acc'],'y-')
        # # plt.plot(metrics.val_f1s,'r.-')
        # # plt.plot(metrics.val_precisions,'g-')
        # # plt.plot(metrics.val_recalls,'c-')
        # plt.title('autospeech2019')
        # plt.ylabel('evaluation')
        # plt.xlabel('epoch')
        # #plt.legend(['train_accuracy', 'val_accuracy','val_f1-score','val_precisions','val_recalls'], loc='lower right')
        # plt.legend(['train_accuracy', 'val_accuracy'], loc='lower right')
        # fig_path = self.train_output_path + '/scoring_output/result_acc.png'
        # plt.savefig(fig_path)
        # plt.show()


       
        model.save(self.train_output_path + '/model.h5')

        with open(self.train_output_path + '/feature.config', 'wb') as f:
            f.write(str(max_len).encode())
            f.close()

        self.done_training=True

    def test(self, test_x, remaining_time_budget=None):
        """
        :param x_test: list of vectors, input test speech raw data.
        :param remaining_time_budget:
        :return: A `numpy.ndarray` matrix of shape (sample_count, class_num).
                     here `sample_count` is the number of examples in this dataset as train
                     set and `class_num` is the same as the class_num in metadata. The
                     values should be binary.
        """
        model = models.load_model(self.test_input_path + '/model.h5')
        with open(self.test_input_path + '/feature.config', 'r') as f:
            max_len = int(f.read().strip())
            f.close()

        #extract test feature
        fea_x = extract_mfcc(test_x)
        fea_x = pad_seq(fea_x, max_len)
        test_x=fea_x[:,:,:, np.newaxis]

        #predict
        y_pred = model.predict_classes(test_x)

        test_num=self.metadata['test_num']
        class_num=self.metadata['class_num']
        y_test = np.zeros([test_num, class_num])
        for idx, y in enumerate(y_pred):
            y_test[idx][y] = 1

        return y_test

