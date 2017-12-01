# -*- coding: utf-8 -*-

#from __future__ import print_function, unicode_literals
import numpy as np
np.random.seed(1337)
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from six.moves import zip
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Input, merge
from keras.layers import GRU, Dense, Embedding, ChainCRF, LSTM, Bidirectional, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Conv1D,ZeroPadding1D
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.utils.data_utils import get_file
from keras.callbacks import Callback
from subprocess import Popen, PIPE, STDOUT
import data
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
# 指定第一块GPU可用
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)

KTF.set_session(sess)

nb_word = len(data.index_word)
nb_tag = len(data.index_tag)
maxlen = 100
word_embedding_dim = 50
lstm_dim = 50
batch_size = 64


word_input = Input(shape=(maxlen,), dtype='int32', name='word_input')
word_emb = Embedding(nb_word, word_embedding_dim, input_length=maxlen, dropout=0.2, name='word_emb')(word_input)
bilstm = Bidirectional(LSTM(lstm_dim, dropout_W=0.1, dropout_U=0.1, return_sequences=True))(word_emb)
bilstm_d = Dropout(0.1)(bilstm)

half_window_size=2

paddinglayer=ZeroPadding1D(padding=half_window_size)(word_emb)
conv=Conv1D(nb_filter=50,filter_length=(2*half_window_size+1),border_mode='valid')(paddinglayer)
conv_d = Dropout(0.1)(conv)
dense_conv = TimeDistributed(Dense(50))(conv_d)
rnn_cnn_merge=merge([bilstm_d,dense_conv], mode='concat', concat_axis=2)


dense = TimeDistributed(Dense(nb_tag))(rnn_cnn_merge)
crf = ChainCRF()
crf_output = crf(dense)

model = Model(input=[word_input], output=[crf_output])

model.compile(loss=crf.sparse_loss,
              optimizer=RMSprop(0.0001),
              metrics=['sparse_categorical_accuracy'])

model.load_weights('./model/weights.50.hdf5')

data.make_test_normal('./data/beida_to_test.txt','./data/beida_to_test_normal.txt')#将测试集转化为标准数据集文件
X_test_cut,X_test_len,X_test_word,X_test_fenge=data.get_X('./data/beida_to_test_normal.txt')#转化test为标准格式

Y_pred = model.predict(X_test_cut)
# import data
# for i1 in range(Y_pred.shape[0]):
#     for i2 in range(Y_pred.shape[1]):
#         tags=Y_pred[i1][i2]
#         tag=0
#         for i3 in range(Y_pred.shape[2]):
#             if(tags[i3]==1):
#                 tag=i3
#         tag=1
#         print(data.index_tag[tag])
data.write_result_to_file('./data/beida_fenci.txt',Y_pred,X_test_len,X_test_word,X_test_fenge)