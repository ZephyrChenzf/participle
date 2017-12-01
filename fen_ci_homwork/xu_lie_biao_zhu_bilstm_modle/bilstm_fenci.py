#-*- coding:utf-8 -*-
import re
import numpy as np
import pandas as pd
import os
import data
# import sys
# reload(sys)
# sys.setdefaultencoding("gbk")
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.preprocessing.sequence import pad_sequences
from sklearn.cross_validation import train_test_split
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.utils import np_utils
# 指定第一块GPU可用
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)

KTF.set_session(sess)

X_train = data.X_train
Y_train = data.Y_train
chars=data.chars
tag=data.tag

# 最大句子长度
maxlen = 32
if(os.path.exists('./model/my_model.h5')==False):
    #生成适合模型输入的格式
    from keras.utils import np_utils
    #填充
    # d['x'] = d['data'].apply(lambda x: np.array(list(chars[x])+[0]*(maxlen-len(x))))
    # d['y'] = d['label'].apply(lambda x: np.array(map(lambda y:np_utils.to_categorical(y,5), tag[x].reshape((-1,1)))+[np.array([[0,0,0,0,1]])]*(maxlen-len(x))))
    X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)
    print('X_train shape:', X_train.shape)
    print('Y_train shape:', Y_train.shape)
    #设计模型
    word_size=64
    maxlen=32
    n_epoch=50
    from keras.layers import Dense,Embedding,LSTM,TimeDistributed,Input,Bidirectional
    from keras.models import Model

    sequence=Input(shape=(maxlen,),dtype='int32')
    embedded=Embedding(len(chars)+1,word_size,input_length=maxlen,mask_zero=True)(sequence)
    blstm=Bidirectional(LSTM(64,return_sequences=True),merge_mode='sum')(embedded)
    output=TimeDistributed(Dense(5,activation='softmax'))(blstm)
    model=Model(input=sequence,output=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    checkpoint = ModelCheckpoint('model/weights.{epoch:02d}.hdf5', monitor='val_loss', verbose=1, save_best_only=True,
                                 mode='auto')
    earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
    batch_size=1024
    #history = model.fit(np.array(list(d['x'])), np.array(list(d['y'])).reshape((-1,maxlen,5)), batch_size=batch_size, nb_epoch=50)
    history = model.fit(X_train, Y_train, batch_size=batch_size,nb_epoch=n_epoch,callbacks=[checkpoint,earlyStopping],validation_data=(X_test, Y_test))
    model.save('./model/my_model.h5')

from keras.models import load_model
model=load_model('./model/my_model.h5')

#转移概率，单纯用了等概率
zy = {'be':0.5,
      'bm':0.5,
      'eb':0.5,
      'es':0.5,
      'me':0.5,
      'mm':0.5,
      'sb':0.5,
      'ss':0.5
     }

zy = {i:np.log(zy[i]) for i in zy.keys()}

def viterbi(nodes):
    paths = {'b':nodes[0]['b'], 's':nodes[0]['s']}
    for l in range(1,len(nodes)):
        paths_ = paths.copy()
        paths = {}
        for i in nodes[l].keys():
            nows = {}
            for j in paths_.keys():
                if j[-1]+i in zy.keys():
                    nows[j+i]= paths_[j]+nodes[l][i]+zy[j[-1]+i]
            k = np.argmax(nows.values())
            paths[nows.keys()[k]] = nows.values()[k]
    return paths.keys()[np.argmax(paths.values())]

def simple_cut(s):
    if s:
        #选择未填补部分
        # 将数据整理成适合输入的格式
        X_test = []
        X_line=[]
        for word in s:
            X_line.append(chars[word])
        #print X_test
        X_test.append(X_line)
        X_test = pad_sequences(X_test, maxlen=maxlen, padding='post')
        #r = model.predict(np.array([list(chars[list(s)].fillna(0).astype(int))+[0]*(maxlen-len(s))]), verbose=False)[0][:len(s)]
        r = model.predict(X_test, verbose=False)[0][:len(s)]
        r = np.log(r)
        nodes = [dict(zip(['s','b','m','e'], i[1:5])) for i in r]
        #返回维特比最大可能路径（序列）
        t = viterbi(nodes)
        #print "vbt return:"+t
        words = []
        for i in range(len(s)):
            # print t
            #print s
            if t[i] in ['s', 'b']:
                words.append(s[i])
            else:
                words[-1] += s[i]
        # word_ss=""
        # fenge=" "
        # for w in words:
        #     word_ss=word_ss+w+fenge
        # print word_ss.encode('utf-8').decode('utf-8')
        return words
    else:
        return []

not_cuts = re.compile(u'([\da-zA-Z ]+)|[。；，、？！\.\?,!/]')
def cut_word(s):
    result = []
    j = 0
    for i in not_cuts.finditer(s):
        if(i.start()-j)<=maxlen:#如果分割的单句小于32
            result.extend(simple_cut(s[j:i.start()]))
        else:
            result.extend(simple_cut(s[j:(j+maxlen)]))
            if(i.start()-(j+maxlen))<=maxlen:#再次判定
                result.extend(simple_cut(s[(j+maxlen):i.start()]))
            else:
                result.extend(simple_cut(s[(j+maxlen):(j+maxlen+maxlen)]))
                result.extend(simple_cut(s[(j+maxlen+maxlen):i.start()]))
        result.append(s[i.start():i.end()])
        j = i.end()
    if(len(s)-j)<=maxlen:#如果最后分割的单句小于32
        result.extend(simple_cut(s[j:]))
    else:
        result.extend(simple_cut(s[j:(j+maxlen)]))
        if (len(s) - (j + maxlen)) <= maxlen:  # 再次判定
            result.extend(simple_cut(s[(j + maxlen):]))
        else:
            result.extend(simple_cut(s[(j + maxlen):(j + maxlen + maxlen)]))
            result.extend(simple_cut(s[(j + maxlen + maxlen):]))
    return result

f=open('data/testset.txt').read().decode('gbk')
out_f=open('data/2017140437.txt','w')
#print f
sentences=f.split('\n')
rss = ""
for sentence in sentences:
    sentence=sentence.strip()
    result = cut_word(sentence)
    fenge = " "
    for word in result:
        rss = rss + word + fenge
    rss=rss+'\n'
out_f.write(rss.encode('gbk'))

#使用北大的数据集分词
# f=open('data/beida_to_test.txt').read().decode('utf-8')
# out_f=open('data/beida_fenci.txt','w')
# #print f
# sentences=f.split('\n')
# rss = ""
# for sentence in sentences:
#     sentence=sentence.strip()
#     result = cut_word(sentence)
#     fenge = " "
#     for word in result:
#         rss = rss + word + fenge
#     rss=rss+'\n'
# out_f.write(rss.encode('utf-8'))