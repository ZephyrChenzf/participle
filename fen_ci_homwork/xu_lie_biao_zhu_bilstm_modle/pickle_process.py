#coding=utf-8
import numpy as np
import pickle
import pandas as pd
import re
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
# 最大句子长度
maxlen = 32
s = open('./data/train.txt').read().decode('utf-8')
s = s.split('\r\n')


# print(s[3])
def clean(s):  # 整理一下数据，有些不规范的地方
    if u'“/s' not in s:
        return s.replace(u' ”/s', '')
    elif u'”/s' not in s:
        return s.replace(u'“/s ', '')
    elif u'‘/s' not in s:
        return s.replace(u' ’/s', '')
    elif u'’/s' not in s:
        return s.replace(u'‘/s ', '')
    else:
        return s


s = u''.join(map(clean, s))
s = re.split(u'[，。！？、]/[bems]', s)  # 按标点符号进行切分

data = []  # 生成训练样本
label = []


def get_xy(s):  # 将字与标注分离
    s = re.findall('(.)/(.)', s)
    if s:
        s = np.array(s)
        return list(s[:, 0]), list(s[:, 1])


a, b = get_xy(s[0])
# print a[0].encode('utf8').decode('utf8')
for i in s:
    x = get_xy(i)
    if x:
        data.append(x[0])
        label.append(x[1])

d = pd.DataFrame(index=range(len(data)))
d['data'] = data
d['label'] = label
#print d['label']
d = d[d['data'].apply(len) <= maxlen]
d.index = range(len(d))
#print d['data']
tag = pd.Series({'x': 0, 's': 1, 'b': 2, 'm': 3, 'e': 4})
chars = []  # 统计所有字，跟每个字编号
for i in data:
    chars.extend(i)

chars = pd.Series(chars).value_counts()

chars[:] = range(1, len(chars) + 1)

# 将数据整理成适合输入的格式
X_train = []
for line in d['data']:
    X_line = []
    for word in line:
        X_line.append(chars[word])
    X_train.append(X_line)
# X_train=pad_sequences(X_train,maxlen=maxlen,padding='post')
# print X_train
# print chars
Y_train = []
for line in d['label']:
    Y_line = []
    for word in line:
        Y_line.append(tag[word])
    Y_train.append(Y_line)

X_train = pad_sequences(X_train, maxlen=maxlen, padding='post')
Y_train = pad_sequences(Y_train, maxlen=maxlen, padding='post')
Y_train = np_utils.to_categorical(Y_train, len(tag))
#print d['data']
#pickle.dump(d['data'],open('./data/data.pkl','wb'),2)
#pickle.dump(d['lable'],open('./data/label.pkl','wb'),2)
pickle.dump(chars,open('./data/chars.pkl','wb'),2)
pickle.dump(tag,open('./data/tag.pkl','wb'),2)
pickle.dump(X_train,open('./data/X_train.pkl','wb'),2)
pickle.dump(Y_train,open('./data/Y_train.pkl','wb'),2)