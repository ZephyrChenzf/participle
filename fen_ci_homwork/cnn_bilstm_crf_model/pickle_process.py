#coding=utf-8
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import pickle
max_sen_len=100
f=open('./data/train.txt','r')
X_train=[]#总X训练集
Y_train=[]#总Y训练集
x_sen=[]#存一句话的id组
y_sen=[]#存一句话的标签id组
dict_word={}#词字典，带有id
dict_tag={}#标签字典，带有id
dict_word['PAD']=0
dict_word['UNK']=1
dict_tag['PAD']=0
for line in f:
    line=line.strip()
    if (line=="" or line=="\n" or line=="\r\n"):#一句话结束了X_train和Y_train加一次
        X_train.append((x_sen))
        x_sen=[]
        Y_train.append((y_sen))
        y_sen=[]
        continue
    line=line.split(' ')
    if (len(line)<2):
        continue
    if line[0] in dict_word:#如果在词典中有，将id给x_sen
        x_sen.append(dict_word[line[0]])
    else:#若没有，为词新建id并给x_sen
        index=len(dict_word)
        dict_word[line[0]]=index
        x_sen.append(index)
    # if(len(line)<2):
    #     print line
    if line[1] in dict_tag:#同理,注意不同标签对应的id与初始碰到的标签有关
        y_sen.append(dict_tag[line[1]])
    else:
        index=len(dict_tag)
        dict_tag[line[1]]=index
        y_sen.append(index)
index_word={}#与词典反过来，id对应字
for word in dict_word:
    index_word[dict_word[word]]=word

X_train_cut=[]
Y_train_cut=[]
for i in range(len(X_train)):#每句话
    if len(X_train[i])<=max_sen_len:#如果句子长度小于max_sen_len
        X_train_cut.append(X_train[i])
        Y_train_cut.append(Y_train[i])
        continue
    while len(X_train[i])>max_sen_len:#超过100，使用标点符号拆分句子，将前面部分加入训练集，若后面部分仍超过100，继续拆分
        flag=False
        for j in reversed(range(max_sen_len)):#反向访问，99、98、97...
            if X_train[i][j]==dict_word['，'] or X_train[i][j]==dict_word['、']:
                X_train_cut.append(X_train[i][:j+1])
                Y_train_cut.append(Y_train[i][:j+1])
                X_train[i]=X_train[i][j+1:]
                Y_train[i]=Y_train[i][j+1:]
                break
            if j==0:
                flag=True
        if flag:
            X_train_cut.append(X_train[i][:max_sen_len])
            Y_train_cut.append(Y_train[i][:max_sen_len])
            X_train[i]=X_train[i][max_sen_len:]
            Y_train[i]=Y_train[i][max_sen_len:]
    if len(X_train[i])<=max_sen_len:#如果句子长度小于max_sen_len
        X_train_cut.append(X_train[i])
        Y_train_cut.append(Y_train[i])
        #continue
X_train=pad_sequences(X_train_cut,maxlen=max_sen_len,padding='post')
print(X_train.shape)
Y_train=pad_sequences(Y_train_cut,maxlen=max_sen_len,padding='post')
print(Y_train.shape)

index_tag={}
for tag in dict_tag:
    index_tag[dict_tag[tag]]=tag

pickle.dump(index_word,open('./data/index_word.pkl','wb'),2)
pickle.dump(index_tag,open('./data/index_tag.pkl','wb'),2)
pickle.dump(dict_word,open('./data/dict_word.pkl','wb'),2)
pickle.dump(dict_tag,open('./data/dict_tag.pkl','wb'),2)
pickle.dump(X_train,open('./data/X_train.pkl','wb'),2)
pickle.dump(Y_train,open('./data/Y_train.pkl','wb'),2)
