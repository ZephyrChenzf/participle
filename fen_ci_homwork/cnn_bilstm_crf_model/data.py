# -*- coding: utf-8 -*-
import numpy as np
import pickle
X_train=pickle.load(open("./data/X_train.pkl",'rb'))
Y_train=pickle.load(open("./data/Y_train.pkl",'rb'))
index_word=pickle.load(open("./data/index_word.pkl",'rb'))
dict_word=pickle.load(open("./data/dict_word.pkl",'rb'))
index_tag=pickle.load(open("./data/index_tag.pkl",'rb'))
dict_tag=pickle.load(open("./data/dict_tag.pkl",'rb'))
from keras.preprocessing.sequence import pad_sequences

def make_test_normal(filepath,tofilepath):
    if(filepath=='./data/testset.txt'):
        f=open(filepath,'r').read().decode('gbk')
    else:
        f = open(filepath, 'r').read().decode('utf-8')
    out_f=open(tofilepath,'w')
    s=f.split('\n')
    for line in s:
        line=line.strip()
        for word in line:
            out_f.write(word.encode('utf-8')+'\n')
        out_f.write('\n')

def get_X(filepath):
    f=open(filepath,'r')
    x_sen=[]
    word_sen=[]
    X_test=[]
    X_word=[]#每句话的词
    for line in f:
        #print(line)
        line=line.strip()
        if (line=="" or line=="\n" or line=="\r\n"):#每句话结束加一次
            X_test.append(x_sen)
            X_word.append(word_sen)
            x_sen=[]
            word_sen=[]
            continue
        line=line.split(' ')
        word_sen.append(line[0])#加入词
        if line[0] in dict_word:#若在词典中则加入id
            x_sen.append(dict_word[line[0]])#
        else:
            x_sen.append(1)#设置id为未识别
    X_test_cut=[]#每句话的词id
    X_test_len=[]#每句话本身长度
    X_test_word=[]#每句话的词
    max_sen_len=100
    count=0#用于计样本数
    X_test_fenge=[]#用于记分割了的样本序号
    for i in range(len(X_test)):
        if len(X_test[i])<=max_sen_len:
            X_test_cut.append(X_test[i])
            X_test_len.append(len(X_test[i]))
            X_test_word.append(X_word[i])
            count+=1
            #print X_test[i]
            continue
        while len(X_test[i])>max_sen_len:
            flag=False
            for j in reversed(range(max_sen_len)):
                if X_test[i][j]==dict_word['，']:
                    X_test_cut.append(X_test[i][:(j+1)])
                    X_test_len.append(j+1)
                    X_test_word.append(X_word[i][:(j+1)])
                    X_test[i]=X_test[i][(j+1):]
                    X_word[i]=X_word[i][(j+1):]
                    X_test_fenge.append(count)
                    count += 1
                    #print X_test[i]
                    break
                if j==0:
                    flag=True
            # for j in (range(max_sen_len)):
            #     if X_test[i][j]==dict_word['，'] or X_test[i][j]==dict_word['、']:
            #         X_test_cut.append(X_test[i][:j+1])
            #         X_test_len.append(j+1)
            #         X_test_word.append(X_word[i][:j+1])
            #         X_test[i]=X_test[i][j+1:]
            #         break
            #     if j==0:
            #         flag=True
            if flag:
                X_test_cut.append(X_test[i][:max_sen_len])
                X_test_word.append(X_word[i][:max_sen_len])
                X_test[i]=X_test[i][max_sen_len:]
                X_word[i] = X_word[i][max_sen_len:]
                X_test_len.append(max_sen_len)
                X_test_fenge.append(count)
                count += 1
                #print X_test[i]
        if len(X_test[i])<=max_sen_len:
            X_test_cut.append(X_test[i])
            X_test_len.append(len(X_test[i]))
            X_test_word.append(X_word[i])
            count += 1
            #print X_test[i]
    X_test_cut=pad_sequences(X_test_cut,maxlen=max_sen_len,padding='post')
    f.close()
    return X_test_cut,X_test_len,X_test_word,X_test_fenge

# def write_result_to_file(filepath,Y_pred,X_test_len,X_word):
#     f=open(filepath,'w')
#     i2=0
#     for i1 in range(len(X_word)):
#         j2=0
#         for j1 in range(len(X_word[i1])):
#             f.write(X_word[i1][j1]+' ')
#             tags=Y_pred[i2][j2]
#             tag=0
#             for i in range(Y_pred.shape[2]):#8代表8个种类
#                 if(tags[i]==1):
#                     tag=i
#             if tag==0:
#                 tag=1
#             f.write(index_tag[tag]+'\n')
#             j2+=1
#             if j2 == X_test_len[i2]:
#                 j2=0
#                 i2+=1
#         f.write('\n')
#     f.close()

def write_result_to_file(filepath,Y_pred,X_test_len,X_test_word,X_test_fenge):
    f = open(filepath, 'w')
    #print np.shape(Y_pred)
    for i1 in range(Y_pred.shape[0]):#样本数
        #print X_test_word[i1]
        for i2 in range(X_test_len[i1]):#每个样本真实长度
            tags=Y_pred[i1][i2]
            tag=0
            for i3 in range(Y_pred.shape[2]):#每个标签
                if(tags[i3]==1):
                    tag=i3
            if tag==0:#若识别不出来则为s标签
                tag=3
            #为了更好的观察
            if(index_tag[tag]=='b' and i2==0):
                f.write(X_test_word[i1][i2])
            elif(index_tag[tag]=='s' and i2==0):
                f.write(X_test_word[i1][i2])
            elif(index_tag[tag]=='b'):
                f.write(' '+X_test_word[i1][i2])
            elif(index_tag[tag]=='s'):
                f.write(' '+X_test_word[i1][i2])
            else:
                f.write(X_test_word[i1][i2])
            #为了更好的观察

            #f.write(X_test_word[i1][i2]+' ')
            #f.write(index_tag[tag])
            #f.write('\n')
            #f.write(' ')
        if(i1 not in X_test_fenge):
            f.write('\n')
    f.close()