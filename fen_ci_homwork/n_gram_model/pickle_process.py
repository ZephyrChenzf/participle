# -*- coding: utf-8 -*-
import numpy as np
import pickle

f=open('./data/train.txt').read().decode('utf-8')
all_s=f.split('\r\n')
word_count_dict={}#统计词频
word_word_count_dict={}#统计两个词连接的频率

for s in all_s:
    s=s.split(' ')
    for i,word in enumerate(s):
        if word in word_count_dict:#统计词频
            word_count_dict[word]+=1
        else:
            word_count_dict[word]=1
        if i==len(s)-1:#为了不使序号达到length
            break
        if word in word_word_count_dict:#统计两个词连接的频率
            if(s[i+1] in word_word_count_dict[word]):
                word_word_count_dict[s[i]][s[i+1]]+=1
            else:
                word_word_count_dict[s[i]][s[i+1]]=1
        else:
            word_word_count_dict[s[i]]={s[i+1]:1}

print ("LEN :", len(word_count_dict))
print ("LEN :", len(word_word_count_dict))

pickle.dump(word_count_dict,open('./data/word_count_dict.pkl','wb'),2)
pickle.dump(word_word_count_dict,open('./data/word_word_count_dict.pkl','wb'),2)
