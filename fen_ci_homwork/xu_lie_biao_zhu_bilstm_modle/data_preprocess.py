# -*- coding: utf-8 -*-
#from __future__ import print_function
import numpy as np
import re
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

f=open(u'data/北大(人民日报)语料库199801.txt').read().decode('utf-8')
all_s=f.split('\r\n')
#分离单个词为分词标注
f_w=open('data/train.txt','w')
for s in all_s:
    if (s=='' or s=='\n' or s=='\r\n' or len(s)==0):
        pass
    else:
        s=s.strip()
        #print s
        s=s.split('  ')[1:]
        s_t=""
        for word in s:#一句话里的每个词
            word=re.findall('(.*)/[a-zA-Z]{1,5}',word)[0]
            #print word
            if len(word)>=2:
                #print word[0].encode('utf-8').decode('utf-8')
                for i,single_word in enumerate(word):#每个词里的每个字
                    if(single_word!='/'):#去掉多余的'/'
                        #print single_word
                        if(i==0):
                            s_t=s_t+single_word+'/b'+'  '
                        elif(i==len(word)-1):
                            s_t=s_t+single_word+'/e'+'  '
                        else:
                            s_t=s_t+single_word+'/m'+'  '
            else:
                if(word!='/'):
                    s_t=s_t+word+'/s'+'  '
        f_w.write(s_t+'\n')

#将北大的语料训练数据转化为测试集beida_to_test.txt,与答案集beida_to_test_answer.txt
#答案集beida_to_test_answer.txt
f=open(u'data/北大(人民日报)语料库199801.txt').read().decode('utf-8')
all_s=f.split('\r\n')
#分离单个词为分词标注
f_w=open('data/beida_to_test_answer.txt','w')
for s in all_s:
    if (s=='' or s=='\n' or s=='\r\n' or len(s)==0):
        pass
    else:
        s=s.strip()
        #print s
        s=s.split('  ')[1:]
        s_t=""
        for word in s:#一句话里的每个词
            word=re.findall('(.*)/[a-zA-Z]{1,5}',word)[0]
            s_t=s_t+word+' '
        f_w.write(s_t+'\n')

#测试集beida_to_test.txt
import codecs
f=codecs.open('data/beida_to_test_answer.txt','r','utf-8').read()
all_s=f.split('\r\n')
f_w=open('data/beida_to_test.txt','w')
for s in all_s:
    if (s=='' or s=='\n' or s=='\r\n' or len(s)==0):
        pass
    else:
        s=s.strip()
        s=s.replace(' ','')
        f_w.write(s.encode('utf-8')+'\n')
