# -*- coding: utf-8 -*-
import numpy as np

#读取北大数据使用模型得出的预测集
predict=open('./data/beida_fenci.txt','r').read().decode('utf-8')
#读取北大数据答案集
answer=open('./data/beida_to_test_answer.txt','r').read().decode('utf-8')

p_s=predict.split('\n')[1]
a_s=answer.split('\n')[1]
print p_s
print a_s

p_s_num={}
p_s_no_blank=p_s.replace(' ','')
for i,word in enumerate(p_s_no_blank):
    p_s_num[word]=i
print p_s_num[u'中']
print len(p_s_num)