# -*- coding: utf-8 -*-
import numpy as np
import re
import copy
import data
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

class Two_gram():
    def __init__(self):
        # 载入词频和两个词连接的频率
        self.word_count_dict = data.word_count_dict
        self.word_word_count_dict = data.word_word_count_dict
        # 统计所有词的总次数
        self.all_word_length = 0
        for word in self.word_count_dict:
            self.all_word_length = self.all_word_length + self.word_count_dict[word]

        # 加一平滑，即所有词数分别加1
        self.all_word_length = self.all_word_length + len(self.word_count_dict)

        # 统计每个词后继词的总个数,后面可以用来计算条件概率
        self.word_word_num = {}
        for word in self.word_word_count_dict:
            self.word_word_num[word] = 0
            for word_in in self.word_word_count_dict[word]:
                self.word_word_num[word] += self.word_word_count_dict[word][word_in]

    #获得所有可能的分词路径
    def get_all_paths(self,context, start=0, result=[]):
        #context=context.decode('utf-8')
        length = len(context)
        if start == length:
            self.all_paths.append(copy.deepcopy(result))
        else:
            flag = False
            if start + 6 <= length and context[start: start + 6] in self.word_count_dict:
                flag = True
                result.append(context[start: start + 6])
                self.get_all_paths(context, start + 6, result)
                result.pop()
                return

            if start + 5 <= length and context[start: start + 5] in self.word_count_dict:
                flag = True
                result.append(context[start: start + 5])
                self.get_all_paths(context, start + 5, result)
                result.pop()
                return

            if start + 4 <= length and context[start: start + 4] in self.word_count_dict:
                flag = True
                result.append(context[start: start + 4])
                self.get_all_paths(context, start + 4, result)
                result.pop()

            if start + 3 <= length and context[start: start + 3] in self.word_count_dict:
                flag = True
                result.append(context[start: start + 3])
                self.get_all_paths(context, start + 3, result)
                result.pop()

            if start + 2 <= length and context[start: start + 2] in self.word_count_dict:
                flag = True
                result.append(context[start: start + 2])
                self.get_all_paths(context, start + 2, result)
                result.pop()

            if context[start] in self.word_count_dict:
                flag = True
                result.append(context[start: start + 1])
                self.get_all_paths(context, start + 1, result)
                result.pop()

            if flag == False:
                result.append(context[start: start + 1])
                self.get_all_paths(context, start + 1, result)
                result.pop()


    #预处理，去掉每段的空字符
    def preprocess(self, context):
        tmp = []
        for line in context:
            line=line.strip()
            line=line.replace(' ','')
            if len(line) != 0:
                tmp.append(line)
        return tmp

    #获得分词结果所得到的概率
    def get_possibility(self,result):
        p = 1.0
        length = len(result)
        for index in range(length):
            if index == 0:#如果是第一个，直接计算词频概率
                if result[index] in self.word_count_dict:
                    p = p * (float(self.word_count_dict[result[index]] + 1) / self.all_word_length)
                else:
                    p = p * (float(1) / self.all_word_length)
            else:#计算两词连接占单词出现的概率
                if result[index - 1] in self.word_word_count_dict and result[index] in self.word_word_count_dict[result[index - 1]]:#出现在双词连接表中计算其概率
                    p = p * (float(self.word_word_count_dict[result[index - 1]][result[index]] + 1) /
                             self.word_word_num[result[index - 1]])
                elif result[index - 1] in self.word_word_count_dict:#有单词则取1
                    p = p * (float(1) / self.word_word_num[result[index - 1]])
                else:
                    p = p * pow(float(0.1), 10)#如果两个词都未出现在词表中，则此路径乘以一个极小概率
        return p

    #对每段进行分词
    def simple_cut(self, context):
        self.get_all_paths(context)
        Max = 0
        Max_result=[]
        for result in self.all_paths:
            p = self.get_possibility(result)
            if p > Max:
                Max = p
                Max_result = result
        self.all_paths=[]
        return Max_result

    #将一段话进行拆分
    def cut(self,context):
        self.result = []
        self.all_paths = []
        not_cuts=re.compile(u'：|-|/|【|？|】|\?|。|，|\.|、|《|》| |（|）|”|“|；|\n|\r\n')#存储标点符号
        biaodian=[]
        for i in not_cuts.finditer(context):
            biaodian.append(context[i.start():i.end()])
        context = re.split(u'：|-|/|【|？|】|\?|。|，|\.|、|《|》| |（|）|”|“|；|\n|\r\n', context)
        # print (context)
        context = self.preprocess(context)
        for i,line in enumerate(context):
            if i<len(biaodian):#防止末尾没标点越界
                self.result = self.result + self.simple_cut(line)+[biaodian[i]]
            else:
                self.result = self.result + self.simple_cut(line)
        return self.result


#开始分词
tang = Two_gram()
# f=open('data/testset.txt','r').read().decode('gbk')
# out_f=open('data/2017140437.txt','w')
f=open('data/testset.txt','r').read().decode('gbk')
out_f=open('data/2017140437.txt','w')
#string = u"本报北京１２月３１日讯新华社记者陈雁、本报记者何加正报道：在度过了非凡而辉煌的１９９７年，迈向充满希望的１９９８年之际，’９８北京新年音乐会今晚在人民大会堂举行。党和国家领导人江泽民、李鹏、乔石、朱镕基、李瑞环、刘华清、尉健行、李岚清与万名首都各界群众和劳动模范代表一起，在激昂奋进的音乐声中辞旧迎 新"
sentences=f.split('\n')
result=""
for sentence in sentences:
    res = tang.cut(sentence)
    for word in res:
        result+=word.encode('gbk').decode('gbk')+' '
    result+='\n'
out_f.write(result.encode('gbk'))