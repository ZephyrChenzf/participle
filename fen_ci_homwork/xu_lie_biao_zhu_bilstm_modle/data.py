# -*- coding: utf-8 -*-
import numpy as np
import pickle
X_train=pickle.load(open("./data/X_train.pkl",'rb'))
Y_train=pickle.load(open("./data/Y_train.pkl",'rb'))
#data=pickle.load(open("./data/data.pkl",'rb'))
#label=pickle.load(open("./data/label.pkl",'rb'))
chars=pickle.load(open("./data/chars.pkl",'rb'))
tag=pickle.load(open("./data/tag.pkl",'rb'))