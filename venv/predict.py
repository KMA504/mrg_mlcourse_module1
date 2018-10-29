
# coding: utf-8

from pyunpack import Archive
import glob
import os
import struct
import numpy as np
import random
import math
from sklearn.metrics import classification_report
import pickle
import pdb
import sys


#парсинг опций
varss = {}
args = sys.argv
for i in args[1:]:
    arg_name = i[1:(i.find('='))]
    arg_value = i[(i.find('='))+1:]
    varss[arg_name] = arg_value



#способ распаковки посмотрел тут - https://gist.github.com/akesling/5358964
def unpack_labels(path):
    with open(path, 'rb') as f1:
        _, num = struct.unpack(">II", f1.read(8))
        cls = np.fromfile(f1, dtype=np.int8)
    return cls, num



def unpack_set(path, obs):
    with open(path, 'rb') as f2:
        _, num, num_r, num_c = struct.unpack(">IIII", f2.read(16))
        img = np.fromfile(f2, dtype=np.uint8).reshape(obs, num_r, num_c)
    return img


def extraction(file):
    Archive(file).extractall('')
    list_of_files = glob.glob('*')
    latest_file = max(list_of_files, key=os.path.getctime)
    path = ''
    fin_path = os.path.join(path, latest_file)
    return fin_path


def unpack_file(labels, dataset):
    lab_path = extraction(labels)
    dat_path = extraction(dataset)
    labs, len_labs = unpack_labels(lab_path)
    data = unpack_set(dat_path, len_labs)
    return labs, data

#загрузка модели
with open(varss['model_input_dir'], 'rb') as ff:
    model_params = pickle.load(ff)

#загрузка выборки
y_test, X_test = unpack_file(labels=varss['y_test_dir'], dataset=varss['x_test_dir'])


def OneHot(y, num_classes):
    ar = np.zeros((y.shape[0], num_classes))
    ar[np.arange(y.shape[0]), y] = 1
    return ar

#кодирование выходной переменной
y_testb = OneHot(y_test, model_params['classes'])

X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])

def Standardizer(X, mean, std, eps=1e-6):
    return (X-mean)/(std+eps)

#стандартизация перед выделением главных компонент
X_test = Standardizer(X_test, model_params['mean1'], model_params['std1'])

#выделение главных компонент
X_test = np.dot(X_test, model_params['components'][:, :model_params['num_compomemts']])

#стандартизация перед прогнозированием
X_test = Standardizer(X_test, model_params['mean2'], model_params['std2'])


#вычисление софтмакса
def Softmax(W, x):
    #для достижения численной стабильности пользуюсь свойством softmax(x)=softmax(x+c)
    power = np.dot(W, x) - np.max(np.dot(W, x))
    exps = np.exp(power)
    sfmx = exps / np.sum(exps)
    sfmx.astype(np.float64)
    return sfmx


#предсказание
#print(model_params['W'])
preds=[]
for i in range(X_test.shape[0]):
    y_pred = Softmax(model_params['W'], X_test[i])
    cls_pred = np.argmax(y_pred)
    preds.append(cls_pred)


print(classification_report(y_test, preds))

