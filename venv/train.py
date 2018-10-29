
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
np.random.seed(167643)


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


#распаковка файлов с выборкой
y, X = unpack_file(labels=varss['y_train_dir'], dataset=varss['x_train_dir'])
X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])


def StratifiedSplit(X, y, rate=0.2):
    unq, coun = np.unique(y, return_counts=True)
    num = X.shape[0]
    inds = [i for i in range(num)]
    max_test = (coun*rate).round().astype(int)
    test = []
    for i in unq:
        j=0
        while True:
            ind=random.choice(inds)
            if y[ind] == i:
                j += 1
                inds.remove(ind)
                test.append(ind)
            if j >= max_test[i]:
                break
    np.random.shuffle(test)
    X_test, y_test = X[test], y[test]
    tr = [i for i in range(num)]
    train = [x for x in tr if x not in test]
    np.random.shuffle(train)
    X_train, y_train = X[train], y[train]
    return X_train, y_train, X_test, y_test


#стратифицированное разбиение на трэйн и валидэйт
X_train, y_train, X_val, y_val = StratifiedSplit(X, y)

X_train_mean = X_train.mean(axis=0)
X_train_std = X_train.std(axis=0)

def Standardizer(X, mean, std, eps=1e-6):
    return (X-mean)/(std+eps)

#стандартизация
X_train = Standardizer(X_train, X_train_mean, X_train_std)
X_val = Standardizer(X_val, X_train_mean, X_train_std)


def tSVD(X_train):
    cov = np.dot(X_train.T, X_train) / X_train.shape[0]
    U,S,V = np.linalg.svd(cov)
    i=0
    while i<S.shape[0] and abs(S[i]-S[i+1])>=0.001:
        i += 1
    num_comp = i+1
    X_train = np.dot(X_train, U[:, :num_comp])
    return X_train, U, num_comp


#выделение главных компонент
#остановка по критерию каменистой осыпи - считаем, что вышли на плато, когда разница между соседними числами меньше 0.001
X_train, U, num_comp = tSVD(X_train)
X_val = np.dot(X_val, U[:, :num_comp])


X_train_mean2 = X_train.mean(axis=0)
X_train_std2 = X_train.std(axis=0)


#стандартизация после выделения главных компонент
X_train = Standardizer(X_train, X_train_mean2, X_train_std2)
X_val = Standardizer(X_val, X_train_mean2, X_train_std2)


def OneHot(y, num_classes):
    ar = np.zeros((y.shape[0], num_classes))
    ar[np.arange(y.shape[0]), y] = 1
    return ar


#кодирование выходной переменной
num_classes = max(y_train)+1
y_trainb = OneHot(y_train, num_classes)
y_valb = OneHot(y_val, num_classes)


#инициализация весов по аналогии с методом инициализации Ксавьера в нейронных сетях
def init_weights(features_num, num_labels):
    return [np.random.normal(loc=0, scale = np.sqrt(2/(features_num+num_labels))) for _ in range(features_num)]


#вычисление софтмакса
def Softmax(W, x):
    #для достижения численной стабильности пользуюсь свойством softmax(x)=softmax(x+c)
    power = np.dot(W, x) - np.max(np.dot(W, x))
    exps = np.exp(power)
    sfmx = exps / np.sum(exps)
    sfmx.astype(np.float64)
    return sfmx


#вычисление функции потерь
def CrossEntropy(y_true, y_pred, x):
    CE =- np.sum(y_true*np.log(y_pred+1e-10))
    grad = np.outer(y_pred-y_true, x)
    return CE, grad


#проверка достоверности предсказаний
def accuracy(W, val_X, val_Y):
    a = 0.
    for i in range(val_X.shape[0]):
        y_pred = Softmax(W, val_X[i])
        cls_pred = np.argmax(y_pred)
        if(cls_pred == np.argmax(val_Y[i])):
            a += 1
    return a / val_X.shape[0]



#стохастический градиентный спуск для софтмакс
def SGD(lr, num_iter, X_train, y_trainb, X_val, y_valb):
    W = np.array([init_weights(num_comp, num_classes) for _ in range(num_classes)], dtype=np.float64)
    for _ in range(num_iter):
        Loss = 0.
        mix = list(range(X_train.shape[0]))
        np.random.shuffle(mix)
        for i in range(X_train.shape[0]):
            ind = mix[i]
            x = X_train[ind]
            y = y_trainb[ind]
            y_pred = Softmax(W, x)
            loss, grad = CrossEntropy(y, y_pred, x)
            Loss = Loss+loss
            W -= lr*grad
    fin_acc = accuracy(W, X_val, y_valb)
    return W, fin_acc


#сетка параметра
learning_rates = [0.1, 0.01, 0.001, 0.0001]
num_iter = 10


#подбор оптимальной скорости обучения, смотрим на достоверность на валидационном множестве
accs=[]
for lr in learning_rates:
    W, acc = SGD(lr, num_iter, X_train, y_trainb, X_val, y_valb)
    #print('LR: ', lr, ' Accuracy: ', acc)
    accs.append(acc)


#сохраняем лучший результат
best_lr = learning_rates[np.argmax(accs)]
best_W, _ = SGD(best_lr, num_iter, X_train, y_trainb, X_val, y_valb)
#print(best_W)


#словарь параметров всей модели в целом
params={'classes': num_classes,'mean1': X_train_mean, 'std1': X_train_std, 'components': U,\
        'num_compomemts': num_comp,  'mean2': X_train_mean2, 'std2': X_train_std2, 'W': best_W}


#сохранение модели
with open(varss['model_output_dir'], 'wb') as f:
    pickle.dump(params, f)


#предсказания на обучающей выборке
preds = []
for i in range(X_train.shape[0]):
    y_pred = Softmax(params['W'], X_train[i])
    cls_pred = np.argmax(y_pred)
    preds.append(cls_pred)


print(classification_report(y_train, preds))

